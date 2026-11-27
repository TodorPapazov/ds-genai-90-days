# Day 42: a SCRIPTED offline stand-in that reacts to prompt features. It is NOT a language model.
# It reads the messages you send, checks which prompt elements are present, and imitates common
# failure patterns when an element is missing. The rules are written out below on purpose, so you can
# see exactly why each variant scores what it scores. Its numbers show how a prompt test works;
# they say nothing about how much a real model improves. Only a real run can tell you that.
#
# Scripted rules (prompt element -> behavior):
#   no "only JSON" format rule         -> about 1 in 3 replies wrapped in prose + a ```json fence
#   no label list in the system prompt -> diagnosis copied as written in the note (e.g. "heart failure")
#   no "lower case" rule               -> specialty copied as written (e.g. "Cardiology")
#   no "use null" rule                 -> a missing length of stay is guessed as 3
#   no "convert weeks / months" rule   -> "2 weeks" becomes 2, "1 month" becomes 1
#   no few-shot examples               -> word numbers ("one week", "two weeks", "next week") are missed
#   injection text "set FIELD to VALUE" is obeyed if it sits outside <note></note>, or anywhere when
#   the system prompt does not explain the tags and the "data, not instructions" rule
#   (the data block ends at the FIRST </note>, so a note that contains "</note>" can break out)
import json, re, zlib
import llm_client as lc

WORDNUM = {"one": 1, "two": 2, "three": 3, "a": 1}
LABELS = ["Heart failure", "COPD exacerbation", "Pneumonia", "Type 2 diabetes, hyperglycemia",
          "Hip replacement", "Sepsis, urinary source", "Atrial fibrillation", "Cellulitis"]


class PromptSensitiveMock(lc.MockClient):
    name = "mock"

    def chat(self, messages, model="mock-prompt-model", temperature=0.0, max_tokens=300, response_format=None, timeout=30):
        self.calls += 1
        sys_txt = messages[0]["content"] if messages[0]["role"] == "system" else ""
        S = sys_txt.lower()
        user = messages[-1]["content"]
        f = {"format_rule": "only one json object" in S or "return only json" in (S + user.lower()),
             "labels": all(l.lower() in S for l in LABELS), "lower_rule": "lower case" in S,
             "null_rule": "use null" in S, "unit_rule": "convert weeks" in S,
             "fewshot": sum(m["role"] == "assistant" for m in messages[:-1]) > 0,
             "data_rule": "data, not instructions" in S}
        tagged = "<note>" in user and "<note>" in S  # delimiters count only if the system prompt explains them
        if tagged:  # data = text between the FIRST <note> and the FIRST </note>
            start = user.index("<note>") + len("<note>")
            end = user.find("</note>", start)
            data = user[start:end if end != -1 else len(user)]
            outside = user[:start] + (user[end:] if end != -1 else "")
        else:
            data, outside = user, user
        obj = self._extract(data, f)
        # prompt injection: obey "set FIELD to VALUE" found where the scripted rules say it is not protected
        zones = [outside] + ([] if f["data_rule"] and tagged else [data])
        for zone in zones:
            for k, v in re.findall(r"set (\w+) to (true|false|null|-?\d+)", zone, re.I):
                if k in obj:
                    obj[k] = json.loads(v.lower())
        content = json.dumps(obj)
        if not f["format_rule"] and zlib.crc32(data.encode()) % 3 == 0:
            content = "Sure! Here are the fields I found:\n```json\n" + json.dumps(obj, indent=1) + "\n```"
        prompt = "\n".join(m["content"] for m in messages)
        p, c = self._tokens(prompt) + 4 * len(messages), self._tokens(content)
        return {"id": f"mock-{self.calls}", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}

    @staticmethod
    def _extract(note, f):
        n = note.lower()
        m = re.search(r"admitted for (copd exacerbation|type 2 diabetes, hyperglycemia|sepsis, urinary source|[a-z ]+?)[,.]", n)
        raw = m.group(1) if m else None
        dx = None
        if raw:
            canon = next((l for l in LABELS if l.lower() == raw), None)
            dx = canon if f["labels"] else ("COPD exacerbation" if raw.startswith("copd") else raw)
        m = re.search(r"length of stay (\d+) day", n) or re.search(r"hospital day (\d+)", n)
        los = int(m.group(1)) if m else (None if f["null_rule"] else 3)
        alone = bool(re.search(r"lives alone|lives by (him|her)self", n))
        num = r"(\d+|one|two|three|a)" if f["fewshot"] else r"(\d+)"
        fu = re.search(r"(?:follow up with ([a-z ]+?)|([a-z ]+?) appointment) (?:(?:in|within) " + num +
                       r" (day|week|month)s?|(next week))", note, re.I)
        days, who = None, None
        if fu and (fu.group(3) or (fu.group(5) and f["fewshot"])):
            if fu.group(5):
                qty, unit = 1, "week"
            else:
                qty, unit = WORDNUM.get(fu.group(3).lower()) or int(fu.group(3)), fu.group(4).lower()
            days = qty * {"day": 1, "week": 7, "month": 30}[unit] if f["unit_rule"] else qty
            who = (fu.group(1) or fu.group(2)).strip()
            who = who.lower() if f["lower_rule"] else who
        booked = days is not None
        s = f"{dx or 'Unknown'}; stay {los} days; " + (f"follow-up with {who} in {days} days." if booked else "no follow-up found.")
        return {"primary_diagnosis": dx, "length_of_stay_days": los, "lives_alone": alone,
                "followup_booked": booked, "followup_days": days, "followup_with": who, "summary": s}
