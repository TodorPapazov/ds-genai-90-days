# Day 41: extract fields from SIMULATED discharge notes into JSON with an LLM API from Python.
#   python day41_extract.py --mock            offline, no key, all 30 notes (fake client, same response shape)
#   python day41_extract.py                   real API if LLM_API_KEY or XAI_API_KEY is set; first 3 notes only
#   python day41_extract.py --limit 30        real API, all 30 notes (about 30 small calls; see the cost line)
#   python day41_extract.py --mock --max-tokens 40   watch max_tokens cut the JSON (finish_reason = length)
# Writes extractions.jsonl, call_log.csv and prints accuracy vs the answer key (note_labels.csv).
# Data is simulated. Do not send real patient notes to an API without your employer's approval and a BAA.
import argparse, csv, json, math, time
import llm_client as lc

SCHEMA = {
    "type": "object",
    "properties": {
        "primary_diagnosis": {"type": ["string", "null"], "enum": [
            "Heart failure", "COPD exacerbation", "Pneumonia", "Type 2 diabetes, hyperglycemia",
            "Hip replacement", "Sepsis, urinary source", "Atrial fibrillation", "Cellulitis", None]},
        "length_of_stay_days": {"type": ["integer", "null"]},
        "lives_alone": {"type": "boolean"},
        "followup_booked": {"type": "boolean"},
        "followup_days": {"type": ["integer", "null"]},
        "followup_with": {"type": ["string", "null"]},
        "summary": {"type": "string"},
    },
    "required": ["primary_diagnosis", "length_of_stay_days", "lives_alone", "followup_booked",
                 "followup_days", "followup_with", "summary"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "discharge_fields", "strict": True, "schema": SCHEMA}}

SYSTEM = ("You extract fields from simulated hospital discharge notes for a care management team. "
          "Use only facts stated in the note. If a fact is missing, use null. "
          "followup_days is the number of days until the booked visit (one week = 7, two weeks = 14, next week = 7). "
          "followup_with is the clinic or specialty in lower case. summary is one plain sentence under 25 words. "
          "Return only JSON that matches the schema.")


def messages_for(note_text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Extract the fields from this note.\n<note>{note_text}</note>"}]


def validate(obj):
    """Tiny schema check without extra packages: required keys, no extras, types, enum."""
    errs = []
    if not isinstance(obj, dict):
        return ["not an object"]
    for k in SCHEMA["required"]:
        if k not in obj:
            errs.append(f"missing {k}")
    for k, v in obj.items():
        spec = SCHEMA["properties"].get(k)
        if spec is None:
            errs.append(f"extra key {k}"); continue
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        ok = any((t == "null" and v is None) or (t == "string" and isinstance(v, str)) or
                 (t == "boolean" and isinstance(v, bool)) or
                 (t == "integer" and isinstance(v, int) and not isinstance(v, bool)) for t in types)
        if not ok:
            errs.append(f"{k} has wrong type")
        if "enum" in spec and v not in spec["enum"]:
            errs.append(f"{k} not in enum")
    return errs


def explain_temperature():
    """Toy next-token scores after 'Follow up with cardiology in ...'. Temperature divides the scores
    before softmax: low T sharpens toward the top token, high T spreads the odds out."""
    toks, logits = ["7", "14", "3", "next", "two"], [3.0, 2.0, 1.2, 0.6, 0.2]
    for t in [0.2, 0.7, 1.0, 1.5]:
        e = [math.exp(x / t) for x in logits]; p = [v / sum(e) for v in e]
        print(f"T={t:<4}", "  ".join(f"{k}:{v:.2f}" for k, v in zip(toks, p)))
    print("T=0 means always pick the top token (greedy). For extraction, use 0 or close to it.")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="force the offline fake client")
    ap.add_argument("--limit", type=int, default=None, help="notes to send (real mode default 3, mock default all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--show-request", action="store_true", help="print the first request body (no key) and the raw reply")
    ap.add_argument("--explain-temperature", action="store_true", help="print how temperature reshapes token odds, then exit")
    a = ap.parse_args()
    if a.explain_temperature:
        explain_temperature(); return

    client = lc.get_client(force_mock=a.mock)
    model = "mock-extractor" if client.name == "mock" else lc.MODEL
    notes = load("discharge_notes.csv")
    keyd = {r["NOTE_ID"]: r for r in load("note_labels.csv")}
    limit = a.limit or (len(notes) if client.name == "mock" else 3)
    notes = notes[:limit]
    print(f"client = {client.name} | model = {model} | base_url = {lc.BASE_URL if client.name == 'real' else '(offline)'}")
    print(f"notes = {len(notes)} | temperature = {a.temperature} | max_tokens = {a.max_tokens} | timeout = {a.timeout} s")
    print(f"price card = ${lc.PRICE_IN}/M input, ${lc.PRICE_OUT}/M output (cost column is an estimate)\n")

    log, out = [], []
    for r in notes:
        msgs = messages_for(r["NOTE_TEXT"])
        try:
            resp = lc.chat_with_retries(client, msgs, timeout=a.timeout, model=model, temperature=a.temperature,
                                        max_tokens=a.max_tokens, response_format=RESPONSE_FORMAT)
        except Exception as e:  # give up on this note, keep going, log it
            print(f"  note {r['NOTE_ID']}: FAILED after retries: {e}")
            log.append({"note_id": r["NOTE_ID"], "status": "failed", "finish_reason": "", "prompt_tokens": 0,
                        "completion_tokens": 0, "latency_s": 0, "retries": "", "cost_usd": 0})
            continue
        ch = resp["choices"][0]
        u = resp.get("usage", {})
        status, obj = "ok", None
        try:
            obj = json.loads(ch["message"]["content"])
            errs = validate(obj)
            if errs:
                status = "invalid: " + "; ".join(errs)
        except json.JSONDecodeError:
            status = "bad_json"
        if ch.get("finish_reason") == "length":
            status = "cut_by_max_tokens"
        if a.show_request and r is notes[0]:
            body = {"model": model, "messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens,
                    "response_format": "(the JSON schema above)"}
            print("REQUEST BODY (the key travels only in the Authorization header, never in the body):")
            print(json.dumps(body, indent=1)[:1200])
            print("RAW REPLY:", json.dumps({k: v for k, v in resp.items() if not k.startswith("_")}, indent=1)[:900], "\n")
        c = lc.cost_usd(u)
        log.append({"note_id": r["NOTE_ID"], "status": status, "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                    "latency_s": resp["_latency_s"], "retries": resp["_retries"], "cost_usd": round(c, 6)})
        out.append({"note_id": r["NOTE_ID"], "status": status, "fields": obj})

    with open("extractions.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    with open("call_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)

    ok = [o for o in out if o["status"] == "ok"]
    pt = sum(x["prompt_tokens"] for x in log); ctk = sum(x["completion_tokens"] for x in log)
    cost = sum(x["cost_usd"] for x in log)
    print("status counts:", {s: sum(1 for x in log if x["status"] == s) for s in sorted({x["status"] for x in log})})
    print(f"retries used: {sum(int(x['retries'] or 0) for x in log)} | tokens in {pt:,} out {ctk:,} | "
          f"estimated cost ${cost:.4f} ({'mock tokens, priced as if real' if client.name == 'mock' else 'real usage'})")
    lat = [x["latency_s"] for x in log if x["status"] != "failed"]
    if lat:
        print(f"latency per call: median {sorted(lat)[len(lat)//2]:.2f} s, max {max(lat):.2f} s (includes retry waits)")
    if ok:
        print(f"scale-up: 10,000 notes at this mix = about ${cost / len(log) * 10000:.2f}")

    def same(field, truth, got):
        if field in ("lives_alone", "followup_booked"):
            return int(bool(got)) == int(truth)
        if truth == "":
            return got is None
        return str(got).lower() == str(truth).lower()

    pairs = [("primary_diagnosis", "TRUE_DIAGNOSIS"), ("length_of_stay_days", "TRUE_LOS_DAYS"),
             ("lives_alone", "TRUE_LIVES_ALONE"), ("followup_booked", "TRUE_FOLLOWUP_BOOKED"),
             ("followup_days", "TRUE_FOLLOWUP_DAYS"), ("followup_with", "TRUE_FOLLOWUP_WITH")]
    if not ok:
        print("\nno valid JSON answers to grade (check the status counts above)"); return
    print(f"\naccuracy vs answer key on {len(ok)} valid JSON answers (of {len(notes)} notes):")
    acc = {}
    for fld, col in pairs:
        hits = sum(same(fld, keyd[o["note_id"]][col], o["fields"][fld]) for o in ok)
        acc[fld] = hits / len(ok) if ok else math.nan
        print(f"  {fld:<22} {hits:>2}/{len(ok)} = {acc[fld]:.3f}")
    json.dump({"client": client.name, "n": len(notes), "valid": len(ok), "acc": acc, "tokens_in": pt,
               "tokens_out": ctk, "cost": cost, "log": log}, open("run_summary.json", "w"), indent=1)
    if client.name == "mock":
        print("\nNote: mock = regex, not an LLM. Misses like 'next week' or 'two weeks' are the point: compare with a real run.")


if __name__ == "__main__":
    main()
