# Day 41: LLM APIs from Python: messages and roles, temperature and max tokens, structured JSON, retries, and a token cost log on simulated discharge notes

## Concept
Day 41 opens the LLM APIs, prompt engineering, embeddings and vector search block (Days 41-54). On Day 37 Snowflake ran the model for you inside SQL (SNOWFLAKE.CORTEX.COMPLETE, AI_COMPLETE). Today you make the same kind of call yourself from Python. A chat-completions call is one HTTPS POST to /chat/completions on an OpenAI-compatible API (default here: xAI at https://api.x.ai/v1, model grok-4.3; any provider works by changing the base URL and model). You send messages with roles: system (standing rules and format), user (the request, here the note inside <note> markers) and assistant (earlier replies). The API is stateless: to continue a chat you resend the history and pay for it again. Knobs: temperature (0 for extraction, so reruns match), max_tokens (a hard cap on output; too low and the JSON is cut, finish_reason = "length"), response_format with a JSON schema (strict), and a timeout on every call. The reply carries choices[0].message.content and usage (prompt and completion tokens); cost = input tokens x input price + output tokens x output price. Retries: back off with jitter on 408, 409, 429, 5xx and timeouts (honor Retry-After), never on 400 / 401 / 403. Keys live only in an environment variable (LLM_API_KEY or XAI_API_KEY, e.g. conda env config vars set); never in code, notebooks, logs or git. Exercise: 30 simulated discharge notes to JSON (diagnosis, length of stay, lives alone, follow-up booked / days / with, a one-line summary), validated, logged per call and graded against an answer key. A mock client returns the same response shape offline (a regex, not a model; it fakes one 429 and one timeout so you can watch retries). Mock run on the lesson box: 30 of 30 valid, 2 retries, 5,679 in + 1,836 out tokens (estimate), $0.0117 at grok-4.3 list price; follow-up fields 0.833, the rest 1.000. xAI docs (checked Nov 24, 2026) mark chat completions as legacy and recommend /v1/responses; chat completions stays the portable pattern. What ran on the lesson box: mock mode only, plus a plumbing test of the real-API code against a local fake server. No real API was called and no key exists there. Next: Day 42, prompt engineering basics.

## Key numbers
- **Data:** 30 simulated discharge notes, Nov 9-15, 2026 (seed 41), with an answer key; 8 diagnoses; varied follow-up wording
- **Mock run (real output):** 30 of 30 valid JSON; 2 retries (one faked 429, one faked timeout); 5,679 input + 1,836 output tokens (chars / 4)
- **Mock accuracy vs answer key:** diagnosis, length of stay, lives alone 1.000; follow-up booked / days / with 0.833 / 0.833 / 0.833 (regex misses 'next week', 'two weeks')
- **max_tokens demo:** max_tokens = 40 on 3 notes: 3 of 3 cut_by_max_tokens (finish_reason = length, json.loads fails)
- **Cost math:** $0.0117 for 30 notes at grok-4.3 list ($1.25 in / $2.50 out per 1M, docs Nov 24, 2026); about $3.90 per 10,000 notes at the mock token mix
- **Real-API plumbing test:** RealClient against a local fake server: correct path, Bearer header, body; 503 retried after Retry-After; 401 not retried
- **Not run here:** Any real LLM API call (no key on the lesson box); real tokens, latency and accuracy are yours to measure

## Why it matters for healthcare data science jobs
Healthcare analytics and data science postings increasingly list Python with REST APIs, Gen AI or LLM experience (for example summarizing or extracting fields from clinical text), prompt design, evaluation of model output, and responsible handling of PHI and secrets. Today covers the basic mechanics behind those lines: calling an LLM API from Python, getting validated JSON back, handling rate limits and timeouts, keeping keys out of code, and estimating cost before scaling up. After you run it (and ideally the real-API path), you can say, honestly, that you built a small practice pipeline that extracts structured fields from simulated discharge notes through an LLM API with schema validation, retries, a cost log and accuracy against an answer key, and that you compared it with Snowflake Cortex functions. It is a learning project on simulated data, not production experience.

## Docs checked (Nov 24, 2026)
- xAI models and pricing: https://docs.x.ai/docs/models
- xAI structured outputs (response_format json_schema): https://docs.x.ai/docs/guides/structured-outputs
- xAI Chat Completions (legacy) REST reference: https://docs.x.ai/developers/rest-api-reference/inference/chat-completions
- xAI Responses vs Chat Completions: https://docs.x.ai/developers/model-capabilities/text/comparison
- Uncertain: model names, prices and the chat completions status change over time. Check the provider's pages before a real run; the price card in llm_client.py is overridable with env vars.

## Hands-on (60-90 min)
1. Copy the Day 41 exercise files (including .gitignore and .env.example) into day41/ and run python gen_notes.py (30 notes + answer key).
2. Run snippet 2: python first_call.py --mock. Read the roles in order (system, user, assistant, user) and why history is resent.
3. python day41_extract.py --explain-temperature: how temperature reshapes the next-token odds. Pick T for extraction (0).
4. python day41_extract.py --mock --limit 2 --show-request: find messages, temperature, max_tokens, finish_reason and usage in the request and reply.
5. python day41_extract.py --mock: expect 30 ok, 2 retries (faked 429 and timeout), follow-up fields 0.833, cost line about $0.0117.
6. Open call_log.csv in pandas: total tokens, cost, slowest call, which notes needed a retry.
7. python day41_extract.py --mock --max-tokens 40 --limit 3: expect 3 x cut_by_max_tokens. Explain why.
8. Read llm_client.py (snippet 1): RETRYABLE, chat_with_retries, and the one line that reads the key (it is never printed).
9. Optional, real API: set LLM_API_KEY with conda env config vars set (snippet 4), run python first_call.py, then python day41_extract.py (3 notes).
10. Optional: --limit 30. Compare follow-up accuracy with the mock 0.833, and write tokens, cost and median latency in the README.
11. Read cortex_vs_python.md (snippet 5): pick one job you would keep in Snowflake and one you would run from Python.
12. Fill in 'My results' in day41/README.md, check git status (no .env, no logs), then commit and push (snippet 6).

## Code
Exercise files: `exercise/gen_notes.py`, `exercise/llm_client.py`, `exercise/first_call.py`, `exercise/day41_extract.py`, `exercise/cortex_vs_python.md`, `exercise/discharge_notes.csv`, `exercise/note_labels.csv`, `exercise/.env.example`, `exercise/.gitignore`, `exercise/run_output.txt`, `exercise/README.md`.

### 1. The client: real (urllib, OpenAI-compatible), mock (offline), retries with backoff, cost (Python)
```python
# Day 41: a small, dependency-free client for chat-completions style LLM APIs.
# Real mode: an OpenAI-compatible endpoint (default xAI, https://api.x.ai/v1) using the key in an env var.
# Mock mode: a fake client that returns the SAME response shape, so the exercise runs offline with no key.
# Keys live ONLY in environment variables. Never paste a key into this file, a notebook, or git.
import json, os, random, re, time, urllib.error, urllib.request

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.x.ai/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "grok-4.3")
# Price card in US dollars per 1 million tokens. Defaults: xAI grok-4.3 list price, short prompts
# (docs.x.ai/docs/models, checked Nov 24, 2026). Prices change: override with env vars if needed.
PRICE_IN = float(os.environ.get("LLM_PRICE_IN_PER_M", "1.25"))
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT_PER_M", "2.50"))
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def api_key():
    """Return the key from the environment, or None. The key is never printed or logged."""
    return os.environ.get("LLM_API_KEY") or os.environ.get("XAI_API_KEY")


def cost_usd(usage):
    """Dollars for one call: input tokens x input price + output tokens x output price."""
    return usage.get("prompt_tokens", 0) / 1e6 * PRICE_IN + usage.get("completion_tokens", 0) / 1e6 * PRICE_OUT


class APIError(Exception):
    def __init__(self, status, msg, retry_after=None):
        super().__init__(f"HTTP {status}: {msg}")
        self.status, self.retry_after = status, retry_after


class RealClient:
    """POST /chat/completions on any OpenAI-compatible API (xAI, OpenAI, and others)."""
    name = "real"

    def __init__(self, key, base_url=BASE_URL):
        self._key, self.base_url = key, base_url

    def chat(self, messages, model=MODEL, temperature=0.0, max_tokens=300, response_format=None, timeout=30):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            body["response_format"] = response_format
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After")
            raise APIError(e.code, e.read()[:200].decode(errors="replace"), float(ra) if ra and ra.isdigit() else None)
        # socket timeouts and connection errors bubble up as TimeoutError / URLError


class MockClient:
    """Offline stand-in. Reads the note with regular expressions and answers in the chat-completions
    shape (choices, message, finish_reason, usage). It is NOT a language model. It also fakes, on
    purpose and deterministically, one rate limit (429) and one timeout so you can watch retries work,
    and it honors max_tokens by cutting the output (finish_reason = "length") like a real API."""
    name = "mock"

    def __init__(self, seed=41):
        self.rng = random.Random(seed)
        self.calls = 0

    @staticmethod
    def _tokens(text):
        return max(1, round(len(text) / 4))  # rough rule of thumb: about 4 characters per token in English

    def _read(self, note):
        n = note.lower()
        dx = None
        for d in ["Heart failure", "COPD exacerbation", "Pneumonia", "Type 2 diabetes, hyperglycemia",
                  "Hip replacement", "Sepsis, urinary source", "Atrial fibrillation", "Cellulitis"]:
            if d.lower() in n:
                dx = d
        m = re.search(r"length of stay (\d+) day", n) or re.search(r"hospital day (\d+)", n)
        los = int(m.group(1)) if m else None
        alone = bool(re.search(r"lives alone|lives by (him|her)self", n))
        fu = re.search(r"(?:follow up with ([a-z ]+?)|([a-z ]+?) appointment) in (\d+) (day|week|month)", n)
        days, who = None, None
        if fu:
            mult = {"day": 1, "week": 7, "month": 30}[fu.group(4)]
            days, who = int(fu.group(3)) * mult, (fu.group(1) or fu.group(2)).strip()
        booked = days is not None  # misses "next week", "one week", "two weeks" on purpose: regex is brittle
        s = f"{dx or 'Unknown condition'}; stay {los} days; " + (f"follow-up with {who} in {days} days." if booked else "no follow-up found.")
        return {"primary_diagnosis": dx, "length_of_stay_days": los, "lives_alone": alone,
                "followup_booked": booked, "followup_days": days, "followup_with": who, "summary": s}

    def chat(self, messages, model="mock-extractor", temperature=0.0, max_tokens=300, response_format=None, timeout=30):
        self.calls += 1
        if self.calls == 4:
            raise APIError(429, "mock rate limit: slow down", retry_after=0.2)
        if self.calls == 9:
            raise TimeoutError("mock timeout: no answer within the time limit")
        time.sleep(0.01)
        prompt = "\n".join(m["content"] for m in messages)
        note = re.search(r"<note>(.*?)</note>", prompt, re.S)
        if response_format is None:  # plain chat: echo back so you can see the roles arrive in order
            roles = ", ".join(m["role"] for m in messages)
            content = (f"(mock reply) I got {len(messages)} messages in this order: {roles}. "
                       f"Your last question was: {messages[-1]['content'][:80]}")
        else:
            content = json.dumps(self._read(note.group(1) if note else prompt))
        out_tok = self._tokens(content)
        finish = "stop"
        if out_tok > max_tokens:
            content, out_tok, finish = content[: max_tokens * 4], max_tokens, "length"
        p = self._tokens(prompt) + 4 * len(messages)  # small overhead per message, like real chat formats
        return {"id": f"mock-{self.calls}", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": p, "completion_tokens": out_tok, "total_tokens": p + out_tok}}


def get_client(force_mock=False):
    key = api_key()
    if force_mock or not key:
        return MockClient()
    return RealClient(key)


def chat_with_retries(client, messages, max_retries=4, base_delay=0.5, timeout=30, **kw):
    """Call client.chat, retrying rate limits, server errors and timeouts with exponential backoff
    plus jitter. Errors such as 400 (bad request) or 401 (bad key) are not retried: fix the cause."""
    attempt, t0 = 0, time.perf_counter()
    while True:
        try:
            resp = client.chat(messages, timeout=timeout, **kw)
            resp["_latency_s"] = round(time.perf_counter() - t0, 3)
            resp["_retries"] = attempt
            return resp
        except (APIError, TimeoutError, ConnectionError, urllib.error.URLError) as e:
            status = getattr(e, "status", None)
            if isinstance(e, APIError) and status not in RETRYABLE:
                raise
            if attempt >= max_retries:
                raise
            wait = getattr(e, "retry_after", None) or base_delay * (2 ** attempt) * (1 + random.random() * 0.25)
            print(f"    retry {attempt + 1}/{max_retries} after {type(e).__name__} ({e}); waiting {wait:.2f} s")
            time.sleep(wait)
            attempt += 1
```

### 2. First call: system, user and assistant roles, temperature, max_tokens, usage (Python)
```python
# Day 41, step 1: one chat call from Python. Roles, temperature, max_tokens, usage and cost.
#   python first_call.py          real API if LLM_API_KEY or XAI_API_KEY is set, otherwise the offline mock
#   python first_call.py --mock   always offline
import sys
import llm_client as lc

client = lc.get_client(force_mock="--mock" in sys.argv)
messages = [
    # system: the standing instructions (who the assistant is, rules, format)
    {"role": "system", "content": "You are a concise assistant for a hospital data analytics team. Answer in 2 sentences."},
    # user / assistant pairs: earlier turns. The API is stateless, so YOU resend the history every call.
    {"role": "user", "content": "What is a 30-day readmission?"},
    {"role": "assistant", "content": "A return inpatient stay within 30 days of discharge. Hospitals track it as a quality measure."},
    # the new question
    {"role": "user", "content": "Why would a care manager want follow-up dates pulled out of discharge notes?"},
]
resp = lc.chat_with_retries(client, messages, model="mock-extractor" if client.name == "mock" else lc.MODEL,
                            temperature=0.2, max_tokens=120, timeout=30)
choice = resp["choices"][0]
print(f"client: {client.name} | model: {resp.get('model')} | finish_reason: {choice['finish_reason']}")
print("reply:", choice["message"]["content"])
u = resp["usage"]
print(f"tokens: {u['prompt_tokens']} in + {u['completion_tokens']} out = {u['total_tokens']} | "
      f"cost about ${lc.cost_usd(u):.6f} | latency {resp['_latency_s']} s | retries {resp['_retries']}")
```

### 3. Notes to JSON: schema, validation, per-call log, accuracy vs the answer key (Python)
```python
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
```

### 4. Optional: put the key in your conda env and run the real API (bash)
```bash
conda activate ds90
conda env config vars set LLM_API_KEY="paste-key-here"   # lives in the conda env, outside the repo
conda deactivate && conda activate ds90                  # reload so the variable is visible
python -c "import os; print('key set:', bool(os.environ.get('LLM_API_KEY')))"   # True/False, never the key
# optional: another OpenAI-compatible provider
# conda env config vars set LLM_BASE_URL="https://api.openai.com/v1" LLM_MODEL="your-model"
python first_call.py
python day41_extract.py              # 3 notes first
python day41_extract.py --limit 30   # then all 30
# leaked a key? revoke it in the provider console first, then rotate
```

### 5. Cortex COMPLETE in SQL vs an LLM API from Python (Day 37 reference, do not run today)
```markdown
# Same job, two places: Cortex COMPLETE in SQL (Day 37) vs an LLM API from Python (Day 41)

Reference only. Do not run the SQL today; it is the Day 37 pattern, shown so you can map each part.

```sql
-- Day 37 (Snowflake): the model runs next to the table. One statement, many rows.
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
         [ {'role': 'system', 'content': 'You extract fields from simulated discharge notes.'},
           {'role': 'user',   'content': 'Extract the fields: ' || NOTE_TEXT} ],
         {'temperature': 0, 'max_tokens': 120}) AS R      -- returns choices + usage, like an API
FROM DISCHARGE_NOTES WHERE NOTE_ID <= 5;
```

```python
# Day 41 (Python): you send each request yourself, over HTTPS, with your own key from an env var.
resp = lc.chat_with_retries(client, messages_for(note), model=lc.MODEL,
                            temperature=0, max_tokens=300, response_format=RESPONSE_FORMAT)
fields = json.loads(resp["choices"][0]["message"]["content"])
```

| | Cortex COMPLETE / AI_COMPLETE (SQL) | LLM API from Python |
|---|---|---|
| Where the note goes | Stays inside your Snowflake account and its governance | Leaves your machine for the provider's servers |
| Who can call it | Snowflake roles (CORTEX_USER) | Anyone holding the API key: guard it |
| Loop over rows | The SQL engine does it: one SELECT over a table | You write the loop, batching and rate-limit handling |
| Retries, timeouts | Handled for you inside the query | Your code: backoff on 429 / 5xx, a timeout on every call |
| Same knobs | messages with roles, temperature, max_tokens, JSON schema | messages with roles, temperature, max_tokens, JSON schema |
| Billing | Snowflake credits per token (plus the warehouse) | Provider dollars per token (input and output priced separately) |
| Models | The Cortex model list in your region | Whatever the provider offers; switch with base URL + model name |
| Results land in | A table (CTAS) | Files (JSONL, CSV), then pandas or a load back into the warehouse |
| Good fit | Batch enrichment of warehouse tables, governed data | Apps, notebooks, pipelines outside the warehouse, provider choice |

Rule for real work: patient data goes only where your employer has approved it (for an outside API that usually means a signed BAA and an approved account). This exercise uses simulated notes only.
```

### 6. Push to GitHub
```bash
cd ~/ds-genai-90-days
mkdir -p day41        # copy the exercise files, including .gitignore and .env.example, into day41/
cd day41 && python gen_notes.py && python day41_extract.py --mock && cd ..
git status            # check: no .env, no keys, no call_log.csv / extractions.jsonl / run_summary.json
git add day41
git commit -m "Day 41: LLM API calls from Python, structured JSON extraction from simulated discharge notes, retries and cost log"
git push
```

## Local run output (lesson box, mock mode)
```
$ python gen_notes.py
wrote 30 simulated notes to discharge_notes.csv and the answer key to note_labels.csv

$ python first_call.py --mock
client: mock | model: mock-extractor | finish_reason: stop
reply: (mock reply) I got 4 messages in this order: system, user, assistant, user. Your last question was: Why would a care manager want follow-up dates pulled out of discharge notes?
tokens: 88 in + 44 out = 132 | cost about $0.000220 | latency 0.01 s | retries 0

$ python day41_extract.py --explain-temperature
T=0.2  7:0.99  14:0.01  3:0.00  next:0.00  two:0.00
T=0.7  7:0.73  14:0.18  3:0.06  next:0.02  two:0.01
T=1.0  7:0.59  14:0.22  3:0.10  next:0.05  two:0.04
T=1.5  7:0.46  14:0.24  3:0.14  next:0.09  two:0.07
T=0 means always pick the top token (greedy). For extraction, use 0 or close to it.

$ python day41_extract.py --mock --max-tokens 40 --limit 3
client = mock | model = mock-extractor | base_url = (offline)
notes = 3 | temperature = 0.0 | max_tokens = 40 | timeout = 30 s
price card = $1.25/M input, $2.5/M output (cost column is an estimate)

status counts: {'cut_by_max_tokens': 3}
retries used: 0 | tokens in 568 out 120 | estimated cost $0.0010 (mock tokens, priced as if real)
latency per call: median 0.01 s, max 0.01 s (includes retry waits)

no valid JSON answers to grade (check the status counts above)

$ python day41_extract.py --mock
client = mock | model = mock-extractor | base_url = (offline)
notes = 30 | temperature = 0.0 | max_tokens = 300 | timeout = 30 s
price card = $1.25/M input, $2.5/M output (cost column is an estimate)

    retry 1/4 after APIError (HTTP 429: mock rate limit: slow down); waiting 0.20 s
    retry 1/4 after TimeoutError (mock timeout: no answer within the time limit); waiting 0.54 s
status counts: {'ok': 30}
retries used: 2 | tokens in 5,679 out 1,836 | estimated cost $0.0117 (mock tokens, priced as if real)
latency per call: median 0.01 s, max 0.55 s (includes retry waits)
scale-up: 10,000 notes at this mix = about $3.90

accuracy vs answer key on 30 valid JSON answers (of 30 notes):
  primary_diagnosis      30/30 = 1.000
  length_of_stay_days    30/30 = 1.000
  lives_alone            30/30 = 1.000
  followup_booked        25/30 = 0.833
  followup_days          25/30 = 0.833
  followup_with          25/30 = 0.833

Note: mock = regex, not an LLM. Misses like 'next week' or 'two weeks' are the point: compare with a real run.
```

## What ran, and what did not (note)
The lesson computer has no LLM API key, and none was created. The bot ran gen_notes.py, first_call.py --mock and day41_extract.py in mock mode; the mock client is a regex with the same response shape as a real API, not a language model, and its token counts are a characters-divided-by-4 estimate. The real-API code path (RealClient and chat_with_retries) was exercised only against a local fake server on the box to check the request path, headers, body, a 503 retry and a 401 no-retry. No real model was called, so real accuracy, tokens, latency and cost are yours to measure. Write them in your README.

## Coming up: Days 42-54
Planned outline for Days 42-54 (details may shift):
- Day 42: prompt engineering basics: clear system prompts, few-shot examples, delimiters, and a small prompt test set on the same simulated notes.
- Days 43-44: structured output with JSON schemas and validation (Pydantic), prompt regression tests.
- Days 45-46: cost, latency and safety habits: batching and caching, PHI redaction, prompt injection basics.
- Days 47-49: embeddings and cosine similarity (Cortex embedding functions and a small local model on CPU), chunking choices.
- Days 50-53: vector search (Snowflake VECTOR type and similarity functions, a local vector index), hybrid search, reranking, retrieval metrics.
- Day 54: mini project, semantic search over simulated clinical documents with an evaluation and a cost log.

## Tomorrow (Day 42)
Prompt engineering basics: clear system prompts, a few worked examples (few-shot), delimiters, and a small test set so you can measure whether a prompt change actually helped.

## Printables
- Cheat sheet: `day41_v1_cheat_sheet.png`
- Infographic: `day41_v2_infographic.png`
- Code worksheet: `day41_v3_worksheet.png`
- Practice quiz: `day41_v4_quiz.png`
- PDF: `day41_lesson.pdf`

## Podcast script
(Audio: `day41_podcast.mp3`)

Welcome to day forty one of the ninety day Data Analyst to Data Scientist plan. Last week we lived inside Snowflake. Today we step outside and talk to a language model directly from Python. This starts a new two week block on language model APIs, prompt engineering, embeddings, and vector search.

Here is the good news. You already know the idea. On day thirty seven you called Cortex COMPLETE from SQL. You gave it a system message and a user message, you set the temperature to zero, you capped the length, and you got back an answer plus a count of tokens. A language model API works the same way. The difference is who does the chores. In Snowflake, the SQL engine looped over the table, retried when something went wrong, and kept the data inside your account. From Python, you do those chores yourself.

So what is a call, really? It is one web request. You send a list of messages. The system message holds the standing rules, like, you extract fields from discharge notes, use only facts in the note, and return only JSON. The user message holds the actual request, here the note itself, wrapped in clear markers. And assistant messages are earlier replies from the model. That matters because the API has no memory. Every call starts fresh. If you want a conversation, you send the whole history again, and you pay for those tokens again.

Next come the knobs. Temperature controls how adventurous the model is when it picks the next word. Near zero, it almost always picks its top choice, so reruns match. That is what you want for extraction. Higher values spread the odds out, which is nice for drafting text and bad for data. Max tokens is a hard cap on the length of the answer. Set it too low and the model stops in the middle of your JSON. The reply tells you why it stopped. Stop means it finished. Length means it got cut off. You will see that happen on purpose today.

Then structured output. Instead of hoping for clean JSON, you send a schema: the field names, their types, which ones are required. Many providers will then return JSON that matches. Still, check every reply in your code. Parse it, confirm the keys and types, and record a status for each note.

Now the boring parts that make you look professional. Calls fail. You might hit a rate limit, or the server is busy, or nothing comes back in time. So every call gets a timeout, and you retry only those temporary errors, waiting a little longer each time, with a bit of randomness. You never retry a bad request or a bad key. That just repeats the mistake. And you log every call: tokens in, tokens out, how long it took, how many retries, and what it cost. Cost is simple math. Input tokens times the input price, plus output tokens times the output price. Our offline run of thirty notes came to about one cent at list prices, and ten thousand notes would be a few dollars.

One rule above all. Your API key lives in an environment variable, never in your code, your notebook, or your git history. And real patient notes only go where your employer has approved it.

Today's exercise runs fully offline. A mock client answers in the same shape as a real API, so you can learn the loop with no key and no cost. It even fakes one rate limit and one timeout, so you can watch the retries work. The mock is just pattern matching, so it misses phrases like next week. If you have a key, flip to the real API and see if the model does better.

Tomorrow, on day forty two, we work on the words we send. Prompt engineering basics: clear system prompts, a few worked examples, delimiters, and a small test set so you can measure whether a prompt change actually helped. See you then.
