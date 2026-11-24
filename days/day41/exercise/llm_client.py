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
