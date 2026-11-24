# Day 41: calling an LLM API from Python (discharge notes to JSON)

First day of the LLM APIs block (Days 41-54). On Day 37 Snowflake ran the model for you inside SQL (`SNOWFLAKE.CORTEX.COMPLETE`, `AI_COMPLETE`). Today you make the same kind of call yourself from Python: build the **messages** (system, user, assistant), set **temperature** and **max_tokens**, ask for **structured JSON** with a schema, handle **retries and timeouts**, read **usage** for tokens and cost, and keep the **API key in an environment variable**, never in the repo.

All data is SIMULATED (seed 41): 30 discharge notes, Nov 9-15, 2026, with an answer key. No real patients, no real hospital. Do not send real patient notes to any API from a personal account; at work that needs your employer's approval and, for an outside provider, a signed BAA.

What ran where: on the lesson box the scripts ran in **mock mode** only (no key exists there, no real API was called). The real-API code path was also exercised against a local fake server on the box to check the request shape, the 503 retry and the 401 no-retry; that is a plumbing test, not a model result. Your real run is the one that counts.

## Files

- `gen_notes.py`: writes `discharge_notes.csv` (30 notes) and `note_labels.csv` (answer key)
- `llm_client.py`: `RealClient` (POST `/chat/completions` on any OpenAI-compatible API, stdlib only), `MockClient` (offline, same response shape), `chat_with_retries` (backoff + jitter, timeouts), `cost_usd`
- `first_call.py`: one chat call with all three roles; prints the reply, tokens, cost, latency
- `day41_extract.py`: the main exercise: notes to JSON with a schema, validation, a per-call log, accuracy vs the answer key
- `cortex_vs_python.md`: Day 37 SQL vs today's Python, side by side
- `.env.example`, `.gitignore`: key placeholder and the ignore list (`.env`, logs)
- `run_output.txt`: the real mock-mode output from the lesson box

## Offline first (no key, no cost, about 15 minutes)

- [ ] `python gen_notes.py`
- [ ] `python first_call.py --mock`: roles arrive in order system, user, assistant, user
- [ ] `python day41_extract.py --explain-temperature`: how T reshapes the odds of the next token
- [ ] `python day41_extract.py --mock --limit 2 --show-request`: the request body and the raw reply
- [ ] `python day41_extract.py --mock`: expect 30 ok, 2 retries (one 429, one timeout, both faked on purpose), follow-up fields 0.833
- [ ] `python day41_extract.py --mock --max-tokens 40 --limit 3`: expect 3 x `cut_by_max_tokens`

Lesson-box results (mock, real run): 30 of 30 valid JSON, 2 retries, 5,679 input + 1,836 output tokens (chars / 4 estimate), $0.0117 if priced at grok-4.3 list prices. Accuracy vs the answer key: diagnosis, length of stay and lives alone 1.000; follow-up booked, days and with 0.833 (the regex misses "next week", "one week" and "two weeks"). The mock is a regex, not a language model.

## Real API (optional, a few cents for 30 notes at the defaults, estimate)

The defaults point at xAI (`https://api.x.ai/v1`, model `grok-4.3`). Any OpenAI-compatible provider works: set `LLM_BASE_URL` and `LLM_MODEL`.

```bash
conda activate ds90
conda env config vars set LLM_API_KEY="paste-key-here"   # stored in the conda env, not in the repo
conda deactivate && conda activate ds90
python -c "import os; print('key set:', bool(os.environ.get('LLM_API_KEY')))"
python first_call.py
python day41_extract.py              # 3 notes first
python day41_extract.py --limit 30   # then all 30
```

Notes (xAI docs, checked Nov 24, 2026): grok-4.3 lists $1.25 input / $2.50 output per 1M tokens; grok-4.7 lists $2.00 / $6.00. xAI marks `/v1/chat/completions` as legacy and recommends `/v1/responses` (`messages` becomes `input`, `max_tokens` becomes `max_output_tokens`, and `store` defaults to true, so send `store: false` for sensitive text). Chat completions stays the portable pattern across providers, which is why the lesson uses it. Reasoning models count hidden reasoning tokens as output, so a real bill can be higher than the visible JSON suggests; read `usage`. Real requests also carry the JSON schema, so real input tokens will be higher than the mock estimate. If a key ever lands in a file or a commit: revoke it in the provider console first, then rotate.

## My results (fill in)

- Model ___ ; notes ___ ; status counts ___
- Accuracy: followup_booked ___ , followup_days ___ , followup_with ___ (mock 0.833 each)
- Tokens in ___ , out ___ ; cost $ ___ ; median latency ___ s ; retries ___
- 10,000 notes would cost about $ ___ ; I would run that job in (Cortex / Python) because ___
- One takeaway: ___

## Commit (your own repo)

```bash
cd ~/ds-genai-90-days
mkdir -p day41        # copy the exercise files, including .gitignore and .env.example, into day41/
cd day41 && python gen_notes.py && python day41_extract.py --mock && cd ..
git status            # check: no .env, no keys, no call_log.csv / extractions.jsonl / run_summary.json
git add day41
git commit -m "Day 41: LLM API calls from Python, structured JSON extraction from simulated discharge notes, retries and cost log"
git push
```
