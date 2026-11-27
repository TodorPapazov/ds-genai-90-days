# Day 42: prompt engineering basics (measure whether a prompt change helped)

Day 41 got JSON out of simulated discharge notes with an LLM API. Today the model settings stay the same and only the **prompt** changes. You write four versions, run each on the same **test set** of 36 notes with an answer key, and compare **per-field accuracy**, **fixed vs broken cells** and **cost per variant**.

All data is SIMULATED: the 30 Day 41 notes (seed 41) plus 6 hand-written edge cases. No real patients. Do not send real patient notes to any API from a personal account; at work that needs your employer's approval and, for an outside provider, a signed BAA.

What ran where: on the lesson box everything ran in **mock mode** (no key exists there, no real API or model was called). The mock in `prompt_mock.py` is a **scripted** stand-in: each missing prompt part triggers one failure pattern on purpose, and the rules are listed at the top of the file. Its numbers show how the test works. They are **not** real model accuracy. The real-API path of `day42_prompt_test.py` was also checked against a local fake server (request path, header, roles, no response_format); that is a plumbing test, not a model result.

## Files

- `llm_client.py`: copied unchanged from Day 41 (`RealClient`, `MockClient`, `chat_with_retries`, `cost_usd`)
- `discharge_notes.csv`, `note_labels.csv`: the Day 41 notes and answer key
- `build_test_set.py`: writes `prompt_test_set.csv` (36 notes) and `test_labels.csv` (answer key + `CASE_TYPE`)
- `prompts.py`: the variants. `v1` zero-shot, `v2a` + system prompt, delimiters and format rules, `v2` + 3 few-shot examples, `v3` + sanitize the note
- `prompt_mock.py`: `PromptSensitiveMock`, the scripted offline stand-in (not a model)
- `day42_prompt_test.py`: runs variants on the test set, grades per field (exact match), prints fixed vs broken and cost
- `cortex_prompt_note.md`: the same prompt in Cortex COMPLETE (Day 37 reference)
- `.env.example`, `.gitignore`: key placeholder and the ignore list (`.env`, `results_*.csv`, `run_summary.json`)
- `run_output.txt`: the mock-mode output from the lesson box

## Steps (60-90 minutes)

- [ ] Copy the files into `day42/` and run `python build_test_set.py` (36 notes)
- [ ] Read `prompts.py`: find the role, the numbered rules, the `<note>` delimiters, the output format line and the examples
- [ ] Read rows 31-36 of `prompt_test_set.csv`: three injection lines, a missing stay, "2 weeks", "next week"
- [ ] `python day42_prompt_test.py --mock --show v1 --note 31`: the injected line flips `followup_booked`
- [ ] `python day42_prompt_test.py --mock --show v2a --note 33`: the note contains `</note>`, so the data block ends early and the injection works anyway
- [ ] `python day42_prompt_test.py --mock --variants v1 v2`: the main comparison
- [ ] `python day42_prompt_test.py --mock`: all four variants; which single change fixed the most cells?
- [ ] Open `results_by_note.csv` in pandas and list what v2 still gets wrong
- [ ] Your turn: add `v4` to `prompts.py` (one change only) and rerun. Write fixed, broken and cost change below
- [ ] Optional, real API: `python day42_prompt_test.py --variants v1 v2` (3 notes each), then `--limit 36`
- [ ] Fill in "My results", check `git status`, commit and push

Lesson-box results (MOCK, scripted, not a model): v1 strict JSON 0.75, all 6 fields right on 4/36 notes, 0/3 injection notes correct, 4,103 input tokens, $0.0110. v2 strict JSON 1.00, all 6 right on 35/36, 2/3 injection notes, 23,877 input tokens, $0.0355 (about 3.2x v1). v1 to v2: 55 cells fixed, 0 broken. v3 (sanitize) 36/36. In a real run expect some broken cells too; that is why you count both directions.

```python
import pandas as pd
r = pd.read_csv("results_by_note.csv")
fields = ["primary_diagnosis", "length_of_stay_days", "lives_alone", "followup_booked", "followup_days", "followup_with"]
print(r.groupby("variant")[fields + ["cost_usd"]].agg({**{f: "mean" for f in fields}, "cost_usd": "sum"}).round(3))
print(r[(r.variant == "v2") & (r[fields].min(axis=1) == 0)][["note_id", "case_type"] + fields])
```

## Real API (optional)

Same setup as Day 41: the key lives in `LLM_API_KEY` (or `XAI_API_KEY`) in your conda env, never in a file in git. Defaults point at xAI (`https://api.x.ai/v1`, model `grok-4.3`); set `LLM_BASE_URL` and `LLM_MODEL` for another OpenAI-compatible provider. Today no `response_format` is sent, so you see what the prompt words alone do; Day 43 adds JSON schemas and Pydantic. v1 + v2 on 36 notes is 72 small calls; the mock token estimate prices that at about 5 cents at grok-4.3 list prices (real requests will differ). Some providers discount a repeated prompt prefix (prompt caching); keep the static system prompt and examples first if yours does.

## My results (fill in)

- Model ___ ; notes ___
- v1 real: strict JSON ___ , all 6 right ___ , injection notes ___ , cost $ ___
- v2 real: strict JSON ___ , all 6 right ___ , injection notes ___ , cost $ ___
- v1 to v2: fixed ___ , broken ___ ; was the extra cost worth it? ___
- My v4 change: ___ ; fixed ___ , broken ___
- One surprise: ___

## Commit (your own repo)

```bash
cd ~/ds-genai-90-days
mkdir -p day42        # copy the exercise files, including .gitignore and .env.example, into day42/
cd day42 && python build_test_set.py && python day42_prompt_test.py --mock && cd ..
git status            # check: no .env, no keys, no results_by_note.csv / run_summary.json
git add day42
git commit -m "Day 42: prompt engineering basics, system prompt, delimiters, few-shot, and a prompt test set with per-field accuracy and cost"
git push
```
