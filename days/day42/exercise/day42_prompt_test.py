# Day 42: measure whether a prompt change helped. Same test set, same model settings, one prompt per run.
#   python day42_prompt_test.py --mock                       all variants, all 36 notes, offline (scripted mock)
#   python day42_prompt_test.py --mock --variants v1 v2      just the main comparison
#   python day42_prompt_test.py --mock --show v2 --note 33   print the messages and the reply for one note
#   python day42_prompt_test.py --variants v1 v2             real API if LLM_API_KEY / XAI_API_KEY is set (3 notes each)
#   python day42_prompt_test.py --variants v1 v2 --limit 36  real API, all notes (2 x 36 small calls; see the cost line)
# No response_format today: we test what the prompt words alone do. Day 43 adds JSON schemas + Pydantic.
# Data is simulated. Do not send real patient notes to an API without your employer's approval and a BAA.
import argparse, csv, json, re
import llm_client as lc
import prompts
from prompt_mock import PromptSensitiveMock

FIELDS = [("primary_diagnosis", "TRUE_DIAGNOSIS"), ("length_of_stay_days", "TRUE_LOS_DAYS"),
          ("lives_alone", "TRUE_LIVES_ALONE"), ("followup_booked", "TRUE_FOLLOWUP_BOOKED"),
          ("followup_days", "TRUE_FOLLOWUP_DAYS"), ("followup_with", "TRUE_FOLLOWUP_WITH")]


def parse_reply(text):
    """Strict first (the reply must be pure JSON). If that fails, try to rescue the first {...} block."""
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0)), "rescued"  # extra text or a code fence around the JSON
            except json.JSONDecodeError:
                pass
    return None, "bad_json"


def same(field, truth, got):
    """Exact match on the value your pipeline would store (no lower-casing to hide format drift)."""
    if field in ("lives_alone", "followup_booked"):
        return isinstance(got, bool) and int(got) == int(truth)
    if truth == "":
        return got is None
    if field in ("length_of_stay_days", "followup_days"):
        return isinstance(got, int) and not isinstance(got, bool) and got == int(truth)
    return got == truth


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="force the offline scripted mock")
    ap.add_argument("--variants", nargs="+", default=list(prompts.VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="notes per variant (real default 3, mock default all)")
    ap.add_argument("--show", default=None, help="variant to print messages + reply for (with --note)")
    ap.add_argument("--note", default="1")
    a = ap.parse_args()

    real = (not a.mock) and lc.api_key()
    client = lc.RealClient(lc.api_key()) if real else PromptSensitiveMock()
    model = lc.MODEL if real else "mock-prompt-model"
    notes = load("prompt_test_set.csv")
    key = {r["NOTE_ID"]: r for r in load("test_labels.csv")}

    if a.show:
        r = next(x for x in notes if x["NOTE_ID"] == a.note)
        msgs = prompts.VARIANTS[a.show](r["NOTE_TEXT"])
        for m in msgs:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        resp = lc.chat_with_retries(client, msgs, model=model, temperature=0, max_tokens=300)
        print("--- reply ---\n" + resp["choices"][0]["message"]["content"])
        print(f"\ntokens: {resp['usage']['prompt_tokens']} in + {resp['usage']['completion_tokens']} out")
        return

    limit = a.limit or (len(notes) if not real else 3)
    notes = notes[:limit]
    print(f"client = {'real' if real else 'mock (scripted, not a model)'} | model = {model} | notes = {len(notes)} | temperature = 0")
    print(f"price card = ${lc.PRICE_IN}/M input, ${lc.PRICE_OUT}/M output (estimate)\n")

    rows, summary = [], {}
    for v in a.variants:
        build = prompts.VARIANTS[v]
        for r in notes:
            msgs = build(r["NOTE_TEXT"])
            try:
                resp = lc.chat_with_retries(client, msgs, model=model, temperature=0, max_tokens=300)
                text, u = resp["choices"][0]["message"]["content"], resp.get("usage", {})
            except Exception as e:
                print(f"  {v} note {r['NOTE_ID']}: FAILED after retries: {e}")
                text, u = "", {}
            obj, status = parse_reply(text)
            row = {"variant": v, "note_id": r["NOTE_ID"], "case_type": r["CASE_TYPE"], "status": status,
                   "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                   "cost_usd": round(lc.cost_usd(u), 6)}
            for fld, col in FIELDS:
                got = obj.get(fld) if isinstance(obj, dict) else None
                row[fld] = int(isinstance(obj, dict) and same(fld, key[r["NOTE_ID"]][col], got))
            rows.append(row)
        vr = [x for x in rows if x["variant"] == v]
        n = len(vr)
        acc = {fld: sum(x[fld] for x in vr) / n for fld, _ in FIELDS}
        inj = [x for x in vr if x["case_type"] == "injection"]
        summary[v] = {"n": n, "strict_json": sum(x["status"] == "ok" for x in vr) / n,
                      "parsed": sum(x["status"] != "bad_json" for x in vr) / n, "acc": acc,
                      "all_fields_right": sum(all(x[f] for f, _ in FIELDS) for x in vr) / n,
                      "injection_notes": len(inj), "injection_safe": sum(all(x[f] for f, _ in FIELDS) for x in inj),
                      "tokens_in": sum(x["prompt_tokens"] for x in vr), "tokens_out": sum(x["completion_tokens"] for x in vr),
                      "cost": sum(x["cost_usd"] for x in vr)}

    print(f"{'variant':<5} {'what changed':<42} {'strict':>6} {'diag':>5} {'los':>5} {'alone':>5} {'booked':>6} "
          f"{'days':>5} {'with':>5} {'all6':>5} {'inj ok':>6} {'tok in':>7} {'cost $':>8}")
    for v, s in summary.items():
        A = s["acc"]
        print(f"{v:<5} {prompts.DESCRIBE[v]:<42} {s['strict_json']:>6.2f} {A['primary_diagnosis']:>5.2f} "
              f"{A['length_of_stay_days']:>5.2f} {A['lives_alone']:>5.2f} {A['followup_booked']:>6.2f} "
              f"{A['followup_days']:>5.2f} {A['followup_with']:>5.2f} {s['all_fields_right']:>5.2f} "
              f"{s['injection_safe']:>2}/{s['injection_notes']:<3} {s['tokens_in']:>7,} {s['cost']:>8.4f}")

    # paired view: the same note and field under two prompts. Fixed = wrong before, right after.
    cmp = []
    vs = a.variants
    for old, new in zip(vs, vs[1:]):
        o = {x["note_id"]: x for x in rows if x["variant"] == old}
        nw = {x["note_id"]: x for x in rows if x["variant"] == new}
        fixed = sum(1 for k in o for f, _ in FIELDS if not o[k][f] and nw[k][f])
        broke = sum(1 for k in o for f, _ in FIELDS if o[k][f] and not nw[k][f])
        cmp.append({"from": old, "to": new, "fixed": fixed, "broken": broke})
        extra = summary[new]["cost"] - summary[old]["cost"]
        print(f"{old} -> {new}: {fixed} field values fixed, {broke} broken; cost {'+' if extra >= 0 else ''}{extra:.4f} USD on {len(o)} notes")
    if "v1" in summary and "v2" in summary:
        o = {x["note_id"]: x for x in rows if x["variant"] == "v1"}
        nw = {x["note_id"]: x for x in rows if x["variant"] == "v2"}
        fixed = sum(1 for k in o for f, _ in FIELDS if not o[k][f] and nw[k][f])
        broke = sum(1 for k in o for f, _ in FIELDS if o[k][f] and not nw[k][f])
        if ("v1", "v2") not in [(c["from"], c["to"]) for c in cmp]:
            cmp.append({"from": "v1", "to": "v2", "fixed": fixed, "broken": broke})
        print(f"v1 -> v2 (the main comparison): {fixed} fixed, {broke} broken")
        for v in ("v1", "v2"):
            print(f"  {v}: cost per note ${summary[v]['cost'] / summary[v]['n']:.6f}, "
                  f"per 10,000 notes about ${summary[v]['cost'] / summary[v]['n'] * 10000:.2f}")

    with open("results_by_note.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"client": "real" if real else "mock", "model": model, "summary": summary, "compare": cmp},
              open("run_summary.json", "w"), indent=1)
    print("\nwrote results_by_note.csv and run_summary.json")
    if not real:
        print("Note: mock = scripted rules, not an LLM. These numbers show how the test works, not real prompt gains.")


if __name__ == "__main__":
    main()
