# Day 38: a LOCAL retrieval baseline you can run anywhere (no Snowflake, no tokens, no cost).
# Keyword-only search (TF-IDF and BM25) over the same simulated chunks, scored on the labeled questions
# with hit rate@k and MRR. Then: attribute filters, chunks vs whole documents, a sample grounded prompt,
# and a size/cost estimate for the Cortex Search service. Compare with 07_evaluate.sql on your trial.
import csv, json, math, re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

chunks = list(csv.DictReader(open("doc_chunks.csv")))
docs = list(csv.DictReader(open("clinical_docs.csv")))
allq = list(csv.DictReader(open("eval_questions.csv")))
qs = [q for q in allq if q["Q_TYPE"] != "unanswerable"]
K = [1, 3, 5, 10]

def tok(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in ENGLISH_STOP_WORDS]

class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.d = [Counter(tok(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.d]); self.avg = self.len.mean()
        df = Counter(w for c in self.d for w in c)
        n = len(texts); self.idf = {w: math.log(1 + (n - f + .5) / (f + .5)) for w, f in df.items()}
        self.k1, self.b = k1, b
    def scores(self, q):
        s = np.zeros(len(self.d))
        for w in set(tok(q)):
            if w not in self.idf: continue
            tf = np.array([c.get(w, 0) for c in self.d])
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s

class Tfidf:
    def __init__(self, texts, **kw):
        self.v = TfidfVectorizer(sublinear_tf=True, **kw); self.m = self.v.fit_transform(texts)
    def scores(self, q):
        return (self.m @ self.v.transform([q]).T).toarray().ravel()   # rows are L2-normalised: cosine

def passes(row, flt):
    """Apply a Cortex Search style filter (@eq, @gte, @lte, @and, @or, @not) to one row."""
    if not flt: return True
    (op, arg), = flt.items()
    if op == "@and": return all(passes(row, f) for f in arg)
    if op == "@or": return any(passes(row, f) for f in arg)
    if op == "@not": return not passes(row, arg)
    (colname, val), = arg.items()
    return {"@eq": row[colname] == val, "@gte": row[colname] >= val, "@lte": row[colname] <= val}[op]

def rank(model, rows, q, flt=None, key="CHUNK_ID", n=10):
    s = model.scores(q)
    order = [i for i in np.argsort(-s, kind="stable") if passes(rows[i], flt)]
    return [rows[i][key] for i in order[:n]]

def evaluate(model, rows, use_filter, key="CHUNK_ID", gold="GOLD_CHUNK_ID", subset=None):
    rr, hits = [], {k: [] for k in K}
    for q in (subset or qs):
        flt = json.loads(q["FILTER_JSON"]) if (use_filter and q["FILTER_JSON"]) else None
        got = rank(model, rows, q["QUESTION"], flt, key)
        r = got.index(q[gold]) + 1 if q[gold] in got else None
        rr.append(1 / r if r else 0.0)
        for k in K: hits[k].append(1 if r and r <= k else 0)
    return {**{f"hit@{k}": round(float(np.mean(hits[k])), 3) for k in K}, "mrr": round(float(np.mean(rr)), 3)}

texts = [c["CHUNK_TEXT"] for c in chunks]
models = {"TF-IDF (words)": Tfidf(texts, stop_words="english", ngram_range=(1, 2)),
          "BM25": BM25(texts),
          "TF-IDF (char 3-5)": Tfidf(texts, analyzer="char_wb", ngram_range=(3, 5))}
print(f"chunks: {len(chunks)} from {len(docs)} simulated documents | labeled questions: {len(qs)} "
      f"({sum(q['Q_TYPE']=='keyword' for q in qs)} keyword, {sum(q['Q_TYPE']=='paraphrase' for q in qs)} paraphrase, "
      f"{sum(bool(q['FILTER_JSON']) for q in qs)} with a filter) + {len(allq)-len(qs)} unanswerable for the RAG test")

card = {"n_chunks": len(chunks), "n_docs": len(docs), "n_q": len(qs), "methods": {}, "by_type": {}, "filter": {}}
print("\n== chunk retrieval, filters applied where the question has one")
for name, m in models.items():
    r = evaluate(m, chunks, True); card["methods"][name] = r
    print(f"  {name:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))

print("\n== by question type (BM25 and TF-IDF words, hit@3 / MRR)")
for name in ["TF-IDF (words)", "BM25"]:
    card["by_type"][name] = {}
    for t in ["keyword", "paraphrase"]:
        r = evaluate(models[name], chunks, True, subset=[q for q in qs if q["Q_TYPE"] == t])
        card["by_type"][name][t] = r
        print(f"  {name:15s} {t:10s} hit@3 {r['hit@3']:.3f}  MRR {r['mrr']:.3f}")

fq = [q for q in qs if q["FILTER_JSON"]]
print(f"\n== the {len(fq)} filtered questions, BM25 with and without the attribute filter")
for use in [False, True]:
    r = evaluate(models["BM25"], chunks, use, subset=fq); card["filter"]["with" if use else "without"] = r
    print(f"  filter {'ON ' if use else 'OFF'}  hit@1 {r['hit@1']:.3f}  hit@3 {r['hit@3']:.3f}  MRR {r['mrr']:.3f}")
for q in fq:
    a = rank(models["BM25"], chunks, q["QUESTION"], None)[:2]
    b = rank(models["BM25"], chunks, q["QUESTION"], json.loads(q["FILTER_JSON"]))[:2]
    print(f"  Q{q['QID']} gold {q['GOLD_CHUNK_ID']} | no filter top2 {a} | {q['FILTER_JSON']} top2 {b}")

print("\n== chunks vs whole documents (BM25, find the right DOCUMENT, k = 3)")
dtexts = [f"{d['TITLE']}. " + d["DOC_TEXT"] for d in docs]
dm = BM25(dtexts)
whole = evaluate(dm, docs, True, key="DOC_ID", gold="GOLD_DOC_ID")
def chunk_doc_hit(k):
    h = []
    for q in qs:
        flt = json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None
        got = rank(models["BM25"], chunks, q["QUESTION"], flt, "DOC_ID")[:k]
        h.append(q["GOLD_DOC_ID"] in got)
    return round(float(np.mean(h)), 3)
avg_chunk = np.mean([len(t) for t in texts]); avg_doc = np.mean([len(t) for t in dtexts])
card["chunk_vs_doc"] = {"doc_hit3": whole["hit@3"], "chunk_doc_hit3": chunk_doc_hit(3),
                        "ctx_chars_chunks_k3": int(3 * avg_chunk), "ctx_chars_docs_k3": int(3 * avg_doc)}
print(f"  whole docs : right doc in top 3 {whole['hit@3']:.3f}, context sent to the LLM at k=3 about {int(3*avg_doc)} chars")
print(f"  chunks     : right doc in top 3 {card['chunk_vs_doc']['chunk_doc_hit3']:.3f}, context at k=3 about {int(3*avg_chunk)} chars "
      f"(and the exact passage, not the whole document)")

best = "BM25"
print("\n== misses: gold chunk not in BM25 top 3")
miss = []
for q in qs:
    flt = json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None
    got = rank(models[best], chunks, q["QUESTION"], flt)[:3]
    if q["GOLD_CHUNK_ID"] not in got:
        miss.append(q["QID"]); print(f"  Q{q['QID']:>2} [{q['Q_TYPE']}] {q['QUESTION'][:70]} -> got {got[0]}")
card["misses_k3"] = miss

# a grounded prompt, exactly what 06_rag_complete.sql builds (not sent to any model here)
qx = next(q for q in qs if q["QID"] == "29")
top = rank(models[best], chunks, qx["QUESTION"], None)[:3]
byid = {c["CHUNK_ID"]: c for c in chunks}
ctx = "\n".join(f"[{i}] {byid[i]['CHUNK_TEXT']}" for i in top)   # in rank order
prompt = ("You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. "
          "If the context does not contain the answer, reply exactly: Not found in the documents.\n\n"
          f"Context:\n{ctx}\n\nQuestion: {qx['QUESTION']}\nAnswer:")
print(f"\n== sample grounded prompt for Q{qx['QID']} (BM25 top 3; gold {qx['GOLD_CHUNK_ID']} "
      f"{'found' if qx['GOLD_CHUNK_ID'] in top else 'NOT found'})\n{prompt}")
card["prompt_example"] = prompt; card["prompt_tokens_est"] = len(prompt) // 4

# size and cost estimate for the Cortex Search service (formulas from the Snowflake cost docs)
emb_tokens = sum(len(t) for t in texts) // 4
row_bytes = np.mean([sum(len(v) for v in c.values()) for c in chunks])
gb = len(chunks) * (768 * 4 + row_bytes) / 1e9
card.update(emb_tokens=int(emb_tokens), index_gb=gb, row_bytes=int(row_bytes),
            avg_chunk_chars=int(avg_chunk), chunk_chars=[len(t) for t in texts],
            dept_counts=dict(Counter(c["DEPARTMENT"] for c in chunks)),
            type_counts=dict(Counter(c["DOC_TYPE"] for c in chunks)))
print("\n== size and cost estimate (rates NOT copied: see the Snowflake Service Consumption Table)")
print(f"  embedding tokens to build the index: about {emb_tokens:,} (chars / 4) = {emb_tokens/1e6:.4f} million")
print(f"  indexed data: {len(chunks)} rows x (768 dims x 4 bytes + about {int(row_bytes)} bytes of columns) = {gb*1e3:.3f} MB = {gb:.6f} GB")
print(f"  serving credits per month = {gb:.6f} GB x (credits per GB-month). With the docs example rate of 6.3: {gb*6.3:.4f} credits/month")
print("  warehouse: X-Small uses 1 credit per hour while running, billed per second, 60 second minimum per resume")
print(f"  grounded prompt above: about {len(prompt)//4} input tokens per question for COMPLETE")
json.dump(card, open("baseline_card.json", "w"), indent=1)
print("wrote baseline_card.json")
