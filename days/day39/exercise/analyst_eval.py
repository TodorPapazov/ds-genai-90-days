"""Day 39 helpers shared by local_eval.py (runs offline) and day39_analyst.py (runs on your trial).
compare_results: is the generated answer the same as the gold answer?  guard_sql: is the SQL safe to run?"""
import re
import pandas as pd

BLOCKED = r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|COPY|PUT|GET|CALL|EXECUTE|USE)\b"


def guard_sql(sql, allowed_tables=("ENCOUNTERS", "PATIENTS", "DRG_GROUPS", "__ENCOUNTERS", "__PATIENTS", "__DRG_GROUPS")):
    """Return (ok, reason). One read-only statement, no DDL/DML, only the three practice tables."""
    if not sql or not sql.strip():
        return False, "no SQL returned"
    s = re.sub(r"--[^\n]*", " ", sql)               # drop line comments
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S).strip().rstrip(";")
    if ";" in s:
        return False, "more than one statement"
    if not re.match(r"^\s*(WITH|SELECT)\b", s, re.I):
        return False, "not a SELECT"
    if re.search(BLOCKED, s, re.I):
        return False, "blocked keyword: " + re.search(BLOCKED, s, re.I).group(1).upper()
    ctes = {m.upper() for m in re.findall(r"(?:WITH|,)\s*([A-Za-z_][\w]*)\s+AS\s*\(", s, re.I)}
    refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.\"]*)", s, re.I)
    for t in refs:
        base = t.replace('"', "").split(".")[-1].upper()
        if base not in allowed_tables and base not in ctes:
            return False, f"table not allowed: {t}"
    return True, "ok"


def _norm_col(col):
    """Turn one column into a comparable list: numbers rounded, dates as ISO text, strings stripped."""
    out = []
    for v in col:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append("NULL")
        elif hasattr(v, "isoformat"):
            out.append(str(v.isoformat())[:10])
        else:
            try:
                out.append(f"{round(float(v), 3):.3f}")
            except (TypeError, ValueError):
                out.append(str(v).strip())
    return out


def compare_results(gold, gen, order_matters=False):
    """Result-set equivalence that ignores column NAMES and column ORDER, and row order unless order_matters.
    Extra columns in the generated answer are allowed if every gold column is matched. Returns (match, reason)."""
    if gen is None:
        return False, "no result"
    if len(gold) != len(gen):
        return False, f"row count {len(gen)} vs gold {len(gold)}"
    g = {c: _norm_col(gold[c]) for c in gold.columns}
    h = {c: _norm_col(gen[c]) for c in gen.columns}
    if not order_matters:                              # sort rows by the gold row tuples, then match columns
        key_g = sorted(range(len(gold)), key=lambda i: tuple(g[c][i] for c in gold.columns))
        g = {c: [v[i] for i in key_g] for c, v in g.items()}
    used, mapping = set(), {}
    for c, vals in g.items():
        for d, wals in h.items():
            if d in used:
                continue
            same = sorted(vals) == sorted(wals) if not order_matters else vals == wals
            if same:
                mapping[c] = d; used.add(d); break
        else:
            if len(vals) == 1 and len(h) >= 1:
                return False, f"{c}: got {list(h.values())[-1][0]}, gold {vals[0]}"
            return False, f"no generated column matches gold column {c}"
    if not order_matters:                              # rows must line up as whole tuples, not column by column
        rows_g = sorted(zip(*[g[c] for c in gold.columns]))
        rows_h = sorted(zip(*[h[mapping[c]] for c in gold.columns]))
        if rows_g != rows_h:
            return False, "same values per column but rows do not line up"
    extra = len(gen.columns) - len(mapping)
    return True, "match" + (f" (+{extra} extra column{'s' if extra > 1 else ''})" if extra else "")
