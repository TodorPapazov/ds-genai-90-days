# Day 12: Confounding and risk adjustment: Simpson's paradox, standardization and O/E ratios for a fair hospital comparison

## Concept
Today you compare 30-day surgical mortality at two simulated hospitals (6,000 admissions, seed 12). The crude numbers say Hospital A, a referral center, is almost twice as deadly: 6.8% vs. 3.6%. But A's patients are 61% high severity vs. 14% at B, and high-severity patients die far more often (11.7% vs. 1.3%). Severity is a confounder: it is tied to both the hospital and the outcome. Stratify and the story flips: A is lower in both groups (low 1.2% vs. 1.4%, high 10.4% vs. 17.1%). That is Simpson's paradox: the crude rate is a weighted average, and A's weights lean toward the deadly stratum. Three ways to get one fair number: direct standardization applies each hospital's stratum rates to one standard case mix (A 4.6% vs. B 7.2%); the Mantel-Haenszel odds ratio pools across strata (crude OR 1.98, MH OR 0.62, 95% CI 0.47-0.81), and logistic regression with severity and age agrees (0.60); and observed/expected (O/E) uses a case-mix model without hospital to predict expected deaths: A 202 vs. 226 expected (O/E 0.89), B 108 vs. 84 expected (O/E 1.29). Adjust only for things that exist before the exposure (severity, age), never for a mediator like an in-stay complication. Adjustment fixes only the confounders you measured, so always show crude and adjusted side by side and state the limits.

## Key numbers
- **Admissions:** 6,000 simulated surgical admissions (seed 12), two hospitals
- **Crude 30-day mortality:** A 6.8% (202/2,964); B 3.6% (108/3,036)
- **High-severity share:** A 61%; B 14%
- **Mortality by severity:** high 11.7%; low 1.3%
- **Stratified mortality:** low: A 1.2% vs. B 1.4%; high: A 10.4% vs. B 17.1%
- **Directly standardized:** A 4.6%; B 7.2% (standard = combined mix, 63% low / 37% high)
- **Odds ratio A vs. B:** crude 1.98 (1.56-2.52); Mantel-Haenszel 0.62 (0.47-0.81); logistic adjusted 0.60 (0.46-0.79)
- **Breslow-Day homogeneity:** p = 0.24 (stratum ORs similar enough to pool)
- **Observed / expected:** A 202 / 226 = 0.89; B 108 / 84 = 1.29

## Hands-on
1. In your repo, create day12/ and open day12/confounding_risk_adjustment.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate surgical_outcomes.csv (seed 12). Write down which hospital looks worse on crude mortality, then check the hospital by severity crosstab.
3. Run snippet 2: mortality by hospital within each severity level. Confirm severity is tied to both the hospital and death.
4. Write two sentences explaining Simpson's paradox with these numbers, as if to a non-technical manager.
5. Run snippet 3: direct standardization to the combined case mix, then the crude vs. Mantel-Haenszel odds ratio and the Breslow-Day test.
6. Run snippet 4: logistic regression for hospital, crude vs. adjusted for severity and age. Compare the adjusted OR to the Mantel-Haenszel OR.
7. Run snippet 5: fit a case-mix model without hospital, sum predicted risks to get expected deaths, and compute O/E per hospital.
8. Make one chart for a quality director: crude vs. adjusted mortality side by side for both hospitals.
9. Add a Findings markdown cell: 5 plain-language bullets, plus a limits bullet (simulated, unmeasured confounders, severity coding differences, don't adjust for mediators).
10. Commit and push (snippet 6).
11. Stretch: load surgical_outcomes.csv into Snowflake and compute stratified rates and an O/E ratio in SQL using the overall severity-level rates as the expected risk (snippet 7).

## Code
### 1. Simulate two hospitals, crude comparison, case mix
```python
import numpy as np
import pandas as pd

# Simulate 6,000 surgical admissions at two hospitals (fixed seed)
rng = np.random.default_rng(12)
n = 6000
hospital = np.where(rng.random(n) < 0.5, "A (referral center)", "B (community)")
is_a = hospital == "A (referral center)"
# Case mix: the referral center gets far more high-severity patients
p_high = np.where(is_a, 0.60, 0.15)
severity = np.where(rng.random(n) < p_high, "high", "low")
age = np.where(severity == "high", rng.normal(71, 9, n), rng.normal(58, 12, n)).clip(18, 95).round()
# True 30-day mortality depends on severity and age; hospital A is actually a bit BETTER
logit = -4.0 + 2.0 * (severity == "high") + 0.03 * (age - 60) - 0.45 * is_a
died = rng.random(n) < 1 / (1 + np.exp(-logit))
df = pd.DataFrame({"hospital": hospital, "severity": severity, "age": age, "died": died.astype(int)})
df.to_csv("surgical_outcomes.csv", index=False)

# Crude (unadjusted) comparison: what a basic dashboard shows
df = pd.read_csv("surgical_outcomes.csv")
crude = df.groupby("hospital")["died"].agg(["size", "sum", "mean"])
print(crude.round(4))                                           # A looks worse
print(pd.crosstab(df.hospital, df.severity, normalize="index").round(3))  # ...but look at who A treats
```

### 2. Stratify: Simpson's paradox and the confounder check
```python
# Stratify: compare like with like, within each severity level
strat = df.pivot_table(index="severity", columns="hospital", values="died", aggfunc="mean")
print(strat.round(4))                                           # A is LOWER in both strata: Simpson's paradox
counts = df.pivot_table(index="severity", columns="hospital", values="died", aggfunc="size")
print(counts)

# Confounder check: severity is linked to BOTH the hospital and the outcome
print(df.groupby("severity")["died"].mean().round(4).rename("mortality by severity"))
print(df.groupby("hospital")["severity"].apply(lambda s: (s == "high").mean()).round(3).rename("share high severity"))
```

### 3. Direct standardization and the Mantel-Haenszel odds ratio
```python
# Direct standardization: apply each hospital's stratum rates to ONE standard population
standard = df["severity"].value_counts(normalize=True)          # combined case mix as the standard
adj = (strat.mul(standard, axis=0)).sum()
print("Standard mix:", standard.round(3).to_dict())
print("Crude:   ", crude["mean"].round(4).to_dict())
print("Adjusted:", adj.round(4).to_dict())

# Mantel-Haenszel: one odds ratio for A vs B, pooled across severity strata
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
tables = []
def two_by_two(g):                                             # rows: A, B; columns: died, survived
    return pd.crosstab(g.hospital, g.died).loc[["A (referral center)", "B (community)"], [1, 0]].values
tables = [two_by_two(g) for _, g in df.groupby("severity")]
crude_t = two_by_two(df)
st = StratifiedTable(tables)
print(f"Crude OR A vs B: {Table2x2(crude_t).oddsratio:.2f}")
print(f"MH OR A vs B: {st.oddsratio_pooled:.2f}, 95% CI {np.round(st.oddsratio_pooled_confint(), 2)}")
print(f"Breslow-Day homogeneity p = {st.test_equal_odds().pvalue:.3f}")   # are the stratum ORs similar?
```

### 4. Regression adjustment: crude vs. adjusted odds ratio
```python
import statsmodels.formula.api as smf

# Regression adjustment: add the confounders, read the hospital coefficient
df["hosp_a"] = (df.hospital == "A (referral center)").astype(int)
m_crude = smf.logit("died ~ hosp_a", data=df).fit(disp=0)
m_adj = smf.logit("died ~ hosp_a + C(severity) + age", data=df).fit(disp=0)
for name, m in [("crude", m_crude), ("adjusted", m_adj)]:
    or_ = np.exp(m.params["hosp_a"]); lo, hi = np.exp(m.conf_int().loc["hosp_a"])
    print(f"{name:9s} OR A vs B = {or_:.2f} (95% CI {lo:.2f}-{hi:.2f})")
```

### 5. Observed / expected (O/E) ratios, like hospital quality reports
```python
# Observed / Expected (O/E): how hospital quality reports are built
# 1) fit a case-mix model WITHOUT hospital, 2) expected deaths = sum of predicted risks, 3) O/E = observed / expected
risk = smf.logit("died ~ C(severity) + age", data=df).fit(disp=0)
df["expected"] = risk.predict(df)
oe = df.groupby("hospital").agg(observed=("died", "sum"), expected=("expected", "sum"))
oe["O/E"] = oe.observed / oe.expected
oe["risk_adj_rate"] = oe["O/E"] * df.died.mean()               # O/E times the overall rate
print(oe.round(3))                                              # O/E < 1: fewer deaths than its case mix predicts
```

### 6. Push to GitHub
```bash
git add day12
git commit -m "Day 12: confounding, Simpson's paradox and risk adjustment on surgical mortality"
git push
```

### Stretch: Snowflake SQL (stratified rates and O/E)
```sql
-- stratified mortality: compare like with like
SELECT severity, hospital,
       COUNT(*)                   AS admissions,
       ROUND(AVG(died) * 100, 1)  AS mortality_pct
FROM surgical_outcomes
GROUP BY severity, hospital
ORDER BY severity, hospital;

-- simple O/E: expected risk = overall mortality for the patient's severity level
WITH sev_rate AS (
  SELECT severity, AVG(died) AS exp_risk
  FROM surgical_outcomes
  GROUP BY severity
)
SELECT s.hospital,
       SUM(s.died)                                AS observed,
       ROUND(SUM(r.exp_risk), 1)                  AS expected,
       ROUND(SUM(s.died) / SUM(r.exp_risk), 2)    AS oe_ratio
FROM surgical_outcomes s
JOIN sev_rate r ON s.severity = r.severity
GROUP BY s.hospital;
```

## Printables
- Cheat sheet: `day12_v1_cheat_sheet.png`
- Infographic: `day12_v2_infographic.png`
- Code worksheet: `day12_v3_worksheet.png`
- Practice quiz: `day12_v4_quiz.png`
- PDF: `day12_lesson.pdf`

## Podcast script
(Audio: `day12_podcast.mp3`)

Hey Todor, welcome to day twelve. Today we're tackling one of the most important ideas in healthcare analytics, and one of the easiest to get wrong: confounding. And we're going to do it with a question every hospital system asks: which hospital is safer?

Here's the setup. We simulated six thousand surgical admissions at two hospitals. Hospital A is a big referral center. Hospital B is a community hospital. For each patient we know their severity, low or high, their age, and whether they died within thirty days.

Start with the number a basic dashboard would show. Hospital A's thirty day mortality is about six point eight percent. Hospital B's is about three point six percent. So A looks almost twice as deadly. If you stopped there, you might tell leadership to send patients to B.

But then you ask the question that separates an analyst from a data scientist: do these two hospitals treat the same kind of patients? They don't. About sixty one percent of A's patients are high severity. At B, it's about fourteen percent. And high severity patients die far more often, roughly twelve percent versus about one percent for low severity. That's what a confounder is. It's something tied to both the group you're comparing and the outcome you care about. Severity pushes patients toward the referral center, and severity also drives death.

So you stratify. You compare like with like. Among low severity patients, A's mortality is about one point two percent and B's is one point four. Among high severity patients, A is about ten point four percent and B is seventeen point one. Hospital A is better in both groups, yet worse overall. That flip has a name: Simpson's paradox. It happens because the overall rate is a weighted average, and most of A's patients sit in the dangerous group.

Next, you put one fair number on each hospital. The first tool is direct standardization. You pick one standard case mix, here the combined mix of both hospitals, and you apply each hospital's severity specific rates to it. Now both hospitals are graded on the same patients. A comes out around four point six percent and B around seven point two. The ranking flips.

The second tool is a pooled odds ratio. The crude odds ratio for A versus B is about two, meaning A looks twice as risky. The Mantel Haenszel method pools the odds ratio across the severity groups, and it lands near zero point six, with a confidence interval that stays below one. A logistic regression with hospital, severity and age gives almost the same answer, about zero point six. When two different methods agree, you can trust the direction more.

The third tool is the one real quality reports use: observed over expected. You build a risk model with only the patient factors, severity and age, and leave the hospital out. That model tells you how many deaths an average hospital would see with these exact patients. Then you divide actual deaths by expected deaths. Hospital A had two hundred two deaths where its case mix predicted about two hundred twenty six, so its ratio is about point eight nine, better than expected. Hospital B had one hundred eight deaths where it should have had about eighty four, a ratio near one point three. If you've ever seen a CMS star rating or a standardized mortality ratio, this is the engine underneath.

Two warnings before you start. First, only adjust for things that exist before the exposure, like severity on arrival or age. Don't adjust for a complication that happened during the stay. That's on the path between the hospital and the outcome, so adjusting for it hides part of the very quality difference you want to measure. Second, adjustment only fixes the confounders you measured. Frailty, social factors, or one hospital coding severity more generously can still bend the answer. So always show the crude and adjusted numbers side by side, say what you adjusted for, and list what you couldn't.

Your hands-on today is a notebook in your day twelve folder: simulate the data, find the paradox, standardize, run the Mantel Haenszel and logistic models, compute observed over expected, and finish with five plain bullets a quality director could act on. This is exactly the kind of work that shows up in healthcare data science job descriptions as risk adjustment and outcomes analytics.

Tomorrow, we take everything from week two and practice telling the story: feature level EDA and a data quality report you could hand to a stakeholder. See you then.
