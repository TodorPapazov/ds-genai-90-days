# Day 8: Categorical outcomes: chi-square, relative risk and odds ratios on 30-day readmissions

## Concept
Last week the outcome was a number (cost). Today it is a yes or no: was the patient readmitted within 30 days? On 2,000 simulated discharges (seed 8), the readmission rate is 14.4%. A contingency table (pd.crosstab, with normalize='index' for row rates) shows 12.8% of patients sent home came back vs. 21.6% sent to a skilled nursing facility. The chi-square test of independence compares observed counts with the expected counts you would see if there were no relationship: chi2 = 14.16, dof = 3, p = 0.0027. Real, but Cramer's V is only 0.084 (weak). Chi-square needs every expected count to be at least 5; the 35 patients who left against medical advice (AMA) had 4.5 expected readmissions vs. home, so use Fisher's exact test (OR 2.03, p = 0.120, not significant). For a 2x2 table you report three effect sizes: smokers 21.5% vs. non-smokers 12.8% gives a risk difference of +8.7 points, a relative risk of 1.68 (95% CI 1.33 to 2.12) and an odds ratio of 1.87 (1.40 to 2.49). Odds = p/(1-p), so when an outcome is common the OR overstates the RR; at a 50% baseline the same RR of 1.68 would be an OR of 5.31. Simpson's paradox: patients who got a follow-up call were readmitted more (16.5% vs. 13.4%, crude OR 1.28) because calls went to 77% of high-risk vs. 18% of low-risk patients. Within each risk group the call does better (8.7% vs. 11.9%; 22.2% vs. 28.9%), and the Mantel-Haenszel pooled OR is 0.70 (0.51 to 0.97). Bridge to Week 3: in logistic regression, exp(coef) is an odds ratio; smoker alone gives exactly 1.87, and with call, smoking, chronic conditions and disposition in the model, the call OR is 0.61.

## Key numbers
- **Discharges / readmission rate:** 2,000 / 14.4%
- **Rate by disposition:** home 12.8%, home health 16.9%, SNF 21.6%, AMA 22.9% (n = 35)
- **Chi-square (disposition):** chi2 = 14.16, dof = 3, p = 0.0027; smallest expected 5.04
- **Cramer's V:** 0.084 (real but weak)
- **AMA vs. home (Fisher exact):** expected 4.5 < 5; OR 2.03, p = 0.120
- **Smoker vs. non-smoker:** 21.5% vs. 12.8%; risk difference +8.7 points
- **Relative risk vs. odds ratio:** RR 1.68 (1.33 to 2.12); OR 1.87 (1.40 to 2.49)
- **Follow-up call, crude:** 16.5% vs. 13.4%; OR 1.28 (0.99 to 1.66)
- **Follow-up call, by risk group:** Mantel-Haenszel OR 0.70 (0.51 to 0.97), p = 0.034; Breslow-Day p = 0.99
- **Logistic regression:** smoker OR 1.87 alone; adjusted call OR 0.61 (0.45 to 0.84)

## Hands-on
1. In your repo, create day08/, copy day07/patients.csv into it, and open day08/readmissions_categorical.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate readmissions.csv (seed 8, one discharge per patient) and check the readmission rate is 14.4%. Build pd.crosstab of smoker by readmit_30d with margins=True, then disposition with normalize='index'.
3. Run stats.chi2_contingency on disposition by readmission (snippet 2). Print the expected table, find the smallest expected count (5.04) and compute Cramer's V (0.084). Write one sentence: real or not, big or small?
4. Slice the AMA vs. home 2x2 table, confirm an expected count is under 5, and run stats.fisher_exact. Why is p = 0.12 with an OR of 2.03?
5. Build the smoker 2x2 with rows yes/no and columns readmit 1/0 (snippet 3). Compute the risk difference by hand, then RR and OR with 95% CIs from Table2x2. Run the baseline-risk loop and watch the OR pull away from the RR.
6. Compare readmission rates for followup_call yes vs. no, then split by risk_group (snippet 4). Explain the flip in two sentences, then get the Mantel-Haenszel pooled OR and the Breslow-Day test from StratifiedTable.
7. Fit smf.logit with smoker only and confirm exp(coef) = 1.87, then the adjusted model and build an OR table with CIs (snippet 5).
8. Add a Findings markdown cell: 5 plain-language bullets a care-management manager could act on, plus one Limits bullet (simulated, observational, not causal).
9. Commit and push (snippet 6).
10. Stretch: load readmissions.csv into Snowflake and build a readmission rate table with COUNT_IF and conditional aggregation (snippet 7).

## Code
### 1. Simulate readmissions and build contingency tables
```python
import numpy as np
import pandas as pd

# Start from the Day 7 patients and simulate one discharge per patient (fixed seed)
pt = pd.read_csv("patients.csv").drop_duplicates().reset_index(drop=True)   # 2,000
rng = np.random.default_rng(8)
pt["risk_group"] = np.where(pt["chronic_conditions"] >= 3, "high", "low")
pt["followup_call"] = np.where(rng.random(len(pt)) < np.where(pt["risk_group"] == "high", .75, .20), "yes", "no")
p_snf = np.where(pt["age"] >= 70, .20, .05)
u = rng.random(len(pt))
pt["discharge_disposition"] = np.select(
    [u < .02, u < .02 + p_snf, u < .20 + p_snf], ["AMA", "SNF", "home health"], "home")
logit = (-2.6 + .45 * pt["chronic_conditions"] + .55 * (pt["smoker"] == "yes")
         + .35 * (pt["discharge_disposition"] == "SNF") + .25 * (pt["discharge_disposition"] == "home health")
         + 1.0 * (pt["discharge_disposition"] == "AMA") - .45 * (pt["followup_call"] == "yes"))
pt["readmit_30d"] = (rng.random(len(pt)) < 1 / (1 + np.exp(-logit))).astype(int)
df = pt[["patient_id", "age", "plan", "smoker", "chronic_conditions", "risk_group",
         "followup_call", "discharge_disposition", "readmit_30d"]]
df.to_csv("readmissions.csv", index=False)
print(len(df), "discharges, readmission rate", round(df["readmit_30d"].mean(), 3))

# Contingency tables: counts, then row percentages (readmission rate per group)
print(pd.crosstab(df["smoker"], df["readmit_30d"], margins=True))
print(pd.crosstab(df["discharge_disposition"], df["readmit_30d"], normalize="index").round(3))
```

### 2. Chi-square, expected counts, Cramer's V and Fisher's exact
```python
from scipy import stats

# Chi-square test of independence: is readmission related to discharge disposition?
tab = pd.crosstab(df["discharge_disposition"], df["readmit_30d"])
chi2, p, dof, expected = stats.chi2_contingency(tab)
print(f"chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
print(pd.DataFrame(expected, index=tab.index, columns=tab.columns).round(1))
print("smallest expected count:", expected.min().round(2))   # rule of thumb: all >= 5

# Cramer's V: effect size from 0 (no link) to 1 (perfect link)
n = tab.to_numpy().sum()
cramers_v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
print("Cramer's V =", round(cramers_v, 3))

# Small table (AMA vs home): expected counts under 5, so use Fisher's exact test
small = tab.loc[["AMA", "home"], [1, 0]]          # rows: exposure, cols: readmit yes, no
print(small)
print("expected:", stats.contingency.expected_freq(small).round(1).tolist())
odds, p_fisher = stats.fisher_exact(small)
print(f"Fisher exact: OR = {odds:.2f}, p = {p_fisher:.3f}")
```

### 3. Risk difference, relative risk and odds ratio (Table2x2)
```python
from statsmodels.stats.contingency_tables import Table2x2

# 2x2 table: rows = exposure (smoker yes, no), cols = outcome (readmit yes, no)
t2 = pd.crosstab(df["smoker"], df["readmit_30d"]).loc[["yes", "no"], [1, 0]]
a, b = t2.loc["yes"]; c, d = t2.loc["no"]
risk_s, risk_n = a / (a + b), c / (c + d)
print(f"risk smokers {risk_s:.3f}, non-smokers {risk_n:.3f}")
print(f"risk difference {risk_s - risk_n:+.3f}  (extra readmits per 100 discharges)")

tt = Table2x2(t2.to_numpy())
lo, hi = tt.riskratio_confint()
print(f"relative risk {tt.riskratio:.2f}  (95% CI {lo:.2f} to {hi:.2f})")
lo, hi = tt.oddsratio_confint()
print(f"odds ratio    {tt.oddsratio:.2f}  (95% CI {lo:.2f} to {hi:.2f})")
print(tt.summary())

# Why OR > RR here: odds = p / (1 - p), and the gap grows as the outcome gets common
for base in [.01, .05, risk_n, .30, .50]:
    p1 = min(base * tt.riskratio, .99)
    print(f"baseline {base:.0%}: RR {tt.riskratio:.2f} -> OR {(p1/(1-p1))/(base/(1-base)):.2f}")
```

### 4. Simpson's paradox and Mantel-Haenszel (StratifiedTable)
```python
from statsmodels.stats.contingency_tables import StratifiedTable

# Crude view: patients who got a follow-up call look WORSE
print(df.groupby("followup_call")["readmit_30d"].mean().round(3))
crude = Table2x2(pd.crosstab(df["followup_call"], df["readmit_30d"]).loc[["yes", "no"], [1, 0]].to_numpy())
print(f"crude OR (call vs. no call) = {crude.oddsratio:.2f}")

# Stratify by the confounder: calls went mostly to high-risk patients
print(pd.crosstab(df["risk_group"], df["followup_call"], normalize="index").round(2))
print(df.groupby(["risk_group", "followup_call"])["readmit_30d"].mean().unstack().round(3))

tables = [pd.crosstab(g["followup_call"], g["readmit_30d"]).loc[["yes", "no"], [1, 0]].to_numpy()
          for _, g in df.groupby("risk_group")]
st = StratifiedTable(tables)
lo, hi = st.oddsratio_pooled_confint()
print(f"Mantel-Haenszel OR = {st.oddsratio_pooled:.2f} (95% CI {lo:.2f} to {hi:.2f})")
print(f"MH test p = {st.test_null_odds().pvalue:.4f}")
print(f"Breslow-Day (same OR in each stratum?) p = {st.test_equal_odds().pvalue:.3f}")
```

### 5. Bridge to Week 3: logistic regression odds ratios
```python
import statsmodels.formula.api as smf

# Bridge to Week 3: logistic regression models the log odds, so exp(coef) = odds ratio
m1 = smf.logit("readmit_30d ~ C(smoker, Treatment('no'))", data=df).fit(disp=0)
print("logit OR, smoker:", round(np.exp(m1.params.iloc[1]), 2), " (same as the 2x2 table)")

# Add the confounder and other predictors: each OR is now 'holding the others fixed'
m2 = smf.logit("readmit_30d ~ C(followup_call) + C(smoker) + chronic_conditions"
               " + C(discharge_disposition, Treatment('home'))", data=df).fit(disp=0)
ors = pd.DataFrame({"OR": np.exp(m2.params), "low": np.exp(m2.conf_int()[0]),
                    "high": np.exp(m2.conf_int()[1]), "p": m2.pvalues}).drop("Intercept")
print(ors.round(3))
```

### 6. Push to GitHub
```bash
git add day08
git commit -m "Day 8: chi-square, relative risk and odds ratios on 30-day readmissions"
git push
```

### Stretch: Snowflake SQL (readmission rate table)
```sql
SELECT discharge_disposition,
       COUNT(*)                                        AS discharges,
       COUNT_IF(readmit_30d = 1)                       AS readmits,
       ROUND(COUNT_IF(readmit_30d = 1) / COUNT(*), 3)  AS readmit_rate,
       SUM(IFF(smoker = 'yes', readmit_30d, 0))        AS smoker_readmits,
       ROUND(AVG(CASE WHEN smoker = 'yes' THEN readmit_30d END), 3) AS smoker_rate,
       ROUND(AVG(CASE WHEN smoker = 'no'  THEN readmit_30d END), 3) AS nonsmoker_rate
FROM readmissions
GROUP BY discharge_disposition
ORDER BY readmit_rate DESC;

-- crude vs. stratified view of the follow-up call
SELECT risk_group, followup_call,
       COUNT(*) AS n,
       ROUND(COUNT_IF(readmit_30d = 1) / COUNT(*), 3) AS readmit_rate
FROM readmissions
GROUP BY ROLLUP (risk_group, followup_call)
ORDER BY risk_group, followup_call;
```

## Printables
- Cheat sheet: `day08_v1_cheat_sheet.png`
- Infographic: `day08_v2_infographic.png`
- Code worksheet: `day08_v3_worksheet.png`
- Practice quiz: `day08_v4_quiz.png`
- PDF: `day08_lesson.pdf`

## Podcast script
(Audio: `day08_podcast.mp3`)

Hey Todor, welcome to Day Eight, the start of week two. All of last week, our outcome was a number, like annual cost. Today the outcome is a yes or no: was the patient readmitted to the hospital within thirty days of discharge? That is one of the most watched numbers in healthcare analytics, because hospitals get penalized for high readmission rates, so you will see it in interviews and on the job.

We simulated two thousand discharges, one per patient from our Day Seven table, with a fixed seed so your numbers match mine. Overall, fourteen point four percent of patients came back within thirty days.

Step one is always a contingency table. That is just a count for every combination of two categories, and pandas crosstab builds it in one line. Add normalize equals index and the counts turn into row percentages, which here means a readmission rate for each group. Patients sent home came back about thirteen percent of the time. Patients sent to a skilled nursing facility, about twenty two percent.

Is that difference real or just noise? That is the job of the chi-square test of independence. It asks: if discharge disposition and readmission had nothing to do with each other, what counts would we expect in each cell? Then it measures how far the real counts are from those expected counts. We got a chi-square of fourteen point two with three degrees of freedom and a p-value of about point zero zero three. So the link is real.

But real is not the same as big. For size, use Cramer's V, which runs from zero to one. Ours is point zero eight, which is weak. The honest summary is: disposition matters, but it explains only a small part of who comes back.

One warning. Chi-square needs expected counts of at least five in each cell. Our small group of thirty five patients who left against medical advice had an expected count of just four and a half readmissions. When that happens, switch to Fisher's exact test. Fisher gave a p-value of point one two, so with that few patients we cannot say much.

Now the measures your manager actually wants. Smokers were readmitted twenty one and a half percent of the time, non-smokers twelve point eight percent. The risk difference is almost nine extra readmissions per hundred discharges. The relative risk is one point six eight, meaning smokers are about sixty eight percent more likely to come back. The odds ratio is one point eight seven. Why bigger? Odds are the chance it happens divided by the chance it does not. When an outcome is rare, odds and risk are almost the same. When it is common, like fourteen percent and up, the odds ratio runs ahead of the relative risk. So never describe an odds ratio of one point eight seven as eighty seven percent more likely.

Here is my favorite part: Simpson's paradox. Patients who got a follow-up phone call after discharge were readmitted more often, sixteen and a half percent versus thirteen and a half. Does the call hurt? No. Care managers called the sickest patients first. Seventy seven percent of high risk patients got a call, versus eighteen percent of low risk. Split by risk group, and the call group does better in both: eight point seven versus eleven point nine percent in low risk, twenty two versus twenty nine percent in high risk. The Mantel-Haenszel method pools those strata into one odds ratio of point seven zero. That is about thirty percent lower odds of readmission at the same risk level. Always ask who got the treatment, and why.

Last, a bridge to week three. Logistic regression models the log odds of a yes or no outcome. Fit it with just smoking, take the exponent of the coefficient, and you get exactly one point eight seven, the same odds ratio as the table. Add more predictors, and each odds ratio means holding the others fixed, just like last week.

Tomorrow, on Day Nine, we compare more than two groups at once with ANOVA and the Kruskal-Wallis test, and learn why running twenty tests almost guarantees a false alarm. See you then.
