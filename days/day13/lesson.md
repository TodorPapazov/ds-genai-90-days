# Day 13: Survival analysis: censoring, Kaplan-Meier curves, log-rank and Cox hazard ratios for time to readmission

## Concept
Today you answer a time-to-event question: how fast do heart failure patients come back, and does a clinic visit within 7 days of discharge help? You follow 2,000 simulated discharges (seed 13) for up to 90 days. Many patients are censored: they reach day 90 without a readmission or leave follow-up early (429 did). Censored means "at least this long", not "never" and not missing. Ignore censoring and you get it wrong: the naive rate says 38.0% were readmitted, but the Kaplan-Meier estimate of 90-day risk is 42.8%. Kaplan-Meier walks through event times and multiplies survival by (1 - events / patients still at risk); censored patients leave the risk set without pulling the curve down. By group, 30-day readmission risk is 23.1% without a 7-day visit vs. 13.1% with one, and the log-rank test says the curves differ (chi2 46.1, p = 1e-11). A Cox proportional hazards model adjusts for age and low ejection fraction: HR 0.61 (95% CI 0.53-0.70) for the visit, 1.39 for low EF, 1.03 per year of age. A hazard ratio is a ratio of rates, not probabilities, and it assumes the ratio is steady over time (a split check gives 0.54 in days 0-30 vs. 0.67 in days 30-90, CIs overlap). This is observational: patients who make the visit may be healthier or better supported, so treat it as evidence for a follow-up program, not proof.

## Key numbers
- **Discharges:** 2,000 simulated heart failure patients (seed 13), 90-day follow-up
- **Readmissions observed:** 761 (38.0%)
- **Censored:** 1,239; 429 lost to follow-up before day 90
- **Kaplan-Meier 90-day risk:** 42.8% overall (naive events / all: 38.0%)
- **30-day readmission risk:** no 7-day visit 23.1%; 7-day visit 13.1%
- **Median time to readmission:** no visit about 86 days; visit not reached (64% readmission-free at day 90)
- **Log-rank test:** chi2 46.1, p = 1e-11
- **Cox hazard ratios:** 7-day visit 0.61 (0.53-0.70); low EF 1.39 (1.21-1.61); age 1.03 per year
- **PH check (visit HR):** days 0-30: 0.54; days 30-90: 0.67 (CIs overlap)

## Hands-on
1. In your repo, create day13/ and open day13/survival_readmissions.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate hf_readmissions.csv (seed 13). Count events vs. censored and how many left before day 90.
3. Run snippet 2: see why the naive mean of days is misleading, then build a Kaplan-Meier table by hand (at risk, events, cumulative product).
4. Run snippet 3: Kaplan-Meier with statsmodels per group. Read off 30- and 90-day readmission-free values and plot the two step curves.
5. Write two sentences explaining censoring to a care manager who has never heard the word.
6. Run snippet 4: the log-rank test, then a Cox model for the visit adjusted for age and low EF. Read each hazard ratio and its 95% CI.
7. Run snippet 5: a quick proportional-hazards check (visit HR in days 0-30 vs. 30-90) and plain-language 30-day risk per group.
8. Make one chart for a care management director: the two KM curves with a day-30 marker and the 30-day risks labeled.
9. Add a Findings markdown cell: 5 plain-language bullets, plus a limits bullet (simulated, observational, who shows up to visits, censoring assumed unrelated to risk).
10. Commit and push (snippet 6).
11. Stretch: load hf_readmissions.csv into Snowflake and build a weekly life table (at risk, events, censored, cumulative survival) in SQL (snippet 7).

## Code
### 1. Simulate heart failure discharges with censoring
```python
import numpy as np
import pandas as pd

# Simulate 2,000 heart failure discharges, followed up to 90 days for readmission (fixed seed)
rng = np.random.default_rng(13)
n = 2000
followup_visit = rng.random(n) < 0.55                      # clinic visit within 7 days of discharge
age = rng.normal(72, 10, n).clip(40, 98).round()
low_ef = rng.random(n) < 0.40                              # ejection fraction < 40%
# True readmission hazard: higher for older and low-EF patients, lower with an early visit
hazard = 0.006 * np.exp(0.025 * (age - 72) + 0.50 * low_ef - 0.45 * followup_visit)
t_event = rng.exponential(1 / hazard)                     # days until readmission
t_lost = rng.uniform(10, 300, n)                          # moved away, switched systems, etc.
days = np.minimum.reduce([t_event, t_lost, np.full(n, 90.0)]).round(1).clip(0.5)
readmitted = (t_event <= np.minimum(t_lost, 90)).astype(int)   # 1 = event seen, 0 = censored
df = pd.DataFrame({"followup_visit": followup_visit.astype(int), "age": age,
                   "low_ef": low_ef.astype(int), "days": days, "readmitted": readmitted})
df.to_csv("hf_readmissions.csv", index=False)

df = pd.read_csv("hf_readmissions.csv")
print(df.head())
print(f"Readmitted: {df.readmitted.mean():.1%}   censored: {1 - df.readmitted.mean():.1%}")
print("Censored before day 90 (lost to follow-up):", ((df.readmitted == 0) & (df.days < 90)).sum())
```

### 2. Why naive averages fail, and Kaplan-Meier by hand
```python
# Why you can't just average "days" or use a plain readmission rate
print("Naive mean days (treats censored as events):", round(df.days.mean(), 1))
print("Mean days among readmitted only:", round(df.loc[df.readmitted == 1, "days"].mean(), 1))
# Kaplan-Meier by hand: at each event time, S(t) = S(t-1) * (1 - deaths / at_risk)
km = (df.groupby("days").agg(events=("readmitted", "sum"), leaving=("readmitted", "size"))
        .sort_index())
km["at_risk"] = km["leaving"][::-1].cumsum()[::-1]      # still in follow-up just before t
km["surv"] = (1 - km.events / km.at_risk).cumprod()
print(km.head(8).round(4))
```

### 3. Kaplan-Meier curves per group with statsmodels
```python
import statsmodels.api as sm

# Kaplan-Meier with statsmodels, one curve per group
curves = {}
for g, d in df.groupby("followup_visit"):
    sf = sm.SurvfuncRight(d.days, d.readmitted)
    curves[g] = sf
    s30 = sf.surv_prob[sf.surv_times <= 30][-1]
    s90 = sf.surv_prob[-1]
    print(f"visit={g}: n={len(d)}, readmission-free at 30d {s30:.1%}, at 90d {s90:.1%}")
print(curves[1].summary().head())                       # times, survival, std error, n at risk

# Plot both curves as step functions
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for g, lab in [(0, "no 7-day visit"), (1, "7-day visit")]:
    sf = curves[g]
    ax.step(np.r_[0, sf.surv_times], np.r_[1, sf.surv_prob], where="post", label=lab)
ax.set(xlabel="days since discharge", ylabel="readmission-free", ylim=(0.5, 1)); ax.legend()
plt.show()
```

### 4. Log-rank test and Cox proportional hazards
```python
# Log-rank test: are the two curves different over the whole 90 days?
from statsmodels.duration.survfunc import survdiff
stat, p = survdiff(df.days, df.readmitted, df.followup_visit)
print(f"Log-rank chi2 = {stat:.1f}, p = {p:.2g}")

# Cox proportional hazards: hazard ratios, adjusted for age and low EF
import statsmodels.formula.api as smf
cox = smf.phreg("days ~ followup_visit + age + low_ef", data=df, status=df.readmitted).fit()
hr = pd.DataFrame({"HR": np.exp(cox.params), "lo": np.exp(cox.conf_int()[:, 0]),
                   "hi": np.exp(cox.conf_int()[:, 1]), "p": cox.pvalues}, index=cox.model.exog_names)
print(hr.round(3))                                       # HR < 1 = lower readmission hazard
```

### 5. Proportional-hazards check and 30-day risk
```python
# Quick proportional-hazards check: does the visit effect fade over time?
early = df.assign(days=df.days.clip(upper=30), readmitted=np.where(df.days <= 30, df.readmitted, 0))
late = df[df.days > 30].copy()
for name, d in [("days 0-30", early), ("days 30-90", late)]:
    m = smf.phreg("days ~ followup_visit + age + low_ef", data=d, status=d.readmitted).fit()
    print(f"{name}: HR for visit = {np.exp(m.params[0]):.2f}")   # similar HRs = assumption looks OK
# Plain-language 30-day risk by group (what a care manager wants)
for g in [0, 1]:
    sf = curves[g]; s30 = sf.surv_prob[sf.surv_times <= 30][-1]
    print(f"visit={g}: estimated 30-day readmission risk {1 - s30:.1%}")
```

### 6. Push to GitHub
```bash
git add day13
git commit -m "Day 13: survival analysis, Kaplan-Meier, log-rank and Cox on heart failure readmissions"
git push
```

### Stretch: Snowflake SQL weekly life table
```sql
-- weekly life table: at risk, events, censored, cumulative readmission-free
WITH wk AS (
  SELECT FLOOR(days / 7) AS week, readmitted
  FROM hf_readmissions
), agg AS (
  SELECT week,
         COUNT(*)                        AS leaving,
         SUM(readmitted)                 AS events,
         COUNT(*) - SUM(readmitted)      AS censored
  FROM wk
  GROUP BY week
), lt AS (
  SELECT week, events, censored,
         SUM(leaving) OVER (ORDER BY week DESC) AS at_risk   -- still followed at start of week
  FROM agg
)
SELECT week, at_risk, events, censored,
       ROUND(EXP(SUM(LN(1 - events / (at_risk - censored / 2)))
             OVER (ORDER BY week)), 4)  AS surv   -- actuarial: censored count half a week
FROM lt
ORDER BY week;
```

## Printables
- Cheat sheet: `day13_v1_cheat_sheet.png`
- Infographic: `day13_v2_infographic.png`
- Code worksheet: `day13_v3_worksheet.png`
- Practice quiz: `day13_v4_quiz.png`
- PDF: `day13_lesson.pdf`

## Podcast script
(Audio: `day13_podcast.mp3`)

Hey Todor, welcome to day thirteen. Today we're picking up a tool that shows up all over healthcare analytics, but that most analysts never get formally taught. It's called survival analysis, and despite the name, it's not only about death. It's about any question that sounds like "how long until something happens." How long until a heart failure patient is readmitted. How long until a member drops their plan. How long until a patient refills a prescription.

Here's the scenario. We have two thousand simulated heart failure patients, and we follow each of them for up to ninety days after discharge. About fifty five percent had a clinic visit within seven days of going home. The question a care manager wants answered is simple: do patients who get that early visit come back to the hospital less often, and how much less?

The tricky part is something called censoring. Not every patient gets watched for the full ninety days. Some move away, some switch to another health system, and many simply reach day ninety without being readmitted. For all of those people, we don't know when they would have come back. We only know they made it at least that long. That's a censored observation. It's not a zero, and it's not missing. It's partial information, and survival methods are built to use it.

If you ignore censoring, you get the wrong answer. In our data, thirty eight percent of patients had a readmission we actually saw. But the proper estimate of ninety day readmission risk is closer to forty three percent, because some of the people who left early would have been readmitted if we'd kept watching. Averaging the days column is even worse, because it mixes real event times with times that just mean "we stopped looking."

The fix is the Kaplan-Meier curve. The idea is beautifully simple. Walk through time day by day. Every time someone is readmitted, look at how many patients were still being followed and still readmission free just before that moment. That's the risk set. Multiply your running survival by one minus the share who had the event. When a patient is censored, they quietly leave the risk set without pulling the curve down. Do that for each group and you get two step curves you can put in front of any clinical leader.

In our data, the curves split early. Without a seven day visit, the thirty day readmission risk is about twenty three percent. With one, it's about thirteen percent. A log-rank test asks whether the two curves differ over the whole follow-up, and here the answer is a very confident yes.

But the groups might differ in other ways, like age or heart function. That's where the Cox proportional hazards model comes in. Think of it as regression for time to event data. It gives you hazard ratios. After adjusting for age and low ejection fraction, the hazard ratio for the early visit is about zero point six one, which means a roughly thirty nine percent lower readmission rate at any point in follow-up, among otherwise similar patients. Low ejection fraction pushes the hazard up by about forty percent, and each extra decade of age by about a third.

Two cautions. First, a hazard ratio is a ratio of rates, not a share of patients, so say "lower hazard," not "thirty nine percent fewer readmissions." Second, the Cox model assumes that ratio stays roughly steady over time. We checked by fitting the first thirty days and the last sixty separately, and the two estimates overlap, so we're comfortable. And as always, this is observational. Healthier or better supported patients may be the ones who show up to that visit. Yesterday's lesson on confounding applies here too.

For your hands-on, you'll simulate the data, build a Kaplan-Meier curve by hand, then with statsmodels, run the log-rank test and the Cox model, and write findings a care manager would understand. This is a great one to mention in interviews, because readmission work is everywhere in hospital and payer analytics.

Tomorrow is day fourteen, the end of week two. We'll pull everything together into a week two mini report, a polished, portfolio ready notebook that tells one clear healthcare story from start to finish. See you then.
