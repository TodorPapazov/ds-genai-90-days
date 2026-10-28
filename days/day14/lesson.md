# Day 14: Explain vs predict (week 2 wrap-up): odds ratios, a train/test split, AUC, calibration and lift on 30-day readmissions

## Concept
Week 2 ends with the question every healthcare analytics team gets wrong at least once: are you trying to explain readmissions or predict them? You simulate 4,000 discharges (seed 14) with a 9.0% 30-day readmission rate. To explain, you fit a logistic regression and read adjusted odds ratios: each prior admission 1.59 (95% CI 1.44-1.76), diabetes 1.34, length of stay 1.06 per day, and a 7-day clinic visit 0.76 (0.61-0.95). All have p < 0.05. To predict, you hold out 25% of patients as a test set the model never sees and compare against a baseline that gives everyone the average rate (AUC 0.50). The same logistic regression in scikit-learn scores AUC 0.72 on the test set and a Brier score of 0.0771 vs. 0.0819 for the baseline. Calibration by risk decile shows predicted risks roughly match observed rates, and the top decile is readmitted at 24%, a 2.7x lift, catching 27% of all readmissions. The key lesson: significant is not the same as predictive. The 7-day visit has p = 0.015, yet on its own it gives an AUC of just 0.52; prior admissions alone gives 0.67. You finish by writing a week 2 README in your repo: question, data, three charts, five findings and limits. Tomorrow, week 3 tries to beat this model with trees and random forests.

## Key numbers
- **Discharges:** 4,000 simulated (seed 14), 6 features, 30-day readmission
- **Readmission rate:** 9.0%
- **Split:** 3,000 train / 1,000 test, stratified
- **Odds ratios (explain):** prior admit 1.59 each; diabetes 1.34; LOS 1.06 per day; 7-day visit 0.76 (0.61-0.95)
- **Test AUC (predict):** model 0.723 vs baseline 0.500
- **Brier score:** model 0.0771 vs baseline 0.0819
- **Top risk decile:** 24% readmitted, 2.7x lift, catches 27% of readmissions
- **One-variable AUCs:** 7-day visit 0.52 (p = 0.015); diabetes 0.55; prior admits 0.67

## Hands-on
1. In your repo, create day14/ and open day14/explain_vs_predict.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate readmissions_30d.csv (seed 14). Note the base rate.
3. Write the business question two ways: one explain question for a CMO and one predict question for a care management team.
4. Run snippet 2: a statsmodels logistic regression. Build a table of odds ratios, 95% CIs and p-values.
5. Run snippet 3: a stratified train/test split, a baseline that predicts the training average, and a scikit-learn LogisticRegression. Record test AUC and Brier for both.
6. Run snippet 4: a decile table of predicted vs. observed rates, then top-decile lift and the share of readmissions caught.
7. Plot the ROC curve and a calibration plot side by side.
8. Run snippet 5: one-variable AUCs next to their p-values. Write two sentences on why the 7-day visit is significant but not predictive.
9. Write day14/README.md as your week 2 report: question, data, three charts, five plain-language findings, limits (simulated, no external validation, possible leakage and fairness issues).
10. Add an index to the README linking your day 1-13 notebooks.
11. Commit and push (snippet 6).
12. Stretch: load readmissions_30d.csv plus a risk_score column into Snowflake and build a decile lift table in SQL (snippet 7).

## Code
### 1. Simulate 30-day readmissions
```python
import numpy as np
import pandas as pd

# Simulate 4,000 hospital discharges with a 30-day readmission flag (fixed seed)
rng = np.random.default_rng(14)
n = 4000
age = rng.normal(66, 13, n).clip(18, 98).round()
prior_admits = rng.poisson(0.9, n)                         # admissions in the past 12 months
los_days = rng.gamma(2.0, 2.4, n).round(1).clip(1)         # length of stay
n_meds = rng.poisson(9, n)                                 # discharge medications
diabetes = (rng.random(n) < 0.32).astype(int)
followup_visit = (rng.random(n) < 0.55).astype(int)        # clinic visit within 7 days
# True log-odds of readmission: prior admits and long stays matter most
logit = (-2.6 + 0.45 * prior_admits + 0.06 * (los_days - 5) + 0.012 * (age - 66)
         + 0.03 * (n_meds - 9) + 0.25 * diabetes - 0.40 * followup_visit)
readmit30 = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"age": age, "prior_admits": prior_admits, "los_days": los_days, "n_meds": n_meds,
                   "diabetes": diabetes, "followup_visit": followup_visit, "readmit30": readmit30})
df.to_csv("readmissions_30d.csv", index=False)

df = pd.read_csv("readmissions_30d.csv")
print(df.describe().round(2).T[["mean", "std", "min", "max"]])
print(f"30-day readmission rate: {df.readmit30.mean():.1%}")
```

### 2. Explain: logistic regression odds ratios
```python
# Question 1 (explain): which factors are linked to readmission? Logistic regression odds ratios
import statsmodels.formula.api as smf
logit_m = smf.logit("readmit30 ~ age + prior_admits + los_days + n_meds + diabetes + followup_visit",
                    data=df).fit(disp=0)
ors = pd.DataFrame({"OR": np.exp(logit_m.params), "lo": np.exp(logit_m.conf_int()[0]),
                    "hi": np.exp(logit_m.conf_int()[1]), "p": logit_m.pvalues}).drop("Intercept")
print(ors.round(3))                    # OR > 1 = higher odds; all of these are "significant"
```

### 3. Predict: train/test split, baseline, AUC and Brier
```python
# Question 2 (predict): can we flag who will be readmitted? Hold out data the model never sees
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
features = ["age", "prior_admits", "los_days", "n_meds", "diabetes", "followup_visit"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df.readmit30, test_size=0.25,
                                                    stratify=df.readmit30, random_state=14)
base = np.full(len(y_test), y_train.mean())               # baseline: everyone gets the average rate
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
p_test = clf.predict_proba(X_test)[:, 1]
print(f"Baseline AUC {roc_auc_score(y_test, base):.3f}  Brier {brier_score_loss(y_test, base):.4f}")
print(f"Model    AUC {roc_auc_score(y_test, p_test):.3f}  Brier {brier_score_loss(y_test, p_test):.4f}")
```

### 4. Calibration by decile and top-decile lift
```python
# Is it useful? Top-decile lift and calibration (do predicted risks match observed rates?)
test = X_test.assign(y=y_test.values, p=p_test)
test["decile"] = pd.qcut(test.p, 10, labels=False) + 1
cal = test.groupby("decile").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
print(cal.round(3))
top = test[test.decile == 10]
print(f"Top 10% risk: observed {top.y.mean():.1%} vs overall {test.y.mean():.1%}  "
      f"(lift {top.y.mean() / test.y.mean():.1f}x), catches {top.y.sum() / test.y.sum():.0%} of readmissions")
```

### 5. Significant is not predictive
```python
# Small p-value is not the same as good prediction: one variable at a time
for f in ["followup_visit", "diabetes", "prior_admits"]:
    m = LogisticRegression().fit(X_train[[f]], y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test[[f]])[:, 1])
    print(f"{f:15s} p = {ors.loc[f, 'p']:.1e}   AUC alone = {auc:.3f}")
# Save a tidy summary for your week 2 report
summary = {"n": len(df), "rate": df.readmit30.mean(), "auc": roc_auc_score(y_test, p_test),
           "brier": brier_score_loss(y_test, p_test)}
pd.Series(summary).to_csv("week2_summary.csv")
```

### 6. Push to GitHub
```bash
git add day14
git commit -m "Day 14: explain vs predict on 30-day readmissions, week 2 report"
git push
```

### Stretch: Snowflake SQL decile lift table
```sql
-- decile lift table from scored test patients
WITH ranked AS (
  SELECT readmit30, risk_score,
         NTILE(10) OVER (ORDER BY risk_score) AS decile      -- 10 = highest risk
  FROM readmissions_scored
)
SELECT decile,
       COUNT(*)                                   AS patients,
       ROUND(AVG(risk_score), 3)                  AS mean_predicted,
       ROUND(AVG(readmit30), 3)                   AS observed_rate,
       ROUND(AVG(readmit30) / AVG(AVG(readmit30)) OVER (), 2) AS lift,
       ROUND(SUM(readmit30) / SUM(SUM(readmit30)) OVER (), 3) AS share_of_readmits
FROM ranked
GROUP BY decile
ORDER BY decile DESC;
```

## Printables
- Cheat sheet: `day14_v1_cheat_sheet.png`
- Infographic: `day14_v2_infographic.png`
- Code worksheet: `day14_v3_worksheet.png`
- Practice quiz: `day14_v4_quiz.png`
- PDF: `day14_lesson.pdf`

## Podcast script
(Audio: `day14_podcast.mp3`)

Hey Todor, welcome to day fourteen. That's the end of week two, so give yourself some credit. Two weeks ago you were summarizing skewed claims costs, and since then you've worked through hypothesis tests, confidence intervals, regression, missing data, confounding, and survival curves. Today ties all of that together and opens the door to machine learning.

Here's the big idea. There are two different questions you can ask of the same data. The first is, why does this happen? Which factors are linked to a patient coming back to the hospital within thirty days, and by how much? That's explaining. The second is, who is going to come back? Given a patient walking out the door today, how risky are they? That's predicting. They sound similar, but they need different tools and different ways of grading your work.

For explaining, you already know the tool. You fit a logistic regression and read the odds ratios. In today's simulated data, four thousand discharges with about a nine percent readmission rate, each prior admission in the past year multiplies the odds of coming back by about one point six. Diabetes bumps the odds up by about a third. And a clinic visit within seven days is linked to about twenty four percent lower odds. Every one of those has a small p-value.

Now here's the twist. For predicting, a p-value tells you almost nothing. What matters is how well the model ranks patients it has never seen. So you split the data. You train on three quarters and hide the last quarter as a test set, like an exam the model hasn't studied for. Then you compare against a baseline, the simplest honest guess, which gives everyone the average rate. That baseline has an AUC of point five, which is a coin flip.

AUC is worth saying in plain words. Pick one patient who was readmitted and one who wasn't. AUC is the chance the model gives the readmitted patient the higher risk score. Our simple model gets about point seven two. That's moderate. Useful, not magic.

And here's the lesson that sticks. The seven day visit has a p-value of about point zero one five, so it's a real effect. But use it alone to predict, and the AUC is point five two, barely better than a coin. Real and small is not the same as predictive. Prior admissions, on the other hand, carries most of the signal.

Two more ideas make this practical. Calibration asks whether the risk numbers are honest. If the model says twenty percent, do about twenty percent of those patients actually come back? You check that by grouping patients into ten risk bands and comparing predicted against observed. And lift turns it into an operations story. The top ten percent of risk scores had about a twenty four percent readmission rate, two point seven times the average, and that list held about a quarter of all readmissions. A care team can work with that.

Your hands-on today has a second job. Write a short week two README in your repo, with the question, the data, three charts, five findings, and the limits. That's the kind of artifact a hiring manager at a health system actually opens.

Tomorrow we start week three and classic machine learning for real. We'll take this same readmission problem and see whether decision trees and random forests can beat our logistic regression, and we'll talk about overfitting and cross validation so you can trust the answer. See you then.
