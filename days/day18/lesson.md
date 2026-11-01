# Day 18: Imbalanced outcomes: the accuracy trap, precision-recall, and choosing a threshold on 30-day readmissions

## Concept
Only 9% of patients in our readmissions data come back within 30 days, and a rare outcome changes how you judge a model. A model that says nobody is readmitted is 91% accurate and useless. Logistic regression from day 14 at the default .predict() cutoff of 0.5 does nearly the same: it flags 1 patient and catches 0 of 90 readmissions, because almost no one has a predicted risk above 50%. So you switch to precision (of patients flagged, share truly readmitted) and recall (of patients readmitted, share flagged in advance), and the precision-recall curve. Its random-guess floor is the prevalence, 0.09, not 0.5: average precision 0.23 is about 2.5x chance while ROC AUC is 0.723. The cutoff is a business decision, chosen on out-of-fold training scores and applied to test once. With illustrative costs ($15,000 per readmission, $500 per nurse follow-up, 25% prevented) the break-even risk is 0.133 and the cost curve picks t = 0.12: 201 flagged, 37 of 90 caught (recall 0.41). A capacity rule (call the riskiest 10%) gives t = 0.15: 98 flagged, 24 caught, precision 0.24. Finally, class_weight='balanced' keeps AUC (0.724) but pushes mean predicted risk to 0.47 versus a true 0.09, so its scores are no longer probabilities. Moving the threshold gets the same benefit without breaking calibration. Tomorrow: unsupervised learning with k-means clustering.

## Key numbers
- **Data:** days 14-17 simulated readmissions, 4,000 discharges, 9.0% readmitted, same stratified split
- **Accuracy trap:** 'nobody' model 91.0% accurate; logistic at 0.5 is 90.9% accurate, flags 1, catches 0 of 90
- **Ranking quality:** ROC AUC 0.723; average precision 0.230 vs a random-guess floor of 0.090
- **Cost rule (illustrative):** $15k readmission, $500 call, 25% prevented: break-even 0.133, chosen t = 0.12, 201 flagged, precision 0.18, recall 0.41
- **Capacity rule:** top 10% of out-of-fold scores: t = 0.154, 98 flagged, precision 0.24, recall 0.27
- **class_weight='balanced':** AUC 0.724, AP 0.232, mean risk 0.47 (true 0.09), Brier 0.226 vs 0.077

## Hands-on
1. In your repo, create day18/, run conda activate ds90, and open day18/thresholds.ipynb in VS Code.
2. Run snippet 1 to rebuild the same data and train/test split as days 14-17.
3. Run snippet 2: score the 'nobody is readmitted' model, then logistic regression at the default 0.5 cutoff. Print and label the confusion matrix.
4. Run snippet 3: plot the precision-recall curve with the prevalence baseline, and build a table of flags, precision and recall at five thresholds.
5. Run snippet 4: make out-of-fold scores on the TRAINING data with cross_val_predict, plot the cost curve, check the break-even math, then apply the cost and top-10% rules to the test set once.
6. Run snippet 5: compare plain vs class_weight='balanced' logistic regression (AUC, AP, mean risk, Brier) and plot both risk histograms.
7. Write day18/README.md: three charts, a threshold table, and the rule you would recommend to a care-management team, with reasons.
8. Commit and push (snippet 6).
9. Stretch: compute precision and recall at several thresholds in Snowflake SQL (snippet 7).

## Code
### 1. Same data and split as days 14-17
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14-17, so every model is compared fairly
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
from sklearn.model_selection import train_test_split
features = ["age", "prior_admits", "los_days", "n_meds", "diabetes", "followup_visit"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df.readmit30, test_size=0.25,
                                                    stratify=df.readmit30, random_state=14)
```

### 2. The accuracy trap at the default 0.5 cutoff
```python
# The accuracy trap: a model that never flags anyone is 91% "accurate"
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
print(f"Readmission rate in test: {y_test.mean():.1%}")
print(f"'Nobody is readmitted' accuracy: {accuracy_score(y_test, np.zeros(len(y_test))):.1%}")
lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
p_lr = lr_model.predict_proba(X_test)[:, 1]
pred_05 = (p_lr >= 0.5).astype(int)                       # the default .predict() threshold
print(f"Logistic at 0.5: accuracy {accuracy_score(y_test, pred_05):.1%}, flags {pred_05.sum()} patients, "
      f"recall {recall_score(y_test, pred_05, zero_division=0):.1%}")
print(confusion_matrix(y_test, pred_05))                  # rows = actual (no, yes), cols = predicted
```

### 3. Precision-recall curve and a threshold table
```python
# Precision-recall curve: the honest picture when positives are rare
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
prec, rec, thr = precision_recall_curve(y_test, p_lr)
ap = average_precision_score(y_test, p_lr)
print(f"ROC AUC {roc_auc_score(y_test, p_lr):.3f} | average precision (PR AUC) {ap:.3f} | "
      f"random-guess AP = prevalence {y_test.mean():.3f}")
for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
    pred = (p_lr >= t).astype(int)
    print(f"threshold {t:.2f}: flags {pred.sum():4d}  precision {precision_score(y_test, pred, zero_division=0):.2f}  "
          f"recall {recall_score(y_test, pred):.2f}")
```

### 4. Choose the threshold on out-of-fold training scores: cost and capacity rules
```python
# Pick the threshold on TRAINING data (out-of-fold scores), then apply it once to test
from sklearn.model_selection import cross_val_predict, StratifiedKFold
oof = cross_val_predict(LogisticRegression(max_iter=1000), X_train, y_train,
                        cv=StratifiedKFold(5, shuffle=True, random_state=14), method="predict_proba")[:, 1]
# Rule A, cost (illustrative numbers): a readmission costs ~$15,000, a nurse follow-up program costs ~$500
# per patient, and the program prevents ~25% of readmissions among the patients it reaches
cost_readmit, cost_call, prevent = 15000, 500, 0.25
grid_t = np.round(np.arange(0.02, 0.60, 0.01), 2)
cost = [cost_call * (oof >= t).sum() + cost_readmit * (y_train.sum() - prevent * ((oof >= t) & (y_train == 1)).sum())
        for t in grid_t]
print(f"break-even risk = cost_call / (prevent * cost_readmit) = {cost_call / (prevent * cost_readmit):.3f}")
t_cost = grid_t[int(np.argmin(cost))]
# Rule B, capacity: the care team can only call the riskiest 10% of discharges
t_cap = np.quantile(oof, 0.90)
for name, t in [("default 0.5", 0.5), ("cost-based", t_cost), ("top-10% capacity", t_cap)]:
    pred = (p_lr >= t).astype(int)
    print(f"{name:17s} t={t:.3f}  flags {pred.sum():4d}  precision {precision_score(y_test, pred, zero_division=0):.2f}  "
          f"recall {recall_score(y_test, pred):.2f}")
```

### 5. class_weight='balanced' vs moving the threshold
```python
# class_weight="balanced" vs moving the threshold: same ranking, very different probabilities
from sklearn.metrics import brier_score_loss
bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
p_bal = bal.predict_proba(X_test)[:, 1]
print(f"plain    AUC {roc_auc_score(y_test, p_lr):.3f}  AP {average_precision_score(y_test, p_lr):.3f}  "
      f"mean predicted risk {p_lr.mean():.3f}  Brier {brier_score_loss(y_test, p_lr):.4f}")
print(f"balanced AUC {roc_auc_score(y_test, p_bal):.3f}  AP {average_precision_score(y_test, p_bal):.3f}  "
      f"mean predicted risk {p_bal.mean():.3f}  Brier {brier_score_loss(y_test, p_bal):.4f}")
pred_bal = (p_bal >= 0.5).astype(int)
print(f"balanced at 0.5: flags {pred_bal.sum()}, precision {precision_score(y_test, pred_bal):.2f}, recall {recall_score(y_test, pred_bal):.2f}")
print(f"actual readmission rate {y_test.mean():.3f}: only the plain model's risks can be read as real probabilities")
```

### 6. Push to GitHub
```bash
git add day18
git commit -m "Day 18: imbalanced readmissions, precision-recall and cost/capacity thresholds"
git push
```

### Stretch: Snowflake SQL precision and recall by threshold
```sql
-- after writing test-set scores to MODEL_SCORES(patient_id, risk, readmit30)
WITH t AS (SELECT column1 AS threshold FROM VALUES (0.05), (0.10), (0.12), (0.15), (0.20))
SELECT
  t.threshold,
  COUNT_IF(s.risk >= t.threshold)                                   AS flagged,
  COUNT_IF(s.risk >= t.threshold AND s.readmit30 = 1)               AS caught,
  ROUND(caught / NULLIF(flagged, 0), 3)                             AS precision,
  ROUND(caught / NULLIF(SUM(s.readmit30), 0), 3)                    AS recall
FROM model_scores s CROSS JOIN t
GROUP BY t.threshold
ORDER BY t.threshold;
```

## Printables
- Cheat sheet: `day18_v1_cheat_sheet.png`
- Infographic: `day18_v2_infographic.png`
- Code worksheet: `day18_v3_worksheet.png`
- Practice quiz: `day18_v4_quiz.png`
- PDF: `day18_lesson.pdf`

## Podcast script
(Audio: `day18_podcast.mp3`)

Hey Todor, welcome to day eighteen. Today is about a problem that quietly ruins a lot of healthcare models: the outcome you care about is rare. In our readmissions data, only about nine percent of patients come back within thirty days. That sounds like a detail, but it changes how you should judge every model you build.

Start with the trap. Imagine a model that simply says nobody will be readmitted. It is right for ninety one percent of patients, so its accuracy is ninety one percent. It is also completely useless, because it never finds a single patient who needs help. And here is the surprise: our logistic regression from day fourteen, using the default cutoff of fifty percent, does almost exactly the same thing. It flags one patient out of a thousand and catches zero of the ninety readmissions. The model is not broken. Its risk scores are fine. The problem is the cutoff. When the base rate is nine percent, almost nobody has a predicted risk above fifty percent.

So we switch to two better questions. Precision asks: of the patients we flagged, how many were truly readmitted? Recall asks: of the patients who were readmitted, how many did we flag ahead of time? Moving the cutoff trades one for the other. A low cutoff catches most readmissions but floods the care team with calls. A high cutoff gives cleaner flags but misses most of the patients who come back.

The precision recall curve shows that whole trade off in one picture. One thing to remember: on this curve, a random guess does not score one half. It scores the readmission rate, about nine percent. Our model's average precision is about twenty three percent, roughly two and a half times better than chance. That is a real signal, and also an honest reminder that predicting readmissions is hard.

Now, how do you actually pick the cutoff? Not by staring at the test set. You make out of fold predictions on the training data with cross validation, choose the cutoff there, and then apply it to the test set exactly once. And the choice itself is a business decision. In today's exercise we use made up but realistic numbers: a readmission costs about fifteen thousand dollars, a nurse follow up program costs about five hundred dollars per patient, and it prevents about a quarter of readmissions. Calling a patient pays off when their risk times the savings beats the cost of the call, which works out to a break even risk of about thirteen percent. The cost curve lands right near that, at twelve percent, flagging about two hundred patients and catching thirty seven of the ninety readmissions. The other common rule is capacity. If the team can only call the riskiest ten percent, you flag about a hundred patients and catch twenty four.

Last idea. You will often see people fix imbalance with class weights or resampling. With class weight set to balanced, the ranking barely changes, so the AUC stays the same. But every predicted risk gets pushed up, to an average of forty seven percent when the true rate is nine. That is fine for ranking, but you can no longer tell a clinician that a patient has a forty percent chance of coming back. Moving the threshold usually gets you the same benefit without breaking your probabilities.

In job interviews and in real healthcare teams, being able to say this clearly is a big deal: accuracy is misleading here, here is the precision and recall at our chosen cutoff, and here is why we chose it.

Tomorrow we switch gears to unsupervised learning, and you will group patients into segments with k means clustering, no labels required. See you then.
