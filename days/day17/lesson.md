# Day 17: Gradient boosting with XGBoost: residuals, learning rate, early stopping and CV tuning on 30-day readmissions

## Concept
A random forest grows trees side by side and averages them. Gradient boosting grows them in a sequence: start every patient at the 9% base rate, fit a small tree to the residuals (actual minus predicted risk), add a shrunken slice of it, and repeat. Same simulated data and split as days 14-16 (4,000 discharges, seed 14, 9.0% readmitted), so the scoreboard stays fair. First you boost by hand: 30 depth-2 trees with step 0.3 take test AUC from 0.59 to 0.70. Then XGBoost with learning rate 0.05, depth 3 and early stopping on a validation split carved from training data stops at round 59 of 2,000 (test AUC 0.700). The danger: a greedy model (depth 8, learning rate 0.3, no stopping) reaches train AUC 1.00 and test AUC 0.57, pure memorization. 5-fold CV over max_depth and learning_rate picks depth 1, learning rate 0.01 (CV AUC 0.638, test AUC 0.709): no interactions, because the simulated risk is additive. Scoreboard: logistic 0.723, forest 0.716, XGBoost 0.709, tree 0.649. Boosting ties rather than wins here; on real EHR data with interactions and thresholds it often pulls ahead, which is why you always keep a simple baseline. Gain importance ranks los_days second, permutation importance ranks age second, the same many-valued-column bias as yesterday. Tomorrow: imbalanced outcomes, thresholds and precision-recall.

## Key numbers
- **Data:** days 14-16 simulated readmissions, 4,000 discharges, 9.0% readmitted, same stratified split
- **Boosting by hand:** 30 depth-2 trees, step 0.3: test AUC 0.59 after 1 tree, 0.70 after 30
- **XGBoost + early stopping:** lr 0.05, depth 3: best round 59 of 2,000, validation AUC 0.693, test AUC 0.700
- **Greedy XGBoost:** depth 8, lr 0.3, 300 rounds: train AUC 1.00, test AUC 0.571
- **Tuned by 5-fold CV:** max_depth 1, learning_rate 0.01, CV AUC 0.638, test AUC 0.709
- **Scoreboard (test AUC):** logistic 0.723, random forest 0.716, XGBoost 0.709, tree 0.649
- **Importance:** permutation: prior_admits 0.154, age 0.036, los_days 0.016; gain share ranks los_days 0.23 above age 0.16

## Hands-on
1. In your repo, create day17/, run conda activate ds90 and pip install xgboost, and open day17/xgboost.ipynb in VS Code.
2. Run snippet 1 to rebuild the same data and train/test split as days 14-16.
3. Run snippet 2: boost by hand with 30 depth-2 regression trees fit to residuals. Plot train and test AUC by round.
4. Run snippet 3: split a validation set from the TRAINING data and fit XGBClassifier with early_stopping_rounds=100. Record best_iteration and plot evals_result().
5. Run snippet 4: fit the greedy model (depth 8, learning rate 0.3) and plot its train vs validation curves, then GridSearchCV over max_depth and learning_rate and draw a heatmap.
6. Run snippet 5: the six-model scoreboard and gain vs permutation importance side by side.
7. Write day17/README.md: three charts, four findings, and an honest note on why boosting tied logistic regression here.
8. Commit and push (snippet 6).
9. Stretch: load XGBoost and forest test-set scores into Snowflake and compare top-decile lift (snippet 7).

## Code
### 1. Same data and split as days 14-16
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14-16, so every model is compared fairly
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

### 2. Boosting by hand: small trees fit to residuals
```python
# Boosting by hand: start from the base rate, then each small tree fixes what the model still gets wrong
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
base = np.log(y_train.mean() / (1 - y_train.mean()))       # log-odds of the 9% base rate
F_train = np.full(len(X_train), base); F_test = np.full(len(X_test), base)
lr, hand = 0.3, []
for r in range(1, 31):
    resid = y_train - 1 / (1 + np.exp(-F_train))           # what we still get wrong (actual minus predicted risk)
    stump = DecisionTreeRegressor(max_depth=2, random_state=r).fit(X_train, resid)
    F_train += lr * stump.predict(X_train); F_test += lr * stump.predict(X_test)   # small step toward the fix
    hand.append({"round": r, "train_auc": roc_auc_score(y_train, F_train), "test_auc": roc_auc_score(y_test, F_test)})
hand = pd.DataFrame(hand)
print(hand.iloc[[0, 4, 9, 29]].round(3))
```

### 3. XGBoost with early stopping on a validation split
```python
# XGBoost with early stopping: carve a validation set out of TRAINING data, never the test set
from xgboost import XGBClassifier
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=14)
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=5, eval_metric="auc", early_stopping_rounds=100, random_state=14)
xgb.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
p_xgb = xgb.predict_proba(X_test)[:, 1]
print(f"Stopped at round {xgb.best_iteration + 1}, validation AUC {xgb.best_score:.3f}, test AUC {roc_auc_score(y_test, p_xgb):.3f}")
```

### 4. Watch it overfit, then tune depth and learning rate with CV
```python
# Watch boosting overfit: deep trees + a big learning rate + no early stopping
curves = xgb.evals_result()
greedy = XGBClassifier(n_estimators=300, learning_rate=0.3, max_depth=8, eval_metric="auc", random_state=14)
greedy.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
g = greedy.evals_result()
p_greedy = greedy.predict_proba(X_test)[:, 1]
print(f"Greedy model: train AUC {g['validation_0']['auc'][-1]:.3f}, validation AUC {g['validation_1']['auc'][-1]:.3f}, test AUC {roc_auc_score(y_test, p_greedy):.3f}")
# Tune the two knobs that matter most with 5-fold CV on the training set
from sklearn.model_selection import GridSearchCV, StratifiedKFold
grid = GridSearchCV(XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=14),
                    {"max_depth": [1, 2, 3, 5], "learning_rate": [0.01, 0.03, 0.1]},
                    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=14))
grid.fit(X_train, y_train)
p_best = grid.best_estimator_.predict_proba(X_test)[:, 1]
print(grid.best_params_, f"CV AUC {grid.best_score_:.3f}  test AUC {roc_auc_score(y_test, p_best):.3f}")
```

### 5. Six-model scoreboard and gain vs permutation importance
```python
# Scoreboard vs days 14-16 on the same 1,000 test patients, plus gain vs permutation importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
p_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train).predict_proba(X_test)[:, 1]
p_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=14).fit(X_train, y_train).predict_proba(X_test)[:, 1]
p_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features="sqrt", n_jobs=-1,
                              random_state=14).fit(X_train, y_train).predict_proba(X_test)[:, 1]
board = {"logistic regression (day 14)": p_lr, "tuned tree (day 15)": p_tree, "random forest (day 16)": p_rf,
         "greedy XGBoost": p_greedy, "XGBoost, early stopping": p_xgb, "XGBoost, CV-tuned": p_best}
for name, p in board.items():
    print(f"{name:30s} test AUC {roc_auc_score(y_test, p):.3f}")
best_xgb = grid.best_estimator_
gain = pd.Series(best_xgb.get_booster().get_score(importance_type="gain")).reindex(features).fillna(0)
perm = permutation_importance(best_xgb, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=14)
imp = pd.DataFrame({"gain_share": gain / gain.sum(), "permutation_auc_drop": perm.importances_mean}).sort_values("permutation_auc_drop", ascending=False)
print(imp.round(3))
```

### 6. Push to GitHub
```bash
git add day17
git commit -m "Day 17: gradient boosting with XGBoost, early stopping and CV tuning on readmissions"
git push
```

### Stretch: Snowflake SQL top-decile lift by model
```sql
-- after writing test-set scores to MODEL_SCORES(patient_id, model, risk, readmit30)
WITH ranked AS (
  SELECT *, NTILE(10) OVER (PARTITION BY model ORDER BY risk DESC) AS risk_decile
  FROM model_scores
)
SELECT
  model,
  COUNT_IF(risk_decile = 1)                                   AS top10_patients,
  ROUND(AVG(IFF(risk_decile = 1, readmit30, NULL)), 3)        AS top10_readmit_rate,
  ROUND(AVG(IFF(risk_decile = 1, readmit30, NULL)) / AVG(readmit30), 2) AS top10_lift
FROM ranked
GROUP BY model
ORDER BY top10_lift DESC;
```

## Printables
- Cheat sheet: `day17_v1_cheat_sheet.png`
- Infographic: `day17_v2_infographic.png`
- Code worksheet: `day17_v3_worksheet.png`
- Practice quiz: `day17_v4_quiz.png`
- PDF: `day17_lesson.pdf`

## Podcast script
(Audio: `day17_podcast.mp3`)

Hey Todor, welcome to day seventeen. Yesterday you built a random forest, hundreds of trees that each learn on their own and then vote by averaging. Today you meet the other big family of tree ensembles, and the one you'll see in more healthcare data science job postings than almost any other model: gradient boosting, and its most famous library, XGBoost.

Here's the core idea in plain words. A forest builds its trees side by side, independently. Boosting builds them in a line, like a relay team. You start with a very simple guess: everyone in the training data gets the base rate, about a nine percent chance of being readmitted. Then you look at what that guess gets wrong. Patients who were readmitted are under-predicted, and everyone else is a little over-predicted. That gap, actual outcome minus predicted risk, is called the residual. The first small tree is trained to predict those residuals. You add a small slice of its answer to your guess, look at what's still wrong, and train the next tree on that. Round after round, each tree fixes a bit of what the team still misses.

The word small matters. Each tree's contribution is shrunk by the learning rate. A learning rate of point zero five means every tree only moves the prediction five percent of the way toward its fix. That feels slow, but small careful steps generalize much better than big confident jumps. In today's exercise you actually do this by hand first, with thirty tiny trees in a simple loop, and you watch the test AUC climb from about point five nine to about point seven. No magic, just a loop that keeps correcting itself.

Now the danger. Boosting is so good at fixing errors that it will happily keep fixing noise. You'll prove it with what I call the greedy model: trees eight levels deep, a big learning rate, and three hundred rounds with nobody watching. Its training AUC hits a perfect one point zero. Its test AUC drops to about point five seven, barely better than a coin flip. It didn't learn readmission risk, it memorized three thousand patients.

The fix is early stopping. You carve a validation set out of your training data, never the test set, and tell XGBoost to score itself on that validation set after every round. When the validation score stops improving for a hundred rounds, training stops and keeps the best round. In your run, it stopped around round fifty nine out of a possible two thousand. Then you tune the two knobs that matter most, tree depth and learning rate, with five-fold cross validation.

And here's the honest twist. Cross validation picked trees with a depth of one, the simplest possible setting, and the tuned model scored about point seven one on the test set. That's basically a tie with the random forest and with the logistic regression from day fourteen. Is that a failure? Not at all. We simulated this data with a smooth, additive risk, where each factor just adds its own piece. That's exactly the shape logistic regression assumes, so it's very hard to beat. On real hospital data, with interactions and thresholds, like a long stay only mattering for patients with prior admissions, boosting often pulls ahead. The professional habit is the same either way: keep a simple baseline, let cross validation decide, and report the test score once, at the end.

One more thing to notice. XGBoost's built-in gain importance ranks length of stay second, but permutation importance, which measures how much test AUC drops when you scramble a column, puts age second. Just like yesterday, built-in importance tends to favor columns with lots of distinct values. When you explain a model to a clinical team, permutation importance is the more honest story.

So today's takeaway: forests average independent trees, boosting chains small trees that correct each other, and early stopping plus a small learning rate is what keeps boosting honest. Push your notebook and README to GitHub, and you've got a real XGBoost project in your portfolio.

Tomorrow we tackle a problem that's everywhere in healthcare: rare outcomes. Only nine percent of these patients are readmitted, so we'll look at precision, recall, and how to pick a risk threshold that a care team can actually work with. See you then.
