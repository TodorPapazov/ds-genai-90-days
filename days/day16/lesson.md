# Day 16: Random forests: bagging, random features, out-of-bag AUC and permutation importance on 30-day readmissions

## Concept
Yesterday's single tree was either too blunt or too jumpy. Today you fix that with a random forest: hundreds of trees, each fit on a bootstrap sample (rows drawn with replacement) and allowed to try only a random subset of features at each split, with their risk scores averaged. Same simulated data and split as days 14 and 15 (4,000 discharges, seed 14, 9.0% readmitted), so the scoreboard is fair. First you see the problem: 30 bootstrapped depth-6 trees give one test patient risks from 0.00 to 0.14. Averaging removes that wobble. A 500-tree forest (25 patients per leaf) scores test AUC 0.716; adding trees climbs from 0.65 with 1 tree and flattens after about 50, and never overfits. Leaf size is the real knob: 5-fold CV picks min_samples_leaf 100, max_features sqrt (CV AUC 0.62, test AUC 0.716). Honesty checks: the default out-of-bag score is accuracy, 0.91, which equals predicting nobody is readmitted, so you compute OOB AUC (0.62) instead, which agrees with CV and reminds you a 1,000-patient test set carries luck. Permutation importance (test AUC drop when a column is shuffled) ranks prior_admits 0.14, age 0.04, los_days 0.01, while impurity importance overrates many-valued columns like los_days. Scoreboard: forest 0.716 beats yesterday's tree 0.649 and ties day 14's logistic regression 0.723, because the simulated risk is smooth and additive. Tomorrow: gradient boosting with XGBoost.

## Key numbers
- **Data:** days 14-15 simulated readmissions, 4,000 discharges, 9.0% readmitted, same stratified split
- **One deep tree, 30 bootstraps:** one patient's risk ranged 0.00 to 0.14 (average 0.07)
- **Forest, 500 trees, leaf 25:** train AUC 0.82, test AUC 0.716
- **Out-of-bag:** accuracy 0.91 (= base rate, useless), OOB AUC 0.618
- **Tree count:** 1 tree 0.648, 50 trees 0.713, 500 trees 0.716 (flattens, no overfit)
- **Tuned by 5-fold CV:** min_samples_leaf 100, max_features sqrt, CV AUC 0.623, test AUC 0.716
- **Scoreboard (test AUC):** logistic 0.723, random forest 0.716, tuned tree 0.649
- **Permutation importance:** prior_admits 0.144, age 0.044, los_days 0.014, n_meds 0.014, diabetes 0.008, followup_visit 0.005

## Hands-on
1. In your repo, create day16/ and open day16/random_forest.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to rebuild the same data and train/test split as days 14 and 15.
3. Run snippet 2: fit 30 depth-6 trees on bootstrap samples and record their risk for one test patient. Histogram the 30 values.
4. Run snippet 3: a 500-tree RandomForestClassifier with oob_score=True. Compare OOB accuracy with the 91% baseline, then compute OOB AUC.
5. Run snippet 4: the tree-count sweep (1 to 500) and a line chart of test AUC, then GridSearchCV over min_samples_leaf and max_features.
6. Run snippet 5: the three-model scoreboard and impurity vs permutation importance side by side.
7. Write day16/README.md: three charts, four findings, why the forest only ties logistic regression, and the test-set noise caveat.
8. Commit and push (snippet 6).
9. Stretch: load the forest's test-set risk scores into Snowflake and flag the top 10% for outreach (snippet 7).

## Code
### 1. Same data and split as days 14-15
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14 and 15, so every model is compared fairly
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

### 2. Why one tree is unstable: 30 bootstrap trees, one patient
```python
# Instability: refit one deep tree on bootstrap samples and watch its prediction for ONE patient jump
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
patient = X_test.iloc[[0]]
boot_preds = []
for b in range(30):
    idx = np.random.default_rng(b).integers(0, len(X_train), len(X_train))   # sample rows with replacement
    t = DecisionTreeClassifier(max_depth=6, random_state=b).fit(X_train.iloc[idx], y_train.iloc[idx])
    boot_preds.append(t.predict_proba(patient)[0, 1])
print(f"One patient, 30 bootstrap trees: risk ranges {min(boot_preds):.2f} to {max(boot_preds):.2f}, average {np.mean(boot_preds):.2f}")
```

### 3. Fit a random forest with out-of-bag scoring
```python
# A random forest = many bootstrapped trees, each split only looks at a random subset of features
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=25, max_features="sqrt",
                            oob_score=True, n_jobs=-1, random_state=14).fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
oob_auc = roc_auc_score(y_train, rf.oob_decision_function_[:, 1])   # each row scored only by trees that never saw it
print(f"Out-of-bag accuracy {rf.oob_score_:.3f} (useless: predicting 'no' for everyone gets 0.91)  OOB AUC {oob_auc:.3f}")
print(f"Random forest train AUC {roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]):.3f}  test AUC {roc_auc_score(y_test, p_rf):.3f}")
```

### 4. Tree-count sweep and CV tuning
```python
# More trees: the score climbs, then flattens. It does not overfit by adding trees
rows = []
for n_trees in [1, 5, 10, 25, 50, 100, 200, 500]:
    m = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=25, n_jobs=-1, random_state=14).fit(X_train, y_train)
    rows.append({"n_trees": n_trees, "test_auc": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])})
ntree_tbl = pd.DataFrame(rows)
print(ntree_tbl.round(3))
# Leaf size matters more than tree count: tune it with 5-fold CV on the training set only
from sklearn.model_selection import GridSearchCV, StratifiedKFold
grid = GridSearchCV(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=14),
                    {"min_samples_leaf": [1, 10, 25, 50, 100], "max_features": ["sqrt", None]},
                    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=14))
grid.fit(X_train, y_train)
p_best = grid.best_estimator_.predict_proba(X_test)[:, 1]
print(grid.best_params_, f"CV AUC {grid.best_score_:.3f}  test AUC {roc_auc_score(y_test, p_best):.3f}")
```

### 5. Scoreboard and two kinds of importance
```python
# Scoreboard vs days 14-15, and two kinds of importance
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
p_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train).predict_proba(X_test)[:, 1]
p_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=14).fit(X_train, y_train).predict_proba(X_test)[:, 1]
for name, p in [("logistic regression (day 14)", p_lr), ("tuned tree (day 15)", p_tree), ("tuned random forest", p_best)]:
    print(f"{name:30s} test AUC {roc_auc_score(y_test, p):.3f}")
best_rf = grid.best_estimator_
mdi = pd.Series(best_rf.feature_importances_, index=features)
perm = permutation_importance(best_rf, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=14)
imp = pd.DataFrame({"impurity_mdi": mdi, "permutation_auc_drop": perm.importances_mean}).sort_values("permutation_auc_drop", ascending=False)
print(imp.round(3))     # permutation = how much test AUC drops when that column is shuffled
```

### 6. Push to GitHub
```bash
git add day16
git commit -m "Day 16: random forests, OOB AUC and permutation importance on readmissions"
git push
```

### Stretch: Snowflake SQL flag from the riskiest leaf
```sql
-- after writing test-set scores to RF_SCORES(patient_id, rf_risk, readmit30)
WITH ranked AS (
  SELECT *, NTILE(10) OVER (ORDER BY rf_risk DESC) AS risk_decile
  FROM rf_scores
)
SELECT
  IFF(risk_decile = 1, 'top 10%', 'other 90%') AS outreach_group,
  COUNT(*)                                     AS patients,
  ROUND(AVG(readmit30), 3)                     AS readmit_rate,
  ROUND(AVG(readmit30) / (SELECT AVG(readmit30) FROM rf_scores), 2) AS lift
FROM ranked
GROUP BY 1
ORDER BY 1 DESC;
```

## Printables
- Cheat sheet: `day16_v1_cheat_sheet.png`
- Infographic: `day16_v2_infographic.png`
- Code worksheet: `day16_v3_worksheet.png`
- Practice quiz: `day16_v4_quiz.png`
- PDF: `day16_lesson.pdf`

## Podcast script
(Audio: `day16_podcast.mp3`)

Hey Todor, welcome to day sixteen. Yesterday you built a decision tree, and you saw its big weakness. Let it grow deep and it memorizes the training patients. Keep it small and it's too blunt. Today's fix is one of the most useful ideas in all of machine learning: instead of trusting one tree, you ask hundreds of them and average their answers. That's a random forest.

Let's start with why one tree is shaky. In today's exercise you take the training data and draw a bootstrap sample. That just means you pick three thousand rows at random, with replacement, so some patients show up twice and about a third don't show up at all. You fit a tree on that sample, then do it again, thirty times. Then you ask every tree about the same single patient. The answers ran all the way from zero percent risk to fourteen percent. Same patient, same kind of model, wildly different answers. That wobble is called variance, and it's exactly what averaging removes.

A random forest adds one more twist. When each tree picks a split, it's only allowed to look at a random handful of the features. That sounds like a handicap, but it stops every tree from leaning on the same strong feature, which here is prior admissions. The trees end up making different mistakes, and different mistakes cancel out when you average them.

The results are nice. Five hundred trees reached a test AUC of about point seven two, up from point six five for yesterday's tuned tree. And here's a detail I love: adding trees never makes a forest overfit. You'll chart it. The score climbs fast from one tree to about fifty, then flattens out. The knob that actually controls overfitting is leaf size, and cross validation picked a minimum of one hundred patients per leaf.

Now two honesty checks. First, the forest gives you a free validation score called out of bag. Each patient gets scored only by the trees that never saw them. But the default out of bag score is accuracy, and it came out at ninety one percent. That sounds great until you remember only nine percent of patients are readmitted. Saying nobody comes back is also ninety one percent accurate. So you compute out of bag AUC instead, and it's about point six two, close to the cross validation score. That gap versus the test set is a reminder that one test set of a thousand patients has some luck in it.

Second, feature importance. The built in importance from the forest measures how much each feature cleaned up the splits, and it tends to favor columns with lots of distinct values, like length of stay. Permutation importance asks a more practical question: if I scramble this one column, how much does test AUC drop? By that measure prior admissions dominates, and age matters more than length of stay. Either way, importance tells you what the model leaned on, not what causes readmission.

And the scoreboard? The forest basically ties logistic regression. That's not a failure. We simulated a smooth, additive risk, which is exactly the shape logistic regression is built for. Forests shine when risk has thresholds and interactions. In an interview, being able to say that clearly is worth more than the model itself.

Tomorrow we go from averaging trees to stacking them on purpose. It's gradient boosting with XGBoost, where each new tree studies the mistakes of the ones before it. See you then.
