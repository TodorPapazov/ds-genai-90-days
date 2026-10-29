# Day 15: Decision trees: splits, Gini impurity, overfitting and cross-validated depth on 30-day readmissions

## Concept
Week 3 starts classic machine learning with the easiest model to picture: a decision tree, a stack of yes/no questions that ends in risk groups called leaves. You reuse day 14's simulated data (4,000 discharges, seed 14, 9.0% readmitted) and the same 3,000/1,000 split, so any change in score comes from the model. Each split is chosen to make child groups purer, measured by Gini impurity 2p(1-p): the root scores 0.16, a pure node 0. A readable depth-3 tree (at least 100 patients per leaf) makes 6 risk groups; the riskiest is any prior admission plus a stay over 10.9 days, 23% readmitted (n=111). Test AUC 0.61. Then overfitting: with no depth limit the tree grows 497 leaves, scores train AUC 1.00 but only test AUC 0.56. You tune max_depth and min_samples_leaf with 5-fold cross-validation on the training set only (GridSearchCV picks depth 4, 50 per leaf, CV AUC 0.61) and score the test set once: 0.65. The honest result: day 14's logistic regression still wins at 0.72, because the true risk here is a smooth sum of effects. Feature importance (prior_admits 0.39, los_days 0.25) shows what the tree used, not what causes readmission. Tomorrow, random forests average many trees to fix the instability.

## Key numbers
- **Data:** day 14 simulated readmissions, 4,000 discharges, 9.0% readmitted, same stratified split
- **Root Gini:** 2 x 0.09 x 0.91 = 0.16
- **Readable tree:** depth 3, min 100 per leaf, 6 leaves, test AUC 0.611
- **Riskiest leaf:** prior admission and LOS > 10.9 days: 23% readmitted (n=111)
- **Unlimited tree:** 497 leaves, train AUC 1.00, test AUC 0.56 (overfit)
- **Tuned by 5-fold CV:** max_depth 4, min_samples_leaf 50, CV AUC 0.614, test AUC 0.649
- **Day 14 logistic regression:** test AUC 0.723 (still the best)
- **Importance:** prior_admits 0.39, los_days 0.25, age 0.19, n_meds 0.13, diabetes 0.03, followup_visit 0.00

## Hands-on
1. In your repo, create day15/ and open day15/decision_trees.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to rebuild the day 14 data and the exact same train/test split.
3. Compute the root Gini impurity by hand: 2 x 0.09 x 0.91.
4. Run snippet 2: a depth-3 tree with min_samples_leaf=100. Read the export_text rules and draw it with plot_tree(filled=True, proportion=True).
5. Write the riskiest leaf as one plain-English rule a care manager could use.
6. Run snippet 3: the depth sweep from 1 to unlimited. Chart train vs test AUC and mark where overfitting starts.
7. Run snippet 4: GridSearchCV over max_depth and min_samples_leaf with 5-fold CV on the training set. Score the tuned tree once on the test set.
8. Run snippet 5: compare with logistic regression, plot both ROC curves and a feature importance bar chart.
9. Write day15/README.md: three charts, four findings, why the tree lost here, and what importance does not mean.
10. Commit and push (snippet 6).
11. Stretch: turn the riskiest leaf into a Snowflake SQL flag and compare its rate to the overall rate (snippet 7).

## Code
### 1. Same data and split as day 14
```python
import numpy as np
import pandas as pd

# Same simulated data as day 14 (seed 14) so you can compare models fairly
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
from sklearn.model_selection import train_test_split
features = ["age", "prior_admits", "los_days", "n_meds", "diabetes", "followup_visit"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df.readmit30, test_size=0.25,
                                                    stratify=df.readmit30, random_state=14)
```

### 2. A readable depth-3 tree
```python
# A shallow tree is a set of if/then rules you can read out loud
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score
tree3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=14).fit(X_train, y_train)
print(export_text(tree3, feature_names=features, show_weights=True))
print(f"Depth-3 tree test AUC: {roc_auc_score(y_test, tree3.predict_proba(X_test)[:, 1]):.3f}")
```

### 3. Depth sweep: watch overfitting happen
```python
# Overfitting: deeper trees memorize training patients and do worse on new ones
rows = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=14).fit(X_train, y_train)
    rows.append({"max_depth": d if d else "none", "leaves": t.get_n_leaves(),
                 "train_auc": roc_auc_score(y_train, t.predict_proba(X_train)[:, 1]),
                 "test_auc": roc_auc_score(y_test, t.predict_proba(X_test)[:, 1])})
depth_tbl = pd.DataFrame(rows)
print(depth_tbl.round(3))
```

### 4. Tune with 5-fold cross-validation
```python
# Pick depth and leaf size with 5-fold cross-validation on the TRAINING set only
from sklearn.model_selection import GridSearchCV, StratifiedKFold
grid = GridSearchCV(DecisionTreeClassifier(random_state=14),
                    {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [10, 25, 50, 100]},
                    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=14))
grid.fit(X_train, y_train)
best = grid.best_estimator_
p_tree = best.predict_proba(X_test)[:, 1]
print(grid.best_params_, f"CV AUC {grid.best_score_:.3f}  test AUC {roc_auc_score(y_test, p_tree):.3f}")
```

### 5. Compare with logistic regression and read importance
```python
# Compare with day 14's logistic regression and look at what the tree used
from sklearn.linear_model import LogisticRegression
p_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train).predict_proba(X_test)[:, 1]
print(f"Logistic regression test AUC {roc_auc_score(y_test, p_lr):.3f} vs tuned tree {roc_auc_score(y_test, p_tree):.3f}")
imp = pd.Series(best.feature_importances_, index=features).sort_values(ascending=False)
print(imp.round(3))                       # share of impurity reduction, not cause and effect
```

### 6. Push to GitHub
```bash
git add day15
git commit -m "Day 15: decision trees, Gini, overfitting and CV tuning on readmissions"
git push
```

### Stretch: Snowflake SQL flag from the riskiest leaf
```sql
-- turn the riskiest tree leaf into a rule-based flag
SELECT
  IFF(prior_admits >= 1 AND los_days > 10.85, 1, 0) AS high_risk_leaf,
  COUNT(*)                                          AS patients,
  ROUND(AVG(readmit30), 3)                          AS readmit_rate,
  ROUND(AVG(readmit30) / (SELECT AVG(readmit30) FROM readmissions_30d), 2) AS lift
FROM readmissions_30d
GROUP BY 1
ORDER BY 1 DESC;
```

## Printables
- Cheat sheet: `day15_v1_cheat_sheet.png`
- Infographic: `day15_v2_infographic.png`
- Code worksheet: `day15_v3_worksheet.png`
- Practice quiz: `day15_v4_quiz.png`
- PDF: `day15_lesson.pdf`

## Podcast script
(Audio: `day15_podcast.mp3`)

Hey Todor, welcome to day fifteen, and welcome to week three. For two weeks you've been doing statistics: describing data, testing ideas, and explaining why things happen. This week is classic machine learning, and we're starting with the model that's easiest to picture, the decision tree.

A decision tree is just a stack of yes or no questions. Picture a nurse sorting patients at discharge. First question: has this patient been admitted in the past year? If yes, next question: did they stay more than about eleven days? Each answer sends the patient down a branch, and at the bottom you land in a group called a leaf. The predicted risk for anyone in that leaf is simply the share of training patients in that leaf who came back within thirty days.

How does the tree pick its questions? It tries every feature and every cut point, and it keeps the one that makes the two groups below it as pure as possible. Pure means nearly everyone in the group had the same outcome. The usual way to measure that is called Gini impurity. With two classes it's two times p times one minus p. A group where nine percent were readmitted has a Gini of about point one six. A perfectly pure group scores zero, and a fifty fifty mix scores point five, the worst it can be.

Today you're reusing the exact same simulated readmissions data and the exact same train and test split from yesterday. That's on purpose. It means any change in the score comes from the model, not from the data. A small tree, three questions deep with at least a hundred patients per leaf, gives you six risk groups you can read out loud. The riskiest one is patients with any prior admission and a stay over about eleven days, and about twenty three percent of them were readmitted, more than two and a half times the average.

Now the most important lesson of the day: overfitting. If you let a tree keep asking questions with no limit, it grows to almost five hundred leaves. On the training data it gets an AUC of one point zero, which looks perfect. On the test patients it has never seen, it drops to about point five six, barely better than a coin flip. It didn't learn readmission risk. It memorized the training patients, noise and all. That gap between the training score and the test score is the warning sign, so always print both.

The fix is to limit the tree, with settings like maximum depth and minimum patients per leaf. But how do you choose them? Not with the test set. The test set is your final exam, and if you peek at it while tuning, your reported score becomes too optimistic. Instead you use cross validation. You split the training data into five parts, train on four, check on the fifth, and rotate. Grid search does that for every combination of settings and picks the best. Here it picked a depth of four and at least fifty patients per leaf.

And here's the honest twist. The tuned tree scores about point six five on the test set. Yesterday's plain logistic regression scored about point seven two. The simple model wins. That's because we simulated risk as a smooth sum of effects, and a straight line fits that shape better than a staircase of yes or no steps. In real healthcare data this happens more often than people expect, which is exactly why you always compare against a simple baseline before reaching for something fancier.

One more caution. The tree reports feature importance, and the seven day clinic visit scored zero. That doesn't mean follow up visits don't matter. It only means this tree never chose to split on it. Importance tells you what the model used, not what causes readmission.

Your exercise today is to build the small readable tree, run the depth sweep and chart training versus test scores, tune with cross validation, and write a short README explaining why the tree lost. Tomorrow we take that unstable, overfitting tree and fix it by growing hundreds of them on different samples and averaging their votes. That's the random forest, and we'll see if it can finally beat logistic regression. See you then.
