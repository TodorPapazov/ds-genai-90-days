# Day 20: PCA and dimensionality reduction: turning a correlated 9-test lab panel into kidney, blood-sugar and heart axes

## Concept
Yesterday PCA only drew the segments. Today we learn it properly. We give the same 4,000 simulated patients a simulated 9-test lab panel. Labs come in families: creatinine, BUN and eGFR move together (r about 0.85), and HbA1c tracks glucose (r 0.87), so 9 columns carry only a few ideas. Principal component analysis (PCA) builds new, uncorrelated axes, each a weighted mix of the scaled columns, ordered by how much spread they capture. The scree plot shows PC1 at 36%, PC2 21% and PC3 16% of the variance (73% together); 6 PCs pass 90%. Loadings let you name each axis: PC1 is kidney (creatinine, BUN up, eGFR down), PC2 is blood sugar, PC3 is heart (BNP against sodium). PCA never saw the diabetes flag, yet the mean PC2 score is 0.94 with diabetes vs -0.46 without. Two traps: scale first (unscaled, BNP with sd about 100 takes over PC1), and keep the scaler and PCA inside a Pipeline so cross-validation never peeks at test folds. And the honest lesson: PCA ignores the outcome. Compressing all 15 columns to 3 PCs gives a CV AUC of 0.580 vs 0.641 with no PCA, because readmission risk lives in prior admissions and follow-up, not in the biggest lab variance. Tomorrow: the week 3 model scorecard.

## Key numbers
- **Data:** days 14-19 simulated discharges (4,000 patients) plus a simulated 9-test lab panel, seed 20
- **Correlated families:** creatinine, BUN, eGFR r about 0.85; HbA1c and glucose r 0.87; BNP and sodium r -0.48
- **Explained variance:** PC1 36%, PC2 21%, PC3 16% (73% total); 6 PCs reach 95%
- **Named axes:** PC1 kidney, PC2 blood sugar, PC3 heart; PC2 mean 0.94 with diabetes vs -0.46 without
- **Unscaled PCA:** PC1 is almost 100% BNP, the lab with the biggest raw spread
- **Prediction (5-fold CV AUC):** 3 PCs 0.580, 5 PCs 0.599, 8 PCs 0.648, no PCA 0.641

## Hands-on
1. In your repo, create day20/, run conda activate ds90, and open day20/pca_labs.ipynb in VS Code.
2. Run snippet 1 to rebuild the same simulated data as days 14-19.
3. Run snippet 2 to add the simulated 9-test lab panel, then plot the lab correlation heatmap and find the 3 families.
4. Run snippet 3: scale the labs, fit PCA, print variance per component, and draw the scree plot.
5. Run snippet 4: print loadings for PC1 to PC3, name each one, and scatter PC1 vs PC2 colored by diabetes.
6. Refit PCA on the unscaled labs and note which lab takes over PC1.
7. Run snippet 5: a scaler + PCA + logistic regression pipeline with 5-fold CV AUC for 1 to 15 components, compared with no PCA.
8. Write day20/README.md: three charts, a loadings table, and two sentences on when you would and would not use PCA.
9. Commit and push (snippet 6).
10. Stretch: summarize the lab panel by diabetes status in Snowflake SQL (snippet 7).

## Code
### 1. Same simulated data as days 14-19
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14-19, so today uses the same patients
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

### 2. Add a correlated 9-test lab panel
```python
# Add a 9-test lab panel for the same patients (simulated, fixed seed).
# Labs come in related families, so many columns carry overlapping information.
rng2 = np.random.default_rng(20)
kidney = rng2.normal(0, 1, n) + 0.3 * (df.age - 66) / 13        # hidden kidney strain
glyc = rng2.normal(0, 1, n) + 1.2 * df.diabetes                 # hidden blood-sugar control
heart = rng2.normal(0, 1, n) + 0.25 * df.prior_admits           # hidden heart strain
noise = lambda s: rng2.normal(0, s, n)
labs = pd.DataFrame({
    "creatinine": (1.0 + 0.30 * kidney + noise(0.12)).round(2),
    "bun":        (18 + 6.0 * kidney + noise(3.0)).round(0),
    "egfr":       (75 - 15 * kidney + noise(6.0)).round(0),
    "hba1c":      (6.0 + 0.9 * glyc + noise(0.35)).round(1),
    "glucose":    (120 + 30 * glyc + noise(15)).round(0),
    "bnp":        (150 + 90 * heart + noise(40)).clip(5).round(0),
    "sodium":     (139 - 1.2 * heart + noise(2.0)).round(0),
    "hemoglobin": (13 - 0.5 * kidney + noise(1.0)).round(1),
    "potassium":  (4.2 + 0.15 * kidney + noise(0.35)).round(1)})
df = pd.concat([df, labs], axis=1)
lab_cols = list(labs.columns)
print(df[lab_cols].corr().round(2))       # creatinine, bun, egfr move together; hba1c with glucose
```

### 3. Scale, fit PCA, explained variance
```python
# PCA on the scaled lab panel: how many directions really carry the information?
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
Zlab = StandardScaler().fit_transform(df[lab_cols])
pca_full = PCA(random_state=20).fit(Zlab)
evr = pca_full.explained_variance_ratio_
for i, (v, c) in enumerate(zip(evr, evr.cumsum())):
    print(f"PC{i+1}: {v:.1%} of variance, cumulative {c:.1%}")
n90 = int(np.argmax(evr.cumsum() >= 0.90) + 1)
print("components needed for 90% of the variance:", n90)
```

### 4. Loadings: name the components
```python
# Loadings: what each component is made of (read the biggest numbers, ignore the sign flip)
loadings = pd.DataFrame(pca_full.components_[:3].T, index=lab_cols, columns=["PC1", "PC2", "PC3"]).round(2)
print(loadings)
# PC1 ~ kidney (creatinine, bun, egfr in the opposite direction), PC2 ~ blood sugar, PC3 ~ heart (bnp, sodium)
scores = pca_full.transform(Zlab)[:, :3]
df[["pc_kidney", "pc_sugar", "pc_heart"]] = scores
print(df.groupby("diabetes").pc_sugar.mean().round(2))   # the sugar axis separates diabetes, which PCA never saw
```

### 5. PCA inside a pipeline: does it help prediction?
```python
# Does compressing 15 columns into fewer components hurt prediction? Pipeline keeps PCA inside the train fold.
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
all_cols = features + lab_cols
X_tr, X_te, y_tr, y_te = train_test_split(df[all_cols], df.readmit30, test_size=0.25,
                                          stratify=df.readmit30, random_state=14)
cv = StratifiedKFold(5, shuffle=True, random_state=20)
results = {}
for k in [1, 2, 3, 5, 8, 15]:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=20), LogisticRegression(max_iter=1000))
    results[k] = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    print(f"{k:2d} components: CV AUC {results[k]:.3f}")
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
auc_raw = cross_val_score(base, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
print(f"no PCA, all 15 columns: CV AUC {auc_raw:.3f}")
# Leakage trap: never fit the scaler or PCA on the full data before splitting
```

### 6. Push to GitHub
```bash
git add day20
git commit -m "Day 20: PCA on a correlated lab panel, scree plot, loadings, PCA in a pipeline"
git push
```

### Stretch: Snowflake SQL lab summary by diabetes status
```sql
-- lab panel summary by diabetes status (table LAB_PANEL joined to DISCHARGES)
SELECT
  d.diabetes,
  COUNT(*)                          AS patients,
  ROUND(AVG(l.hba1c), 2)            AS avg_hba1c,
  ROUND(AVG(l.glucose), 0)          AS avg_glucose,
  ROUND(AVG(l.creatinine), 2)       AS avg_creatinine,
  ROUND(CORR(l.creatinine, l.egfr), 2) AS corr_creat_egfr,
  ROUND(AVG(d.readmit30), 3)        AS readmit_rate
FROM lab_panel l
JOIN discharges d USING (patient_id)
GROUP BY d.diabetes
ORDER BY d.diabetes;
```

## Printables
- Cheat sheet: `day20_v1_cheat_sheet.png`
- Infographic: `day20_v2_infographic.png`
- Code worksheet: `day20_v3_worksheet.png`
- Practice quiz: `day20_v4_quiz.png`
- PDF: `day20_lesson.pdf`

## Podcast script
(Audio: `day20_podcast.mp3`)

Hey Todor, welcome to day twenty. Yesterday we drew our patient segments on a flat two dimensional picture and I promised we would open up what those two axes really mean. Today is that day. The method is called principal component analysis, or PCA, and it is the most common way to do dimensionality reduction, which is a fancy phrase for turning many columns into a few.

Here is the setup. We kept the same four thousand simulated patients from the last week, and gave each of them a simulated lab panel with nine tests: creatinine, BUN and eGFR for the kidneys, HbA1c and glucose for blood sugar, BNP and sodium for the heart, plus hemoglobin and potassium. If you have ever looked at real lab data, you know these tests are not independent. When creatinine is high, BUN tends to be high and eGFR tends to be low. In our data those three move together with correlations around point eight five. So nine columns are really telling us about three or so underlying ideas.

PCA finds those ideas for you. It looks for the single direction through the data where patients differ the most, and calls it the first principal component. Then it finds the next direction with the most remaining spread, at a right angle to the first, and so on. Each component is just a weighted mix of the original columns, and the components are uncorrelated with each other. You get as many components as columns, but usually the first few carry most of the information.

The scree plot tells you how much. In our lab panel, the first component holds about thirty six percent of the variance, the second about twenty one, the third about sixteen. Three components together hold about seventy three percent, and six components pass ninety percent. You pick the number where the bars drop off, or where you reach a target like ninety percent.

Then comes the part I like most: naming the components. You do that by reading the loadings, the weights of each lab in each component. Component one puts big weights on creatinine and BUN and a big negative weight on eGFR, so it is a kidney axis. Component two is almost entirely HbA1c and glucose, a blood sugar axis. Component three is BNP against sodium, a heart axis. A quick note: the sign of a component is arbitrary. A direction and its opposite describe the same line, so read the relative sizes, not the plus or minus.

And here is a nice check. PCA never saw the diabetes column, yet patients with diabetes score much higher on the blood sugar axis, about point nine four against minus point four six. PCA found real clinical structure without any labels.

Two traps. First, scale before PCA. PCA chases variance in raw units, so without scaling, BNP, which has a spread of almost a hundred, takes over the whole first component, and creatinine, with a spread of point three, basically disappears. Second, put the scaler and PCA inside a scikit learn pipeline, so cross validation fits them only on the training folds and never peeks at the test data.

Now the honest lesson. We used PCA to compress all fifteen columns and then predicted readmission with logistic regression. With three components the cross validated AUC was about point five eight. With no PCA at all it was about point six four. Why? Because PCA ignores the outcome. The directions where patients vary the most are mostly the lab families, but readmission risk lives in prior admissions and follow up visits. Biggest variance is not the same thing as most predictive. So use PCA to compress redundant panels, to draw pictures of patients, or to steady a linear model with many correlated inputs, but always compare against the plain model.

In an interview, a strong answer sounds like this: I scaled the features, used a scree plot to choose the number of components, named each component from its loadings, kept PCA inside a pipeline, and checked whether it actually helped prediction.

Tomorrow is the end of week three. We will line up every model we have built so far, from logistic regression to trees, forests and XGBoost, in one honest scorecard. See you then.
