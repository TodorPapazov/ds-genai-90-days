# Day 19: k-means clustering: unsupervised patient segments, scaling, choosing k, and profiling on the readmissions data

## Concept
For five days every model had a target. Today we hide the readmit30 label and ask whether patients fall into natural groups using k-means clustering, an unsupervised method: assign each patient to the nearest of k centers, move each center to the mean of its patients, repeat until stable. Scale first. Unscaled, age (sd about 13) swamps prior admissions (sd about 1) and k = 4 just slices patients by age (mean ages 47, 60, 71, 84). After StandardScaler, compare k = 2..8 with inertia (elbow) and the silhouette score. Here silhouette stays between 0.19 and 0.22 with no sharp elbow: the structure is weak, and k-means will always return k groups anyway. We choose k = 4 because the segments are easy to name: no diabetes with a follow-up visit (1,286), diabetes (1,200), no diabetes and no follow-up (1,092), and long stays of about 12 days (422). The yes/no flags drive most of the split. Only after fitting do we bring the label back to describe segments: readmission 6.8%, 10.1%, 9.1% and 12.3% versus 9.0% overall. Refits with 3 other seeds give identical groups (adjusted Rand index 1.00). PCA is used only to draw the segments in 2-D (35% of variance). Clustering describes; supervised models predict. Tomorrow: dimensionality reduction with PCA.

## Key numbers
- **Data:** days 14-18 simulated discharges, 4,000 patients, 6 features; readmit30 left out of the fit
- **Unscaled k = 4:** clusters split almost only on age: mean ages 47, 60, 71, 84
- **Choosing k (scaled):** silhouette 0.19 to 0.22 for k = 2..8, no sharp elbow; k = 4 chosen for interpretability
- **Segments:** no diabetes + follow-up 1,286; diabetes 1,200; no diabetes + no follow-up 1,092; long stay ~12 days 422
- **Readmission by segment:** 6.8%, 10.1%, 9.1%, 12.3% vs 9.0% overall (described after fitting)
- **Stability:** adjusted Rand index 1.00 across 3 other seeds; PCA 2-D view explains 35% of variance

## Hands-on
1. In your repo, create day19/, run conda activate ds90, and open day19/segments.ipynb in VS Code.
2. Run snippet 1 to rebuild the same simulated data as days 14-18. Do not use readmit30 to fit anything today.
3. Run snippet 2: check means and spreads, fit k-means on the raw columns, see that it splits by age, then scale with StandardScaler.
4. Run snippet 3: loop k = 2..8, record inertia and silhouette, and plot the elbow and silhouette charts.
5. Run snippet 4: fit k = 4, profile each segment in real units, and give each one a plain name.
6. Run snippet 5: bring readmit30 back to describe segment readmission rates, draw the segments on 2 PCA axes, and check stability across seeds.
7. Write day19/README.md: three charts, a segment table, one suggested action per segment, and an honest note that silhouette is about 0.2.
8. Commit and push (snippet 6).
9. Stretch: write segment labels to Snowflake and build the profile table in SQL (snippet 7).

## Code
### 1. Same simulated data as days 14-18
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14-18, so today's segments use the same patients
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

### 2. Why scaling matters for k-means
```python
# Unsupervised: use the patient features only, no readmit30 label
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[features]
print(X.describe().loc[["mean", "std"]].round(1))          # age and n_meds have big spreads
# Without scaling, k-means distance is dominated by the widest columns (age)
km_raw = KMeans(n_clusters=4, n_init=10, random_state=19).fit(X)
print("unscaled cluster mean age:", X.groupby(km_raw.labels_).age.mean().round(0).tolist())
scaler = StandardScaler()
Xs = scaler.fit_transform(X)                               # every column: mean 0, sd 1
```

### 3. Choose k: elbow (inertia) and silhouette
```python
# Choose k: inertia (elbow) and silhouette score for k = 2..8
from sklearn.metrics import silhouette_score
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=19).fit(Xs)
    inertias.append(km.inertia_)                           # within-cluster sum of squares
    sils.append(silhouette_score(Xs, km.labels_, sample_size=2000, random_state=19))
    print(f"k={k}  inertia {km.inertia_:8.0f}  silhouette {sils[-1]:.3f}")
```

### 4. Fit k = 4 and profile the segments
```python
# Fit the chosen k and profile each segment in original units
pd.set_option("display.width", 120, "display.max_columns", 20)
k = 4
km = KMeans(n_clusters=k, n_init=10, random_state=19).fit(Xs)
df["segment"] = km.labels_
profile = df.groupby("segment")[features].mean().round(2)
profile["patients"] = df.segment.value_counts().sort_index()
print(profile)
# Centers back in real units (age in years, stay in days)
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features).round(1) + 0.0
print(centers)
```

### 5. Describe segments with the outcome, PCA view, and stability
```python
# AFTER clustering: does readmission differ by segment? (label used only to describe, never to fit)
seg_rate = df.groupby("segment").readmit30.agg(["size", "mean"]).rename(columns={"size": "patients", "mean": "readmit_rate"})
print(seg_rate.round(3))
print(f"overall readmission rate {df.readmit30.mean():.3f}")
# 2-D picture with PCA (only for plotting; clusters were found in all 6 dimensions)
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=19)
Z = pca.fit_transform(Xs)
print(f"PCA: 2 components explain {pca.explained_variance_ratio_.sum():.0%} of the variance")
# Stability check: refit with different seeds and compare labelings
from sklearn.metrics import adjusted_rand_score
for seed in [1, 2, 3]:
    other = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs).labels_
    print(f"seed {seed}: adjusted Rand index vs seed 19 = {adjusted_rand_score(km.labels_, other):.2f}")
```

### 6. Push to GitHub
```bash
git add day19
git commit -m "Day 19: k-means patient segments, scaling, elbow and silhouette, segment profiles"
git push
```

### Stretch: Snowflake SQL precision and recall by threshold
```sql
-- after writing labels to PATIENT_SEGMENTS(patient_id, segment) and joining discharge features
SELECT
  s.segment,
  COUNT(*)                         AS patients,
  ROUND(AVG(d.age), 1)             AS avg_age,
  ROUND(AVG(d.los_days), 1)        AS avg_los_days,
  ROUND(AVG(d.diabetes), 2)        AS share_diabetes,
  ROUND(AVG(d.followup_visit), 2)  AS share_followup,
  ROUND(AVG(d.readmit30), 3)       AS readmit_rate
FROM patient_segments s
JOIN discharges d USING (patient_id)
GROUP BY s.segment
ORDER BY readmit_rate DESC;
```

## Printables
- Cheat sheet: `day19_v1_cheat_sheet.png`
- Infographic: `day19_v2_infographic.png`
- Code worksheet: `day19_v3_worksheet.png`
- Practice quiz: `day19_v4_quiz.png`
- PDF: `day19_lesson.pdf`

## Podcast script
(Audio: `day19_podcast.mp3`)

Hey Todor, welcome to day nineteen. For the last five days every model we built had a target, the thirty day readmission flag, and the model learned to predict it. Today we do something different. We hide the label completely and ask a simpler question: if we only look at the patients themselves, do they fall into natural groups? That is unsupervised learning, and the workhorse method is called k means clustering.

Here is how k means works, in plain terms. You pick a number k, say four. The algorithm drops four center points into the data. Then it repeats two steps. First, every patient is assigned to the closest center. Second, each center moves to the average of the patients assigned to it. It keeps doing that until the centers stop moving. That is the whole idea. Because the starting points are random, scikit learn runs it several times and keeps the best result, which is what the n init setting does.

The first trap is scaling. K means measures distance, so the units matter. In our data, age has a spread of about thirteen years, while prior admissions has a spread of about one. If you run k means on the raw columns, age wins completely. You get four clusters with average ages of forty seven, sixty, seventy one and eighty four, and nothing else matters. So the habit is simple: always standardize first, so every column has an average of zero and a spread of one.

The second question is how many clusters. There are two common checks. Inertia is the total squared distance from each patient to their own center. It always goes down as you add clusters, so you look for an elbow, a point where it stops dropping quickly. The silhouette score runs from minus one to one and asks whether each patient is much closer to its own group than to the next one. Above about point five means clear, well separated groups. In our data the silhouette sits near point two for every k from two to eight, and there is no sharp elbow. That is an honest and important result. It means the patients do not form strong natural types. K means will always hand you k groups, even when the data has no real groups, so you have to check.

We picked four clusters because they are easy to name and act on. One segment is patients without diabetes who had a follow up clinic visit. One is patients with diabetes. One is patients without diabetes who did not get a follow up visit. And a smaller one, about four hundred patients, is long stays of around twelve days. Notice that the yes or no columns, diabetes and follow up, drive most of the split. After scaling, those flags create big jumps in distance, which is worth mentioning when you present.

Only after the clusters are fixed do we bring the readmission label back, just to describe the groups. Patients who got a follow up visit came back about seven percent of the time. Long stay patients came back about twelve percent of the time, against nine percent overall. That is a useful story for a care management team, but remember, clustering is description. If you want to predict who will be readmitted, the supervised models from the last few days are the right tool.

Two more checks before you share. We drew the segments on a two dimensional picture using something called PCA, but that is only for drawing; the clusters were found using all six columns. And we refit with three different random seeds and the groups came out identical, so the segments are stable.

In interviews, a strong answer sounds like this: I scaled the features, compared k using inertia and silhouette, chose four for interpretability, profiled each segment, and was upfront that the structure is weak but stable.

Tomorrow we open up that PCA picture and learn dimensionality reduction properly, what those two axes actually mean and when to use them. See you then.
