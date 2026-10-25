# Day 11: Missing data and outliers: why it's missing, what's impossible, and how to fill the holes honestly

## Concept
Today you clean 2,000 simulated primary care patients (seed 11) with age, diabetes status, BMI, systolic blood pressure (SBP) and hemoglobin A1c. Start every project with df.isna().mean() and describe(): A1c is missing for 45.6% of patients, BMI for 7.8%, and the max BMI is 427, which is a typo, not a patient. Then ask why values are missing. MCAR (missing completely at random): BMI is missing about 8% in every age band, like a skipped scale. MAR (missing at random, given other columns): A1c is missing for 59% of non-diabetics but 8% of diabetics, and for 66% of 18-44 year olds vs. 31% of 65+. MNAR (missing because of the value itself) can't be proven from data, so you ask how it was collected. MAR matters: tested patients are older and sicker, so the complete-case mean A1c is 6.45 vs. a true 6.06, and the share with A1c of 8 or higher is 18.8% among the tested vs. 11.1% overall. Mean imputation keeps the biased mean, adds a fake spike and shrinks the SD (1.20 to 1.03). Group imputation within diabetic status and age band lands on 6.06. Outliers: the IQR rule flags 20 SBPs, but only 6 are impossible (sentinels 0 and 999). Use clinical ranges first (BMI 12-80, SBP 60-260), fix what you can explain (6 dropped decimals, 312 back to 31.2), set sentinels to NaN, and keep real extreme readings. Bridge to ML: fit imputers on training rows only, keep a missing-indicator column, and scale before KNN imputation. On 229 hidden test values, scaled KNN missed by 0.39 A1c points vs. 0.58 for the median.

## Key numbers
- **Patients:** 2,000 simulated (seed 11); diabetic share 17% (18-44), 28% (45-64), 37% (65+)
- **Missing (raw):** A1c 45.6%, BMI 7.8%, SBP 0% (but 6 sentinel codes)
- **A1c missing by group:** non-diabetic 59%, diabetic 8%; age 18-44 66%, 45-64 34%, 65+ 31%
- **BMI missing by age band:** 8.1%, 8.0%, 7.4% (flat, consistent with MCAR)
- **Impossible values:** 6 BMIs 219-427 (dropped decimal); 3 SBP = 999, 3 SBP = 0
- **Outlier flags:** BMI: IQR 12, z>3 6, robust z 6; SBP: IQR 20, z>3 6, robust z 6
- **Mean A1c:** truth 6.06; complete case 6.45; mean impute 6.45; group impute 6.06
- **A1c SD:** truth 1.20; after mean imputation 1.03
- **A1c of 8 or higher:** truth 11.1%; among tested 18.8%
- **Imputation MAE (229 hidden test A1c):** median 0.58; scaled KNN 0.39

## Hands-on
1. In your repo, create day11/ and open day11/missing_outliers.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate patients_labs.csv (seed 11). Keep the truth frame aside: it's your answer key. Run isna().mean() and describe() and list every red flag.
3. Run snippet 2: A1c missing rate by diabetic status and age band, then compare age, SBP and % diabetic for tested vs. untested patients. Check that BMI missingness is flat across age bands.
4. Write one sentence each: which column looks MCAR, which looks MAR, and why you can't rule out MNAR.
5. Run snippet 3: flag BMI and SBP outliers with the IQR rule, z > 3 and a robust (MAD) z-score. Then apply clinical ranges and list the impossible values.
6. Run snippet 4: fix dropped decimals, set sentinels 0 and 999 to NaN, and compare mean A1c under complete case, mean imputation and group imputation against the truth.
7. Plot the true A1c histogram and the mean-imputed histogram on the same axes. Describe the spike and what it does to the SD.
8. Run snippet 5: split train/test, fit SimpleImputer(add_indicator=True) and a scaled KNNImputer on training rows only, and score both on the hidden test A1c values.
9. Add a Findings markdown cell: 5 plain-language bullets a clinic quality manager could act on, plus a limits bullet (simulated, MNAR can't be tested, imputed values are estimates).
10. Commit and push (snippet 6).
11. Stretch: load patients_labs.csv into Snowflake and write a data-quality query: null rate per column, rule breaks per column, and A1c null rate by diabetic status (snippet 7).

## Code
### 1. Simulate patient labs, break them like a real extract, first look
```python
import numpy as np
import pandas as pd

# Simulate 2,000 adult primary care patients with a few lab and vital fields (fixed seed)
rng = np.random.default_rng(11)
n = 2000
age = rng.integers(18, 90, n)
diabetic = rng.random(n) < 0.12 + 0.004 * (age - 18)               # diabetes more common with age
bmi = rng.normal(29, 5.5, n).clip(16, 60)
sbp = (112 + 0.45 * age + 0.6 * (bmi - 29) + rng.normal(0, 13, n)).round()
a1c = np.where(diabetic, rng.normal(7.6, 1.3, n), rng.normal(5.5, 0.45, n)).clip(4.2, 14).round(1)
truth = pd.DataFrame({"age": age, "diabetic": diabetic, "bmi": bmi.round(1), "sbp": sbp, "a1c": a1c})

# Now break it the way real extracts break
df = truth.copy()
df.loc[rng.random(n) < 0.08, "bmi"] = np.nan                         # MCAR: scale skipped at random
p_test = np.where(df.diabetic, 0.92, np.where(df.age >= 45, 0.55, 0.20))
df.loc[rng.random(n) > p_test, "a1c"] = np.nan                       # MAR: A1c ordered mostly for diabetics and older adults
err = rng.choice(n, 12, replace=False)
df.loc[err[:6], "bmi"] = df.loc[err[:6], "bmi"] * 10                 # decimal point dropped: 31.2 typed as 312
df.loc[err[6:9], "sbp"] = 999                                        # sentinel code for "not taken"
df.loc[err[9:], "sbp"] = 0
df.to_csv("patients_labs.csv", index=False)

# First look: how much is missing, and where?
df = pd.read_csv("patients_labs.csv")
print(df.isna().mean().mul(100).round(1).rename("% missing"))
print(df.describe().round(1))                                        # max BMI 500+? SBP 999? red flags
```

### 2. Why is it missing? MCAR vs. MAR checks
```python
# Is A1c missing at random (MCAR) or does missingness depend on other columns (MAR)?
df["a1c_missing"] = df["a1c"].isna()
print(df.groupby("diabetic")["a1c_missing"].mean().round(3).rename("A1c missing rate by diabetic"))
df["age_band"] = pd.cut(df["age"], [17, 44, 64, 90], labels=["18-44", "45-64", "65+"])
print(df.groupby("age_band", observed=True)["a1c_missing"].mean().round(3))

# Compare the patients WITH an A1c to those without: if they differ, complete-case results are biased
print(df.groupby("a1c_missing")[["age", "sbp"]].mean().round(1))
print(df.groupby("a1c_missing")["diabetic"].mean().round(3).rename("share diabetic"))

# BMI should look MCAR: missing rate flat across groups
print(df.groupby("age_band", observed=True)["bmi"].apply(lambda s: s.isna().mean()).round(3))
```

### 3. Outlier flags: IQR, z-score, robust z, and clinical rules
```python
# Three ways to flag outliers in BMI and SBP
def iqr_flags(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))

def z_flags(s, z=3):
    return ((s - s.mean()) / s.std()).abs() > z

def mad_flags(s, z=3.5):                                   # robust z: median and MAD are not dragged by the outliers
    med = s.median(); mad = (s - med).abs().median()
    return (0.6745 * (s - med) / mad).abs() > z

for col in ["bmi", "sbp"]:
    s = df[col].dropna()
    print(col, "IQR:", iqr_flags(s).sum(), " z>3:", z_flags(s).sum(), " robust z:", mad_flags(s).sum())

# Domain rules beat pure statistics: a BMI of 312 or an SBP of 0 or 999 is impossible, not "unusual"
rules = {"bmi": (12, 80), "sbp": (60, 260)}
for col, (lo, hi) in rules.items():
    bad = (df[col] < lo) | (df[col] > hi)
    print(col, "impossible values:", bad.sum(), sorted(df.loc[bad, col].unique())[:6])
```

### 4. Fix, then compare imputation strategies against the truth
```python
clean = df.copy()
# Fix what you can explain, null out what you cannot
fix = clean["bmi"] > 80
clean.loc[fix, "bmi"] = clean.loc[fix, "bmi"] / 10            # dropped decimal point: 312 -> 31.2
clean.loc[clean["sbp"].isin([0, 999]), "sbp"] = np.nan        # sentinels are missing values, not readings
print("BMI fixed:", fix.sum(), "  SBP set to missing:", clean["sbp"].isna().sum())

# How wrong is each strategy for the mean A1c? We know the truth because we simulated it
true_mean = truth["a1c"].mean()
cc = clean["a1c"].mean()                                       # complete case (pandas skips NaN)
overall = clean["a1c"].fillna(clean["a1c"].mean()).mean()      # mean imputation (same mean, shrunk spread)
grp = clean.groupby(["diabetic", "age_band"], observed=True)["a1c"].transform("mean")
group_imp = clean["a1c"].fillna(grp).mean()                    # impute within diabetic x age band
print(f"true {true_mean:.2f} | complete case {cc:.2f} | mean impute {overall:.2f} | group impute {group_imp:.2f}")
print(f"SD true {truth['a1c'].std():.2f} | after mean impute {clean['a1c'].fillna(cc).std():.2f}")

# Diabetes control rate (A1c >= 8) is also biased by who got tested
print(f"A1c>=8: true {(truth.a1c >= 8).mean():.3f} | among tested {(clean.a1c.dropna() >= 8).mean():.3f}")
```

### 5. Bridge to ML: train-only imputers, missing flags, scaled KNN
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Bridge to ML: fit imputers on TRAINING rows only, then apply to test (no peeking)
X = clean[["age", "bmi", "sbp", "a1c"]].assign(diabetic=clean["diabetic"].astype(int))
X_train, X_test = train_test_split(X, test_size=0.25, random_state=11)

med = SimpleImputer(strategy="median", add_indicator=True).fit(X_train)
print(med.get_feature_names_out())                             # adds missingindicator_* columns

# KNN measures distance, so scale first (otherwise age and SBP swamp everything)
sc = StandardScaler().fit(X_train)                             # NaNs are ignored when fitting
knn = KNNImputer(n_neighbors=15).fit(sc.transform(X_train))
Xt = pd.DataFrame(sc.inverse_transform(knn.transform(sc.transform(X_test))), columns=X.columns, index=X_test.index)

# Score the guesses on test rows where A1c was hidden but we know the truth
hid = X_test["a1c"].isna()
t = truth.loc[X_test.index[hid], "a1c"]
print(f"MAE of imputed A1c on {hid.sum()} hidden test values: "
      f"median {(t - X_train['a1c'].median()).abs().mean():.2f}, KNN (scaled) {(t - Xt.loc[hid, 'a1c']).abs().mean():.2f}")
```

### 6. Push to GitHub
```bash
git add day11
git commit -m "Day 11: missing data (MCAR/MAR), outliers and imputation on patient labs"
git push
```

### Stretch: Snowflake SQL (data-quality checks)
```sql
-- null rates and clinical-rule breaks in one pass
SELECT COUNT(*)                                                   AS patients,
       ROUND(AVG(IFF(a1c IS NULL, 1, 0)) * 100, 1)                AS a1c_null_pct,
       ROUND(AVG(IFF(bmi IS NULL, 1, 0)) * 100, 1)                AS bmi_null_pct,
       SUM(IFF(bmi NOT BETWEEN 12 AND 80, 1, 0))                  AS bmi_impossible,
       SUM(IFF(sbp IN (0, 999), 1, 0))                            AS sbp_sentinels,
       SUM(IFF(sbp NOT BETWEEN 60 AND 260 AND sbp NOT IN (0, 999), 1, 0)) AS sbp_other_breaks
FROM patients_labs;

-- is A1c missing at random? null rate by diabetic status
SELECT diabetic,
       COUNT(*)                                     AS patients,
       ROUND(AVG(IFF(a1c IS NULL, 1, 0)) * 100, 1)  AS a1c_null_pct,
       ROUND(AVG(a1c), 2)                           AS mean_a1c_tested
FROM patients_labs
GROUP BY diabetic;
```

## Printables
- Cheat sheet: `day11_v1_cheat_sheet.png`
- Infographic: `day11_v2_infographic.png`
- Code worksheet: `day11_v3_worksheet.png`
- Practice quiz: `day11_v4_quiz.png`
- PDF: `day11_lesson.pdf`

## Podcast script
(Audio: `day11_podcast.mp3`)

Hey Todor, welcome to day eleven. Today is about the part of data science nobody puts on a poster but everybody does all day: missing values and outliers. And we're going to do it with patient data, because in healthcare, the holes in the data are rarely random.

Here's the setup. We simulated two thousand primary care patients with age, diabetes status, body mass index, blood pressure, and hemoglobin A1c, the three month blood sugar test. Then we broke the data the way real extracts break. Some BMIs are missing because the scale got skipped. A lot of A1c values are missing because the doctor never ordered the test. A few BMIs had the decimal point dropped, so thirty one point two became three hundred twelve. And a few blood pressures were stored as zero or nine ninety nine, which is a system's way of saying "not taken."

Step one is always the same. Before any chart, run the missing rate for every column and a quick describe. In this data, A1c is missing for about forty six percent of patients, and the maximum BMI is four hundred twenty seven. No human has a BMI of four hundred twenty seven. That's a typo, and you found it in five seconds.

Step two is the question that separates an analyst from a data scientist: why is it missing? There are three flavors. Missing completely at random means the gap has nothing to do with anything, like the skipped scale. You can check that: BMI is missing about eight percent in every age group. Missing at random means the gap depends on other columns you can see. A1c is the classic case. It's missing for about fifty nine percent of non diabetic patients but only eight percent of diabetic patients, and much more often for younger adults. And then there's missing not at random, where the gap depends on the value itself, like patients skipping a lab because they suspect bad news. You can't prove that one from the data, so you ask how the data gets collected.

Why does this matter? Because the patients who got tested are older and sicker. If you just average the A1c values you have, you get six point four five. The true average for everyone is six point zero six. Worse, the share of patients with an A1c of eight or higher looks like almost nineteen percent among the tested, when the truth is about eleven. Imagine putting that first number in a quality report.

So how do you fill the holes? The lazy fix is to plug in the average. It keeps the biased mean, it creates a fake spike right at that one value, and it shrinks the spread, which makes everything look more certain than it is. A better fix is to impute within similar patients, for example within diabetic status and age band. In our data that lands right on the true mean. For machine learning, you can use a K nearest neighbors imputer, which borrows values from the most similar patients. Just remember that it measures distance, so scale your features first. Scaled K N N cut the imputation error from about point five eight to about point three nine A1c points on held out patients.

Now outliers. Statistical rules like the I Q R fence or a z score are good for flagging, but they don't know medicine. The I Q R rule flagged twenty blood pressures, but only six were impossible. The other fourteen were real people with very high or very low readings, and deleting them would erase exactly the patients a care team cares about. So use clinical ranges first. Fix what you can explain, like the dropped decimal. Turn sentinel codes into missing values. And keep the rest.

Two habits for the road. When you build a model, fit the imputer on the training rows only, so the test set stays truly unseen. And keep a little flag column that says "this value was missing," because sometimes the fact that a test wasn't ordered is itself a strong signal.

Tomorrow, on day twelve, we'll get the data ready for models: skewed features, log transforms, scaling, binning, and turning categories into numbers. See you then.
