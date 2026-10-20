# Day 6: Correlation and simple linear regression: read a slope, R squared and a residual plot

## Concept
Yesterday's scatter plot showed cost drifting up with age; today you measure that link and fit a line through it. Pearson r measures how close points sit to a straight line (-1 to +1); Spearman rho does the same on ranks, so skew and outliers matter less. On the Day 5 table of 2,000 simulated patients, age vs. raw cost gives Pearson 0.33 but age vs. log cost gives 0.48, while Spearman stays at 0.48 both times because a log never changes the ranking. Simple linear regression fits log cost = 7.14 + 0.0264 x age. On a log scale the slope becomes a percent change: exp(0.0264) - 1 = about 2.7% higher cost per year of age, or about 30% per decade (95% CI for the slope 0.0243 to 0.0286). R squared is 0.233: age alone explains about 23% of the variation in log cost. That is incomplete, not useless. A residual plot (actual minus predicted vs. fitted) should be a shapeless band around 0: on raw dollars you get a fan with huge outliers (R squared only 0.11), on log cost an even band. Finally, correlation is not causation: ER visits correlate with log cost (r = 0.22, coefficient +0.267 alone), but after adding chronic conditions and smoking the coefficient is -0.011 (p = 0.63). Smoking and illness drive both; in this simulation ER visits never caused cost.

## Key numbers
- **Age vs. raw cost:** Pearson 0.33, Spearman 0.48
- **Age vs. log cost:** Pearson 0.48, Spearman 0.48
- **Fitted line:** log cost = 7.14 + 0.0264 x age
- **Slope as % change:** +2.7% per year, +30% per decade
- **R squared:** 0.233 (log cost) vs. 0.11 (raw dollars)
- **Typical predicted cost:** age 40 $3,632, age 70 $8,031
- **ER visits coefficient:** +0.267 alone, -0.011 adjusted (p = 0.63)

## Hands-on
1. In your repo, create day06/, copy day05/patients.csv into it, and open day06/regression_patients.ipynb in VS Code; conda activate ds90 and conda install statsmodels scikit-learn if they are not already there.
2. Load the table, drop duplicates (2,000 rows), add log_cost, and compute Pearson and Spearman for age vs. raw cost and vs. log cost (snippet 1). Expect Pearson 0.33 vs. 0.48 and Spearman 0.48 both times.
3. Fit the line with stats.linregress and plot it with sns.regplot (snippet 2). Write down slope (0.0264), intercept (7.14) and R squared (0.233).
4. Translate the slope: np.expm1(b) per year and np.expm1(10*b) per decade. Write one manager-friendly sentence.
5. Run smf.ols("log_cost ~ age") and read the summary: coef, std err, p-value, 95% CI. Confirm scikit-learn LinearRegression gives the same slope and predict ages 40 and 70 (snippet 3).
6. Draw residual plots for raw dollars and for log cost side by side (snippet 4). Which is a shapeless cloud? Compare the two R squared values and the mean residual for smokers vs. non-smokers.
7. Causation test: regress log cost on ER visits alone, then add chronic conditions and smoker (snippet 5). Watch the ER coefficient and its 95% CI.
8. Add a markdown cell with 5 plain-language findings, including the caveat that the data is simulated and that R squared 0.23 means other drivers are missing.
9. Commit and push (snippet 6).
10. Stretch: in Snowflake, compute CORR, REGR_SLOPE, REGR_INTERCEPT and REGR_R2 on LN(annual_cost) vs. age (snippet 7) and check they match Python.

## Code
### 1. Load the patients and compare Pearson vs. Spearman
```python
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the Day 5 simulated patient table (copy day05/patients.csv into day06/)
df = pd.read_csv("patients.csv", parse_dates=["admit_month"])
df = df.drop_duplicates().reset_index(drop=True)      # 2,012 rows -> 2,000 patients
df["log_cost"] = np.log(df["annual_cost"])            # cost is right-skewed: model the log
print(df.shape)

# Pearson = straight-line strength on the raw values; Spearman = same idea on ranks
for y in ["annual_cost", "log_cost"]:
    r, p = stats.pearsonr(df["age"], df[y])
    rho, _ = stats.spearmanr(df["age"], df[y])
    print(f"age vs {y:12s} Pearson r = {r:.3f}  Spearman rho = {rho:.3f}  (p = {p:.1e})")
```

### 2. Fit the line and read the slope
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Simple linear regression: log_cost = intercept + slope * age
fit = stats.linregress(df["age"], df["log_cost"])
print(f"slope = {fit.slope:.4f} per year, intercept = {fit.intercept:.3f}")
print(f"r = {fit.rvalue:.3f}, R squared = {fit.rvalue**2:.3f}")
# On a log scale, a slope b means cost changes by (exp(b) - 1) per one-unit step
print(f"each extra year: {np.expm1(fit.slope):.1%} higher cost; "
      f"each extra decade: {np.expm1(10 * fit.slope):.1%}")

ax = sns.regplot(data=df, x="age", y="log_cost", scatter_kws=dict(s=8, alpha=0.3),
                 line_kws=dict(color="red"))
ax.set_title(f"Cost rises about {np.expm1(10 * fit.slope):.0%} per decade of age "
             f"(R squared = {fit.rvalue**2:.2f})")
plt.show()
```

### 3. statsmodels OLS summary and scikit-learn
```python
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

# Same model, full statistical summary (coef, std err, p-value, 95% CI, R-squared)
m1 = smf.ols("log_cost ~ age", data=df).fit()
print(m1.summary().tables[1])
print("R squared:", round(m1.rsquared, 3), " 95% CI for slope:", m1.conf_int().loc["age"].round(4).tolist())

# scikit-learn gives the same line (it is built for prediction, not p-values)
lr = LinearRegression().fit(df[["age"]], df["log_cost"])
print("sklearn slope:", lr.coef_[0].round(4), "intercept:", round(lr.intercept_, 3))
print("predicted cost at 40 vs 70:", np.exp(lr.predict(pd.DataFrame({"age": [40, 70]}))).round(0))
```

### 4. Residual plots: raw dollars vs. log cost
```python
# Residual = actual - predicted. A good residual plot is a shapeless cloud around 0.
m_raw = smf.ols("annual_cost ~ age", data=df).fit()     # same model on RAW dollars
print("R squared raw $:", round(m_raw.rsquared, 3), " log $:", round(m1.rsquared, 3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(m_raw.fittedvalues, m_raw.resid, s=6, alpha=0.3)
ax[0].axhline(0, color="red"); ax[0].set_title("Raw dollars: fan shape and big outliers")
ax[1].scatter(m1.fittedvalues, m1.resid, s=6, alpha=0.3)
ax[1].axhline(0, color="red"); ax[1].set_title("Log cost: even band around 0 (good)")
for a in ax: a.set_xlabel("fitted value"); a.set_ylabel("residual")
plt.tight_layout(); plt.show()

# Residuals still hold signal: smokers sit above the line
print(m1.resid.groupby(df["smoker"]).mean().round(3))
```

### 5. Correlation is not causation: ER visits
```python
# Correlation is not causation: ER visits vs cost
# (In this simulation, cost was generated WITHOUT er_visits. ER visits and cost
#  share causes: chronic conditions and smoking.)
r_er, _ = stats.pearsonr(df["er_visits"], df["log_cost"])
print(f"ER visits vs log cost: r = {r_er:.3f}")
print(df.groupby("smoker")[["er_visits", "log_cost"]].mean().round(2))

naive = smf.ols("log_cost ~ er_visits", data=df).fit()
adj = smf.ols("log_cost ~ er_visits + chronic_conditions + smoker", data=df).fit()
for name, m in [("naive", naive), ("adjusted", adj)]:
    lo, hi = m.conf_int().loc["er_visits"]
    print(f"{name:9s} er_visits coef = {m.params['er_visits']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  p = {m.pvalues['er_visits']:.3f}")
```

### 6. Push to GitHub
```bash
git add day06
git commit -m "Day 6: correlation and simple linear regression"
git push
```

### Stretch: Snowflake SQL (correlation and regression in SQL)
```sql
SELECT CORR(LN(annual_cost), age)            AS pearson_r,
       REGR_SLOPE(LN(annual_cost), age)      AS slope,
       REGR_INTERCEPT(LN(annual_cost), age)  AS intercept,
       REGR_R2(LN(annual_cost), age)         AS r_squared,
       EXP(10 * REGR_SLOPE(LN(annual_cost), age)) - 1 AS pct_change_per_decade
FROM patients;
-- note: REGR_* functions take (y, x): dependent variable first
```

## Printables
- Cheat sheet: `day06_v1_cheat_sheet.png`
- Infographic: `day06_v2_infographic.png`
- Code worksheet: `day06_v3_worksheet.png`
- Practice quiz: `day06_v4_quiz.png`
- PDF: `day06_lesson.pdf`

## Podcast script
(Audio: `day06_podcast.mp3`)

Hey Todor, welcome to Day Six. Yesterday we drew a scatter plot of age against annual cost and saw the dots drift upward. Today we put a number on that drift and then a line through it. We're covering correlation and simple linear regression. By the end you'll know how to read a slope, an R squared, and a residual plot, and you'll have a real example of why correlation is not causation.

We're reusing the same simulated table of two thousand patients from Day Five. Because cost has that long right tail, we work with the log of cost.

Let's start with correlation. Pearson's r measures how close the points sit to a straight line, on a scale from minus one to plus one. Spearman's rho asks the same question using ranks instead of raw values. Here's the interesting part. Age against raw dollars gives a Pearson r of about point three three. Age against log cost gives about point four eight. Same patients, very different answer. The long tail of expensive patients bends the raw relationship, and Pearson only rewards straight lines. Spearman, meanwhile, sits at about point four eight both times, because taking a log never changes who ranks above whom. So if your data is skewed, check Spearman, or transform first.

Now the line. Simple linear regression finds the straight line that makes the squared misses as small as possible. Our line says log cost equals about seven point one four, plus point zero two six four times age. That slope is the headline. On a log scale, a slope translates into a percent change. Each extra year of age goes with about two point seven percent higher cost, which works out to roughly thirty percent per decade. In dollars, the model's typical cost is about thirty six hundred at age forty and about eight thousand at age seventy. That's a sentence a manager can use.

What about the intercept? It's the predicted value at age zero, about twelve hundred sixty dollars. But nobody in this data is under eighteen, so don't report it as a finding.

Next, R squared. Ours is about point two three. That means age alone explains about twenty three percent of the variation in log cost. That's not a bad model. It's an incomplete one. Smoking, chronic illness, and plain chance explain the rest. When you present it, say what's missing. When you run the statsmodels summary you'll also see a ninety five percent confidence interval for the slope, about point zero two four to point zero two nine. That's your uncertainty, straight from Day Three.

Then the residual plot. A residual is actual minus predicted, the part the line missed. Plot residuals against fitted values. A healthy plot is a shapeless band around zero. On raw dollars you get a fan, with huge outliers above the line, and the R squared drops to about point one one. On log cost you get an even band. And if you split residuals by smoking, smokers sit well above the line on average. The leftovers still hold signal, which is a hint for tomorrow.

Finally, the trap. Patients with more E R visits cost more. The correlation is about point two two, and a simple regression says each extra E R visit goes with about thirty percent higher cost. It's tempting to say cut E R visits and you cut cost. But in this simulation, cost was generated without E R visits at all. Smoking and chronic conditions drive both. Add those two to the model and the E R coefficient drops to about zero, with a p value around point six three. That's confounding. Correlation told you where to look. It didn't tell you why.

For your hands-on, compute both correlations, fit the line with statsmodels and scikit-learn, draw both residual plots, run the E R visits test, write your findings, and push to GitHub.

Tomorrow is Day Seven, our week one wrap-up mini project. We'll move to multiple linear regression, putting age, smoking, and chronic conditions into one model, and you'll see the age slope shrink once its partners join in. Then you'll write a short report that goes from E D A all the way to a regression you can explain. See you then.
