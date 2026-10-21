# Day 7: Multiple regression and dummy variables: holding the others fixed (week 1 mini-report)

## Concept
Day 6 ended with a warning: one predictor can take credit for things it did not do. Multiple regression puts several predictors in one model, and each coefficient becomes the change in the outcome holding the other predictors fixed. On the same 2,000 simulated patients, the age slope on log cost is 0.0264 alone (+30% per decade) but 0.0190 (+21% per decade) once chronic conditions and smoking join, because older patients carry more chronic illness (r = 0.40). R squared jumps from 0.233 to 0.436. Categories become dummy variables: 0/1 columns, one per level minus a reference level. Plan has three levels, so HMO is the reference and PPO and Medicare Advantage each get a column (keeping all three is the dummy trap). On a log scale, exp(coef) - 1 is the % difference: smokers cost +166% (95% CI +142% to +191%) versus non-smokers of the same age and illness burden, and each chronic condition adds +27%. Plan and sex are not significant, but the plan CI (about -14% to +11%) means 'no clear difference', not 'no effect'. Adjusted R squared penalizes extra columns: plain R squared rises even for a column of random noise, while adjusted R squared falls from 0.4349 to 0.4342 to 0.4339. Two more habits: check n after each fit (adding BMI silently drops 142 patients), and check multicollinearity with the VIF (all under 2.4 here). Finally a preview of Week 3: on 500 held-out patients, age alone scores test R squared 0.22 and six features score 0.45. To close Week 1, you turn the analysis into a short REPORT.md a manager could read: your first portfolio piece.

## Key numbers
- **Age slope:** 0.0264 alone, 0.0190 with chronic + smoker
- **Per decade of age:** +30% alone, +21% adjusted
- **R squared:** 0.233 (age) to 0.436 (age + chronic + smoker)
- **Smoker:** +166% (95% CI +142% to +191%)
- **Each chronic condition:** +27% (95% CI +24% to +31%)
- **Plan, sex:** not significant (p 0.56 to 0.91)
- **Adjusted R squared:** 0.4349 (3 cols), 0.4342 (+ plan, sex), 0.4339 (+ noise)
- **Adding BMI:** n drops from 2,000 to 1,858
- **Held-out test R squared:** 0.22 (age) vs. 0.45 (6 features); median miss $2,584 vs. $2,160

## Hands-on
1. In your repo, create day07/, copy day06/patients.csv into it, and open day07/multiple_regression_patients.ipynb in VS Code with conda activate ds90.
2. Load, drop duplicates, add log_cost, count missing BMI (142), then fit log_cost ~ age and log_cost ~ age + chronic_conditions + smoker (snippet 1). Expect the age slope to fall from 0.0264 to 0.0190 and R squared to rise from 0.233 to 0.436.
3. Write one sentence explaining why the age slope shrank. Hint: df[["age","chronic_conditions"]].corr().
4. Fit the full model with C(smoker, Treatment('no')), C(plan, Treatment('HMO')) and C(sex), and build the % change table with np.expm1 on the coefficients and CIs (snippet 2). Also run pd.get_dummies(..., drop_first=True) and look at the column names.
5. Compare R squared and adjusted R squared for four models, including one with a random noise column, then add bmi and check model.nobs (snippet 3).
6. Compute the VIF for every predictor (snippet 4). Are any above 5?
7. Split 75/25 with train_test_split(random_state=42) and compare age-only vs. six features on test R squared and median dollar miss (snippet 5).
8. Week 1 mini-project: write day07/REPORT.md with Question, Data (simulated, 2,000 patients, 12 duplicates and 142 missing BMI), two charts saved as PNG (a Day 5 EDA chart and the coefficient plot), the model in one line, 5 plain-language findings, and a Limits section (simulated data, observational, not causal, R squared 0.44).
9. Commit and push (snippet 6).
10. Stretch: in Snowflake, build dummies with IFF() and compare REGR_SLOPE of log cost on age for smokers and non-smokers (snippet 7).

## Code
### 1. One predictor vs. several
```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Same simulated patient table from Day 5 and 6 (copy day06/patients.csv into day07/)
df = pd.read_csv("patients.csv").drop_duplicates().reset_index(drop=True)   # 2,000 patients
df["log_cost"] = np.log(df["annual_cost"])
print(df["bmi"].isna().sum(), "patients missing BMI")   # OLS silently drops these rows

# Day 6 model (one predictor) vs. a multiple regression (several at once)
m_age = smf.ols("log_cost ~ age", data=df).fit()
m_multi = smf.ols("log_cost ~ age + chronic_conditions + smoker", data=df).fit()
print("age only : slope", round(m_age.params["age"], 4), " R sq", round(m_age.rsquared, 3))
print("multiple : slope", round(m_multi.params["age"], 4), " R sq", round(m_multi.rsquared, 3))
print(m_multi.summary().tables[1])
```

### 2. Dummy variables and the % change table
```python
# Categorical columns become dummy (0/1) columns. C() makes it explicit and lets
# you pick the reference level that every other level is compared against.
m_full = smf.ols(
    "log_cost ~ age + chronic_conditions + C(smoker, Treatment('no'))"
    " + C(plan, Treatment('HMO')) + C(sex)", data=df).fit()

# On a log scale, exp(coef) - 1 is the % difference, holding the other columns fixed
pct = (np.exp(m_full.params) - 1).drop("Intercept")
ci = (np.exp(m_full.conf_int()) - 1).drop("Intercept")
table = pd.DataFrame({"pct_change": pct, "ci_low": ci[0], "ci_high": ci[1],
                      "p_value": m_full.pvalues.drop("Intercept")})
print(table.round(3))

# pandas does the same encoding for scikit-learn: drop_first avoids the dummy trap
X = pd.get_dummies(df[["age", "chronic_conditions", "smoker", "plan", "sex"]],
                   drop_first=True, dtype=int)
print(X.columns.tolist())
```

### 3. Adjusted R squared, noise and missing BMI
```python
# R squared never goes down when you add a column, even a useless one.
# Adjusted R squared charges a small penalty per column.
rng = np.random.default_rng(1)
df["noise"] = rng.normal(size=len(df))            # pure random column
m_noise = smf.ols(m_full.model.formula + " + noise", data=df).fit()
for name, m in [("age only", m_age), ("multiple", m_multi), ("full", m_full), ("full + noise", m_noise)]:
    print(f"{name:13s} R sq = {m.rsquared:.5f}  adj R sq = {m.rsquared_adj:.5f}  n = {int(m.nobs)}")

# BMI has missing values: statsmodels drops those rows, so n changes. Check it!
m_bmi = smf.ols(m_full.model.formula + " + bmi", data=df).fit()
print("with bmi: n =", int(m_bmi.nobs), " bmi coef =", round(m_bmi.params["bmi"], 4),
      " p =", round(m_bmi.pvalues["bmi"], 3))
```

### 4. Multicollinearity check (VIF)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Multicollinearity: when predictors move together, coefficients get shaky.
# VIF above about 5 deserves a look; above 10 is a red flag.
Xv = pd.get_dummies(df[["age", "chronic_conditions", "er_visits", "smoker", "plan", "sex"]],
                    drop_first=True, dtype=float)
Xv.insert(0, "const", 1.0)
vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])],
                index=Xv.columns).drop("const")
print(vif.round(2).sort_values(ascending=False))
print(df[["age", "chronic_conditions", "er_visits"]].corr().round(2))
```

### 5. Preview of Week 3: held-out test set
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Preview of Week 3: judge a model on patients it has NOT seen
X = pd.get_dummies(df[["age", "chronic_conditions", "smoker", "plan", "sex"]], drop_first=True, dtype=int)
y = df["log_cost"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
for cols in [["age"], X.columns.tolist()]:
    lr = LinearRegression().fit(X_tr[cols], y_tr)
    pred = lr.predict(X_te[cols])
    print(f"{len(cols)} feature(s): test R sq = {r2_score(y_te, pred):.3f}, "
          f"median abs error in $ = {np.median(np.abs(np.exp(y_te) - np.exp(pred))):,.0f}")
```

### 6. Push to GitHub
```bash
git add day07
git commit -m "Day 7: multiple regression, dummy variables and week 1 report"
git push
```

### Stretch: Snowflake SQL (dummies and slopes by group)
```sql
SELECT smoker,
       COUNT(*)                              AS n,
       REGR_SLOPE(LN(annual_cost), age)      AS age_slope,
       REGR_R2(LN(annual_cost), age)         AS r_squared
FROM patients
GROUP BY smoker;

-- dummy columns by hand (HMO is the reference level)
SELECT patient_id,
       IFF(smoker = 'yes', 1, 0)             AS smoker_yes,
       IFF(plan = 'PPO', 1, 0)               AS plan_ppo,
       IFF(plan = 'Medicare Adv', 1, 0)      AS plan_medicare_adv
FROM patients;
```

## Printables
- Cheat sheet: `day07_v1_cheat_sheet.png`
- Infographic: `day07_v2_infographic.png`
- Code worksheet: `day07_v3_worksheet.png`
- Practice quiz: `day07_v4_quiz.png`
- PDF: `day07_lesson.pdf`

## Podcast script
(Audio: `day07_podcast.mp3`)

Hey Todor, welcome to Day Seven, the last day of week one. Yesterday we fit a straight line of cost against age, and we ended on a warning: one predictor can take credit for things it didn't do. Today we fix that with multiple regression, which simply means one model with several predictors at once.

We're still using the same simulated table of two thousand patients, and we're still modeling the log of annual cost, because cost has that long right tail.

Here's the headline. On its own, age had a slope of about point zero two six, which works out to roughly thirty percent more cost per decade. Now add chronic conditions and smoking to the same model. The age slope drops to about point zero one nine, or roughly twenty one percent per decade. Why did it shrink? Because older patients tend to have more chronic conditions. When age was alone, it quietly got credit for that extra illness. Once chronic conditions sit in the model, each coefficient means the change in cost when that one thing goes up and everything else is held fixed. That phrase, holding the others fixed, is the single most important idea today. Say it every time you read a coefficient out loud.

And look at what we gain. R squared jumps from about point two three to about point four four. Three simple columns now explain nearly half the variation in log cost.

Next, categories. Smoker, plan, and sex are words, not numbers, so we turn them into dummy variables, which are just zero or one columns. Plan has three levels, HMO, PPO, and Medicare Advantage, so we make two columns and leave HMO out as the reference level. Every plan coefficient is then a comparison against HMO. If you kept all three columns, the third would be perfectly predictable from the other two, and the math breaks. People call that the dummy trap.

Reading these on a log scale works just like yesterday. Take e to the coefficient, subtract one, and you get a percent difference. Smokers cost about one hundred sixty six percent more, so roughly two point seven times as much, compared with non-smokers of the same age and illness burden. Each extra chronic condition adds about twenty seven percent. Plan and sex? Not significant. But careful with the wording. The confidence interval for plan runs from about minus fourteen to plus eleven percent, so the honest phrase is no clear difference, not no effect.

Now a trap that catches a lot of analysts. Regular R squared never goes down when you add a column, even a column of pure random noise. Adjusted R squared charges a small penalty for each extra column, and in our data it actually dips when we add plan, sex, and the noise column. So when two models have a different number of predictors, compare adjusted R squared.

Two quick habits. First, check how many rows the model actually used. When we add BMI, a hundred forty two patients with missing BMI silently disappear, so that model isn't even fit on the same people. Second, check multicollinearity with something called the variance inflation factor. It tells you how much predictors overlap. Above five, take a look. Above ten, worry. Ours all sit under two and a half, so we're fine.

Finally, a sneak peek at week three. We split the patients, train on three quarters, and test on the quarter the model never saw. Age alone scores an R squared of about point two two on new patients. All six features score about point four five, and the typical miss in dollars drops from about two thousand six hundred to about two thousand one hundred. That's the real test of a model: how it does on patients it hasn't met.

For your hands-on, fit the three models, build the percent change table, run the noise and BMI checks, and the train and test split. Then comes the week one mini project I promised: a short report in your repo, called REPORT dot md, that walks from the data, through the EDA charts, to this regression, in plain sentences a manager could read. Five findings, one chart or two, and an honest limits section that says the data is simulated. That report is your first portfolio piece.

That wraps up week one. You've gone from describing skewed costs, to testing differences, to confidence intervals, experiments, charts, and now regression with several predictors. Tomorrow we start week two, and we change the kind of question. Instead of predicting a dollar amount, we'll predict a yes or no, like whether a patient gets readmitted within thirty days, with logistic regression and odds ratios. See you then.
