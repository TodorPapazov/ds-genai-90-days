# Day 3: Confidence intervals and the bootstrap: how big is the effect, give or take?

## Concept
Day 2 asked is it real?; today asks how big, give or take? A confidence interval is estimate ± critical value × standard error (SE = s / √n), so quadrupling n halves the margin. You will build a t-interval for mean length of stay, compare Wald and Wilson intervals for readmission rates (Wald breaks on small samples), and learn the bootstrap: resample your data with replacement, recompute the statistic 5,000 times, and take the 2.5th and 97.5th percentiles. It works for medians, ratios and model metrics with no formula. '95%' describes the method: about 95 of 100 intervals from repeated studies capture the truth, which you will verify with a coverage simulation. A 95% CI for a difference that includes 0 matches p > 0.05, and it also shows the size of the effect.

## Key numbers
- **Mean LOS (95% CI):** 4.75 [4.41, 5.09]
- **Readmit rate, Wilson:** 15.5% [12.3%, 19.4%]
- **Median claim (bootstrap):** $420 [$407, $432]
- **PPO-HMO median gap:** $20 [-$2, $50]
- **CIs covering truth:** 94 / 100

## Hands-on
1. In your repo, create day03/ and open day03/confidence_intervals_bootstrap.ipynb in VS Code; conda activate ds90 (no new packages needed).
2. Simulate 400 heart-failure lengths of stay with seed 6 and build a 95% t-interval by hand, then check it with stats.t.interval (snippet 1). Expect 4.75 [4.41, 5.09].
3. Compare Wald and Wilson intervals for 62/400 and 2/25 readmissions (snippet 2). Note which Wald interval goes below 0 and why.
4. Plot margin of error against n for a 15% rate and work out the n needed for ±2 points (about 1,225).
5. Rebuild the Day 2 claims (seed 42) and bootstrap the median 5,000 times (snippet 3). Expect $420 [$407, $432].
6. Bootstrap the PPO minus HMO median gap and cross-check with scipy.stats.bootstrap (snippet 4). Does the interval include 0? How does that match Day 2's permutation p?
7. Run the coverage experiment: 100 samples of 40 patients, 100 intervals, count how many contain the true mean of 4.8 and plot them (snippet 5).
8. Write 3 stakeholder sentences, each with an estimate, its 95% CI and a plain-English meaning (plus the caveat that the data is simulated).
9. Commit and push (snippet 6).
10. Stretch: in Snowflake, compute mean LOS with a 95% CI by hospital and flag hospitals whose interval sits above the system average (snippet 7).

## Code
### 1. t-interval for mean length of stay
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(6)
# Length of stay (days) for 400 heart-failure admissions: right-skewed
los = rng.gamma(shape=2.0, scale=2.4, size=400)     # true mean = 4.8 days
n, m, s = len(los), los.mean(), los.std(ddof=1)
se = s / np.sqrt(n)                                  # standard error
t_crit = stats.t.ppf(0.975, df=n - 1)                # ~1.97 for n = 400
lo, hi = m - t_crit * se, m + t_crit * se
print(f"Mean LOS = {m:.2f} days, SE = {se:.3f}, 95% CI [{lo:.2f}, {hi:.2f}]")
print(stats.t.interval(0.95, df=n - 1, loc=m, scale=se))   # same answer
```

### 2. Readmission rate: Wald vs. Wilson
```python
from scipy import stats

def wald(k, n, z=1.96):
    p = k / n
    se = (p * (1 - p) / n) ** 0.5
    return p - z * se, p + z * se

for k, n, label in [(62, 400, "Hospital"), (2, 25, "Small clinic")]:
    w = wald(k, n)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    print(f"{label}: rate {k/n:.3f} | Wald [{w[0]:.3f}, {w[1]:.3f}] | "
          f"Wilson [{ci.low:.3f}, {ci.high:.3f}]")
```

### 3. Bootstrap CI for the median claim
```python
import numpy as np
import pandas as pd

# Same simulated claims as Day 2
rng = np.random.default_rng(42)
n = 5000
plan = rng.choice(["HMO", "PPO"], n)
amount = rng.lognormal(mean=6, sigma=1, size=n) * np.where(plan == "PPO", 1.10, 1.0)
df = pd.DataFrame({"plan_type": plan, "amount": amount})
amts = df["amount"].to_numpy()

boot = np.array([np.median(rng.choice(amts, size=len(amts), replace=True))
                 for _ in range(5000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Median claim = ${np.median(amts):.0f}, bootstrap 95% CI [${lo:.0f}, ${hi:.0f}]")
```

### 4. Bootstrap the PPO-HMO median gap
```python
from scipy import stats

hmo = df.loc[df.plan_type == "HMO", "amount"].to_numpy()
ppo = df.loc[df.plan_type == "PPO", "amount"].to_numpy()
gaps = np.array([np.median(rng.choice(ppo, len(ppo))) - np.median(rng.choice(hmo, len(hmo)))
                 for _ in range(5000)])
obs = np.median(ppo) - np.median(hmo)
lo, hi = np.percentile(gaps, [2.5, 97.5])
print(f"Median gap = ${obs:.0f}, bootstrap 95% CI [${lo:.0f}, ${hi:.0f}]")  # includes 0?

# Cross-check with scipy's built-in bootstrap (BCa method by default)
res = stats.bootstrap((amts,), np.median, n_resamples=5000, random_state=1)
print(res.confidence_interval)
```

### 5. Coverage: what 95% really means
```python
import matplotlib.pyplot as plt

true_mean = 2.0 * 2.4                     # gamma(2, 2.4) mean = 4.8 days
hits, rows = 0, []
for i in range(100):
    x = rng.gamma(2.0, 2.4, size=40)
    se = x.std(ddof=1) / np.sqrt(40)
    lo, hi = stats.t.interval(0.95, df=39, loc=x.mean(), scale=se)
    hits += lo <= true_mean <= hi
    rows.append((lo, hi))
print(f"{hits} of 100 intervals contain the true mean")

for i, (lo, hi) in enumerate(rows):
    plt.plot([lo, hi], [i, i], color="tab:blue" if lo <= true_mean <= hi else "red")
plt.axvline(true_mean, color="black"); plt.title("100 samples, 100 CIs"); plt.show()
```

### 6. Push to GitHub
```bash
git add day03
git commit -m "Day 3: confidence intervals and bootstrap"
git push
```

### Stretch: Snowflake SQL (CI by hospital)
```sql
WITH h AS (
  SELECT hospital_id,
         COUNT(*)         AS n,
         AVG(los_days)    AS mean_los,
         STDDEV(los_days) AS sd_los
  FROM admissions
  GROUP BY hospital_id
)
SELECT hospital_id, n, mean_los,
       mean_los - 1.96 * sd_los / SQRT(n) AS ci_low,
       mean_los + 1.96 * sd_los / SQRT(n) AS ci_high,
       ci_low > (SELECT AVG(los_days) FROM admissions) AS above_system_avg
FROM h
ORDER BY mean_los DESC;
```

## Printables
- Cheat sheet: `day03_v1_cheat_sheet.png`
- Infographic: `day03_v2_infographic.png`
- Code worksheet: `day03_v3_worksheet.png`
- Practice quiz: `day03_v4_quiz.png`
- PDF: `day03_lesson.pdf`

## Podcast script
(Audio: `day03_podcast.mp3`)

Welcome back to the Data Science Coach podcast. This is Day Three. Yesterday we asked a yes or no question: is the cost gap between PPO and HMO members real, or just noise? Today we ask a better question, one your stakeholders actually care about. How big is the effect, give or take? The tool for that is the confidence interval.

Here's the idea in one sentence. A confidence interval is your estimate, plus or minus a margin that reflects how much that estimate would wobble if you collected a fresh sample. That wobble has a name, the standard error. For an average, it's the sample standard deviation divided by the square root of the sample size. Notice the square root. If you want to cut your margin of error in half, you don't double your sample. You need four times as many patients.

Let's make it concrete. Picture four hundred heart failure admissions. The average length of stay is about four point seven five days. The standard error is about point one seven days. Multiply that by roughly two, and you get a ninety five percent interval from about four point four to about five point one days. So when the nurse manager asks how long these patients stay, you don't just say four point seven five. You say our best estimate is about four and three quarter days, and the true average is plausibly somewhere between four point four and five point one.

Now, what does ninety five percent actually mean? This trips up a lot of people in interviews. It doesn't mean there's a ninety five percent chance the truth sits inside this particular interval. It describes the method. If you repeated the study over and over, about ninety five out of every hundred intervals you built would capture the true value. In today's exercise you'll see this with your own eyes. You'll draw a hundred samples, build a hundred intervals, and count how many contain the true average. In my run, ninety four did. About five should miss, and that's exactly the point.

Rates need a little extra care. Say sixty two out of four hundred discharges came back within thirty days, a fifteen and a half percent readmission rate. The simple textbook formula, called the Wald interval, works fine here. But try a small clinic with two readmissions out of twenty five patients. The Wald formula gives a lower bound below zero, which is impossible for a rate. The Wilson interval fixes that and gives roughly two percent to twenty five percent. Wide, yes, but honest. Small samples deserve wide intervals.

And now my favorite trick of the day, the bootstrap. What if you want an interval for a median, or a ratio, or a model's accuracy score, and there's no neat formula? You let the computer do the work. Take your data, and draw a new sample of the same size, with replacement, so some rows show up twice and some not at all. Compute your statistic. Do that five thousand times. Then take the middle ninety five percent of those results. That's your interval. For our five thousand simulated claims, the median is about four hundred twenty dollars, and the bootstrap interval runs from about four hundred seven to four hundred thirty two.

Here's where it connects back to yesterday. Bootstrap the gap between the PPO and HMO medians, and the interval runs from about minus two dollars to about fifty dollars. It crosses zero. That lines up with yesterday's permutation test, where the p-value was about point one. When a ninety five percent interval for a difference includes zero, the matching test won't be significant at point oh five. But the interval tells you more. It says the gap could be nothing, or it could be as large as fifty dollars a claim. That's something a finance team can actually plan around.

For your hands-on today, build the length of stay interval by hand and check it with scipy, compare Wald and Wilson intervals for the hospital and the small clinic, bootstrap the median claim and the median gap, run the coverage experiment, and write three sentences a stakeholder could understand. Then commit and push to GitHub.

Tomorrow, we put hypothesis tests and intervals to work in a real decision. We'll design and analyze an A B test, think about sample size and statistical power before the test starts, and see why peeking at results early can fool you. See you on Day Four.
