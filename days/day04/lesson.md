# Day 4: A/B testing, sample size and power: design the experiment before you run it

## Concept
Days 2 and 3 analyzed data you already had; today you design the experiment first. An A/B test randomizes patients to control (A) or a change (B), so the change is the only systematic difference. Before launch you fix the primary metric, the minimum detectable effect (MDE), alpha (false alarm rate, Type I error, usually 5%) and power (chance to catch a real effect, usually 80%; missing it is a Type II error), and those set the sample size. For a clinic cutting no-shows from 20% to 16%, that is about 1,445 patients per arm. Halving the effect needs roughly 4x the patients, and a 500-per-arm test has only about 37% power. You will analyze the test with a two-proportion z-test plus a 95% CI for the gap, check power by simulation, and see why peeking (stopping the first time p < 0.05) pushes the false positive rate from 5% to about 24%.

## Key numbers
- **n per arm (80% power):** 1,445 (formula 1,447)
- **No-show: SMS vs control:** 15.7% vs 20.1%
- **Difference (95% CI):** -4.3 pts [-7.1, -1.6], p = 0.002
- **Power at 500 per arm:** 37%
- **False positives, 15 peeks:** 23.7% (promised 5%)

## Hands-on
1. In your repo, create day04/ and open day04/ab_test_power.ipynb in VS Code; conda activate ds90 and conda install statsmodels.
2. Write a 5-line test plan in a markdown cell: primary metric (no-show rate), MDE (20% to 16%), alpha 0.05, power 0.80, analysis (two-proportion z-test + 95% CI).
3. Compute the sample size per arm by formula and with NormalIndPower (snippet 1). Expect about 1,447 and 1,445.
4. Simulate the test with 1,450 patients per arm (seed 4), run proportions_ztest and get the CI for the gap (snippet 2). Expect 15.7% vs 20.1%, p = 0.0023.
5. Check power by simulation: 2,000 trials at 1,450 and at 500 per arm, plus an A/A test with no effect (snippet 3). Expect about 80%, 37% and 5%.
6. Run the peeking experiment: 15 looks at an A/A test, stop at the first p < 0.05 (snippet 4). Expect about 24% false positives.
7. Plot power curves for 3, 4 and 5 point reductions (snippet 5) and read off n for 80% power.
8. Write a one-paragraph decision memo: recommend SMS or not, with the difference, its 95% CI and the caveat that the data is simulated.
9. Commit and push (snippet 6).
10. Stretch: in Snowflake, compute the no-show rate by arm and the two-proportion z-statistic from an appointments table (snippet 7).

## Code
### 1. Sample size per arm (formula and statsmodels)
```python
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# Clinic no-show rate today: 20%. A new SMS reminder should cut it to 16%.
p_ctrl, p_trt, alpha, power = 0.20, 0.16, 0.05, 0.80

# By formula (two-sided test, equal arms)
z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
p_bar = (p_ctrl + p_trt) / 2
n_formula = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar))
              + z_b * np.sqrt(p_ctrl * (1 - p_ctrl) + p_trt * (1 - p_trt))) ** 2
             / (p_ctrl - p_trt) ** 2)
# With statsmodels (Cohen's h effect size)
h = proportion_effectsize(p_ctrl, p_trt)
n_sm = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power)
print(f"n per arm: formula {np.ceil(n_formula):.0f}, statsmodels {np.ceil(n_sm):.0f}, h = {h:.3f}")
```

### 2. Simulate and analyze the A/B test
```python
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

n = 1450                                   # patients per arm (rounded up)
rng = np.random.default_rng(4)
ctrl = rng.random(n) < p_ctrl              # True = no-show
trt = rng.random(n) < p_trt
counts, nobs = [trt.sum(), ctrl.sum()], [n, n]
z, p = proportions_ztest(counts, nobs)
lo, hi = confint_proportions_2indep(counts[0], n, counts[1], n, method="wald")
print(f"No-show: SMS {trt.mean():.1%} vs control {ctrl.mean():.1%}")
print(f"Difference {trt.mean() - ctrl.mean():+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}], p = {p:.4f}")
```

### 3. Power by simulation
```python
def run_trials(p_a, p_b, n, sims=2000, seed=0):
    r = np.random.default_rng(seed)
    a = r.binomial(n, p_a, sims)          # no-shows in each simulated trial
    b = r.binomial(n, p_b, sims)
    pooled = (a + b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    pvals = 2 * stats.norm.sf(np.abs((b - a) / n / se))
    return (pvals < 0.05).mean()

print(f"Power at n=1450: {run_trials(0.20, 0.16, 1450):.1%}")    # about 80%
print(f"Power at n=500:  {run_trials(0.20, 0.16, 500):.1%}")     # underpowered
print(f"False positives when there is NO effect: {run_trials(0.20, 0.20, 1450):.1%}")
```

### 4. Peeking inflates false positives
```python
# Peeking: no real effect (both arms 20%), but we check after every 100 patients
# per arm and stop the first time p < 0.05.
r = np.random.default_rng(11)
looks = np.r_[np.arange(100, 1401, 100), 1450]   # 15 looks
stopped_early = 0
for _ in range(2000):
    a = r.random(1450) < 0.20
    b = r.random(1450) < 0.20
    for k in looks:
        _, pv = proportions_ztest([b[:k].sum(), a[:k].sum()], [k, k])
        if pv < 0.05:
            stopped_early += 1
            break
print(f"False positive rate with {len(looks)} peeks: {stopped_early / 2000:.1%}  (promised: 5%)")
```

### 5. Power curves
```python
import matplotlib.pyplot as plt

ns = np.arange(100, 3001, 50)
for p_b, label in [(0.16, "20% to 16%"), (0.17, "20% to 17%"), (0.15, "20% to 15%")]:
    h = proportion_effectsize(0.20, p_b)
    plt.plot(ns, NormalIndPower().power(h, nobs1=ns, alpha=0.05), label=label)
plt.axhline(0.8, ls="--", color="gray")
plt.xlabel("patients per arm"); plt.ylabel("power"); plt.legend()
plt.title("Power curve: no-show reduction"); plt.show()
```

### 6. Push to GitHub
```bash
git add day04
git commit -m "Day 4: A/B testing and power"
git push
```

### Stretch: Snowflake SQL (no-show rate and z by arm)
```sql
WITH arms AS (
  SELECT arm,
         COUNT(*)                 AS n,
         AVG(IFF(no_show, 1, 0))  AS rate
  FROM appointments
  WHERE experiment_id = 'sms_reminder_v1'
  GROUP BY arm
),
w AS (
  SELECT MAX(IFF(arm = 'B', rate, NULL)) AS p_b,
         MAX(IFF(arm = 'A', rate, NULL)) AS p_a,
         MAX(IFF(arm = 'B', n, NULL))    AS n_b,
         MAX(IFF(arm = 'A', n, NULL))    AS n_a
  FROM arms
)
SELECT p_b, p_a, p_b - p_a AS diff,
       (p_b * n_b + p_a * n_a) / (n_b + n_a) AS p_pool,
       (p_b - p_a) / SQRT(p_pool * (1 - p_pool) * (1 / n_b + 1 / n_a)) AS z_stat
FROM w;
```

## Printables
- Cheat sheet: `day04_v1_cheat_sheet.png`
- Infographic: `day04_v2_infographic.png`
- Code worksheet: `day04_v3_worksheet.png`
- Practice quiz: `day04_v4_quiz.png`
- PDF: `day04_lesson.pdf`

## Podcast script
(Audio: `day04_podcast.mp3`)

Hey Todor, welcome to Day Four. So far we've asked two questions about data we already had. Is the difference real, and how big is it, give or take. Today we flip it around. Instead of analyzing data after the fact, we design the experiment ourselves, before a single patient is enrolled. That's what an A B test is, and it's one of the most common things data scientists get asked about in interviews.

Here's our scenario. A clinic network has a no-show rate of about twenty percent. Someone on the operations team thinks a new text message reminder could bring that down to around sixteen percent. Instead of rolling it out everywhere and hoping, we randomly split patients into two groups. Group A gets the usual reminder. Group B gets the new text. Because the split is random, the only systematic difference between the groups is the reminder itself. That's what lets us say the reminder caused the change, not just that it showed up alongside it.

Now the big question. How many patients do we need? This is where statistical power comes in. Power is the chance your test catches a real effect of the size you care about. The usual target is eighty percent. You also pick alpha, the false alarm rate you're willing to accept, usually five percent. A false alarm, saying there's an effect when there isn't, is called a Type One error. Missing a real effect is a Type Two error. Alpha controls the first, and power protects you from the second.

Put those pieces together, a twenty percent baseline, a four point drop, five percent alpha, and eighty percent power, and the math says you need about fourteen hundred and fifty patients in each group. You'll get that number two ways today, once by the formula and once with the statsmodels library, and they agree within a couple of patients. Here's the part that surprises people. If you wanted to detect a two point drop instead of four, you'd need about six thousand per group. Cut the effect in half, and you need roughly four times the patients. Small effects are expensive.

What happens if you skip the power calculation and just run a quick test with five hundred patients per group? Your power falls to about thirty seven percent. So even if the text reminder really works, you'd miss it most of the time. And then someone reports no significant difference, and a good idea gets thrown away. A small test that finds nothing hasn't proved nothing is there.

When we simulate the real test with fourteen hundred and fifty per group, the text message group's no-show rate comes out around fifteen point seven percent, versus about twenty point one percent for the control group. That's a drop of about four point three points, with a ninety five percent confidence interval from about one point six to seven point one points, and a p-value of about point oh oh two. That's a result you can take to leadership, with the caveat that our data is simulated.

Now the trap. Peeking. Imagine a manager who checks the dashboard every day and stops the test the moment the p-value dips under point oh five. It sounds harmless. But the p-value wanders around as data comes in, and if you look enough times, it will dip under the line just by chance. In today's simulation there's no real effect at all, both groups have the same twenty percent rate, and we peek fifteen times. The false alarm rate jumps from five percent to about twenty four percent. Nearly one in four tests would declare a winner that doesn't exist. The fix is simple. Decide your sample size up front and look once, or use a sequential design that's built for multiple looks.

For your hands-on today, write a short test plan, calculate the sample size both ways, simulate the test and report the difference with its confidence interval, check power by simulation, run the peeking experiment, plot some power curves, and write a one paragraph decision memo. Then commit and push to GitHub.

Tomorrow, we switch from numbers to pictures. We'll do exploratory data analysis with matplotlib and seaborn on a patient-level dataset, and learn which chart answers which question. See you on Day Five.
