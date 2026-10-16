# Day 2: Probability distributions and hypothesis tests: is the PPO cost gap real?

## Concept
Yesterday you described data; today you make decisions with it. Key ideas: three distributions you will see constantly in healthcare, normal (lab values, blood pressure), binomial (readmissions out of n discharges) and Poisson (ED arrivals per hour), and how to read their tails with scipy.stats (.sf = P(X > x)). Then the hypothesis test recipe: state H0 and H1, set alpha before looking, compute a test statistic, get the p-value (the chance of a result this extreme if H0 were true, not the chance H0 is true), and decide. You will run a Welch t-test on log claim costs and a permutation test on the median gap, see that they answer different questions, and use the Central Limit Theorem to see why t-tests work on skewed data. Always report the effect size, not just p.

## Key numbers
- **PPO vs HMO mean gap:** $64
- **Welch t (log cost):** 2.08
- **Welch p:** 0.037
- **Median gap:** $20
- **Permutation p:** 0.097

## Hands-on
1. In your repo, create day02/ and open day02/distributions_hypothesis.ipynb in VS Code; conda activate ds90 (scipy is already in the Day 1 env).
2. Compute the three tail probabilities with scipy.stats (snippet 1) and write one sentence explaining each in plain English.
3. Plot each distribution and shade the tail you computed.
4. Simulate 5,000 claims with seed 42 and PPO costs 10% higher; add log_amt = np.log1p(amount) (snippet 2).
5. In a markdown cell, write H0, H1 and alpha = 0.05 before running any test.
6. Run Welch's t-test on log_amt (snippet 2). Expect t = 2.08, p = 0.037.
7. Run a permutation test on the median gap with 5,000 shuffles and plot the null histogram with the observed gap (snippet 3). Expect p near 0.097.
8. Show the Central Limit Theorem: 2,000 sample means of 50 claims next to the raw histogram (snippet 4).
9. Write 3 stakeholder-style insights, each with an effect size, a p-value and a caveat (simulated data, which question was tested).
10. Commit and push (snippet 5).
11. Stretch: in Snowflake, compute COUNT, AVG and STDDEV of LN(1 + amount) by plan and compute the Welch t statistic by hand (snippet 6).

## Code
### 1. Normal, binomial and Poisson tails
```python
from scipy import stats

# Normal: systolic blood pressure ~ N(125, 15)
p_sbp = stats.norm.sf(140, loc=125, scale=15)
# Binomial: 30-day readmissions out of 200 discharges, p = 0.15
p_readm = stats.binom.sf(39, n=200, p=0.15)      # P(X >= 40)
# Poisson: ED arrivals per hour, average 12
p_ed = stats.poisson.sf(19, mu=12)               # P(X >= 20)
print(f"P(SBP >= 140)       = {p_sbp:.3f}")
print(f"P(readmits >= 40)   = {p_readm:.3f}")
print(f"P(ED arrivals >= 20) = {p_ed:.3f}")
```

### 2. Simulate claims & Welch t-test
```python
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
n = 5000
plan = rng.choice(["HMO", "PPO"], n)
amount = rng.lognormal(mean=6, sigma=1, size=n) * np.where(plan == "PPO", 1.10, 1.0)
df = pd.DataFrame({"plan_type": plan, "amount": amount})
df["log_amt"] = np.log1p(df["amount"])
print(df.groupby("plan_type")["amount"].agg(["count", "mean", "median"]))

# H0: mean log cost is the same for HMO and PPO. alpha = 0.05
hmo = df.loc[df.plan_type == "HMO", "log_amt"]
ppo = df.loc[df.plan_type == "PPO", "log_amt"]
t, p = stats.ttest_ind(ppo, hmo, equal_var=False)   # Welch's t-test
print(f"Welch t = {t:.2f}, p = {p:.3f}")
```

### 3. Permutation test on the median gap
```python
obs = (df.loc[df.plan_type == "PPO", "amount"].median()
       - df.loc[df.plan_type == "HMO", "amount"].median())
labels = df["plan_type"].to_numpy()
amts = df["amount"].to_numpy()
diffs = []
for _ in range(5000):
    s = rng.permutation(labels)              # break any real link
    diffs.append(np.median(amts[s == "PPO"]) - np.median(amts[s == "HMO"]))
diffs = np.array(diffs)
p_perm = (np.abs(diffs) >= abs(obs)).mean()  # two-sided
print(f"Observed median gap = ${obs:.0f}, permutation p = {p_perm:.3f}")

import matplotlib.pyplot as plt
plt.hist(diffs, bins=50, color="lightgray")
plt.axvline(obs, color="red"); plt.axvline(-obs, color="red", ls="--")
plt.title("Null distribution of the median gap"); plt.show()
```

### 4. Central Limit Theorem
```python
import matplotlib.pyplot as plt

means = [rng.choice(amts, 50).mean() for _ in range(2000)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(amts[amts < 6000], bins=60); ax[0].set_title("Claims: right-skewed")
ax[1].hist(means, bins=40); ax[1].set_title("Means of samples of 50: ~normal")
plt.show()
print(f"mean of sample means = {np.mean(means):.0f}, SE = {np.std(means):.0f}")
```

### 5. Push to GitHub
```bash
git add day02
git commit -m "Day 2: distributions and hypothesis tests"
git push
```

### Stretch: Snowflake SQL (Welch t by hand)
```sql
WITH g AS (
  SELECT plan_type,
         COUNT(*)              AS n,
         AVG(LN(1 + amount))   AS m,
         STDDEV(LN(1 + amount)) AS s
  FROM claims
  GROUP BY plan_type
)
SELECT (p.m - h.m) / SQRT(p.s*p.s/p.n + h.s*h.s/h.n) AS welch_t
FROM g p JOIN g h
  ON p.plan_type = 'PPO' AND h.plan_type = 'HMO';
```

## Printables
- Cheat sheet: `day02_v1_cheat_sheet.png`
- Infographic: `day02_v2_infographic.png`
- Code worksheet: `day02_v3_worksheet.png`
- Practice quiz: `day02_v4_quiz.png`
- PDF: `day02_lesson.pdf`

## Podcast script
(Audio: `day02_podcast.mp3`)

Welcome back to the Data Science Coach podcast. This is Day Two. Yesterday we learned to describe data honestly, with medians, spread, and outliers. Today we take the next step, from describing data to making decisions with it. The big question for today is simple to ask and surprisingly tricky to answer: is a difference real, or is it just noise?

Let's start with probability distributions. A distribution is just a description of which values are likely and which are rare. You'll meet three of them constantly in healthcare data. The first is the normal distribution, the classic bell curve. Think of systolic blood pressure in a population, centered around one hundred twenty-five with a spread of about fifteen. If you ask, what share of people sit at one hundred forty or higher, the normal curve answers about sixteen percent.

The second is the binomial distribution. It counts yes or no events out of a fixed number of tries. Picture two hundred hospital discharges in a month, each with a fifteen percent chance of a thirty-day readmission. On average you'd expect thirty readmissions. Seeing forty or more would happen only about three percent of the time. So if you see forty, that's worth a closer look.

The third is the Poisson distribution, which counts events in a window of time. Emergency department arrivals per hour are a good example. If a department averages twelve arrivals an hour, a rush of twenty or more happens only about two percent of the time. In Python, the scipy stats module gives you all three, and a function called the survival function hands you the chance of landing above a value. That's the tail, and tails are where the interesting questions live.

Now, the main event. Hypothesis testing. Imagine a manager says PPO members cost more than HMO members. In our simulated data of five thousand claims, the average PPO claim is about seven hundred forty-three dollars and the average HMO claim is about six hundred seventy-nine. That's a gap of about sixty-four dollars. But claims are noisy. Could a gap like that show up just by chance?

Here's the recipe. First, write down the null hypothesis, the boring explanation. Here it's that plan type makes no difference to cost. Second, choose your threshold, called alpha, before you look at the results. Point zero five is the usual choice. Third, compute a test statistic. Fourth, get the p-value. And fifth, decide.

The p-value is the most misunderstood number in data science, so let's say it carefully. It's the probability of seeing a result at least this extreme if the null hypothesis were true. It's not the chance the null is true. And it says nothing about how big or important the effect is.

For our test, we use Welch's t-test on the log of the claim amounts. We take the log because, as you saw yesterday, claims are right-skewed. And we use Welch's version because it doesn't assume the two groups have the same spread. The result is a t statistic of about two point oh eight and a p-value of about point oh three seven. That's below point oh five, so we reject the null. The log-scale gap is unlikely to be pure noise.

But here's a twist I built in on purpose. If we instead ask about the median gap, which is only about twenty dollars, and we test it with a permutation test, the p-value comes out near point one. Not significant. A permutation test is beautifully simple. You shuffle the plan labels thousands of times, which creates a world where plan type truly doesn't matter, and you count how often the shuffled gap is as big as the real one. Two tests, two different questions, two different answers. That's why you always say which question you tested, and you always report the size of the effect, not just the p-value.

One more idea ties it all together, the central limit theorem. Even though individual claims are wildly skewed, if you take many samples of fifty claims and average each one, those averages form a near-normal bell shape. That's the reason t-tests work on messy real-world data when samples are reasonably large.

For your hands-on today, compute the three tail probabilities, simulate the claims, write your hypotheses before testing, run the Welch t-test and the permutation test, plot the central limit theorem, and write three plain-English insights that include the effect size and a caveat. Then commit and push.

Tomorrow, we'll go from yes or no answers to ranges. We'll build confidence intervals and learn the bootstrap, a way to measure uncertainty by resampling your own data. See you on Day Three.
