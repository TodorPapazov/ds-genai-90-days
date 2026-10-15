# Day 1: Descriptive statistics on right-skewed healthcare claims

## Concept
Analysts report numbers; data scientists ask how those numbers are distributed and how sure we can be about them. Key ideas: mean vs. median (skew pulls the mean toward the tail), standard deviation, percentiles and IQR (Q3 − Q1), outliers via the 1.5×IQR rule, and the log1p transform to tame right skew. Healthcare claim amounts are heavily right-skewed, so the median and log transforms matter a lot.

## Key numbers
- **Mean:** $653
- **Median:** $402
- **Std dev:** $845
- **IQR:** $556
- **Outliers:** 408 over $1,593

## Hands-on
1. Create a GitHub repo ds-genai-90-days with folders day01/, data/ and a README.md.
2. Set up the conda env (snippet 1).
3. In day01/descriptive_stats.ipynb, simulate 5,000 claims with np.random.lognormal(mean=6, sigma=1, size=5000), plus member_id and plan_type (HMO/PPO).
4. Compute mean, median, std, skew and percentiles overall and by plan type with groupby().describe().
5. Flag outliers with the 1.5×IQR rule.
6. Plot histograms of raw vs. np.log1p amounts.
7. Write 3 stakeholder-style insights in a markdown cell.
8. Commit and push to GitHub.
9. Stretch: reproduce the stats in Snowflake with MEDIAN and PERCENTILE_CONT.

## Code
### 1. Environment
```bash
conda create -n ds90 python=3.11 pandas numpy matplotlib seaborn scipy jupyter -y
conda activate ds90
```

### 2. Simulate claims & describe
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000
df = pd.DataFrame({
    "member_id": rng.integers(1000, 2000, n),
    "plan_type": rng.choice(["HMO", "PPO"], n),
    "amount": np.random.lognormal(mean=6, sigma=1, size=n),
})
print(df["amount"].agg(["mean", "median", "std", "skew"]))
print(df.groupby("plan_type")["amount"].describe())
```

### 3. IQR outliers & log1p histograms
```python
import matplotlib.pyplot as plt

q1, q3 = df["amount"].quantile([0.25, 0.75])
iqr = q3 - q1
cutoff = q3 + 1.5 * iqr
df["is_outlier"] = df["amount"] > cutoff
print(f"IQR={iqr:.0f}, cutoff={cutoff:.0f}, outliers={df.is_outlier.sum()}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df["amount"], bins=60); ax[0].set_title("Raw amounts")
ax[1].hist(np.log1p(df["amount"]), bins=60); ax[1].set_title("log1p(amount)")
plt.show()
```

### 4. Push to GitHub
```bash
git add .
git commit -m "Day 1: descriptive stats"
git push
```

### Stretch: Snowflake SQL
```sql
SELECT plan_type,
       AVG(amount) AS mean_amt,
       MEDIAN(amount) AS median_amt,
       STDDEV(amount) AS std_amt,
       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3
FROM claims
GROUP BY plan_type;
```

## Printables
- Cheat sheet: `day01_v1_cheat_sheet.png`
- Infographic: `day01_v2_infographic.png`
- Code worksheet: `day01_v3_worksheet.png`
- Practice quiz: `day01_v4_quiz.png`
- PDF: `day01_lesson.pdf`

## Podcast script
(Audio: `day01_podcast.mp3`)

Welcome to Day One of the Data Science Coach podcast, where you go from data analyst to data scientist with Gen AI skills, one day at a time. I'm glad you're here. Today's topic is descriptive statistics, and more specifically, why the average can quietly lie to you.

Let's start with a picture. Imagine a table of healthcare claims. Each row is one claim: a member ID, a date, a plan type, and a dollar amount. Most of those claims are small. A few hundred dollars for an office visit, a lab panel, a prescription. But every so often there's a huge one. A surgery. An inpatient stay. An emergency that costs tens of thousands of dollars. When you plot those amounts as a histogram, most of the bars pile up on the left, and a long, thin tail stretches far off to the right. That shape has a name. It's called right skew, and in healthcare data it's the rule, not the exception.

Here's why that matters. The mean, the plain average, gets pulled toward the tail. Every giant claim drags it upward. In our simulated practice data of five thousand claims, the mean comes out to about six hundred fifty-three dollars. But the median, the value sitting right in the middle when you line every claim up from smallest to largest, is only about four hundred two dollars. That's a gap of roughly two hundred fifty dollars. The median doesn't care how extreme the biggest claims are. It only cares about the middle. So if you tell a stakeholder that the typical claim costs six fifty, you're overstating what a typical member actually experiences. With skewed data, the median tells the more honest story. A good habit is to report both, and to explain why they differ.

Next, let's talk about spread. The standard deviation measures how far values typically sit from the mean. In our data it's around eight hundred forty-five dollars, which is larger than the mean itself. That's a red flag that something is skewed. Because the standard deviation squares every distance, a handful of extreme claims inflate it a lot. So we want a sturdier measure.

That measure is the interquartile range, or I Q R. Find the twenty-fifth percentile, called the first quartile, and the seventy-fifth percentile, called the third quartile. Subtract the first from the third. The result is the range that covers the middle half of your data, and it ignores the tails completely. In our practice data, the I Q R is about five hundred fifty-six dollars. In pandas, you get the quartiles with the quantile method, or all at once with describe.

The I Q R also gives us a classic rule for flagging outliers. Take one and a half times the I Q R and add it to the third quartile. Anything above that line gets flagged. In our data, that cutoff lands at roughly sixteen hundred dollars, and it flags about four hundred claims, around eight percent of the total. Now here's the important part. Outliers are not automatically mistakes. Sometimes they're data entry errors, sure. But in healthcare, those flagged claims are often the most important rows in the whole table. They're the high-cost members, the people a care management team most wants to find and support. So don't delete outliers by reflex. Investigate them. Ask what story they tell.

The last tool for today is the log transform. When you take the logarithm of skewed amounts, the long right tail gets squeezed and the bulk of the data spreads out. In numpy, we use the function log one p, which computes the log of one plus x, so a zero dollar claim doesn't break anything. After the transform, our claims histogram starts to look like a symmetric bell curve. That matters a lot later in this program, because many statistical tests and machine learning models behave better when the data is closer to normal. You'll see this trick again and again, especially with costs, incomes, and counts.

So here's the mindset shift for today. An analyst reports the average. A data scientist first asks, what does the distribution look like? Is it symmetric or skewed? Are there outliers, and what are they? Only then do they choose the right summary. That one habit will make your analyses more trustworthy and your stakeholder conversations a lot sharper.

Now let's recap your hands-on. First, create your GitHub repo and your conda environment. Then, in your Day One notebook, simulate five thousand claims using numpy's lognormal function, and add a member ID and a plan type, H M O or P P O. Compute the mean, median, standard deviation, skew, and percentiles, overall and by plan type, using group by and describe. Flag outliers with the one and a half I Q R rule. Plot a histogram of the raw amounts next to the log transformed amounts. Write three plain-English insights, the way you'd explain them to a manager. Then commit, push, and share your post. Learning in public keeps you accountable, and it builds your portfolio at the same time. For a stretch goal, run the same statistics in Snowflake with median and percentile cont, and compare them to pandas.

Tomorrow, we move from describing data to making decisions with it. We'll cover probability distributions and the foundations of hypothesis testing: what a p-value really means, and how to tell a real difference from random noise. Great work today. See you on Day Two.

## X / LinkedIn post

Day 1 of my 90-day journey from Data Analyst to Data Scientist with Gen AI 🚀

Today: descriptive statistics on skewed healthcare claims data. When data is right-skewed, the mean misleads; the median and a log transform tell the real story.

Code + notebook 👉 https://github.com/TodorPapazov/ds-genai-90-days

#DataScience #Python #GenAI #LearningInPublic
