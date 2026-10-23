# Day 9: Comparing many groups: ANOVA, Kruskal-Wallis, post-hoc tests and the multiple-testing trap

## Concept
Today you compare length of stay (LOS) across four hospital sites at once, on 2,000 simulated inpatient stays (seed 9; mean 4.61 days, median 3.8). Running all six pairwise t-tests at 0.05 raises the chance of at least one false alarm to about 26% (64% for 20 tests). Instead, ask one question first with a one-way ANOVA: the F statistic compares the spread between site means with the spread within sites. F = 20.90, p = 2e-13, so the site means are not all equal. But eta squared is only 0.030: site explains about 3% of LOS variation. Check the assumptions: Levene's test p = 7e-15 (Downtown SD 4.09 vs. about 2.4 elsewhere) and residual skew 2.87, so confirm with Welch's ANOVA (F = 17.60), the rank-based Kruskal-Wallis test (H = 33.13, p = 3e-07) and ANOVA on log LOS (F = 12.55). All agree. ANOVA only says "not all equal", so run a post-hoc test: Tukey HSD on log LOS and pairwise Mann-Whitney with Holm-adjusted p-values both show Downtown is longer than each suburban site (mean 5.30 vs. 4.03 to 4.29 days), while Carmel, Fishers and Greenwood do not differ. The multiple-testing trap: 20 pure-noise metrics give 3 "significant" results at 0.05; after Bonferroni, Holm or Benjamini-Hochberg (FDR) adjustment, zero remain. A two-way ANOVA (site x admit type) shows emergency stays are longer at every site with no interaction (p = 0.55).

## Key numbers
- **Stays / LOS:** 2,000 / mean 4.61 days, median 3.8
- **Mean LOS by site:** Downtown 5.30, Greenwood 4.29, Fishers 4.19, Carmel 4.03
- **One-way ANOVA:** F = 20.90, p = 2e-13
- **Eta squared:** 0.030 (site explains about 3% of variation)
- **Assumptions:** Levene p = 7e-15; residual skew 2.87
- **Robust checks:** Welch F = 17.60; Kruskal-Wallis H = 33.13, p = 3e-07; log-LOS ANOVA F = 12.55
- **Post-hoc (Tukey, Holm):** Downtown differs from all 3 suburban sites; suburbs alike
- **False alarm risk:** 1 test 5%; 6 tests 26%; 20 tests 64%
- **20 noise metrics:** 3 raw hits at 0.05; 0 after Bonferroni, Holm or BH
- **Two-way ANOVA:** site p < 0.001, admit type p < 0.001, interaction p = 0.55

## Hands-on
1. In your repo, create day09/, copy day07/patients.csv into it, and open day09/los_anova.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate stays.csv (seed 9) and check mean LOS is 4.61. Build the per-site summary (count, mean, median, SD) and a boxplot of LOS by site on a log axis.
3. Run the one-way ANOVA with stats.f_oneway (snippet 2) and compute eta squared by hand. Write one sentence: real or not, big or small?
4. Check the assumptions with Levene and the residual skew, then run Welch's ANOVA with anova_oneway(..., use_var='unequal').
5. Run Kruskal-Wallis and an ANOVA on log LOS (snippet 3). Compare the geometric means with the arithmetic means.
6. Run pairwise_tukeyhsd on log LOS, then the six pairwise Mann-Whitney tests with Holm-adjusted p-values (snippet 4). Summarize the pattern in one sentence.
7. Simulate 20 pure-noise metrics (snippet 5). Count raw hits at 0.05, then apply Bonferroni, Holm and Benjamini-Hochberg. Explain why the hits vanish.
8. Fit the two-way ANOVA log_los ~ C(site) * C(admit_type) and read the interaction row.
9. Add a Findings markdown cell: 5 plain-language bullets a hospital operations manager could act on, plus a limits bullet (simulated, observational, site not randomized).
10. Commit and push (snippet 6).
11. Stretch: load stays.csv into Snowflake and build a LOS summary by site with MEDIAN and PERCENTILE_CONT (snippet 7).

## Code
### 1. Simulate stays and summarize LOS by site
```python
import numpy as np
import pandas as pd

# Start from the Day 7 patients and simulate one inpatient stay per patient at 4 sites (fixed seed)
pt = pd.read_csv("patients.csv").drop_duplicates().reset_index(drop=True)   # 2,000
rng = np.random.default_rng(9)
sites = ["Downtown", "Carmel", "Fishers", "Greenwood"]
pt["site"] = rng.choice(sites, size=len(pt), p=[.40, .22, .20, .18])
pt["admit_type"] = np.where(rng.random(len(pt)) < .55, "emergency", "elective")
site_eff = pt["site"].map({"Downtown": .12, "Carmel": -.05, "Fishers": 0.0, "Greenwood": .02})
log_los = (1.05 + site_eff + .25 * (pt["admit_type"] == "emergency")
           + .07 * pt["chronic_conditions"] + .004 * (pt["age"] - 50)
           + rng.normal(0, .50 + .10 * (pt["site"] == "Downtown"), len(pt)))
pt["los_days"] = np.round(np.exp(log_los), 1).clip(lower=0.5)
df = pt[["patient_id", "age", "plan", "chronic_conditions", "site", "admit_type", "los_days"]]
df.to_csv("stays.csv", index=False)
print(len(df), "stays, mean LOS", round(df["los_days"].mean(), 2), "median", df["los_days"].median())

# Group summary: n, mean, median, SD per site (LOS is right-skewed, so check both)
print(df.groupby("site")["los_days"].agg(["count", "mean", "median", "std"]).round(2))
```

### 2. One-way ANOVA, eta squared, assumptions and Welch
```python
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway

# One-way ANOVA: are the 4 site means all equal? (H0: mu1 = mu2 = mu3 = mu4)
groups = [g["los_days"].to_numpy() for _, g in df.groupby("site")]
F, p_anova = stats.f_oneway(*groups)
print(f"one-way ANOVA: F = {F:.2f}, p = {p_anova:.2g}")

# Effect size: eta squared = share of LOS variance explained by site
grand = df["los_days"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = ((df["los_days"] - grand) ** 2).sum()
print("eta squared =", round(ss_between / ss_total, 3))

# Check the assumptions: equal variances (Levene) and normal residuals (Shapiro on a sample)
print(f"Levene p = {stats.levene(*groups).pvalue:.2g}")
resid = smf.ols("los_days ~ C(site)", data=df).fit().resid
print(f"Shapiro p (residuals) = {stats.shapiro(resid.sample(500, random_state=9)).pvalue:.2g}, skew = {resid.skew():.2f}")

# Unequal variances? Welch's ANOVA does not assume them
w = anova_oneway(df["los_days"], df["site"], use_var="unequal")
print(f"Welch ANOVA: F = {w.statistic:.2f}, p = {w.pvalue:.2g}")
```

### 3. Kruskal-Wallis and ANOVA on log LOS
```python
# Skewed data: the rank-based Kruskal-Wallis test compares the distributions (often read as medians)
H, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H = {H:.2f}, p = {p_kw:.2g}")
eps2 = (H - len(groups) + 1) / (len(df) - len(groups))
print("epsilon squared (KW effect size) =", round(eps2, 3))

# Or model the log: LOS is roughly log-normal, so ANOVA on log(LOS) compares geometric means
df["log_los"] = np.log(df["los_days"])
F_log, p_log = stats.f_oneway(*[g["log_los"] for _, g in df.groupby("site")])
print(f"ANOVA on log LOS: F = {F_log:.2f}, p = {p_log:.2g}")
print(np.exp(df.groupby("site")["log_los"].mean()).round(2).rename("geometric mean LOS"))
```

### 4. Post-hoc: Tukey HSD and Holm-adjusted Mann-Whitney
```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ANOVA says "not all equal". Post-hoc: WHICH pairs differ? Tukey HSD controls the family-wise error
tk = pairwise_tukeyhsd(df["log_los"], df["site"], alpha=.05)
print(tk.summary())

# Nonparametric version: all 6 pairwise Mann-Whitney tests, then adjust the p-values with Holm
pairs, raw = [], []
for a, b in combinations(sorted(df["site"].unique()), 2):
    pairs.append(f"{a} vs {b}")
    raw.append(stats.mannwhitneyu(df.loc[df.site == a, "los_days"], df.loc[df.site == b, "los_days"]).pvalue)
reject, p_holm, _, _ = multipletests(raw, method="holm")
print(pd.DataFrame({"pair": pairs, "p_raw": raw, "p_holm": p_holm, "significant": reject}).round(4))
```

### 5. The multiple-testing trap and two-way ANOVA
```python
# The multiple-testing trap: 20 quality metrics where the sites are truly identical (pure noise)
rng2 = np.random.default_rng(12)
p_noise = []
for m in range(20):
    noise = rng2.normal(0, 1, len(df))
    p_noise.append(stats.f_oneway(*[noise[df["site"] == s] for s in sites]).pvalue)
p_noise = np.array(p_noise)
print("raw 'significant' metrics:", (p_noise < .05).sum(), "of 20   (every one is a false alarm)")
print("chance of at least one false alarm in 20 tests:", round(1 - .95 ** 20, 2))
for meth in ["bonferroni", "holm", "fdr_bh"]:
    print(f"{meth:>10}: {multipletests(p_noise, method=meth)[0].sum()} significant after adjustment")

# Two-way ANOVA: site and admit type together (does the site gap hold within admit type?)
m = smf.ols("log_los ~ C(site) * C(admit_type)", data=df).fit()

print(sm.stats.anova_lm(m, typ=2).round(4))
```

### 6. Push to GitHub
```bash
git add day09
git commit -m "Day 9: ANOVA, Kruskal-Wallis, post-hoc tests and multiple testing on length of stay"
git push
```

### Stretch: Snowflake SQL (LOS summary by site)
```sql
SELECT site,
       COUNT(*)                                              AS stays,
       ROUND(AVG(los_days), 2)                               AS mean_los,
       MEDIAN(los_days)                                      AS median_los,
       ROUND(STDDEV(los_days), 2)                            AS sd_los,
       PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY los_days) AS p90_los,
       ROUND(EXP(AVG(LN(los_days))), 2)                      AS geo_mean_los
FROM stays
GROUP BY site
ORDER BY mean_los DESC;

-- site by admit type (the two-way view)
SELECT site, admit_type, COUNT(*) AS n, ROUND(EXP(AVG(LN(los_days))), 2) AS geo_mean_los
FROM stays
GROUP BY site, admit_type
ORDER BY site, admit_type;
```

## Printables
- Cheat sheet: `day09_v1_cheat_sheet.png`
- Infographic: `day09_v2_infographic.png`
- Code worksheet: `day09_v3_worksheet.png`
- Practice quiz: `day09_v4_quiz.png`
- PDF: `day09_lesson.pdf`

## Podcast script
(Audio: `day09_podcast.mp3`)

Hey Todor, welcome to Day 9. Yesterday the outcome was a yes or no, readmitted or not. Today we go back to a number, length of stay in days, but with a twist. Instead of comparing two groups, we compare four hospital sites at once. And that opens a trap that catches a lot of analysts, so let's walk through it.

Picture a manager asking you, do our four sites have different lengths of stay? Your first instinct might be to run a t-test for every pair. Four sites means six pairs, so six tests. The problem is that each test has a five percent chance of a false alarm. Stack six of them and your chance of at least one false alarm climbs to about twenty six percent. Stack twenty and it is about sixty four percent. You would almost certainly "find" something, even if nothing is there.

The fix is to ask one question first. That is what one-way ANOVA does. ANOVA stands for analysis of variance, and the idea is simple. It compares how far apart the group averages are with how spread out the patients are inside each group. That ratio is the F statistic. If the sites differ a lot more than the normal patient-to-patient noise, F gets big and the p-value gets small. In our simulated data, two thousand stays, F is about twenty one and the p-value is tiny. So the sites are not all the same.

But hold on to two things. First, ANOVA only tells you "not all equal." It does not tell you which site is different. Second, a tiny p-value is not a big effect. The effect size here is called eta squared, the share of the variation explained by site, and it is only about three percent. Most of the reason one patient stays longer than another has to do with the patient, like an emergency admission or more chronic conditions, not the building.

Next, check the assumptions. ANOVA assumes the groups have similar spread and roughly normal noise. Length of stay breaks both. It is right-skewed, a few patients stay a very long time, and the Downtown site has a wider spread than the suburban sites. You have three good options. Welch's ANOVA drops the equal spread assumption. Kruskal-Wallis works on ranks instead of raw values, so long stays don't dominate. Or you take the log of length of stay, which tames the skew. Here all three agree, and when different methods agree, you can say the finding is solid.

Now, which sites differ? That is the job of a post-hoc test. Tukey's test compares every pair while keeping the overall false alarm rate at five percent. Another option is to run the six pairwise tests and adjust the p-values with the Holm method. Both tell the same story. Downtown stays run about a day longer than each suburban site, and Carmel, Fishers and Greenwood look alike. That is a sentence a manager can actually use.

Then comes the part I really want you to remember. In the exercise you'll create twenty quality metrics that are pure random noise, so the sites are truly identical. Three of them still come out with a p-value under point zero five. Every one of those is a false alarm. After a Bonferroni, Holm, or Benjamini-Hochberg adjustment, all three disappear. Bonferroni and Holm protect you from even one false positive, which fits when a wrong call is costly. Benjamini-Hochberg controls the share of false discoveries, which fits when you are screening lots of metrics, like a dashboard of fifty quality measures.

Finally, you'll try a two-way ANOVA with site and admission type together. Emergency stays are longer at every site, by about the same proportion, and the interaction test says the site gap does not depend on admission type. In regression terms, the lines are roughly parallel.

In healthcare job postings you'll see things like "compare outcomes across facilities" or "identify statistically significant variation." Today's habits are exactly what that means in practice: one test first, check assumptions, report the effect size, and adjust when you test many things.

Tomorrow we bring time into the picture. We'll look at daily emergency department visits, rolling averages, weekly seasonality, and how to tell a real shift from a normal bumpy week. See you then.
