# Day 10: Time series EDA: rolling means, seasonality, and telling a real shift from a bumpy week

## Concept
Today you explore 731 days of simulated emergency department (ED) visits at one hospital (Jan 2024 to Dec 2025, seed 10; mean 189.2 visits a day). First the plumbing: parse dates into a DatetimeIndex and call asfreq("D") to expose the 9 days missing from the extract, then interpolate them, because a missing day silently shortens every rolling window. A 7-day rolling mean cancels the weekday cycle (each window holds one of every weekday) and reveals the slower story: a winter flu peak near 217 visits a day, a summer low near 161, and slow growth. Seasonality: Monday runs 12% above an average day and weekends about 7% below; autocorrelation at lag 7 (0.70) beats lag 1 (0.51), so last Monday predicts next Monday better than yesterday does. STL splits the series into trend, weekly seasonal and remainder. The real question: a new urgent care opened on Sep 1, 2025. Comparing fall with spring mixes in flu season, so compare year over year: each week divided by the same week last year. That ratio ran about 1.07 from January to August; after Sep 1 all 17 weeks fell below the center line and 9 fell below the lower control limit (0.94). One low week in March stayed inside the limits: a bumpy week, not a shift. A run rule (8+ weeks on one side of the center) fires by late October. Sep-Dec 2025 ran 5.1% below 2024 (Welch p = 0.001) after running about 7% above, a swing of roughly 12 points. The timing fits the urgent care, but this is observational. Bridge to ML: build lag features with shift(), split by time (never shuffle), and set baselines to beat. On Nov-Dec 2025, "same day last week" has an MAE of 14.9 visits vs. 18.2 for the training mean.

## Key numbers
- **Days / visits:** 731 days (9 missing, interpolated) / mean 189.2 a day, SD 25.0
- **Flu season:** busiest month Dec 2024, about 217 a day; summer low about 161
- **Weekday index:** Monday 1.12; Saturday 0.93, Sunday 0.92
- **Autocorrelation:** lag 1 = 0.51, lag 7 = 0.70, lag 14 = 0.69
- **STL:** weekly component holds about 22% of daily variance
- **YoY (monthly):** Jan-Aug 2025 +3% to +9%; Sep-Dec 2025 -4% to -7%
- **Control chart:** baseline ratio 1.07, limits 0.94 to 1.21; 9 of 17 weeks below LCL after Sep 1
- **Run rule:** 17 weeks in a row below center; 8th week (flagged) Oct 26, 2025
- **Sep-Dec 2024 vs 2025:** 197.4 vs 187.3 a day, -5.1%, Welch p = 0.001
- **Baseline MAE (Nov-Dec 2025):** training mean 18.2; same day last week 14.9; trailing 7-day mean 16.7

## Hands-on
1. In your repo, create day10/ and open day10/ed_time_series.ipynb in VS Code with conda activate ds90.
2. Run snippet 1 to simulate ed_visits.csv (seed 10). Reload it with parse_dates, set the date index, and confirm asfreq("D") finds 9 missing days. Interpolate them.
3. Plot daily visits with a 7-day and a 28-day centered rolling mean (snippet 2). Mark Sep 1, 2025 with a vertical line.
4. Resample to weekly totals (W-SUN) and monthly means. Build the weekday index and print autocorrelation at lags 1, 7 and 14. Write one sentence on what lag 7 tells you.
5. Run STL with period=7 (snippet 3) and plot the components. Compute the monthly year-over-year % change and note the month it turns negative.
6. Build the weekly year-over-year ratio and control limits from the Jan-Aug 2025 baseline (snippet 4). Count weeks below the LCL after Sep 1 and find the longest run below center.
7. Compare Sep-Dec 2025 with Sep-Dec 2024 (daily means and a Welch t-test). Write down why this beats comparing fall 2025 with spring 2025.
8. Create lag and rolling features with shift(), split at Nov 1, 2025, and compute the MAE of the three baselines (snippet 5).
9. Add a Findings markdown cell: 5 plain-language bullets an ED operations manager could act on, plus a limits bullet (simulated, observational, other possible causes).
10. Commit and push (snippet 6).
11. Stretch: load ed_visits.csv into Snowflake and compute a 7-day moving average, weekday index and YoY weekly ratio with window functions (snippet 7).

## Code
### 1. Simulate ED visits, load with a date index, find the gaps
```python
import numpy as np
import pandas as pd

# Simulate 2 years of daily emergency department (ED) visits at one hospital (fixed seed)
rng = np.random.default_rng(10)
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
t = np.arange(len(dates))
dow_eff = np.array([1.12, 1.04, 1.00, 0.99, 1.00, 0.93, 0.92])[dates.dayofweek]   # Mon busiest, weekend quieter
flu = 1 + 0.14 * np.cos(2 * np.pi * (dates.dayofyear - 20) / 365.25)            # winter peak, late January
trend = 1 + 0.0002 * t                                                         # slow growth, about 7% a year
shift = np.where(dates >= "2025-09-01", 0.90, 1.0)                              # new urgent care opens Sep 1, 2025
visits = rng.poisson(180 * dow_eff * flu * trend * shift)
ed = pd.DataFrame({"visit_date": dates, "visits": visits})
ed = ed.drop(rng.choice(len(ed), 9, replace=False))      # 9 days missing from the extract
ed.to_csv("ed_visits.csv", index=False)

# Load it the right way: parse dates, set a DatetimeIndex, then check for gaps
ed = pd.read_csv("ed_visits.csv", parse_dates=["visit_date"]).set_index("visit_date").sort_index()
full = ed.asfreq("D")                                    # re-inserts the missing days as NaN
print(len(ed), "rows,", full["visits"].isna().sum(), "missing days")
full["visits"] = full["visits"].interpolate()            # fill gaps from the neighbours
print(full["visits"].describe().round(1))
```

### 2. Rolling means, resampling, weekday profile, autocorrelation
```python
# Smooth the daily noise: 7-day and 28-day rolling means (a 7-day window cancels the weekday cycle)
full["ma7"] = full["visits"].rolling(7).mean()
full["ma28"] = full["visits"].rolling(28, center=True).mean()

# Resample to coarser grains: weekly totals and monthly averages
weekly = full["visits"].resample("W-SUN").sum().iloc[1:-1]          # drop partial first/last weeks
monthly = full["visits"].resample("MS").mean()
print(monthly.round(1).head(12))

# Seasonal profiles: day of week and month of year
dow = full.groupby(full.index.day_name())["visits"].mean()
dow = dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
print((dow / dow.mean()).round(3).rename("weekday index"))

# Autocorrelation: today vs 1 day ago and 7 days ago (weekly rhythm shows up at lag 7)
for lag in [1, 7, 14]:
    print(f"lag {lag:>2} autocorrelation = {full['visits'].autocorr(lag):.2f}")
```

### 3. STL decomposition and year-over-year change
```python
from statsmodels.tsa.seasonal import STL

# STL splits the series into trend + weekly seasonal + remainder (robust to odd days)
stl = STL(full["visits"], period=7, seasonal=31, robust=True).fit()
comp = pd.DataFrame({"trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid})
print(comp.describe().round(1))
print("share of daily variance in the weekly pattern:",
      round(comp["seasonal"].var() / full["visits"].var(), 2))

# Year over year: compare the same month across years (removes seasonality without a model)
yoy = monthly.pct_change(12) * 100
print(yoy.dropna().round(1).rename("YoY % change"))
```

### 4. Real shift or bumpy week: YoY control chart and run rule
```python
from scipy import stats

# Real shift or a bumpy week? Compare each week with the same week last year (52 weeks back),
# so the flu season and weekday mix cancel out, then put control limits on that ratio
ratio = (weekly / weekly.shift(52)).dropna()
base = ratio[:"2025-08-31"]                              # baseline: Jan to Aug 2025
center, sd = base.mean(), base.std()
ucl, lcl = center + 3 * sd, center - 3 * sd
print(f"baseline YoY ratio {center:.3f}, control limits {lcl:.3f} to {ucl:.3f}")
after = ratio["2025-09-01":]
print("weeks below the lower limit after Sep 1:", (after < lcl).sum(), "of", len(after))
print("weeks below the center line after Sep 1:", (after < center).sum(), "of", len(after))
print("baseline weeks outside the limits:", ((base > ucl) | (base < lcl)).sum(), "of", len(base))

# Run rule: 8 or more weeks in a row on one side of the center line signals a real shift
below = (ratio < center).astype(int)
run = below.groupby((below != below.shift()).cumsum()).cumsum() * below
print("longest run below center:", run.max(), "weeks, first flagged week:", run[run >= 8].index.min().date())

# Same months, year over year (Sep-Dec 2025 vs Sep-Dec 2024)
a = full.loc["2025-09-01":"2025-12-31", "visits"]
b = full.loc["2024-09-01":"2024-12-31", "visits"]
print(f"Sep-Dec daily mean: 2024 {b.mean():.1f}, 2025 {a.mean():.1f}, change {100 * (a.mean() / b.mean() - 1):.1f}%")
print(f"Welch t-test p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.2g}")
```

### 5. Bridge to ML: lag features, time split, baselines to beat
```python
# Bridge to ML: lag features and a time-based split (never shuffle time series)
feat = full[["visits"]].copy()
feat["lag_1"] = feat["visits"].shift(1)
feat["lag_7"] = feat["visits"].shift(7)
feat["ma7_prev"] = feat["visits"].shift(1).rolling(7).mean()   # shift FIRST, so today never leaks in
train, test = feat[:"2025-10-31"].dropna(), feat["2025-11-01":]

# Two baselines any model must beat: same day last week, and the trailing 7-day mean
mae = lambda pred: (test["visits"] - pred).abs().mean()
print(f"MAE, training mean      : {mae(train['visits'].mean()):.1f} visits")
print(f"MAE, same day last week : {mae(test['lag_7']):.1f} visits")
print(f"MAE, trailing 7-day mean: {mae(test['ma7_prev']):.1f} visits")
```

### 6. Push to GitHub
```bash
git add day10
git commit -m "Day 10: time series EDA on daily ED visits (rolling means, seasonality, YoY control chart)"
git push
```

### Stretch: Snowflake SQL (moving average, weekday index, YoY ratio)
```sql
-- 7-day moving average and weekday index
SELECT visit_date,
       visits,
       DAYNAME(visit_date)                                                         AS weekday,
       ROUND(AVG(visits) OVER (ORDER BY visit_date
                               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1)       AS ma7,
       ROUND(AVG(visits) OVER (PARTITION BY DAYOFWEEK(visit_date))
             / AVG(visits) OVER (), 3)                                             AS weekday_index
FROM ed_visits
ORDER BY visit_date;

-- weekly totals vs. the same week last year
WITH wk AS (
  SELECT DATE_TRUNC('week', visit_date) AS week_start, SUM(visits) AS visits
  FROM ed_visits
  GROUP BY 1
)
SELECT week_start,
       visits,
       LAG(visits, 52) OVER (ORDER BY week_start)                      AS visits_last_year,
       ROUND(visits / NULLIF(LAG(visits, 52) OVER (ORDER BY week_start), 0), 3) AS yoy_ratio
FROM wk
ORDER BY week_start;
```

## Printables
- Cheat sheet: `day10_v1_cheat_sheet.png`
- Infographic: `day10_v2_infographic.png`
- Code worksheet: `day10_v3_worksheet.png`
- Practice quiz: `day10_v4_quiz.png`
- PDF: `day10_lesson.pdf`

## Podcast script
(Audio: `day10_podcast.mp3`)

Hey Todor, welcome to Day 10. Yesterday we compared four hospital sites at once. Today we add the one thing almost every healthcare dashboard has, and that most stats courses skip: time. We're looking at two years of daily emergency department visits at one hospital, and the question is one you'll hear in real meetings. Visits look down lately. Is that real, or just a bumpy few weeks?

Start with the boring part, because it bites people. Load the dates as real dates, make them the index, and check for missing days. Our extract is missing nine days. On a chart, a missing day doesn't show up as a blank. It just disappears, and the line jumps straight to the next day. Worse, a seven-day rolling window that skips a day is now covering eight days. So you ask pandas to re-insert every calendar day, see the gaps as empty values, and fill them from the neighbors before you do anything else.

Next, smoothing. Daily counts are noisy. Some days you get 160 visits, some days 230. A seven-day rolling mean averages each day with the six before it, and that has a neat side effect. Every seven-day window contains exactly one Monday, one Tuesday, and so on, so the weekday pattern cancels out. What's left is the slower story: a flu-season peak every winter of about 217 visits a day, a summer low around 161, and slow growth year over year.

Then look at seasonality directly. Group by day of week and you'll see Monday runs about twelve percent above an average day, and weekends about seven percent below. Autocorrelation confirms it. Today's count looks more like the same day last week than like yesterday. The correlation at a lag of seven days is about point seven, while a lag of one day is about point five. In plain words, if you want to guess next Monday, last Monday is a great place to start.

Now the main question. In September 2025, a new urgent care opened nearby. Did ED volume really drop? Here's the trap. If you compare fall with the spring before it, the flu season gets mixed into your answer. The cleaner move is to compare each week with the same week last year. Season and weekday mix cancel, and you're left with a ratio. From January to August that ratio sat around one point zero seven, meaning about seven percent growth. After September, every single week came in below that line, and nine of seventeen weeks fell below the lower control limit.

That's the difference between a bumpy week and a real shift. Back in March there was one week that dipped well below normal, but it stayed inside the limits, and the next week bounced back. That's ordinary noise. A shift looks different. It's a run. The common rule of thumb is eight or more points in a row on one side of the center line. We got seventeen. Sep through December came in about five percent below last year, when the year had been running seven percent above. That's a swing of roughly twelve points.

One honest caution. The timing fits the urgent care opening, but this is observational data. Before you tell leadership the urgent care caused it, check for other explanations, like a coding change, a closure, or simply a mild flu season.

Last piece, and it's your bridge into machine learning next week. When you build features for a forecasting model, like yesterday's count or last week's count, always split by time. Train on the past, test on the future, never shuffle. And build rolling features from shifted data, so today's number never leaks into today's prediction. You'll also set two simple baselines. Guessing the same day last week misses by about fifteen visits a day, and the training average misses by about eighteen. Any fancy model you build later has to beat those, or it isn't worth shipping.

In healthcare job postings, this shows up as "trend analysis," "volume forecasting," or "monitor operational metrics." Today's habits are exactly that: check the dates, compare like with like, and tell a shift from a bumpy week.

Tomorrow we roll up our sleeves on data cleaning. We'll take a messy claims extract with duplicates, missing values, and impossible numbers, and turn it into something you'd actually trust. See you then.
