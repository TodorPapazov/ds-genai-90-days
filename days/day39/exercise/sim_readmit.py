"""Day 30 readmission generator as a function, with drift knobs (all data simulated for teaching).
simulate(1000, 30) with the defaults rebuilds the Day 30 table exactly."""
import numpy as np
import pandas as pd


def simulate(n_pat, seed, start="2024-01-01", days=546, pid0=0,
             age_mean=64.0, creat_report=1.0, care_effect=0.0, clip=False):
    """age_mean: population age (data drift). creat_report: the lab reports creatinine x this factor
    (a new assay; outcomes still follow the TRUE value).
    clip: keep only stays inside the window (start + days). care_effect: change in log-odds of readmission
    for Home health / SNF discharges (a transitional care program: concept drift, P(y | x) changes)."""
    if seed != 30 and not hasattr(simulate, "day30_eff"):
        simulate(1000, 30)                           # learn the Day 30 DRG list and effects once
    rng = np.random.default_rng(seed)
    visits = np.clip(1 + rng.poisson(1.6, n_pat), 1, 6)
    pid = np.repeat(np.arange(n_pat), visits) + pid0
    n = len(pid)
    frailty = np.repeat(rng.normal(0, 1, n_pat), visits)                 # hidden, never a feature
    age = np.repeat(rng.normal(age_mean, 15, n_pat).clip(18, 99).round(), visits)
    first = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n_pat), unit="D")
    gap = rng.exponential(110 * np.exp(-0.5 * frailty)) + 3
    order = pd.Series(pid).groupby(pid).cumcount().values
    offset = pd.Series(np.where(order == 0, 0, gap)).groupby(pid).cumsum().values
    admit_date = np.repeat(first.values, visits) + pd.to_timedelta(offset.round(), unit="D").values
    top = ["291", "871", "470", "190", "392", "683", "690", "193", "603", "378"]
    tail = [f"{c:03d}" for c in rng.choice(np.setdiff1d(np.arange(1, 999), np.array(top, int)), 90, replace=False)]
    if seed != 30:                                   # keep the Day 30 DRG list and effects for every batch
        tail = simulate.day30_tail
    drg_codes = np.array(top + tail)
    p = 1 / np.arange(1, len(drg_codes) + 1) ** 1.1; p /= p.sum()
    drg = rng.choice(drg_codes, n, p=p)
    eff = rng.normal(0, 0.35, len(drg_codes))
    if seed != 30:
        eff = simulate.day30_eff
    else:
        simulate.day30_tail, simulate.day30_eff = tail, eff
    drg_eff = dict(zip(drg_codes, eff))
    drg_eff.update({"291": 0.7, "871": 0.6, "470": -1.0, "190": 0.6, "392": -0.3, "683": 0.4})
    chap_of = {"291": "I", "871": "A", "470": "M", "190": "J", "392": "K", "683": "N", "690": "N", "193": "J", "603": "L", "378": "K"}
    chapters = np.array(list("IJKNAEMFL"))
    chap = np.array([chap_of.get(d, chapters[int(d) % 9]) for d in drg])
    icd10 = np.array([f"{c}{rng.integers(10, 99)}.{rng.integers(0, 9)}" for c in chap])
    payer = np.where(age >= 65, rng.choice(["Medicare", "Commercial", "Medicaid"], n, p=[.85, .1, .05]),
                     rng.choice(["Commercial", "Medicaid", "Self-pay"], n, p=[.6, .3, .1]))
    admit_type = np.where(drg == "470", "Elective", rng.choice(["Emergency", "Urgent", "Elective"], n, p=[.62, .2, .18]))
    discharge = rng.choice(["Home", "Home health", "SNF", "AMA"], n, p=[.6, .18, .17, .05])
    los_days = np.round(np.exp(1.2 + 0.25 * frailty + rng.normal(0, 0.5, n)), 1)
    creatinine = np.round(np.exp(0.05 + 0.25 * frailty + rng.normal(0, 0.4, n)), 2)
    hemoglobin = np.round(12.6 - 0.8 * frailty + rng.normal(0, 1.4, n), 1)
    sodium = np.round(139 + rng.normal(0, 3.5, n))
    creatinine[rng.random(n) < np.where(admit_type == "Elective", 0.25, 0.05)] = np.nan
    hemoglobin[rng.random(n) < 0.10] = np.nan
    sodium[rng.random(n) < 0.06] = np.nan
    df = pd.DataFrame({"patient_id": pid, "admit_date": admit_date, "age": age, "payer": payer, "admit_type": admit_type,
                       "discharge": discharge, "drg": drg, "icd10": icd10, "los_days": los_days,
                       "creatinine": creatinine, "hemoglobin": hemoglobin, "sodium": sodium, "_frailty": frailty})
    df = df.sort_values(["patient_id", "admit_date"]).reset_index(drop=True)
    gap_true = df.groupby("patient_id").admit_date.diff().dt.days
    dow, mon, hgb = df.admit_date.dt.dayofweek, df.admit_date.dt.month, df.hemoglobin.fillna(12.5)
    logit = (-3.1 + 0.012 * (df.age - 64) + 0.7 * np.log(df.creatinine.fillna(1.0)) + 1.1 * (hgb < 10) + 0.3 * hgb.between(10, 11.5)
             + df.drg.map(drg_eff) + df.discharge.map({"Home": 0, "Home health": .25, "SNF": .45, "AMA": 1.0})
             + 0.35 * (df.payer == "Medicaid") + 0.4 * np.log(df.los_days) + 0.25 * (dow >= 5) + 0.2 * mon.isin([12, 1, 2])
             + np.where(gap_true < 30, 0.9, np.where(gap_true < 90, 0.35, 0)) + 0.3 * df._frailty
             + care_effect * df.discharge.isin(["Home health", "SNF"]))
    df["readmit30"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["creatinine"] = (df.creatinine * creat_report).round(2)        # what the NEW assay reports
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    df = df.drop(columns="_frailty")
    return df[df.admit_date < end].reset_index(drop=True) if clip else df
