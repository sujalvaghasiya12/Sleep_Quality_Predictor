# scripts/generate_dataset.py
"""
Improved synthetic dataset generator for Sleep Quality Predictor.

Features:
- numeric: hours_of_sleep, screen_time, caffeine_intake, steps_walked, stress_level,
           exercise_minutes, alcohol_units, bedtime_hour
- categorical: ambient_light, weekday
- derived: late_bed (bool), active (bool)
- target: sleep_quality ('good' or 'poor') and sleep_good_prob (float)

Usage:
    from scripts.generate_dataset import generate_sleep_dataset
    df = generate_sleep_dataset(n_samples=2000, out_path='data/sleep_data.csv',
                                random_state=42, noise_scale=0.03, balance_classes=False)
"""
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_sleep_dataset(
    n_samples: int = 2000,
    out_path: str = "data/sleep_data.csv",
    random_state: Optional[int] = 42,
    noise_scale: float = 0.03,
    balance_classes: bool = False,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for sleep-quality prediction.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    out_path : str
        CSV file path to write the dataset.
    random_state : Optional[int]
        Random seed for reproducibility.
    noise_scale : float
        Stddev of gaussian noise added to the label probability (0..1).
    balance_classes : bool
        If True, adjust threshold so roughly half the dataset is 'good' and half 'poor'.

    Returns
    -------
    pd.DataFrame
        Generated dataframe (also saved to out_path).
    """
    rng = np.random.RandomState(random_state)

    Path("data").mkdir(parents=True, exist_ok=True)

    # ---------- Generate base features (realistic-looking distributions) ----------
    # hours_of_sleep: normal around 7, clamp to [3, 11]
    hours_of_sleep = rng.normal(loc=7.0, scale=1.25, size=n_samples).clip(3.0, 11.0)

    # screen_time: right-skewed (many low users, some high)
    screen_time = rng.gamma(shape=2.0, scale=1.5, size=n_samples).clip(0.0, 16.0)

    # caffeine_intake: small integers, Poisson
    caffeine_intake = rng.poisson(lam=1.0, size=n_samples).clip(0, 6)

    # steps_walked: skewed; clamp to plausible range
    steps_walked = rng.normal(loc=6000, scale=3000, size=n_samples).clip(0, 25000).astype(int)

    # stress_level: discrete 1..10 (uniform-ish with slight bias to lower)
    stress_level = rng.choice(np.arange(1, 11), size=n_samples, p=None)

    # exercise_minutes: many with low activity, some high
    exercise_minutes = rng.normal(loc=25, scale=25, size=n_samples).clip(0, 240).astype(int)

    # alcohol_units: mostly zero, some 1-3
    alcohol_units = rng.binomial(n=3, p=0.1, size=n_samples)

    # categorical
    ambient_light = rng.choice(["low", "medium", "high"], size=n_samples, p=[0.62, 0.30, 0.08])
    weekday = rng.choice(["weekday", "weekend"], size=n_samples, p=[0.7, 0.3])

    # bedtime_hour: centered around 23 for many, but some early/late
    bedtime_hour = (rng.normal(loc=23.0, scale=1.5, size=n_samples) % 24).round(2)

    # ---------- Derived features: make model-building easier ----------
    late_bed = (bedtime_hour >= 24 - 2) | (bedtime_hour >= 23.0)  # boolean-ish: true if late
    active = (steps_walked >= 8000) | (exercise_minutes >= 30)

    # ---------- Combine into DataFrame ----------
    df = pd.DataFrame(
        {
            "hours_of_sleep": hours_of_sleep,
            "screen_time": screen_time,
            "caffeine_intake": caffeine_intake,
            "steps_walked": steps_walked,
            "stress_level": stress_level,
            "exercise_minutes": exercise_minutes,
            "alcohol_units": alcohol_units,
            "ambient_light": ambient_light,
            "weekday": weekday,
            "bedtime_hour": bedtime_hour,
            "late_bed": late_bed.astype(int),
            "active": active.astype(int),
        }
    )

    # ---------- Scoring function (higher score => worse sleep) ----------
    # Use scaled contributions so features are comparable. Tuned coefficients for realism.
    # Positive coefficients increase the 'badness' score.
    score = (
        1.2 * (7.0 - df["hours_of_sleep"])  # less sleep -> worse
        + 0.6 * (df["screen_time"] / 3.0)  # more screen time -> worse
        + 1.0 * df["caffeine_intake"]  # caffeine -> worse
        + 0.9 * (df["stress_level"] / 10.0)  # stress (0..1)
        + 0.6 * ((df["bedtime_hour"] - 22.0).clip(lower=0) / 3.0)  # late bedtime -> worse
        + 0.85 * df["alcohol_units"]  # alcohol -> worse
        - 0.35 * (df["exercise_minutes"] / 30.0)  # exercise -> better
        - 0.25 * (df["steps_walked"] / 8000.0)  # steps -> better
        - 0.4 * df["active"]  # being 'active' gives additional benefit
        + 0.15 * (df["late_bed"])  # slight penalty for late bedding
    )

    # convert score to probability of *bad* sleep using a sigmoid, then invert for "good" prob
    prob_bad = _sigmoid(score - 0.4)  # shift controls baseline prevalence
    # add a small gaussian noise to the probability (keeps realism)
    prob_bad = (prob_bad + rng.normal(scale=noise_scale, size=n_samples)).clip(0.0, 1.0)
    prob_good = 1.0 - prob_bad

    # If user asked for balanced classes, choose threshold at median(prob_good)
    if balance_classes:
        threshold = float(np.median(prob_good))
    else:
        threshold = 0.5

    # Label: 'good' if prob_good >= threshold, else 'poor'
    df["sleep_good_prob"] = prob_good
    df["sleep_quality"] = np.where(df["sleep_good_prob"] >= threshold, "good", "poor")

    # ---------- Save & report ----------
    df.to_csv(out_path, index=False)
    n_good = (df["sleep_quality"] == "good").sum()
    n_poor = (df["sleep_quality"] == "poor").sum()
    print(f"✅ Saved synthetic dataset to {out_path} with {n_samples} rows")
    print(f"   -> good: {n_good} rows ({n_good / n_samples:.2%}), poor: {n_poor} rows ({n_poor / n_samples:.2%})")
    if balance_classes:
        print("   -> classes balanced by thresholding at median probability (approx 50/50).")

    return df


if __name__ == "__main__":
    # quick test / demo
    _ = generate_sleep_dataset(n_samples=2000, out_path="data/sleep_data.csv", random_state=42, balance_classes=False)
