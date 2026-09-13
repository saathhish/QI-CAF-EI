"""
utils/synthetic_data.py
═══════════════════════════════════════════════════════════════════════════════
Synthetic data generators that faithfully mirror the schema of restricted
healthcare datasets. These allow full pipeline testing without requiring
PhysioNet credentialing or large downloads.

Datasets simulated:
  - MIMIC-III Clinical Database (binary mortality, 17 tabular features, 11.5% pos)
  - PhysioNet 2012 Challenge (48h ICU time-series, 42 variables)

Usage:
  from utils.synthetic_data import generate_mimic3_synthetic, generate_physionet_synthetic
  X, y = generate_mimic3_synthetic(n_samples=10000, random_state=42)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ─── MIMIC-III Schema ─────────────────────────────────────────────────────────
MIMIC3_FEATURES = [
    # Demographics
    "age", "gender",
    # Vital signs (mean over ICU stay)
    "heart_rate_mean", "sysbp_mean", "diasbp_mean", "meanbp_mean",
    "resprate_mean", "tempc_mean", "spo2_mean",
    # Lab values (mean over ICU stay)
    "glucose_mean", "sodium_mean", "potassium_mean", "creatinine_mean",
    "bun_mean", "wbc_mean", "hgb_mean",
    # Clinical
    "icu_los_hours",
]

# Clinically plausible ranges for each feature [mean, std]
MIMIC3_DISTRIBUTIONS = {
    "age":              (65.0, 16.0),
    "gender":           None,                   # Bernoulli 0.43 female
    "heart_rate_mean":  (83.0, 17.0),
    "sysbp_mean":       (121.0, 22.0),
    "diasbp_mean":      (64.0, 13.0),
    "meanbp_mean":      (82.0, 16.0),
    "resprate_mean":    (18.5, 4.5),
    "tempc_mean":       (36.8, 0.7),
    "spo2_mean":        (97.2, 2.1),
    "glucose_mean":     (132.0, 55.0),
    "sodium_mean":      (138.5, 5.2),
    "potassium_mean":   (4.1, 0.6),
    "creatinine_mean":  (1.5, 1.4),
    "bun_mean":         (23.0, 17.0),
    "wbc_mean":         (11.2, 5.8),
    "hgb_mean":         (10.4, 2.1),
    "icu_los_hours":    (78.0, 80.0),
}

# Shift parameters for non-survivors (approximate clinical differences)
MIMIC3_MORTALITY_SHIFT = {
    "age":              +8.0,
    "heart_rate_mean":  +12.0,
    "sysbp_mean":       -15.0,
    "resprate_mean":    +4.0,
    "spo2_mean":        -3.5,
    "creatinine_mean":  +0.9,
    "bun_mean":         +10.0,
    "icu_los_hours":    +30.0,
}


def generate_mimic3_synthetic(
    n_samples: int = 20000,
    positive_rate: float = 0.115,
    missing_rate: float = 0.12,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic MIMIC-III tabular data.

    Args:
        n_samples:      Total number of patient records.
        positive_rate:  Fraction of mortality=1 (death). Paper: 11.5%.
        missing_rate:   Fraction of lab values to randomly set as NaN (≈12% in real data).
        random_state:   NumPy random seed.

    Returns:
        X: DataFrame of shape (n_samples, 17) with feature names.
        y: Series of binary labels (0=survived, 1=died).
    """
    rng = np.random.RandomState(random_state)
    n_pos = int(n_samples * positive_rate)
    n_neg = n_samples - n_pos

    records = []
    labels  = []

    for label, n_group in [(1, n_pos), (0, n_neg)]:
        for _ in range(n_group):
            row = {}
            for feat, dist in MIMIC3_DISTRIBUTIONS.items():
                if dist is None:
                    row[feat] = float(rng.binomial(1, 0.43))
                else:
                    mu, sigma = dist
                    shift = MIMIC3_MORTALITY_SHIFT.get(feat, 0.0) if label == 1 else 0.0
                    val = rng.normal(mu + shift, sigma)
                    # Clip to plausible ranges
                    row[feat] = float(max(val, 0.0))
            records.append(row)
            labels.append(label)

    X = pd.DataFrame(records, columns=MIMIC3_FEATURES)

    # Inject random missing values (lab features only)
    lab_features = [f for f in MIMIC3_FEATURES if f.endswith("_mean") and
                    f not in ("heart_rate_mean", "sysbp_mean", "diasbp_mean",
                               "meanbp_mean", "resprate_mean", "tempc_mean", "spo2_mean")]
    mask = rng.rand(n_samples, len(lab_features)) < missing_rate
    X[lab_features] = X[lab_features].where(~mask, other=np.nan)

    y = pd.Series(labels, name="hospital_mortality", dtype=np.int64)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


# ─── PhysioNet 2012 Schema ────────────────────────────────────────────────────
PHYSIONET_STATIC_FEATURES = [
    "RecordID", "Age", "Gender", "Height", "ICUType", "Weight"
]

PHYSIONET_TIME_SERIES_VARS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS",
    "Glucose", "HCO3", "HCT", "HR", "K", "Lactate",
    "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets",
    "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC",
]

PHYSIONET_VAR_STATS = ["min", "max", "mean", "std", "first", "last"]


def generate_physionet_synthetic(
    n_samples: int = 4000,
    n_time_steps: int = 48,
    positive_rate: float = 0.14,
    missing_rate: float = 0.60,    # highly sparse in real data
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic PhysioNet 2012 time-series data.

    Returns aggregated feature statistics (min/max/mean/std/first/last per variable)
    to produce a fixed-size feature vector suitable for MLP/BiLSTM after aggregation.

    Args:
        n_samples:    Number of ICU patient records.
        n_time_steps: Time bins (48 hours).
        positive_rate: Fraction dying in-hospital.
        missing_rate: Fraction of time-series entries missing (real ≈60%).
        random_state: Seed.

    Returns:
        X: ndarray of shape (n_samples, n_vars * n_stats + n_static)
           where n_vars=36, n_stats=6, n_static=5 → 221 features
        y: ndarray of shape (n_samples,) binary mortality labels
    """
    rng = np.random.RandomState(random_state)
    n_vars = len(PHYSIONET_TIME_SERIES_VARS)
    n_stats = len(PHYSIONET_VAR_STATS)
    n_pos = int(n_samples * positive_rate)
    n_neg = n_samples - n_pos

    all_X, all_y = [], []

    for label, n_group in [(1, n_pos), (0, n_neg)]:
        for _ in range(n_group):
            # Static features
            age     = rng.normal(63, 17)
            gender  = rng.binomial(1, 0.43)
            icu_type = rng.randint(1, 5)
            height  = rng.normal(170, 12)
            weight  = rng.normal(82, 22)
            static  = [age, gender, icu_type, height, weight]

            # Time-series: irregular, sparse
            ts_feats = []
            for _v in range(n_vars):
                # Random number of observations (Poisson, mean=6 per 48h)
                n_obs = rng.poisson(6)
                if n_obs == 0 or rng.rand() < missing_rate:
                    # No observations → fill with NaN stats
                    ts_feats.extend([np.nan] * n_stats)
                else:
                    vals = rng.normal(5.0, 2.0, size=n_obs)
                    if label == 1:
                        vals *= rng.uniform(0.8, 1.3)
                    ts_feats.extend([
                        float(np.min(vals)),
                        float(np.max(vals)),
                        float(np.mean(vals)),
                        float(np.std(vals) if len(vals) > 1 else 0.0),
                        float(vals[0]),
                        float(vals[-1]),
                    ])

            all_X.append(static + ts_feats)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ─── UCI Heart Disease Synthetic ──────────────────────────────────────────────
UCI_HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def generate_uci_heart_synthetic(
    n_samples: int = 2000,
    positive_rate: float = 0.55,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic UCI Heart Disease data matching the 13-feature schema."""
    rng = np.random.RandomState(random_state)
    n_pos = int(n_samples * positive_rate)
    n_neg = n_samples - n_pos
    records, labels = [], []

    for label, n_group in [(1, n_pos), (0, n_neg)]:
        for _ in range(n_group):
            row = {
                "age":       int(rng.normal(54 + label * 2, 9)),
                "sex":       int(rng.binomial(1, 0.68)),
                "cp":        int(rng.randint(0, 4)),
                "trestbps":  int(rng.normal(131 + label * 5, 17)),
                "chol":      int(rng.normal(246, 52)),
                "fbs":       int(rng.binomial(1, 0.15)),
                "restecg":   int(rng.randint(0, 3)),
                "thalach":   int(rng.normal(150 - label * 15, 23)),
                "exang":     int(rng.binomial(1, 0.23 + label * 0.2)),
                "oldpeak":   float(max(0, rng.normal(1.0 + label * 0.6, 1.2))),
                "slope":     int(rng.randint(0, 3)),
                "ca":        int(rng.randint(0, 4)),
                "thal":      int(rng.choice([1, 2, 3])),
            }
            records.append(row)
            labels.append(label)

    X = pd.DataFrame(records, columns=UCI_HEART_FEATURES)
    y = pd.Series(labels, name="target", dtype=np.int64)
    idx = rng.permutation(n_samples)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
