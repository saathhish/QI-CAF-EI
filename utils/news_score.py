"""
utils/news_score.py
═══════════════════════════════════════════════════════════════════════════════
National Early Warning Score (NEWS / NEWS2) computation.

The NEWS score is used as the clinical urgency component u_i in the
context vector c_i = [s_i, d_i, l_i, b_i, u_i]^T for the Context Controller.

References:
  Royal College of Physicians. National Early Warning Score (NEWS) 2.
  RCP, London, 2017. https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

Score interpretation:
  0–4   → Low risk
  5–6   → Medium risk (or any single score of 3)
  7+    → High risk (urgent clinical review)
  Max   → 20 points

Usage:
  from utils.news_score import compute_news_score, normalize_news
  score = compute_news_score(resp_rate=22, spo2=95, on_o2=False,
                              systolic_bp=100, hr=110, consciousness="A",
                              temp=37.5)
  u_i = normalize_news(score)   # → [0, 1]
"""

from typing import Optional, Union


# ─── NEWS2 Scoring Tables ─────────────────────────────────────────────────────

def _score_resp_rate(rr: float) -> int:
    """Respiratory rate (breaths/min)."""
    if rr <= 8:   return 3
    if rr <= 11:  return 1
    if rr <= 20:  return 0
    if rr <= 24:  return 2
    return 3


def _score_spo2_scale1(spo2: float) -> int:
    """SpO2 — Scale 1 (not hypercapnic; standard)."""
    if spo2 <= 91:  return 3
    if spo2 <= 93:  return 2
    if spo2 <= 95:  return 1
    return 0


def _score_spo2_scale2(spo2: float, on_o2: bool) -> int:
    """SpO2 — Scale 2 (hypercapnic / COPD patients)."""
    if spo2 <= 83:   return 3
    if spo2 <= 85:   return 2
    if spo2 <= 87:   return 1
    if spo2 <= 92 and not on_o2:  return 0
    if spo2 <= 92 and on_o2:      return 0
    if spo2 <= 94 and on_o2:      return 1
    if spo2 <= 96 and on_o2:      return 2
    if on_o2:                      return 3
    return 0


def _score_supplemental_o2(on_o2: bool) -> int:
    """Supplemental oxygen (2 points if on O2)."""
    return 2 if on_o2 else 0


def _score_systolic_bp(sbp: float) -> int:
    """Systolic blood pressure (mmHg)."""
    if sbp <= 90:   return 3
    if sbp <= 100:  return 2
    if sbp <= 110:  return 1
    if sbp <= 219:  return 0
    return 3


def _score_heart_rate(hr: float) -> int:
    """Heart rate (beats/min)."""
    if hr <= 40:   return 3
    if hr <= 50:   return 1
    if hr <= 90:   return 0
    if hr <= 110:  return 1
    if hr <= 130:  return 2
    return 3


def _score_consciousness(avpu: str) -> int:
    """
    Level of consciousness (AVPU scale).
    'A' = Alert (0), 'C' = Confused (3), 'V' = Voice (3),
    'P' = Pain (3), 'U' = Unresponsive (3).
    """
    avpu_upper = str(avpu).strip().upper()
    return 0 if avpu_upper == "A" else 3


def _score_temperature(temp_c: float) -> int:
    """Temperature (°C)."""
    if temp_c <= 35.0:  return 3
    if temp_c <= 36.0:  return 1
    if temp_c <= 38.0:  return 0
    if temp_c <= 39.0:  return 1
    return 2


# ─── Main Scoring Function ────────────────────────────────────────────────────

def compute_news_score(
    resp_rate:    float,
    spo2:         float,
    on_o2:        bool,
    systolic_bp:  float,
    hr:           float,
    consciousness: Union[str, int],   # AVPU string or integer 0/3
    temp:         float,
    hypercapnic:  bool = False,
) -> int:
    """
    Compute the National Early Warning Score (NEWS2).

    Args:
        resp_rate:      Respiratory rate (breaths/min).
        spo2:           Peripheral oxygen saturation (%).
        on_o2:          True if patient is on supplemental oxygen.
        systolic_bp:    Systolic blood pressure (mmHg).
        hr:             Heart rate (beats/min).
        consciousness:  AVPU string ('A','C','V','P','U') or 0/3 integer.
        temp:           Temperature (°C).
        hypercapnic:    True to use SpO2 Scale 2 (COPD / hypercapnic).

    Returns:
        Total NEWS score (integer, 0–20).
    """
    score_rr  = _score_resp_rate(resp_rate)
    score_o2  = (
        _score_spo2_scale2(spo2, on_o2) if hypercapnic
        else _score_spo2_scale1(spo2)
    )
    score_sup = _score_supplemental_o2(on_o2)
    score_sbp = _score_systolic_bp(systolic_bp)
    score_hr  = _score_heart_rate(hr)
    score_con = (
        _score_consciousness(consciousness)
        if isinstance(consciousness, str)
        else int(consciousness)
    )
    score_tmp = _score_temperature(temp)

    total = (score_rr + score_o2 + score_sup +
             score_sbp + score_hr + score_con + score_tmp)
    return int(total)


def normalize_news(score: int, max_score: int = 20) -> float:
    """
    Normalize NEWS score to [0, 1] for use as clinical urgency u_i.

    Args:
        score:      Raw NEWS integer score.
        max_score:  Maximum possible NEWS score (default 20).

    Returns:
        Normalized urgency in [0.0, 1.0].
    """
    return float(min(score, max_score)) / float(max_score)


def get_news_risk_level(score: int) -> str:
    """Return clinical risk level string for logging/reporting."""
    if score <= 4:  return "low"
    if score <= 6:  return "medium"
    return "high"


def simulate_news_for_node(
    node_id:      int,
    dataset_name: str,
    random_state: Optional[int] = None,
) -> float:
    """
    Simulate a plausible NEWS-based urgency u_i for a federated node.

    For datasets where live vital signs are not available, this generates
    a plausible clinical urgency score based on dataset characteristics:
      - MIMIC-III / PhysioNet: ICU patients → higher urgency
      - ChestX-ray14: outpatient radiology → moderate urgency
      - UCI Heart: cardiac risk → moderate-high urgency
      - WISDM: community/wearable → low urgency

    Args:
        node_id:      Node index (used for seeded reproducibility).
        dataset_name: One of "mimic3","physionet2012","chestxray14","uci_heart","wisdm".
        random_state: Base random seed.

    Returns:
        Normalized urgency u_i in [0, 1].
    """
    import numpy as np
    seed = (random_state or 0) + node_id * 17
    rng  = np.random.RandomState(seed)

    # Dataset-specific NEWS distribution parameters [mean, std]
    NEWS_DIST = {
        "mimic3":        (8.5,  3.5),    # ICU: high urgency
        "physionet2012": (9.0,  3.0),    # ICU: high urgency
        "chestxray14":   (4.0,  2.5),    # Outpatient radiology: moderate
        "uci_heart":     (5.5,  2.5),    # Cardiac clinic: moderate-high
        "wisdm":         (2.0,  1.5),    # Community wearable: low
    }

    mu, sigma = NEWS_DIST.get(dataset_name, (5.0, 3.0))
    raw_score = int(np.clip(rng.normal(mu, sigma), 0, 20))
    return normalize_news(raw_score)
