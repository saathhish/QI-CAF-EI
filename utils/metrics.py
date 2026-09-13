"""
utils/metrics.py
═══════════════════════════════════════════════════════════════════════════════
Multi-Objective Performance Metrics for QI-CAF-EI

Implements:
  1. Polygon Area Metric (PAM) — multi-objective radar chart area
  2. Standard classification metrics (AUC-ROC, F1, accuracy, AUPRC)
  3. Multi-label metrics (for ChestX-ray14)
  4. Privacy consumption metrics
  5. Communication efficiency metrics
  6. Fairness metrics (per-node accuracy variance)

Polygon Area Metric (PAM):
  PAM = (1/2) Σ_k m_k × m_{k+1} × sin(2π/K)
  Where m_k ∈ [0,1] are normalized objective scores on K axes:
    - Accuracy/AUC
    - Privacy strength (1/ε_mean)
    - Communication efficiency (compression ratio)
    - Fairness (1 - std_dev of node accuracies)
    - XAI consistency (1/L_XAI)

Author: QI-CAF-EI Research Framework
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    classification_report,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Standard Classification Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_proba:   Optional[np.ndarray] = None,
    multiclass: bool = False,
    multilabel: bool = False,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true:     True labels (n_samples,) or (n_samples, n_classes) for multi-label.
        y_pred:     Predicted labels/classes.
        y_proba:    Predicted probabilities (for AUC computation).
        multiclass: True for multi-class (>2 classes).
        multilabel: True for multi-label (ChestX-ray14).

    Returns:
        Dict with accuracy, f1, auc_roc, auprc, etc.
    """
    metrics = {}

    if multilabel:
        # ChestX-ray14: per-label averaging
        metrics["accuracy"] = float(np.mean(y_pred == y_true))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0))
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro",
                                             zero_division=0))
        if y_proba is not None:
            try:
                metrics["auc_roc_macro"] = float(
                    roc_auc_score(y_true, y_proba, average="macro"))
                metrics["auprc_macro"]   = float(
                    average_precision_score(y_true, y_proba, average="macro"))
            except Exception:
                metrics["auc_roc_macro"] = 0.5

    elif multiclass:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted",
                                                zero_division=0))
        if y_proba is not None:
            try:
                metrics["auc_roc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, average="macro",
                                  multi_class="ovr"))
            except Exception:
                metrics["auc_roc_ovr"] = 0.5

    else:
        # Binary classification
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1"]       = float(f1_score(y_true, y_pred, zero_division=0))
        if y_proba is not None:
            try:
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["auc_roc"] = float(roc_auc_score(y_true, proba))
                metrics["auprc"]   = float(average_precision_score(y_true, proba))
            except Exception:
                metrics["auc_roc"] = 0.5
                metrics["auprc"]   = 0.5

    return metrics


@torch.no_grad()
def evaluate_model(
    model:        nn.Module,
    dataloader,
    dataset_name: str,
    device:       torch.device,
    loss_fn:      Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a DataLoader and compute all relevant metrics.

    Args:
        model:        PyTorch model.
        dataloader:   Evaluation DataLoader.
        dataset_name: Dataset identifier (for metric selection).
        device:       Torch device.
        loss_fn:      Loss function (optional, for loss reporting).

    Returns:
        Dict of metric names → values.
    """
    model.eval()
    all_preds, all_true, all_proba, total_loss = [], [], [], 0.0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device).float()
        batch_y = batch_y.to(device)
        output  = model(batch_X)

        if loss_fn is not None:
            loss = loss_fn(output,
                           batch_y.float() if dataset_name == "chestxray14"
                           else batch_y)
            total_loss += float(loss.item()) * len(batch_y)

        if dataset_name == "chestxray14":
            proba = torch.sigmoid(output).cpu().numpy()
            preds = (proba > 0.5).astype(int)
            all_proba.append(proba)
            all_preds.append(preds)
            all_true.append(batch_y.cpu().numpy())
        else:
            proba = torch.softmax(output, dim=-1).cpu().numpy()
            preds = output.argmax(dim=-1).cpu().numpy()
            all_proba.append(proba)
            all_preds.append(preds)
            all_true.append(batch_y.cpu().numpy())

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)
    n_samples = len(y_true)

    is_multilabel  = (dataset_name == "chestxray14")
    is_multiclass  = (dataset_name == "wisdm")

    mets = compute_classification_metrics(
        y_true, y_pred, y_proba,
        multiclass = is_multiclass,
        multilabel = is_multilabel,
    )

    if n_samples > 0 and loss_fn is not None:
        mets["loss"] = total_loss / n_samples

    return mets


# ═══════════════════════════════════════════════════════════════════════════════
# Polygon Area Metric (PAM)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pam(
    accuracy:            float,   # e.g., AUC-ROC or classification accuracy
    compression_ratio:   float,   # MPS compression ratio (dense/mps params)
    epsilon_mean:        float,   # Mean ε across nodes (lower = more private)
    epsilon_max_budget:  float,   # Maximum allowed ε for normalization
    accuracy_std:        float,   # Std dev of per-node accuracy (fairness)
    xai_loss:            float,   # Mean L_XAI (lower = better consistency)
    xai_loss_max:        float = 10.0,  # Normalization upper bound for L_XAI
    compression_ref:     float = 10.0, # Reference compression ratio for normalization
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Polygon Area Metric (PAM).

    The PAM summarizes multi-objective performance as the area of a
    regular K-gon radar chart with K=5 axes, each normalized to [0, 1]:

    Axes:
      m_1: Accuracy/AUC     (higher = better)
      m_2: Privacy strength  (1 - ε_mean/ε_max, higher = better)
      m_3: Compression       (compression_ratio/ref, higher = better)
      m_4: Fairness          (1 - accuracy_std/0.5, higher = better)
      m_5: XAI consistency   (1 - L_XAI/L_XAI_max, higher = better)

    Area formula:
      PAM = (1/2) × Σ_k m_k × m_{k+1} × sin(2π/K)
      (cyclic: m_{K+1} = m_1)

    Args:
        accuracy:           Global accuracy or AUC-ROC (0–1).
        compression_ratio:  MPS compression ratio (e.g., 12.5×).
        epsilon_mean:       Mean privacy budget ε across active nodes.
        epsilon_max_budget: Reference maximum ε for normalization.
        accuracy_std:       Standard deviation of per-node accuracy.
        xai_loss:           Mean L_XAI over this round.
        xai_loss_max:       Upper bound for L_XAI normalization.
        compression_ref:    Reference compression ratio (max expected).

    Returns:
        (pam_score, axis_scores_dict)
    """
    K = 5
    angle = 2.0 * math.pi / K

    # ── Normalize all axes to [0, 1] ──────────────────────────────────────
    m1_accuracy    = float(np.clip(accuracy, 0.0, 1.0))
    m2_privacy     = float(np.clip(1.0 - epsilon_mean / max(epsilon_max_budget, 1e-6), 0.0, 1.0))
    m3_compression = float(np.clip(compression_ratio / max(compression_ref, 1.0), 0.0, 1.0))
    m4_fairness    = float(np.clip(1.0 - accuracy_std / 0.5, 0.0, 1.0))
    m5_xai         = float(np.clip(1.0 - xai_loss / max(xai_loss_max, 1e-6), 0.0, 1.0))

    axes = [m1_accuracy, m2_privacy, m3_compression, m4_fairness, m5_xai]

    # ── Shoelace area formula for regular polygon ──────────────────────────
    pam = 0.0
    for k in range(K):
        m_k  = axes[k]
        m_k1 = axes[(k + 1) % K]
        pam += m_k * m_k1 * math.sin(angle)
    pam = 0.5 * pam

    axis_scores = {
        "m1_accuracy":    m1_accuracy,
        "m2_privacy":     m2_privacy,
        "m3_compression": m3_compression,
        "m4_fairness":    m4_fairness,
        "m5_xai":         m5_xai,
        "pam":            pam,
    }

    return pam, axis_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Communication Efficiency Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_communication_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Compute communication efficiency metrics for MPS-compressed models.

    Returns:
        Dict with total_params, mps_params, compression_ratio,
        transmission_size_kb (estimated).
    """
    from src.quantum_mps import compute_mps_compression_stats
    stats = compute_mps_compression_stats(model)

    mps_params    = stats.get("total_mps_params", 0)
    dense_params  = stats.get("total_dense_params", 0)
    ratio         = stats.get("overall_ratio", 1.0)

    # Estimate transmission size (float32 = 4 bytes)
    tx_size_kb = (mps_params * 4) / 1024

    return {
        "dense_params":       dense_params,
        "mps_params":         mps_params,
        "compression_ratio":  ratio,
        "param_reduction_pct": 100.0 * (1.0 - mps_params / max(dense_params, 1)),
        "tx_size_kb":         tx_size_kb,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Federated Round Summary
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedMetricsTracker:
    """
    Tracks and aggregates metrics across federated rounds.

    Maintains per-round history for:
      - Global model accuracy/AUC
      - Per-node accuracy (for fairness)
      - Privacy budget consumption
      - XAI consistency loss
      - PAM score

    Usage:
      tracker = FederatedMetricsTracker(n_nodes, dataset_name)
      tracker.record_round(round_idx, global_metrics, node_metrics, privacy, xai_loss)
      tracker.print_summary()
      tracker.save(path)
    """

    def __init__(self, n_nodes: int, dataset_name: str):
        self.n_nodes      = n_nodes
        self.dataset_name = dataset_name
        self.history: List[dict] = []

    def record_round(
        self,
        round_idx:        int,
        global_metrics:   Dict[str, float],
        node_metrics:     Optional[Dict[int, Dict[str, float]]] = None,
        privacy_summary:  Optional[Dict[str, float]] = None,
        xai_loss:         float = 0.0,
        compression:      Optional[Dict[str, float]] = None,
    ):
        """Record all metrics for one federated round."""
        # Per-node accuracy for fairness metric
        if node_metrics:
            node_accs = [m.get("accuracy", 0.0) for m in node_metrics.values()]
            acc_std   = float(np.std(node_accs)) if node_accs else 0.0
            acc_mean  = float(np.mean(node_accs)) if node_accs else 0.0
        else:
            acc_std  = 0.0
            acc_mean = global_metrics.get("accuracy", 0.0)

        # PAM computation
        pam, pam_axes = compute_pam(
            accuracy          = global_metrics.get("accuracy",
                                global_metrics.get("auc_roc",
                                global_metrics.get("auc_roc_macro", 0.5))),
            compression_ratio = compression.get("compression_ratio", 1.0)
                                if compression else 1.0,
            epsilon_mean      = privacy_summary.get("mean", 1.0)
                                if privacy_summary else 1.0,
            epsilon_max_budget= 10.0,
            accuracy_std      = acc_std,
            xai_loss          = xai_loss,
        )

        record = {
            "round":          round_idx,
            "global":         global_metrics,
            "fairness_std":   acc_std,
            "fairness_mean":  acc_mean,
            "privacy":        privacy_summary or {},
            "xai_loss":       xai_loss,
            "compression":    compression or {},
            "pam":            pam,
            "pam_axes":       pam_axes,
        }
        self.history.append(record)

        logger.info(
            "[Round %d] acc=%.4f, AUC=%.4f, PAM=%.4f, eps_mean=%.3f, L_XAI=%.4f",
            round_idx,
            global_metrics.get("accuracy", 0.0),
            global_metrics.get("auc_roc",
                global_metrics.get("auc_roc_ovr",
                global_metrics.get("auc_roc_macro", 0.0))),
            pam,
            privacy_summary.get("mean", 0.0) if privacy_summary else 0.0,
            xai_loss,
        )

    def print_summary(self):
        """Print a formatted metrics table for all rounds."""
        try:
            from tabulate import tabulate
            rows = []
            for r in self.history:
                rows.append([
                    r["round"],
                    f"{r['global'].get('accuracy', 0):.4f}",
                    f"{r['global'].get('auc_roc', r['global'].get('auc_roc_ovr', 0)):.4f}",
                    f"{r['pam']:.4f}",
                    f"{r['privacy'].get('mean', 0):.3f}",
                    f"{r['xai_loss']:.4f}",
                    f"{r['fairness_std']:.4f}",
                ])
            headers = ["Round", "Acc", "AUC", "PAM", "eps_mean", "L_XAI", "Fair_std"]
            print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
        except ImportError:
            for r in self.history:
                print(f"Round {r['round']:3d}: "
                      f"acc={r['global'].get('accuracy',0):.4f}  "
                      f"PAM={r['pam']:.4f}  "
                      f"ε={r['privacy'].get('mean',0):.3f}")

    def save(self, path: str):
        """Save metrics history to numpy .npz file."""
        import json
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)
        logger.info("Metrics saved to %s", path)

    def get_best_round(self, metric: str = "pam") -> dict:
        """Return the round with the best value for a given metric."""
        if not self.history:
            return {}
        return max(self.history, key=lambda r: r.get(metric, 0.0))

    def plot_metrics(self, output_dir: str = "results/"):
        """Generate and save metric evolution plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            import os
            os.makedirs(output_dir, exist_ok=True)

            rounds     = [r["round"] for r in self.history]
            accs       = [r["global"].get("accuracy", 0) for r in self.history]
            pams       = [r["pam"] for r in self.history]
            epsilons   = [r["privacy"].get("mean", 0) for r in self.history]
            xai_losses = [r["xai_loss"] for r in self.history]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"QI-CAF-EI Training Metrics — {self.dataset_name}",
                         fontsize=14, fontweight="bold")

            axes[0, 0].plot(rounds, accs, "b-o", linewidth=2)
            axes[0, 0].set_title("Global Accuracy")
            axes[0, 0].set_xlabel("Round"); axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(rounds, pams, "g-s", linewidth=2)
            axes[0, 1].set_title("Polygon Area Metric (PAM)")
            axes[0, 1].set_xlabel("Round"); axes[0, 1].set_ylabel("PAM")
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(rounds, epsilons, "r-^", linewidth=2)
            axes[1, 0].set_title("Mean Privacy Budget ε")
            axes[1, 0].set_xlabel("Round"); axes[1, 0].set_ylabel("ε (lower = better)")
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(rounds, xai_losses, "m-D", linewidth=2)
            axes[1, 1].set_title("XAI Consistency Loss (L_XAI)")
            axes[1, 1].set_xlabel("Round"); axes[1, 1].set_ylabel("L_XAI")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"metrics_{self.dataset_name}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Metrics plot saved to %s", plot_path)
            return plot_path

        except Exception as e:
            logger.warning("Failed to generate metrics plot: %s", e)
            return None

    def plot_pam_radar(self, round_idx: int = -1, output_dir: str = "results/") -> Optional[str]:
        """Generate radar chart showing PAM axes for a specific round."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            import os
            os.makedirs(output_dir, exist_ok=True)

            r = self.history[round_idx]
            axes_dict = r.get("pam_axes", {})

            labels  = ["Accuracy", "Privacy", "Compression", "Fairness", "XAI"]
            values  = [
                axes_dict.get("m1_accuracy", 0),
                axes_dict.get("m2_privacy", 0),
                axes_dict.get("m3_compression", 0),
                axes_dict.get("m4_fairness", 0),
                axes_dict.get("m5_xai", 0),
            ]

            # Close the radar chart
            values += values[:1]
            labels_plot = labels + labels[:1]

            angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6),
                                   subplot_kw=dict(projection="polar"))
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)

            ax.plot(angles, values, "o-", linewidth=2, color="#1f77b4")
            ax.fill(angles, values, alpha=0.25, color="#1f77b4")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_title(
                f"PAM Radar — {self.dataset_name} (Round {r['round']})\n"
                f"PAM = {r['pam']:.4f}",
                size=12, fontweight="bold", pad=20,
            )

            plot_path = os.path.join(
                output_dir, f"pam_radar_{self.dataset_name}_r{r['round']}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("PAM radar saved to %s", plot_path)
            return plot_path

        except Exception as e:
            logger.warning("Failed to generate PAM radar: %s", e)
            return None
