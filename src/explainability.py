"""
src/explainability.py
═══════════════════════════════════════════════════════════════════════════════
Phase 4: Federated SHAP Explainability & XAI Consistency Loss

Implements:
  1. Local SHAP attribution computation per edge node:
       E_i ∈ R^{d_features}  (feature importance for local model/data)
  2. Federated XAI Consistency Loss (HOD PAPER §5):
       L_XAI = Σ_i ‖E_i - E_global‖₂²
  3. Global SHAP vector aggregation (weighted average from server)
  4. Fallback KernelExplainer for non-differentiable models

Clinical motivation:
  Isolated hospital models may develop "spurious" feature attributions that
  are locally consistent but globally inconsistent (e.g., node A attributes
  chest X-rays primarily to texture while node B attributes to shape).
  The L_XAI penalty forces all nodes toward a consensus clinical explanation,
  improving trustworthiness and regulatory compliance.

Usage:
  explainer = SHAPExplainer(model, background_data, dataset_name)
  E_i = explainer.compute_attributions(X_batch)
  L_XAI = explainer.xai_loss(E_i, E_global)

Author: QI-CAF-EI Research Framework
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP Explainer
# ═══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    Local SHAP attribution engine for a single federated node.

    Supports:
      - DeepExplainer: PyTorch-native, exact Shapley via backpropagation
        (used for MLP, BiLSTM, CNN1D with tabular/sensor data)
      - GradientExplainer: Gradient-based approximation (faster)
      - KernelExplainer: Model-agnostic fallback (slow but universal)

    The explainer type is automatically selected based on model architecture
    and falls back gracefully if SHAP is not available.

    Args:
        model:          PyTorch model (with MPS layers).
        background_X:   Background dataset tensor (n_bg, n_features).
        dataset_name:   Dataset identifier for explainer selection.
        n_bg_samples:   Number of background samples to use.
        explainer_type: "deep"|"gradient"|"kernel"|"auto"
        device:         Torch device.
    """

    def __init__(
        self,
        model:          nn.Module,
        background_X:   torch.Tensor,
        dataset_name:   str = "mimic3",
        n_bg_samples:   int = 50,
        explainer_type: str = "auto",
        device:         Optional[torch.device] = None,
    ):
        self.model         = model
        self.dataset_name  = dataset_name
        self.device        = device or next(model.parameters(), torch.zeros(1)).device
        self.n_bg_samples  = n_bg_samples
        self._explainer    = None
        self._shap_values_cache: Optional[np.ndarray] = None

        # Subsample background
        n_bg = min(n_bg_samples, len(background_X))
        idx  = np.random.choice(len(background_X), n_bg, replace=False)
        self.background_X = background_X[idx].to(self.device)

        # Select explainer type
        if explainer_type == "auto":
            if dataset_name in ("mimic3", "physionet2012", "uci_heart"):
                self.explainer_type = "gradient"   # avoid DeepExplainer batch-size issue
            elif dataset_name == "wisdm":
                self.explainer_type = "gradient"
            else:
                self.explainer_type = "kernel"   # chestxray14: too large for deep
        else:
            self.explainer_type = explainer_type

        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer; fall back gracefully if unavailable."""
        try:
            import shap
            self.model.eval()

            if self.explainer_type == "deep":
                try:
                    self._explainer = shap.DeepExplainer(
                        self.model,
                        self.background_X,
                    )
                    logger.info("SHAPExplainer: DeepExplainer initialized for %s",
                                self.dataset_name)
                except Exception as e:
                    logger.warning("DeepExplainer failed (%s), trying GradientExplainer", e)
                    self.explainer_type = "gradient"

            if self.explainer_type == "gradient":
                try:
                    self._explainer = shap.GradientExplainer(
                        self.model,
                        self.background_X,
                    )
                    logger.info("SHAPExplainer: GradientExplainer initialized for %s",
                                self.dataset_name)
                except Exception as e:
                    logger.warning("GradientExplainer failed (%s), trying KernelExplainer", e)
                    self.explainer_type = "kernel"

            if self.explainer_type == "kernel":
                # KernelExplainer requires numpy wrapper
                def model_predict(x_np):
                    with torch.no_grad():
                        t = torch.tensor(x_np, dtype=torch.float32, device=self.device)
                        out = self.model(t)
                        if out.shape[-1] > 1:
                            out = torch.softmax(out, dim=-1)
                        return out.cpu().numpy()

                bg_np = self.background_X.cpu().numpy()
                self._explainer = shap.KernelExplainer(model_predict, bg_np)
                logger.info("SHAPExplainer: KernelExplainer initialized for %s",
                            self.dataset_name)

        except ImportError:
            logger.warning("SHAP not installed. XAI loss will use zero attributions.")
            self._explainer = None

    def compute_attributions(
        self,
        X:        torch.Tensor,
        n_samples: int = 30,
    ) -> np.ndarray:
        """
        Compute SHAP feature attributions for input X.

        Args:
            X:         Input tensor (batch, n_features) or (batch, ch, len).
            n_samples: Number of SHAP samples (for KernelExplainer).

        Returns:
            Attribution array E_i of shape (n_features,) — mean |SHAP| per feature.
            Returns zero vector if SHAP computation fails.
        """
        if self._explainer is None:
            # Return zero attributions (no penalty applied)
            n_features = self._infer_n_features(X)
            return np.zeros(n_features, dtype=np.float32)

        try:
            import shap
            self.model.eval()
            X_sample = X[:min(n_samples, len(X))].to(self.device)

            if self.explainer_type in ("deep", "gradient"):
                with torch.no_grad():
                    shap_vals = self._explainer.shap_values(X_sample)
            else:
                # KernelExplainer: numpy input/output
                X_np = X_sample.cpu().numpy()
                shap_vals = self._explainer.shap_values(X_np, nsamples=50)

            # shap_vals may be list (one per output class) or array
            if isinstance(shap_vals, list):
                # Take absolute values across classes
                shap_arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            else:
                shap_arr = np.abs(shap_vals)

            # Mean attribution per feature (shape: n_features)
            if shap_arr.ndim > 1:
                E_i = shap_arr.mean(axis=0)
            else:
                E_i = shap_arr

            # Flatten for CNN/image models
            E_i = E_i.ravel().astype(np.float32)

            self._shap_values_cache = E_i
            return E_i

        except Exception as e:
            logger.warning("SHAP computation failed (%s); using cached/zeros.", e)
            if self._shap_values_cache is not None:
                return self._shap_values_cache
            return np.zeros(self._infer_n_features(X), dtype=np.float32)

    def _infer_n_features(self, X: torch.Tensor) -> int:
        """Infer feature dimensionality from input tensor."""
        if X.ndim == 2:
            return X.shape[1]
        elif X.ndim == 3:
            return X.shape[1] * X.shape[2]  # C × L for 1D CNN
        elif X.ndim == 4:
            return X.shape[1] * X.shape[2] * X.shape[3]  # C × H × W
        return int(np.prod(X.shape[1:]))


# ═══════════════════════════════════════════════════════════════════════════════
# XAI Consistency Loss
# ═══════════════════════════════════════════════════════════════════════════════

def xai_consistency_loss(
    E_local:  torch.Tensor,
    E_global: torch.Tensor,
) -> torch.Tensor:
    """
    Compute XAI consistency loss for a single node.

    L_XAI_i = ‖E_i - E_global‖₂²

    Where:
      E_i      = local SHAP attribution vector (node i)
      E_global = global reference SHAP attribution (from server)

    Args:
        E_local:  Local attribution vector (n_features,).
        E_global: Global attribution vector (n_features,).

    Returns:
        Scalar loss tensor.
    """
    return torch.sum((E_local - E_global) ** 2)


def xai_global_consistency_loss(
    E_locals: List[torch.Tensor],
    E_global: torch.Tensor,
) -> torch.Tensor:
    """
    Compute total federated XAI consistency loss.

    L_XAI = Σ_i ‖E_i - E_global‖₂²   (HOD PAPER §5)

    Args:
        E_locals: List of local attribution vectors from all nodes.
        E_global: Global reference attribution vector.

    Returns:
        Scalar sum of per-node consistency losses.
    """
    total = torch.tensor(0.0, requires_grad=False)
    for E_i in E_locals:
        total = total + xai_consistency_loss(E_i, E_global)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Approximate Local SHAP Loss (differentiable, for in-loop training)
# ═══════════════════════════════════════════════════════════════════════════════

class SHAPConsistencyLoss(nn.Module):
    """
    Differentiable approximation of SHAP consistency loss for inclusion
    in the local training objective L_total = L_acc + λ × L_XAI.

    Since true SHAP values are non-differentiable, this module uses
    gradient-based saliency (input × gradient) as a fast, differentiable
    approximation of feature importance.

    E_approx_i = |X ⊙ ∂L/∂X|   (Input × Gradient saliency)

    The XAI loss is then:
      L_XAI = ‖mean_batch(E_approx_i) - E_global‖₂²

    This is computed per mini-batch and backpropagated through the model.

    Args:
        E_global:  Global reference attribution (n_features,) tensor or None.
                   If None, the loss is always 0 (no penalty applied).
        lambda_xai: Weight of XAI loss in total loss.
    """

    def __init__(
        self,
        E_global:   Optional[torch.Tensor] = None,
        lambda_xai: float = 0.1,
    ):
        super().__init__()
        self.lambda_xai = lambda_xai
        self.register_buffer(
            "E_global",
            E_global.clone().float() if E_global is not None
            else torch.zeros(1),
        )
        self._E_global_initialized = (E_global is not None)

    def update_global(self, E_global: np.ndarray):
        """Update the global reference attribution vector."""
        t = torch.tensor(E_global, dtype=torch.float32)
        self.E_global = t.to(next(self.buffers()).device
                             if len(list(self.buffers())) > 0
                             else torch.device("cpu"))
        self._E_global_initialized = True

    def forward(
        self,
        X:      torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable XAI consistency loss.

        Args:
            X:      Input tensor (batch, features...) with requires_grad=True.
            output: Model output logits (batch, classes).

        Returns:
            Scalar XAI loss (0 if E_global not yet initialized).
        """
        if not self._E_global_initialized:
            return torch.tensor(0.0, device=X.device, requires_grad=True)

        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        # Compute gradient of scalar output w.r.t. input
        try:
            # Use max-class output as scalar for gradient
            scalar_output = output.max(dim=-1).values.sum()
            grads = torch.autograd.grad(
                outputs=scalar_output,
                inputs=X,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
        except Exception as e:
            logger.debug("XAI gradient failed: %s", e)
            return torch.tensor(0.0, device=X.device, requires_grad=True)

        if grads is None:
            return torch.tensor(0.0, device=X.device, requires_grad=True)

        # Saliency: |X ⊙ grad|
        saliency = (X * grads).abs()
        # Mean over batch → (n_features,)
        E_local = saliency.view(saliency.shape[0], -1).mean(dim=0)

        # Match dimensionality of E_global
        n_local  = E_local.numel()
        n_global = self.E_global.numel()
        E_global = self.E_global.to(X.device)

        if n_local != n_global:
            # Resize via interpolation for mismatched dims (e.g., image models)
            E_global = torch.nn.functional.interpolate(
                E_global.view(1, 1, -1),
                size=n_local, mode="linear", align_corners=False,
            ).view(-1)

        loss = torch.sum((E_local - E_global) ** 2)
        return self.lambda_xai * loss


# ═══════════════════════════════════════════════════════════════════════════════
# Global SHAP Aggregation (Server-side)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalSHAPAggregator:
    """
    Server-side aggregator for SHAP attribution vectors.

    Maintains a weighted running average of local SHAP attributions
    received from participating nodes. The aggregated E_global is
    broadcast back to all nodes at the start of each round.

    Weighted average:
      E_global = Σ_i (n_i / n) × E_i
    """

    def __init__(self, n_features: Optional[int] = None):
        self.n_features    = n_features
        self.E_global:     Optional[np.ndarray] = None
        self._round_buffer: Dict[int, Tuple[np.ndarray, int]] = {}

    def submit_local(
        self,
        node_id:  int,
        E_local:  np.ndarray,
        n_local:  int,
    ):
        """
        Accept a local SHAP vector from node i.

        Args:
            node_id: Node identifier.
            E_local: Local attribution vector.
            n_local: Number of local samples (for weighting).
        """
        self._round_buffer[node_id] = (E_local, n_local)
        if self.n_features is None:
            self.n_features = len(E_local)

    def aggregate(self) -> np.ndarray:
        """
        Compute weighted average E_global from buffered local vectors.

        E_global = Σ_i (n_i / n_total) × E_i

        Returns:
            Global attribution vector.
        """
        if not self._round_buffer:
            if self.E_global is not None:
                return self.E_global
            return np.zeros(self.n_features or 1, dtype=np.float32)

        # Align feature dimensions (some nodes may have different dims)
        n_total = sum(n for _, n in self._round_buffer.values())
        E_agg   = None

        for node_id, (E_i, n_i) in self._round_buffer.items():
            weight = n_i / max(n_total, 1)
            if E_agg is None:
                E_agg = weight * E_i
            else:
                # Handle dimension mismatch
                if len(E_i) != len(E_agg):
                    E_i_resized = np.interp(
                        np.linspace(0, 1, len(E_agg)),
                        np.linspace(0, 1, len(E_i)),
                        E_i,
                    )
                    E_agg = E_agg + weight * E_i_resized
                else:
                    E_agg = E_agg + weight * E_i

        self.E_global = E_agg.astype(np.float32)
        self._round_buffer.clear()

        logger.debug(
            "GlobalSHAP updated: top-3 features = %s",
            np.argsort(self.E_global)[-3:][::-1].tolist(),
        )
        return self.E_global

    def get_global(self) -> Optional[np.ndarray]:
        """Return current global SHAP vector (None if not yet computed)."""
        return self.E_global

    def get_feature_ranking(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return top-k most important global features."""
        if self.E_global is None:
            return []
        names = feature_names or [f"feature_{i}" for i in range(len(self.E_global))]
        ranked = np.argsort(self.E_global)[::-1][:top_k]
        return [(names[i], float(self.E_global[i])) for i in ranked]
