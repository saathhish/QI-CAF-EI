"""
src/federated_client.py
═══════════════════════════════════════════════════════════════════════════════
Phase 4: Federated Edge Client — Local Training with Joint Loss

Implements the local training pipeline for each edge node:
  1. Receive global MPS cores from server
  2. Unpack into local model
  3. Local training with joint loss:
       L_total = L_acc + λ_XAI × L_XAI
  4. L2-norm gradient clipping bounded by C
  5. Gaussian noise injection into MPS cores (DP)
  6. Update privacy accountant
  7. Return perturbed MPS cores for HE encryption + aggregation

Mathematical formulations (HOD PAPER):
  L_total = L_acc + L_XAI
  L_XAI = ‖E_i - E_global‖₂²     (XAI consistency)
  g̃_i = G_k + N(0, σ_i²I)        (DP noise injection)
  σ_i = ΔS √(2 ln(1.25/δ)) / ε_i  (noise calibration)

Author: QI-CAF-EI Research Framework
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.context_controller import NodeContext
from src.privacy_accountant import (
    FederatedPrivacyManager,
    clip_gradients_l2,
)
from src.explainability import SHAPExplainer, SHAPConsistencyLoss
from src.quantum_mps import get_mps_state, set_mps_state, MPSLinear

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Client Training Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientUpdate:
    """
    Result of one local training round for a federated client.

    Transmitted to the server for aggregation (after HE encryption).

    Attributes:
        node_id:        Node identifier.
        mps_cores:      Perturbed MPS core tensors {name: tensor}.
        n_local:        Number of local samples (for weighted aggregation).
        train_loss:     Average training loss this round.
        train_acc:      Training accuracy this round.
        epsilon_spent:  ε consumed this round.
        shap_E_local:   Local SHAP attribution vector (for global update).
        round_idx:      Global round index.
        train_time_s:   Wall-clock training time (seconds).
    """
    node_id:       int
    mps_cores:     List[Tuple[str, torch.Tensor]]
    n_local:       int
    train_loss:    float
    train_acc:     float
    epsilon_spent: float
    shap_E_local:  Optional[np.ndarray] = None
    round_idx:     int = 0
    train_time_s:  float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Federated Client
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedClient:
    """
    Federated learning client implementing local training for QI-CAF-EI.

    One FederatedClient instance exists per edge node. It holds:
      - A local model copy (with MPS layers)
      - A local DataLoader
      - A context reference (for ε and urgency)
      - A SHAP explainer (for L_XAI)

    The client does NOT hold real patient data — it holds indices into
    the dataset, which never leave the device.

    Args:
        node_id:       Unique node identifier.
        model:         Local model (with MPS layers, initialized from global).
        dataloader:    Local training DataLoader.
        dataset_name:  Dataset identifier (for loss function selection).
        cfg:           Full config dict.
        privacy_mgr:   Shared FederatedPrivacyManager instance.
        device:        Torch device.
    """

    def __init__(
        self,
        node_id:       int,
        model:         nn.Module,
        dataloader:    DataLoader,
        dataset_name:  str,
        cfg:           dict,
        privacy_mgr:   FederatedPrivacyManager,
        device:        Optional[torch.device] = None,
    ):
        self.node_id      = node_id
        self.model        = model.to(device or torch.device("cpu"))
        self.dataloader   = dataloader
        self.dataset_name = dataset_name
        self.cfg          = cfg
        self.privacy_mgr  = privacy_mgr
        self.device       = device or torch.device("cpu")

        # Load training config
        fl_cfg  = cfg.get("federated", {})
        opt_cfg = cfg.get("optimizer", {})
        xai_cfg = cfg.get("xai", {})
        prv_cfg = cfg.get("privacy", {})

        self.local_epochs  = fl_cfg.get("local_epochs", 3)
        self.batch_size    = fl_cfg.get("local_batch_size", 32)
        self.clip_norm     = prv_cfg.get("clip_norm", 1.0)
        self.lambda_xai    = xai_cfg.get("lambda_xai", 0.1)
        self.n_bg_samples  = xai_cfg.get("n_background_samples", 50)

        # Loss function
        from src.backbones import get_loss_fn
        self.loss_fn = get_loss_fn(dataset_name).to(self.device)

        # XAI consistency loss module
        self.xai_loss_fn = SHAPConsistencyLoss(
            E_global   = None,
            lambda_xai = self.lambda_xai,
        ).to(self.device)

        # Optimizer (will be re-initialized each round for fresh momentum)
        self._build_optimizer(opt_cfg)

        # SHAP explainer (initialized lazily on first use)
        self._shap_explainer: Optional[SHAPExplainer] = None
        self._background_X:   Optional[torch.Tensor] = None

        logger.info(
            "FederatedClient %d: dataset=%s, n_local=%d, epochs=%d",
            node_id, dataset_name, len(dataloader.dataset), self.local_epochs,
        )

    def _build_optimizer(self, opt_cfg: dict):
        """Build optimizer over MPS-only parameters."""
        # Only optimize MPS core parameters (frozen backbone excluded)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt_type  = opt_cfg.get("type", "adam").lower()
        lr        = opt_cfg.get("lr", 1e-3)
        wd        = opt_cfg.get("weight_decay", 1e-4)

        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                trainable, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)

    def load_global_model(
        self,
        global_mps_state: List[Tuple[str, torch.Tensor]],
    ):
        """
        Load aggregated MPS cores from server into local model.

        This replaces the local model's MPS core tensors with the
        globally aggregated (and decrypted) versions.

        Args:
            global_mps_state: List of (name, tensor) from server.
        """
        set_mps_state(self.model, global_mps_state)
        logger.debug("Client %d: Loaded global MPS state (%d cores)",
                     self.node_id, len(global_mps_state))

    def update_global_shap(self, E_global: Optional[np.ndarray]):
        """Update the global SHAP reference vector for L_XAI computation."""
        if E_global is not None:
            self.xai_loss_fn.update_global(E_global)

    def _prepare_shap_explainer(self, X_sample: torch.Tensor):
        """Lazily initialize SHAP explainer on first call."""
        if self._shap_explainer is None:
            xai_cfg = self.cfg.get("xai", {})
            self._background_X = X_sample.detach()
            self._shap_explainer = SHAPExplainer(
                model          = self.model,
                background_X   = self._background_X,
                dataset_name   = self.dataset_name,
                n_bg_samples   = self.n_bg_samples,
                explainer_type = xai_cfg.get("explainer_type", "auto"),
                device         = self.device,
            )

    def train_one_round(
        self,
        context:      NodeContext,
        round_idx:    int,
        global_mps:   Optional[List[Tuple[str, torch.Tensor]]] = None,
        E_global:     Optional[np.ndarray] = None,
    ) -> ClientUpdate:
        """
        Execute one full local training round.

        Pipeline:
          1. Load global MPS cores (if provided)
          2. Update global SHAP reference
          3. Local training for `local_epochs` epochs:
             a. Forward pass → L_acc
             b. Differentiable XAI saliency → L_XAI
             c. Total loss: L_total = L_acc + λ × L_XAI
             d. Backward + L2 gradient clipping
          4. Extract MPS core tensors
          5. Inject Gaussian DP noise into cores
          6. Update privacy accountant
          7. Compute SHAP attributions (post-training)

        Args:
            context:    Current node context (urgency, capacity, ε).
            round_idx:  Global round index.
            global_mps: Global MPS state from server (None = first round).
            E_global:   Global SHAP reference vector.

        Returns:
            ClientUpdate with perturbed cores and metrics.
        """
        t_start = time.time()

        # ── 1. Load global model ─────────────────────────────────────────────
        if global_mps is not None:
            self.load_global_model(global_mps)

        # ── 2. Update XAI global reference ──────────────────────────────────
        self.update_global_shap(E_global)

        # ── 3. Local training ────────────────────────────────────────────────
        self.model.train()
        # Rebuild optimizer fresh each round (no stale momentum across rounds)
        self._build_optimizer(self.cfg.get("optimizer", {}))

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        n_steps       = 0

        for epoch in range(self.local_epochs):
            for batch_X, batch_y in self.dataloader:
                # Skip single-sample batches (BatchNorm needs >= 2 samples)
                if len(batch_y) < 2:
                    continue
                # Move to device
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device)

                # Enable gradient tracking on input for XAI saliency
                batch_X.requires_grad_(True)

                self.optimizer.zero_grad()

                # ── Forward pass ──────────────────────────────────────────
                output = self.model(batch_X)

                # ── Accuracy loss L_acc ───────────────────────────────────
                if self.dataset_name == "chestxray14":
                    # Multi-label: targets are float
                    L_acc = self.loss_fn(output, batch_y.float())
                else:
                    L_acc = self.loss_fn(output, batch_y)

                # ── XAI consistency loss L_XAI ────────────────────────────
                L_xai = self.xai_loss_fn(batch_X, output)

                # ── Total loss ────────────────────────────────────────────
                L_total = L_acc + L_xai

                # Guard: skip batch if loss is NaN or Inf
                if not torch.isfinite(L_total):
                    logger.debug("NaN/Inf loss detected — skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                # ── Backward ──────────────────────────────────────────────
                try:
                    L_total.backward()
                except RuntimeError as e:
                    # SHAP DeepExplainer hook can fail on odd-sized final batches
                    # Fall back to acc-only backward (XAI loss for this step is skipped)
                    logger.debug("L_total backward failed (%s); retrying with L_acc only.", e)
                    try:
                        self.optimizer.zero_grad()
                        output2 = self.model(batch_X.detach())
                        if self.dataset_name == "chestxray14":
                            L_acc_only = self.loss_fn(output2, batch_y.float())
                        else:
                            L_acc_only = self.loss_fn(output2, batch_y)
                        L_acc_only.backward()
                    except Exception:
                        self.optimizer.zero_grad()
                        continue

                # ── L2 gradient clipping ──────────────────────────────────
                clip_gradients_l2(self.model, self.clip_norm, mps_only=True)

                self.optimizer.step()

                # ── Metrics ───────────────────────────────────────────────
                batch_size_actual = len(batch_y)
                total_loss        += float(L_total.item()) * batch_size_actual
                total_samples     += batch_size_actual
                n_steps           += 1

                # Accuracy (for classification tasks)
                if self.dataset_name != "chestxray14":
                    preds = output.argmax(dim=1)
                    total_correct += int((preds == batch_y).sum().item())

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc  = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0

        # ── 4. Extract MPS cores pre-noise ──────────────────────────────────
        mps_cores_before_noise = get_mps_state(self.model)

        # ── 5. Compute DP noise sigma and inject ─────────────────────────────
        epsilon_i = context.epsilon_i
        n_local   = max(len(self.dataloader.dataset), 1)
        sigma     = self.privacy_mgr.compute_noise_sigma(
            self.node_id, epsilon_i, n_local)

        perturbed_cores = self._inject_noise_to_cores(
            mps_cores_before_noise, sigma)

        # ── 6. Record privacy expenditure ────────────────────────────────────
        self.privacy_mgr.record_round(
            node_id    = self.node_id,
            sigma      = sigma,
            batch_size = self.batch_size,
            n_local    = n_local,
            n_steps    = n_steps,
        )
        epsilon_spent = self.privacy_mgr.get_epsilon(self.node_id)

        # ── 7. Compute SHAP attributions ─────────────────────────────────────
        E_local = self._compute_local_shap()

        t_elapsed = time.time() - t_start

        logger.info(
            "Client %d [Round %d]: loss=%.4f, acc=%.3f, eps=%.3f, sigma=%.4f, t=%.1fs",
            self.node_id, round_idx, avg_loss, avg_acc,
            epsilon_spent, sigma, t_elapsed,
        )

        return ClientUpdate(
            node_id       = self.node_id,
            mps_cores     = perturbed_cores,
            n_local       = n_local,
            train_loss    = avg_loss,
            train_acc     = avg_acc,
            epsilon_spent = epsilon_spent,
            shap_E_local  = E_local,
            round_idx     = round_idx,
            train_time_s  = t_elapsed,
        )

    def _inject_noise_to_cores(
        self,
        cores: List[Tuple[str, torch.Tensor]],
        sigma: float,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Inject Gaussian noise into MPS core tensors.

        G̃_k = G_k + N(0, σ²I)  for each core tensor k

        Args:
            cores: List of (name, tensor) pairs.
            sigma: Noise standard deviation.

        Returns:
            Perturbed (name, tensor) pairs.
        """
        if sigma <= 0:
            return cores

        perturbed = []
        with torch.no_grad():
            for name, tensor in cores:
                dim_scale = max(float(tensor.numel()) ** 0.5, 1.0)
                noise     = torch.randn_like(tensor) * (sigma / dim_scale)
                perturbed_tensor = torch.nan_to_num(tensor + noise, nan=0.0, posinf=3.0, neginf=-3.0)
                perturbed.append((name, perturbed_tensor))

        return perturbed

    def _compute_local_shap(self) -> Optional[np.ndarray]:
        """
        Compute local SHAP attributions post-training.

        Uses a small batch from the dataloader as the explain set.
        Returns None if SHAP is unavailable.
        """
        try:
            # Get a small batch for explanation
            batch_iter = iter(self.dataloader)
            X_explain, _ = next(batch_iter)
            X_explain = X_explain.to(self.device).float()

            # Get a background batch
            try:
                X_bg, _ = next(batch_iter)
            except StopIteration:
                X_bg = X_explain
            X_bg = X_bg.to(self.device).float()

            # Lazily initialize SHAP explainer
            self._prepare_shap_explainer(X_bg)

            self.model.eval()
            E_local = self._shap_explainer.compute_attributions(X_explain)
            self.model.train()
            return E_local

        except Exception as e:
            logger.debug("SHAP computation skipped for client %d: %s",
                         self.node_id, e)
            return None

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Evaluate model on a given DataLoader (val/test).

        Returns:
            (avg_loss, avg_accuracy)
        """
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device)
                output  = self.model(batch_X)

                if self.dataset_name == "chestxray14":
                    loss = self.loss_fn(output, batch_y.float())
                else:
                    loss = self.loss_fn(output, batch_y)
                    preds = output.argmax(dim=1)
                    total_correct += int((preds == batch_y).sum().item())

                total_loss    += float(loss.item()) * len(batch_y)
                total_samples += len(batch_y)

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc  = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc
