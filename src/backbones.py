"""
src/backbones.py
===============================================================================
Phase 2: Neural Network Backbones with MPS-Compressed Layers

Provides four model architectures, each with standard dense layers replaced
by MPSLinear for federated compression:

  MLPWithMPS     -> MIMIC-III (tabular), UCI Heart (tabular)
  BiLSTMWithMPS  -> PhysioNet 2012 (time-series aggregated features)
  ResNet50WithMPS -> NIH ChestX-ray14 (medical images)
  CNN1DWithMPS   -> WISDM IoMT (1D sensor time-series)

All models expose:
  .get_mps_cores()   — extract cores for FL communication
  .set_mps_cores()   — load received aggregated cores
  .compression_stats() — compression ratio report

Author: QI-CAF-EI Research Framework
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantum_mps import (
    MPSLinear, MPSConv1D, replace_linear_with_mps,
    get_mps_state, set_mps_state, compute_mps_compression_stats,
)

logger = logging.getLogger(__name__)


# ===============================================================================
# Base Mixin: MPS Core Management
# ===============================================================================

class MPSModelMixin:
    """
    Mixin providing MPS core extraction/loading utilities for federated learning.
    """

    def get_mps_cores(self) -> List[Tuple[str, torch.Tensor]]:
        """Return all MPS core tensors for FL communication."""
        return get_mps_state(self)

    def set_mps_cores(self, state: List[Tuple[str, torch.Tensor]]):
        """Load aggregated MPS core tensors from server."""
        set_mps_state(self, state)

    def compression_stats(self) -> dict:
        """Return compression statistics for all MPS layers."""
        return compute_mps_compression_stats(self)

    def count_parameters(self) -> int:
        """Total trainable parameters (MPS, not dense)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===============================================================================
# 1. MLP with MPS — MIMIC-III & UCI Heart
# ===============================================================================

class MLPWithMPS(nn.Module, MPSModelMixin):
    """
    3-Layer Multi-Layer Perceptron with all Linear layers replaced by MPSLinear.

    Architecture:
      Input -> MPSLinear(in->256) -> BN -> ReLU -> Dropout ->
              MPSLinear(256->128) -> BN -> ReLU -> Dropout ->
              MPSLinear(128->n_classes)

    Used for:
      - MIMIC-III (in=17, n_classes=2, binary BCE)
      - UCI Heart Disease (in~22, n_classes=2, binary BCE)

    Args:
        in_features:  Input feature dimension.
        n_classes:    Output classes (2 for binary; 1 for single-logit output).
        hidden_dims:  List of hidden dimensions (default [256, 128]).
        dropout:      Dropout rate.
        bond_rank:    MPS bond rank.
        n_cores:      MPS core count.
    """

    def __init__(
        self,
        in_features: int,
        n_classes:   int = 2,
        hidden_dims: List[int] = None,
        dropout:     float = 0.3,
        bond_rank:   int = 4,
        n_cores:     int = 4,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.in_features = in_features
        self.n_classes   = n_classes

        dims = [in_features] + hidden_dims

        layers = []
        for i in range(len(dims) - 1):
            d_in  = dims[i]
            d_out = dims[i + 1]
            # Use MPSLinear for large enough layers, nn.Linear for small
            if d_in >= 32 and d_out >= 32:
                layers.append(MPSLinear(d_in, d_out, bond_rank, n_cores))
            else:
                layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.BatchNorm1d(d_out))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))

        self.feature_extractor = nn.Sequential(*layers)

        # Output head (kept as MPSLinear if large enough)
        d_last = hidden_dims[-1]
        if d_last >= 32 and n_classes >= 2:
            self.classifier = MPSLinear(d_last, n_classes, bond_rank, n_cores)
        else:
            self.classifier = nn.Linear(d_last, n_classes)

        self._log_stats()

    def _log_stats(self):
        stats = self.compression_stats()
        logger.info(
            "MLPWithMPS: %d params, %.1fx avg compression",
            self.count_parameters(),
            stats.get("overall_ratio", 1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            logits: (batch, n_classes)
        """
        h = self.feature_extractor(x)
        return self.classifier(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate feature representation (for SHAP background)."""
        return self.feature_extractor(x)


# ===============================================================================
# 2. BiLSTM with MPS — PhysioNet 2012
# ===============================================================================

class BiLSTMWithMPS(nn.Module, MPSModelMixin):
    """
    Bidirectional LSTM with MPS-compressed fully-connected heads.

    Architecture:
      Input -> BiLSTM(hidden=64, layers=2) -> last-timestep concat ->
              MPSLinear(128->64) -> ReLU -> Dropout ->
              MPSLinear(64->n_classes)

    Note: PhysioNet 2012 features are pre-aggregated statistics (221 dims),
    so this model treats the feature vector as a 1D sequence of stats.
    For raw time-series, set use_sequence=True.

    Used for:
      - PhysioNet 2012 ICU (in=221, n_classes=2)
    """

    def __init__(
        self,
        in_features:    int,
        n_classes:      int = 2,
        hidden_size:    int = 64,
        num_layers:     int = 2,
        dropout:        float = 0.2,
        bond_rank:      int = 4,
        n_cores:        int = 4,
        use_sequence:   bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.use_sequence = use_sequence
        self.hidden_size  = hidden_size

        if use_sequence:
            # True bidirectional time-series mode
            self.lstm = nn.LSTM(
                input_size   = in_features,
                hidden_size  = hidden_size,
                num_layers   = num_layers,
                dropout      = dropout if num_layers > 1 else 0.0,
                bidirectional = True,
                batch_first  = True,
            )
            lstm_out_dim = hidden_size * 2
        else:
            # Treat flattened feature vector as single timestep
            self.input_proj = nn.Linear(in_features, hidden_size * 2)
            lstm_out_dim    = hidden_size * 2
            self.lstm       = None

        # MPS-compressed heads
        self.fc1 = (MPSLinear(lstm_out_dim, 64, bond_rank, n_cores)
                    if lstm_out_dim >= 32 else nn.Linear(lstm_out_dim, 64))
        self.bn1  = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = (MPSLinear(64, n_classes, bond_rank, n_cores)
                    if n_classes >= 2 else nn.Linear(64, n_classes))

        logger.info("BiLSTMWithMPS: %d params", self.count_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) or (batch, seq_len, in_features)
        Returns:
            logits: (batch, n_classes)
        """
        if self.use_sequence and self.lstm is not None:
            # x: (batch, seq_len, in_features)
            out, _ = self.lstm(x)
            h      = out[:, -1, :]   # Last timestep
        else:
            # x: (batch, in_features)
            h = self.input_proj(x)

        h = self.fc1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop(h)
        return self.fc2(h)


# ===============================================================================
# 3. ResNet-50 with MPS — NIH ChestX-ray14
# ===============================================================================

class ResNet50WithMPS(nn.Module, MPSModelMixin):
    """
    ResNet-50 backbone (ImageNet pretrained) with MPS-compressed classifier.

    Strategy:
      1. Load torchvision ResNet-50 with pretrained weights
      2. Freeze all backbone layers (feature extraction only)
      3. Replace final FC layer with MPSLinear for the 14-class multi-label output
      4. Add additional MPS projection layers for richer representation

    This "feature-extraction + MPS head" approach:
      - Leverages ImageNet features for medical imaging (transfer learning)
      - Only MPS layers are trained and communicated in FL
      - Massive reduction in communication cost

    Used for:
      - NIH ChestX-ray14 (n_classes=14, multi-label sigmoid output)
    """

    def __init__(
        self,
        n_classes:        int = 14,
        pretrained:       bool = True,
        freeze_backbone:  bool = True,
        bond_rank:        int = 6,    # Higher rank for vision
        n_cores:          int = 4,
        dropout:          float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Load ResNet-50
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet50(weights=weights)
        except ImportError:
            raise ImportError("torchvision required for ResNet50WithMPS.")

        # Remove final FC
        backbone_feat_dim = backbone.fc.in_features  # 2048
        backbone.fc       = nn.Identity()

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone

        # MPS-compressed head
        self.head = nn.Sequential(
            MPSLinear(backbone_feat_dim, 512, bond_rank, n_cores),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            MPSLinear(512, n_classes, bond_rank, n_cores),
        )

        logger.info(
            "ResNet50WithMPS: backbone frozen=%s, head params=%d",
            freeze_backbone,
            sum(p.numel() for p in self.head.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits: (batch, 14)  — raw logits for multi-label BCE
        """
        feats = self.backbone(x)   # (batch, 2048)
        return self.head(feats)    # (batch, 14)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (for SHAP)."""
        with torch.no_grad():
            return self.backbone(x)

    def get_mps_cores(self):
        """Only return cores from trainable head (backbone is frozen)."""
        return get_mps_state(self.head)

    def set_mps_cores(self, state):
        set_mps_state(self.head, state)

    def compression_stats(self):
        return compute_mps_compression_stats(self.head)


# ===============================================================================
# 4. 1D CNN with MPS — WISDM IoMT
# ===============================================================================

class CNN1DWithMPS(nn.Module, MPSModelMixin):
    """
    1D Convolutional Neural Network with MPS-compressed FC layers.

    Architecture:
      Input (batch, channels, window) ->
        Conv1D(ch, 64, k=5) -> BN -> ReLU ->
        Conv1D(64, 128, k=5) -> BN -> ReLU -> MaxPool ->
        Conv1D(128, 256, k=3) -> BN -> ReLU -> GlobalAvgPool ->
        MPSLinear(256->128) -> ReLU -> Dropout ->
        MPSLinear(128->n_classes)

    Standard conv layers are kept (small parameter count).
    Only FC layers use MPS compression.

    Used for:
      - WISDM IoMT (in_channels=3, window=200, n_classes=18)
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes:   int = 18,
        channels:    List[int] = None,
        kernel_size: int = 5,
        dropout:     float = 0.2,
        bond_rank:   int = 4,
        n_cores:     int = 4,
    ):
        super().__init__()
        channels = channels or [64, 128, 256]
        self.n_classes = n_classes

        # Convolutional feature extractor (standard conv, small params)
        conv_layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(channels):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i == 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch

        self.conv_features = nn.Sequential(*conv_layers)
        self.global_avg    = nn.AdaptiveAvgPool1d(1)

        feat_dim = channels[-1]

        # MPS-compressed classifier
        self.classifier = nn.Sequential(
            MPSLinear(feat_dim, 128, bond_rank, n_cores)
            if feat_dim >= 32 else nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            MPSLinear(128, n_classes, bond_rank, n_cores)
            if n_classes >= 2 else nn.Linear(128, n_classes),
        )

        logger.info("CNN1DWithMPS: %d params", self.count_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, window_size)
        Returns:
            logits: (batch, n_classes)
        """
        h = self.conv_features(x)   # (batch, 256, T')
        h = self.global_avg(h)      # (batch, 256, 1)
        h = h.squeeze(-1)           # (batch, 256)
        return self.classifier(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return conv features (for SHAP)."""
        with torch.no_grad():
            h = self.conv_features(x)
            h = self.global_avg(h).squeeze(-1)
        return h


# ===============================================================================
# Model Factory
# ===============================================================================

def build_model(
    dataset_name: str,
    cfg:          dict,
    n_features:   Optional[int] = None,
    n_classes:    Optional[int] = None,
) -> nn.Module:
    """
    Build the appropriate backbone model for a given dataset.

    Args:
        dataset_name: One of mimic3|physionet2012|chestxray14|uci_heart|wisdm
        cfg:          Full config dict from config.yaml
        n_features:   Input feature dimension (auto-inferred if None)
        n_classes:    Number of output classes (auto-inferred if None)

    Returns:
        Instantiated nn.Module with MPS layers.
    """
    mps_cfg  = cfg.get("mps", {})
    ds_cfg   = cfg.get("datasets", {}).get(dataset_name, {})
    bb_cfg   = cfg.get("backbones", {})

    rank_tabular = mps_cfg.get("bond_rank_tabular", 4)
    rank_vision  = mps_cfg.get("bond_rank_vision", 6)
    n_cores      = mps_cfg.get("n_cores", 4)

    if dataset_name == "mimic3":
        in_f  = n_features or 17
        n_cls = n_classes  or 2
        mlp   = bb_cfg.get("mlp", {})
        return MLPWithMPS(
            in_features = in_f,
            n_classes   = n_cls,
            hidden_dims = mlp.get("hidden_dims", [256, 128]),
            dropout     = mlp.get("dropout", 0.3),
            bond_rank   = rank_tabular,
            n_cores     = n_cores,
        )

    elif dataset_name == "physionet2012":
        in_f  = n_features or 221   # 36 vars x 6 stats + 5 static
        n_cls = n_classes  or 2
        bl    = bb_cfg.get("bilstm", {})
        return BiLSTMWithMPS(
            in_features = in_f,
            n_classes   = n_cls,
            hidden_size = bl.get("hidden_size", 64),
            num_layers  = bl.get("num_layers", 2),
            dropout     = bl.get("dropout", 0.2),
            bond_rank   = rank_tabular,
            n_cores     = n_cores,
        )

    elif dataset_name == "chestxray14":
        n_cls = n_classes or 14
        rn    = bb_cfg.get("resnet50", {})
        return ResNet50WithMPS(
            n_classes       = n_cls,
            pretrained      = rn.get("pretrained", True),
            freeze_backbone = rn.get("freeze_backbone", True),
            bond_rank       = rank_vision,
            n_cores         = n_cores,
        )

    elif dataset_name == "uci_heart":
        in_f  = n_features or 22   # after one-hot encoding
        n_cls = n_classes  or 2
        mlp   = bb_cfg.get("mlp", {})
        return MLPWithMPS(
            in_features = in_f,
            n_classes   = n_cls,
            hidden_dims = mlp.get("hidden_dims", [128, 64]),
            dropout     = mlp.get("dropout", 0.3),
            bond_rank   = rank_tabular,
            n_cores     = n_cores,
        )

    elif dataset_name == "wisdm":
        n_cls = n_classes or 18
        cn    = bb_cfg.get("cnn1d", {})
        return CNN1DWithMPS(
            in_channels = 3,
            n_classes   = n_cls,
            channels    = cn.get("channels", [64, 128, 256]),
            kernel_size = cn.get("kernel_size", 5),
            dropout     = cn.get("dropout", 0.2),
            bond_rank   = rank_tabular,
            n_cores     = n_cores,
        )

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")


# ===============================================================================
# Loss Functions
# ===============================================================================

def get_loss_fn(dataset_name: str, class_weights: Optional[torch.Tensor] = None):
    """
    Return the appropriate loss function for each dataset task.

    - Binary: CrossEntropyLoss (with optional class weights for imbalance)
    - Multi-label: BCEWithLogitsLoss (ChestX-ray14)
    - Multi-class: CrossEntropyLoss (WISDM)
    """
    if dataset_name == "chestxray14":
        return nn.BCEWithLogitsLoss()
    elif dataset_name in ("mimic3", "physionet2012", "uci_heart"):
        return nn.CrossEntropyLoss(weight=class_weights)
    elif dataset_name == "wisdm":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()
