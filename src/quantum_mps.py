"""
src/quantum_mps.py
===============================================================================
Phase 2: Quantum-Inspired Tensor Compression via Matrix Product States (MPS)

Implements the Tensor Train (TT) / Matrix Product State decomposition as a
drop-in replacement for nn.Linear layers. This compresses parameter count from
O(d_in x d_out) down to O(r² x log d) where r is the bond (virtual) rank.

Mathematical formulation (from HOD PAPER §2):
  W ~ Σ_k G_k^i,  G_k^i ∈ R^{r_{k-1} x I_k x r_k}
  Bond ranks: r=4 (tabular/sensor), r=6 (vision)

Key classes:
  MPSLinear     — Drop-in replacement for nn.Linear using MPS cores
  MPSConv       — MPS-compressed 1D convolution for temporal models
  TTDecomposer  — TT-SVD initialization from pre-trained dense weights

Communication: Only the MPS cores {G_k} are serialized and sent during FL,
               never the reconstructed dense weight matrix W.

Author: QI-CAF-EI Research Framework
"""

import math
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===============================================================================
# TT-SVD Initialization
# ===============================================================================

class TTDecomposer:
    """
    Tensor Train SVD (TT-SVD) decomposition.

    Decomposes a dense weight tensor into MPS/TT cores using sequential
    SVD with rank truncation. This provides a warm-start initialization
    that is far superior to random initialization for fine-tuning.

    Reference:
        Oseledets, I.V. (2011). Tensor-Train Decomposition. SIAM J. Sci. Comput.
    """

    @staticmethod
    def decompose(
        W:          torch.Tensor,
        shape:      List[int],
        bond_rank:  int,
        max_rank:   Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Decompose weight tensor W into TT cores via TT-SVD.

        Args:
            W:          Weight tensor of any shape (will be reshaped to `shape`).
            shape:      Target mode dimensions [I_1, I_2, ..., I_K].
            bond_rank:  Maximum virtual bond rank r.
            max_rank:   Override max rank (None -> use bond_rank throughout).

        Returns:
            List of K core tensors, each of shape (r_{k-1}, I_k, r_k).
            r_0 = r_K = 1 (boundary conditions).
        """
        r = bond_rank if max_rank is None else max_rank
        W_np = W.detach().cpu().numpy().reshape(shape)
        K    = len(shape)
        cores = []

        C = W_np.copy()
        r_left = 1

        for k in range(K - 1):
            n_k   = shape[k]
            # Reshape C to (r_left * n_k, -1)
            C = C.reshape(r_left * n_k, -1)
            # Truncated SVD
            try:
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
            except np.linalg.LinAlgError:
                U, S, Vt = np.linalg.svd(C + 1e-10 * np.eye(*C.shape[:2], C.shape[1]),
                                          full_matrices=False)

            r_new = min(r, len(S))
            # Truncate
            U   = U[:, :r_new]
            S   = S[:r_new]
            Vt  = Vt[:r_new, :]

            # Core k: (r_left, n_k, r_new)
            core = U.reshape(r_left, n_k, r_new)
            cores.append(torch.tensor(core, dtype=torch.float32))

            # Absorb singular values into next factor
            C       = np.diag(S) @ Vt
            r_left  = r_new

        # Last core: (r_left, I_K, 1)
        cores.append(torch.tensor(
            C.reshape(r_left, shape[-1], 1), dtype=torch.float32))

        return cores

    @staticmethod
    def reconstruct(cores: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct weight tensor from TT cores via sequential contraction.

        Args:
            cores: List of K cores, core[k] has shape (r_{k-1}, I_k, r_k).

        Returns:
            Dense tensor of shape (I_1, I_2, ..., I_K).
        """
        result = cores[0]   # (r0=1, I_1, r_1) -> squeeze to (I_1, r_1)
        result = result.squeeze(0)  # (I_1, r_1)

        for k in range(1, len(cores)):
            core = cores[k]             # (r_{k-1}, I_k, r_k)
            # result: (..., I_1..I_{k-1}, r_{k-1})
            # Contract last dim of result with first dim of core
            result = torch.tensordot(result, core, dims=([-1], [0]))
            # result: (..., I_1..I_{k-1}, I_k, r_k)

        # Squeeze trailing r_K=1 dim
        result = result.squeeze(-1)
        return result


# ===============================================================================
# MPSLinear Layer
# ===============================================================================

class MPSLinear(nn.Module):
    """
    MPS/Tensor-Train replacement for nn.Linear.

    Factorizes the weight matrix W ∈ R^{d_in x d_out} into a chain of
    K core tensors G_k ∈ R^{r_{k-1} x I_k x r_k}, where:
        - d_in  = I_1 x I_2 x ... x I_{K/2}
        - d_out = I_{K/2+1} x ... x I_K
        - r_0 = r_K = 1 (boundary conditions)

    During forward pass: W̃ = contract(G_1, ..., G_K), then y = x @ W̃ + b

    During FL: only the ParameterList of cores is serialized and transmitted.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bond_rank:    Maximum virtual bond dimension r.
        n_cores:      Number of MPS cores K.
        bias:         Include bias term.
        tt_svd_init:  Initialize via TT-SVD (better) vs random.

    Attributes:
        cores:        nn.ParameterList of K core tensors.
        compression_ratio: Dense / MPS parameter count.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bond_rank:    int = 4,
        n_cores:      int = 4,
        bias:         bool = True,
        tt_svd_init:  bool = True,
    ):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.bond_rank     = bond_rank
        self.n_cores       = n_cores

        # -- Factor dimensions ------------------------------------------------
        # Split (in_features, out_features) into n_cores mode dimensions
        self.mode_dims = self._compute_mode_dims(in_features, out_features, n_cores)
        self.total_dim = int(np.prod(self.mode_dims))

        # Pad in/out features to align with mode product
        self.padded_in  = int(np.prod(self.mode_dims[:n_cores//2]))
        self.padded_out = int(np.prod(self.mode_dims[n_cores//2:]))

        # -- Bond dimensions: [1, r, r, ..., r, 1] --------------------------
        self.ranks = self._compute_ranks(self.mode_dims, bond_rank)

        # -- Initialize cores -------------------------------------------------
        self.cores = nn.ParameterList()
        if tt_svd_init:
            self._tt_svd_initialize()
        else:
            self._random_initialize()

        # -- Bias -------------------------------------------------------------
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        logger.debug(
            "MPSLinear(%d->%d): %d dense params -> %d MPS params (%.1fx compression)",
            in_features, out_features,
            self.dense_param_count(), self.mps_param_count(),
            self.compression_ratio(),
        )

    @staticmethod
    def _compute_mode_dims(d_in: int, d_out: int, n_cores: int) -> List[int]:
        """
        Find mode dimensions that approximately factor d_in and d_out
        into n_cores/2 dimensions each.

        Strategy: factorize into approximately equal factors.
        """
        def factor_roughly(n: int, k: int) -> List[int]:
            """Factor n into k roughly equal integer factors."""
            factors = []
            remaining = n
            for i in range(k - 1):
                f = max(2, round(remaining ** (1.0 / (k - i))))
                # Find nearest factor
                while remaining % f != 0 and f > 1:
                    f -= 1
                if f <= 1:
                    f = remaining
                factors.append(f)
                remaining //= f
            factors.append(remaining)
            return factors

        half = n_cores // 2
        in_factors  = factor_roughly(d_in,  half)
        out_factors = factor_roughly(d_out, n_cores - half)
        return in_factors + out_factors

    @staticmethod
    def _compute_ranks(mode_dims: List[int], bond_rank: int) -> List[int]:
        """Compute bond dimensions [r_0=1, r_1, ..., r_{K-1}, r_K=1]."""
        K = len(mode_dims)
        ranks = [1]
        for k in range(K - 1):
            ranks.append(bond_rank)
        ranks.append(1)
        return ranks

    def _random_initialize(self):
        """Random Gaussian initialization for MPS cores."""
        K = len(self.mode_dims)
        for k in range(K):
            r_l, I_k, r_r = self.ranks[k], self.mode_dims[k], self.ranks[k + 1]
            std = 1.0 / math.sqrt(self.bond_rank * I_k)
            core = nn.Parameter(torch.randn(r_l, I_k, r_r) * std)
            self.cores.append(core)

    def _tt_svd_initialize(self):
        """TT-SVD initialization from random dense weight (warm start)."""
        # Initialize a small dense weight and decompose it
        W_dense = torch.randn(self.padded_in, self.padded_out)
        nn.init.kaiming_uniform_(W_dense, a=math.sqrt(5))
        W_flat = W_dense.reshape(self.mode_dims)

        try:
            core_tensors = TTDecomposer.decompose(
                W_flat, self.mode_dims, self.bond_rank)
            for ct in core_tensors:
                self.cores.append(nn.Parameter(ct.clone()))
        except Exception as e:
            logger.warning("TT-SVD init failed (%s), falling back to random.", e)
            self._random_initialize()

    def get_weight_matrix(self) -> torch.Tensor:
        """
        Reconstruct effective weight matrix W̃ ∈ R^{d_in x d_out}
        by contracting all MPS cores.
        """
        W = TTDecomposer.reconstruct(list(self.cores))
        if not torch.isfinite(W).all():
            W = torch.nan_to_num(W, nan=0.0, posinf=5.0, neginf=-5.0)
        # W has shape (I_1, I_2, ..., I_K) -> reshape to (padded_in, padded_out)
        W = W.reshape(self.padded_in, self.padded_out)
        # Trim to actual in/out features
        return W[:self.in_features, :self.out_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass via weight reconstruction and standard matmul.
        x: (..., in_features)
        """
        W = self.get_weight_matrix()              # (in_features, out_features)
        out = x @ W                               # (..., out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def dense_param_count(self) -> int:
        """Parameter count of equivalent dense layer."""
        return self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0)

    def mps_param_count(self) -> int:
        """Total MPS core parameter count."""
        total = sum(p.numel() for p in self.cores)
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def compression_ratio(self) -> float:
        """Ratio of dense params / MPS params (higher = better compression)."""
        mps = self.mps_param_count()
        return self.dense_param_count() / max(mps, 1)

    def get_cores_flat(self) -> torch.Tensor:
        """
        Return all cores concatenated as a 1D tensor.
        Used for serialization during federated communication.
        """
        return torch.cat([c.data.reshape(-1) for c in self.cores])

    def set_cores_from_flat(self, flat: torch.Tensor):
        """
        Set core parameters from a flattened 1D tensor.
        Used to load received server aggregation results.
        """
        offset = 0
        for core in self.cores:
            numel = core.numel()
            core.data.copy_(flat[offset:offset + numel].reshape(core.shape))
            offset += numel

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.bond_rank}, K={self.n_cores}, "
                f"compression={self.compression_ratio():.1f}x")


# ===============================================================================
# MPSConv1D Layer
# ===============================================================================

class MPSConv1D(nn.Module):
    """
    MPS-compressed 1D Convolution for temporal/sensor data (WISDM).

    Replaces the weight tensor W ∈ R^{C_out x C_in x kernel_size}
    with an MPS factorization over (C_out, C_in, kernel_size).

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
        kernel_size:  Convolutional kernel size.
        bond_rank:    Virtual bond dimension.
        stride:       Convolution stride.
        padding:      Padding.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        bond_rank:    int = 4,
        stride:       int = 1,
        padding:      int = 0,
        bias:         bool = True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.bond_rank    = bond_rank

        # MPS over flattened weight dimensions
        self.mps_linear = MPSLinear(
            in_features  = in_channels * kernel_size,
            out_features = out_channels,
            bond_rank    = bond_rank,
            n_cores      = 4,
            bias         = False,
            tt_svd_init  = True,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def get_weight(self) -> torch.Tensor:
        """Reconstruct conv weight tensor (out_channels, in_channels, kernel_size)."""
        W = self.mps_linear.get_weight_matrix()   # (C_in * K, C_out)
        W = W.T.reshape(self.out_channels, self.in_channels, self.kernel_size)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, length)"""
        W = self.get_weight()
        return F.conv1d(x, W, self.bias, self.stride, self.padding)

    def compression_ratio(self) -> float:
        dense = self.out_channels * self.in_channels * self.kernel_size
        mps   = self.mps_linear.mps_param_count()
        return dense / max(mps, 1)


# ===============================================================================
# MPS Utilities
# ===============================================================================

def replace_linear_with_mps(
    model:      nn.Module,
    bond_rank:  int = 4,
    n_cores:    int = 4,
    min_size:   int = 64,
    exclude:    Optional[List[str]] = None,
) -> nn.Module:
    """
    Recursively replace all nn.Linear layers in a model with MPSLinear.

    Only replaces layers where min(in_features, out_features) >= min_size
    to avoid compressing tiny layers where overhead exceeds benefit.

    Args:
        model:     PyTorch model to modify in-place.
        bond_rank: MPS bond rank.
        n_cores:   Number of MPS cores.
        min_size:  Minimum feature dimension to replace.
        exclude:   List of module names to skip.

    Returns:
        Modified model (same object, in-place).
    """
    exclude = exclude or []
    replaced_count = 0
    total_dense    = 0
    total_mps      = 0

    def _replace(parent: nn.Module, prefix: str = ""):
        nonlocal replaced_count, total_dense, total_mps
        for name, module in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if any(ex in full_name for ex in exclude):
                continue

            if isinstance(module, nn.Linear):
                if (module.in_features >= min_size and
                        module.out_features >= min_size):
                    mps_layer = MPSLinear(
                        in_features  = module.in_features,
                        out_features = module.out_features,
                        bond_rank    = bond_rank,
                        n_cores      = n_cores,
                        bias         = module.bias is not None,
                        tt_svd_init  = True,
                    )
                    total_dense += mps_layer.dense_param_count()
                    total_mps   += mps_layer.mps_param_count()
                    setattr(parent, name, mps_layer)
                    replaced_count += 1
                    logger.debug("Replaced %s (Linear %d->%d) with MPS (rank=%d)",
                                 full_name, module.in_features,
                                 module.out_features, bond_rank)
            else:
                _replace(module, full_name)

    _replace(model)
    ratio = total_dense / max(total_mps, 1)
    logger.info(
        "MPS replacement: %d layers replaced, %d dense -> %d MPS params (%.1fx overall)",
        replaced_count, total_dense, total_mps, ratio,
    )
    return model


def get_mps_state(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """
    Extract all MPS core tensors from a model for federated communication.

    Returns:
        List of (layer_name.core_k, core_data_flat) tuples.
    """
    state = []
    for name, module in model.named_modules():
        if isinstance(module, MPSLinear):
            for k, core in enumerate(module.cores):
                state.append((f"{name}.core_{k}", core.data.clone()))
    return state


def set_mps_state(model: nn.Module, state: List[Tuple[str, torch.Tensor]]):
    """
    Load MPS core tensors into a model from federated state.

    Args:
        model: Model whose MPSLinear layers will be updated.
        state: List returned by get_mps_state() (possibly from server).
    """
    state_dict = dict(state)
    for name, module in model.named_modules():
        if isinstance(module, MPSLinear):
            for k, core in enumerate(module.cores):
                key = f"{name}.core_{k}"
                if key in state_dict:
                    received = state_dict[key]
                    if not torch.isfinite(received).all():
                        received = torch.nan_to_num(received, nan=0.0, posinf=2.0, neginf=-2.0)
                    core.data.copy_(received.reshape(core.shape))


def compute_mps_compression_stats(model: nn.Module) -> dict:
    """
    Compute overall compression statistics for a model with MPS layers.

    Returns:
        Dict with total_dense_params, total_mps_params, overall_ratio,
        per_layer breakdown.
    """
    stats = {
        "total_dense_params": 0,
        "total_mps_params":   0,
        "per_layer": {},
    }
    for name, module in model.named_modules():
        if isinstance(module, MPSLinear):
            dense = module.dense_param_count()
            mps   = module.mps_param_count()
            stats["total_dense_params"] += dense
            stats["total_mps_params"]   += mps
            stats["per_layer"][name] = {
                "dense": dense, "mps": mps,
                "ratio": dense / max(mps, 1),
            }
    total_d = stats["total_dense_params"]
    total_m = stats["total_mps_params"]
    stats["overall_ratio"]        = total_d / max(total_m, 1)
    stats["parameter_reduction_pct"] = 100.0 * (1 - total_m / max(total_d, 1))
    return stats
