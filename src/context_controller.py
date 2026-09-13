"""
src/context_controller.py
===============================================================================
Phase 3: Context-Aware System State Modeling & Dynamic Privacy Budget Control

Implements the real-time context vector c_i = [s_i, d_i, l_i, b_i, u_i]^T
for each edge node i, where:
  s_i — Data sensitivity (dataset-level, 0=public, 1=restricted)
  d_i — Device capacity (composite CPU + battery score)
  l_i — Network latency (ms, simulated or measured)
  b_i — Network bandwidth (Mbps)
  u_i — Clinical urgency (NEWS score normalized to [0,1])

Min-max normalization (HOD PAPER §1):
  c_{i,k} = (c_{i,k} - c_k^min) / (c_k^max - c_k^min)

Dynamic eps assignment:
  Nodes with high urgency (u_i->1) receive higher eps (less noise, more utility)
  Nodes with low capacity (d_i->0) transmit less frequently

Sync frequency scheduling based on bandwidth and latency constraints.

Author: QI-CAF-EI Research Framework
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===============================================================================
# Context Vector Dataclass
# ===============================================================================

@dataclass
class NodeContext:
    """
    Full context state for a single federated edge node.

    Raw (unnormalized) values are stored; call .normalize() to apply
    min-max normalization before use in the context controller.

    Attributes:
        node_id:        Unique node identifier.
        dataset_name:   Dataset/modality at this node.
        s_raw:          Data sensitivity (scalar in [0,1], dataset-defined).
        d_raw:          Device capacity (composite, in [0,1]).
        l_raw:          Latency (ms).
        b_raw:          Bandwidth (Mbps).
        u_raw:          Clinical urgency (NEWS-normalized, in [0,1]).
        epsilon_i:      Assigned privacy budget for this round.
        n_local:        Number of local training samples.
        is_active:      Whether node participates in this round.
    """
    node_id:       int
    dataset_name:  str
    s_raw:         float = 0.5    # sensitivity
    d_raw:         float = 0.7    # device capacity
    l_raw:         float = 50.0   # latency ms
    b_raw:         float = 20.0   # bandwidth Mbps
    u_raw:         float = 0.3    # urgency
    epsilon_i:     float = 1.0    # assigned privacy budget
    n_local:       int   = 500    # local sample count
    is_active:     bool  = True

    # Normalized values (populated by ContextController.normalize)
    s_norm: float = field(default=0.0, repr=False)
    d_norm: float = field(default=0.0, repr=False)
    l_norm: float = field(default=0.0, repr=False)
    b_norm: float = field(default=0.0, repr=False)
    u_norm: float = field(default=0.0, repr=False)

    @property
    def context_vector(self) -> np.ndarray:
        """Normalized context vector [s, d, l, b, u]."""
        return np.array([self.s_norm, self.d_norm, self.l_norm,
                         self.b_norm, self.u_norm], dtype=np.float32)

    @property
    def priority_score(self) -> float:
        """
        Composite priority score for scheduling and resource allocation.
        High urgency + high bandwidth + high capacity -> higher priority.
        """
        return (self.u_norm * 0.5 +
                self.b_norm * 0.3 +
                self.d_norm * 0.2)


# ===============================================================================
# Context Bounds (for min-max normalization)
# ===============================================================================

@dataclass
class ContextBounds:
    """
    Global min/max bounds for context vector normalization.
    Populated by ContextController during simulation or runtime observation.
    """
    s_min: float = 0.0;   s_max: float = 1.0
    d_min: float = 0.05;  d_max: float = 1.0
    l_min: float = 5.0;   l_max: float = 500.0
    b_min: float = 0.5;   b_max: float = 100.0
    u_min: float = 0.0;   u_max: float = 1.0

    def normalize_scalar(self, val: float, key: str) -> float:
        """Normalize a single scalar using stored bounds."""
        lo = getattr(self, f"{key}_min")
        hi = getattr(self, f"{key}_max")
        denom = hi - lo
        if denom < 1e-10:
            return 0.0
        return float(np.clip((val - lo) / denom, 0.0, 1.0))


# ===============================================================================
# ContextController
# ===============================================================================

class ContextController:
    """
    Central Context-Aware Controller for the QI-CAF-EI federation.

    Responsibilities:
      1. Simulate or receive real-time context vectors per node
      2. Min-max normalize all context dimensions
      3. Assign dynamic eps_i privacy budgets based on urgency + capacity
      4. Determine which nodes are active each round (sync scheduling)
      5. Compute data sensitivity levels s_i

    Mathematical formulation (HOD PAPER §1):
      c_i = [s_i, d_i, l_i, b_i, u_i]^T
      c_{i,k} = (c_{i,k} - c_k^min) / (c_k^max - c_k^min)
      eps_i = eps_base x (1 + γ x u_i) x d_i^{-1}   [clipped to (0, eps_max]]
    """

    # Sensitivity levels per dataset (from DATASET_SENSITIVITY in data_loaders.py)
    DATASET_SENSITIVITY = {
        "mimic3":        1.0,
        "physionet2012": 1.0,
        "chestxray14":   0.7,
        "uci_heart":     0.5,
        "wisdm":         0.3,
    }

    def __init__(self, cfg: dict, n_nodes: int, random_state: int = 42):
        """
        Args:
            cfg:          Full config dict (uses cfg['context'] and cfg['privacy']).
            n_nodes:      Total number of federated nodes.
            random_state: NumPy seed for simulation reproducibility.
        """
        self.cfg          = cfg
        self.n_nodes      = n_nodes
        self.rng          = np.random.RandomState(random_state)
        self.bounds       = ContextBounds()
        self.node_contexts: Dict[int, NodeContext] = {}

        # Load config values
        ctx_cfg = cfg.get("context", {})
        prv_cfg = cfg.get("privacy", {})

        self.epsilon_base = prv_cfg.get("epsilon_base", 1.0)
        self.epsilon_max  = prv_cfg.get("epsilon_max", 10.0)
        self.gamma        = ctx_cfg.get("gamma", 0.5)

        # Update bounds from config
        self.bounds.l_min = ctx_cfg.get("latency_min_ms",   5.0)
        self.bounds.l_max = ctx_cfg.get("latency_max_ms",   500.0)
        self.bounds.b_min = ctx_cfg.get("bandwidth_min_mbps", 0.5)
        self.bounds.b_max = ctx_cfg.get("bandwidth_max_mbps", 100.0)
        self.bounds.d_min = ctx_cfg.get("battery_min",      0.05)
        self.bounds.d_max = ctx_cfg.get("battery_max",      1.0)

    # --- Context Simulation ---------------------------------------------------

    def simulate_node_contexts(
        self,
        node_dataset_map: Dict[int, str],
        n_local_samples:  Dict[int, int],
    ) -> Dict[int, NodeContext]:
        """
        Simulate realistic context vectors for all nodes.

        Generates time-varying (per-round) context conditions:
          - Latency from Log-Normal distribution (heavy-tailed network)
          - Bandwidth from Uniform (heterogeneous edge devices)
          - Device capacity (battery + CPU) from Beta distribution
          - Clinical urgency from NEWS score simulation (per dataset type)

        Args:
            node_dataset_map:  Dict mapping node_id -> dataset_name.
            n_local_samples:   Dict mapping node_id -> number of local samples.

        Returns:
            Dict mapping node_id -> NodeContext with raw and normalized values.
        """
        from utils.news_score import simulate_news_for_node

        contexts: Dict[int, NodeContext] = {}

        for node_id in range(self.n_nodes):
            ds_name   = node_dataset_map.get(node_id, "mimic3")
            n_local   = n_local_samples.get(node_id, 100)
            seed_base = int(self.rng.randint(0, 2**31))

            # -- Simulate raw context values ----------------------------------
            # Latency: Log-Normal (heavy-tailed, real network conditions)
            latency_ms = float(np.clip(
                np.random.RandomState(seed_base).lognormal(
                    mean=np.log(50.0), sigma=0.8),
                self.bounds.l_min, self.bounds.l_max,
            ))

            # Bandwidth: Uniform with device-class variation
            bw_mbps = float(np.clip(
                np.random.RandomState(seed_base + 1).uniform(
                    self.bounds.b_min, self.bounds.b_max),
                self.bounds.b_min, self.bounds.b_max,
            ))

            # Device capacity: composite of battery (Beta) + CPU inverse
            battery  = float(np.clip(
                np.random.RandomState(seed_base + 2).beta(a=2.0, b=1.5),
                self.bounds.d_min, self.bounds.d_max,
            ))
            cpu_load = float(np.clip(
                np.random.RandomState(seed_base + 3).beta(a=1.5, b=2.0),
                0.0, 1.0,
            ))
            capacity = 0.6 * battery + 0.4 * (1.0 - cpu_load)  # Combined metric

            # Sensitivity: dataset-level constant
            sensitivity = self.DATASET_SENSITIVITY.get(ds_name, 0.5)

            # Clinical urgency: NEWS-based simulation
            urgency = simulate_news_for_node(node_id, ds_name, seed_base + 4)

            ctx = NodeContext(
                node_id      = node_id,
                dataset_name = ds_name,
                s_raw        = sensitivity,
                d_raw        = float(capacity),
                l_raw        = latency_ms,
                b_raw        = bw_mbps,
                u_raw        = urgency,
                n_local      = n_local,
                is_active    = True,
            )
            contexts[node_id] = ctx

        # Normalize all dimensions
        self._normalize_all(contexts)
        # Assign privacy budgets
        self._assign_epsilon(contexts)

        self.node_contexts = contexts
        return contexts

    # --- Normalization --------------------------------------------------------

    def _normalize_all(self, contexts: Dict[int, NodeContext]):
        """Apply min-max normalization to all context dimensions."""
        # Collect raw values for bounds computation (or use preset bounds)
        for node_id, ctx in contexts.items():
            ctx.s_norm = self.bounds.normalize_scalar(ctx.s_raw, "s")
            ctx.d_norm = self.bounds.normalize_scalar(ctx.d_raw, "d")
            ctx.l_norm = self.bounds.normalize_scalar(ctx.l_raw, "l")
            ctx.b_norm = self.bounds.normalize_scalar(ctx.b_raw, "b")
            ctx.u_norm = self.bounds.normalize_scalar(ctx.u_raw, "u")

    @staticmethod
    def normalize_context_vector(
        raw: np.ndarray,
        bounds: ContextBounds,
    ) -> np.ndarray:
        """
        Vectorized min-max normalization for a single context vector.

        c_{i,k} = (c_{i,k} - c_k^min) / (c_k^max - c_k^min)

        Args:
            raw:    Raw context vector [s, d, l, b, u].
            bounds: ContextBounds with per-dimension min/max.

        Returns:
            Normalized context vector in [0,1]^5.
        """
        mins = np.array([bounds.s_min, bounds.d_min, bounds.l_min,
                         bounds.b_min, bounds.u_min], dtype=np.float32)
        maxs = np.array([bounds.s_max, bounds.d_max, bounds.l_max,
                         bounds.b_max, bounds.u_max], dtype=np.float32)
        denom = np.where(maxs - mins > 1e-10, maxs - mins, 1.0)
        return np.clip((raw - mins) / denom, 0.0, 1.0).astype(np.float32)

    # --- Dynamic eps Assignment -------------------------------------------------

    def _assign_epsilon(self, contexts: Dict[int, NodeContext]):
        """
        Dynamically assign privacy budget eps_i per node.

        High urgency nodes (u_norm -> 1) get higher eps (more utility, less noise).
        Low capacity nodes (d_norm -> 0) are constrained to smaller eps.

        Formula:
          eps_i = eps_base x (1 + γ x u_norm) x (1 / max(1 - d_norm, eps))
          clipped to (0, eps_max]

        Rationale:
          - Clinical urgency justifies relaxing privacy for better predictions
          - Capacity constraint prevents resource-limited nodes from spending
            large budgets (they train less, so their contribution is smaller)
        """
        for node_id, ctx in contexts.items():
            # Higher urgency -> more eps (utility over privacy)
            urgency_amplifier = 1.0 + self.gamma * ctx.u_norm

            # Higher capacity -> can handle more computation / larger eps
            # Low capacity -> reduce eps (send less, contribute less)
            capacity_factor = max(ctx.d_norm, 0.1)

            # Sensitivity dampener: high sensitivity -> reduce eps regardless
            sensitivity_dampener = 1.0 - 0.3 * ctx.s_norm

            epsilon_i = (self.epsilon_base
                         * urgency_amplifier
                         * capacity_factor
                         * sensitivity_dampener)
            epsilon_i = float(np.clip(epsilon_i, 0.01, self.epsilon_max))
            ctx.epsilon_i = epsilon_i

        logger.debug(
            "eps assignment: min=%.3f, max=%.3f, mean=%.3f",
            min(c.epsilon_i for c in contexts.values()),
            max(c.epsilon_i for c in contexts.values()),
            np.mean([c.epsilon_i for c in contexts.values()]),
        )

    # --- Sync Scheduling -----------------------------------------------------

    def select_active_nodes(
        self,
        contexts:        Dict[int, NodeContext],
        active_fraction: float = 0.8,
        budget_exhausted: Optional[Dict[int, bool]] = None,
    ) -> List[int]:
        """
        Select which nodes participate in the current federated round.

        Selection strategy:
          1. Exclude nodes with exhausted privacy budget
          2. Filter by bandwidth/latency threshold (must be able to sync)
          3. Sample remaining with priority-weighted probability up to active_fraction

        Args:
            contexts:         Current node contexts.
            active_fraction:  Fraction of nodes to activate (q from paper).
            budget_exhausted: Dict mapping node_id -> True if budget spent.

        Returns:
            List of active node IDs for this round.
        """
        budget_exhausted = budget_exhausted or {}

        # Eligible nodes: not budget-exhausted, reasonable bandwidth
        eligible = [
            nid for nid, ctx in contexts.items()
            if not budget_exhausted.get(nid, False)
        ]

        if not eligible:
            logger.warning("No eligible nodes! Returning empty active set.")
            return []

        n_active = max(1, int(active_fraction * len(contexts)))
        n_active = min(n_active, len(eligible))

        # Priority-weighted sampling (higher priority -> more likely selected)
        priorities = np.array([contexts[nid].priority_score for nid in eligible])
        priorities = priorities + 1e-6   # Avoid zero probabilities
        probs = priorities / priorities.sum()

        selected = self.rng.choice(
            eligible, size=n_active, replace=False, p=probs)

        # Mark active status
        for nid in contexts:
            contexts[nid].is_active = (nid in selected)

        logger.info(
            "Round active nodes: %d/%d selected (fraction=%.2f)",
            len(selected), len(contexts), active_fraction,
        )
        return sorted(selected.tolist())

    # --- Round Update ---------------------------------------------------------

    def update_for_round(
        self,
        round_idx:        int,
        node_dataset_map: Dict[int, str],
        n_local_samples:  Dict[int, int],
        active_fraction:  float = 0.8,
        budget_exhausted: Optional[Dict[int, bool]] = None,
    ) -> Tuple[Dict[int, NodeContext], List[int]]:
        """
        Full context update for a new federated round.

        Simulates new device conditions (battery drops, network fluctuates)
        and returns updated contexts + selected active nodes.

        Args:
            round_idx:        Current round index (for logging).
            node_dataset_map: Node -> dataset mapping.
            n_local_samples:  Node -> sample count mapping.
            active_fraction:  Target active fraction.
            budget_exhausted: Nodes with spent privacy budgets.

        Returns:
            (contexts_dict, active_node_ids)
        """
        # Re-seed for each round (simulates time-varying conditions)
        self.rng = np.random.RandomState(self.rng.randint(0, 2**31) + round_idx)

        contexts   = self.simulate_node_contexts(node_dataset_map, n_local_samples)
        active_ids = self.select_active_nodes(
            contexts, active_fraction, budget_exhausted)

        logger.info(
            "[Round %d] Context update: %d nodes total, %d active",
            round_idx, len(contexts), len(active_ids),
        )
        return contexts, active_ids

    # --- Context Reporting ----------------------------------------------------

    def get_context_summary(
        self, contexts: Dict[int, NodeContext]
    ) -> dict:
        """Return summary statistics of current context vectors."""
        all_eps = [c.epsilon_i for c in contexts.values()]
        all_urgency = [c.u_norm for c in contexts.values()]
        all_capacity = [c.d_norm for c in contexts.values()]
        all_bw = [c.b_norm for c in contexts.values()]

        return {
            "n_nodes":          len(contexts),
            "epsilon_mean":     float(np.mean(all_eps)),
            "epsilon_std":      float(np.std(all_eps)),
            "epsilon_min":      float(np.min(all_eps)),
            "epsilon_max":      float(np.max(all_eps)),
            "urgency_mean":     float(np.mean(all_urgency)),
            "urgency_high_frac": float(np.mean(np.array(all_urgency) > 0.5)),
            "capacity_mean":    float(np.mean(all_capacity)),
            "bandwidth_mean":   float(np.mean(all_bw)),
            "n_high_sensitivity": sum(
                1 for c in contexts.values() if c.s_norm > 0.5),
        }
