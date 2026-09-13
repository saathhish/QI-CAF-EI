"""
main_train.py
═══════════════════════════════════════════════════════════════════════════════
QI-CAF-EI: Main Federated Training Orchestrator

This is the central entry point for the QI-CAF-EI federated learning system.
It implements the complete federated learning loop across all phases:

  Phase 1: Dataset loading + non-IID partitioning (Dirichlet + Log-Normal)
  Phase 2: Model initialization with MPS compression
  Phase 3: Context controller + dynamic ε assignment
  Phase 4: Local training (L_acc + L_XAI) + DP noise + HE aggregation

Federated Loop (R rounds, q=0.8 active fraction):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ For round r = 1..R:                                                     │
  │   1. Context update → simulate node conditions → assign ε_i             │
  │   2. Select q×N active clients (priority-weighted)                       │
  │   3. Broadcast: global MPS cores + E_global (SHAP reference)            │
  │   4. Clients: local training → gradient clipping → DP noise injection   │
  │   5. Clients: HE encrypt perturbed cores → transmit to server           │
  │   6. Server: weighted HE sum → decrypt C_global                         │
  │   7. Server: update E_global from received SHAP vectors                 │
  │   8. Evaluate on global val set → log metrics → PAM                     │
  └─────────────────────────────────────────────────────────────────────────┘

Usage:
  # Full training with synthetic data (no real datasets required):
  python main_train.py --dataset mimic3 --nodes 20 --rounds 50

  # Smoke test (fast, 5 nodes, 3 rounds):
  python main_train.py --dataset mimic3 --nodes 5 --rounds 3 --smoke-test

  # With real dataset path:
  python main_train.py --dataset mimic3 --data-path data/raw/mimic3 --nodes 50

  # Multiple datasets in sequence:
  python main_train.py --dataset all --nodes 20 --rounds 30

Author: QI-CAF-EI Research Framework
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows (avoids cp1252 Unicode errors)
if sys.platform == "win32":
    import io, os as _os
    # Reconfigure stdout/stderr to UTF-8
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Also set env var for child processes
    _os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
import yaml

# ─── Setup Python path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ─── Internal imports ─────────────────────────────────────────────────────────
from src.data_loaders import (
    build_federated_dataloaders,
    DATASET_REGISTRY,
    DATASET_SENSITIVITY,
)
from src.backbones import build_model, get_loss_fn
from src.quantum_mps import (
    get_mps_state, set_mps_state, compute_mps_compression_stats
)
from src.context_controller import ContextController, NodeContext
from src.privacy_accountant import FederatedPrivacyManager
from src.federated_client import FederatedClient, ClientUpdate
from src.secure_aggregator import SecureAggregator, encrypt_client_update
from src.explainability import GlobalSHAPAggregator
from utils.metrics import (
    FederatedMetricsTracker, evaluate_model, compute_communication_metrics
)

# ─── Logging setup ────────────────────────────────────────────────────────────
def setup_logging(log_dir: str = "logs/", level: str = "INFO"):
    """Configure colored console + file logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")),
    ]

    try:
        import colorlog
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(name)s: %(message)s",
            log_colors={"DEBUG": "cyan", "INFO": "green",
                        "WARNING": "yellow", "ERROR": "red"},
        )
        handlers[0].setFormatter(formatter)
    except ImportError:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handlers[0].setFormatter(formatter)

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers[1].setFormatter(file_formatter)

    logging.basicConfig(level=log_level, handlers=handlers)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)

logger = logging.getLogger("qi-caf-ei")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values from command-line arguments."""
    if hasattr(args, "nodes") and args.nodes:
        cfg.setdefault("federated", {})["n_nodes"] = args.nodes
    if hasattr(args, "rounds") and args.rounds:
        cfg.setdefault("federated", {})["global_rounds"] = args.rounds
    if hasattr(args, "alpha") and args.alpha:
        cfg.setdefault("partition", {})["dirichlet_alpha"] = args.alpha
    if hasattr(args, "epsilon") and args.epsilon:
        cfg.setdefault("privacy", {})["epsilon_base"] = args.epsilon
    if hasattr(args, "smoke_test") and args.smoke_test:
        cfg.setdefault("datasets", {}).setdefault("mimic3", {})["use_synthetic"] = True
        cfg["federated"]["n_nodes"]      = min(cfg["federated"].get("n_nodes", 5), 5)
        cfg["federated"]["global_rounds"] = 3
        cfg["federated"]["local_epochs"]  = 1
        cfg.setdefault("xai", {})["lambda_xai"] = 0.0   # Disable XAI for speed
        logger.info("SMOKE TEST mode: n_nodes=5, rounds=3, epochs=1")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Single Dataset Training Run
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_training(
    dataset_name: str,
    cfg:          dict,
    device:       torch.device,
    output_dir:   str = "results/",
) -> FederatedMetricsTracker:
    """
    Execute the complete federated training loop for a single dataset.

    Args:
        dataset_name: One of mimic3|physionet2012|chestxray14|uci_heart|wisdm
        cfg:          Configuration dictionary.
        device:       Torch device.
        output_dir:   Directory for saving results.

    Returns:
        FederatedMetricsTracker with full round history.
    """
    fl_cfg     = cfg.get("federated", {})
    part_cfg   = cfg.get("partition", {})
    xai_cfg    = cfg.get("xai", {})
    out_cfg    = cfg.get("output", {})

    n_nodes        = fl_cfg.get("n_nodes", 20)
    global_rounds  = fl_cfg.get("global_rounds", 50)
    active_frac    = fl_cfg.get("active_fraction", 0.80)
    batch_size     = fl_cfg.get("local_batch_size", 32)
    seed           = fl_cfg.get("seed", 42)
    alpha          = part_cfg.get("dirichlet_alpha", 0.5)
    sigma2_qty     = part_cfg.get("lognormal_sigma2", 0.5)
    save_every     = out_cfg.get("save_every_n_rounds", 10)
    update_shap_every = xai_cfg.get("update_global_shap_every", 5)

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "\n====================================================\n"
        "  QI-CAF-EI Federated Training\n"
        "  Dataset:  %s\n"
        "  Nodes:    %d (active=%.0f%%)\n"
        "  Rounds:   %d\n"
        "  Device:   %s\n"
        "=====================================================",
        dataset_name, n_nodes, active_frac * 100, global_rounds, device,
    )

    # ── Phase 1: Build DataLoaders ───────────────────────────────────────────
    logger.info("[Phase 1] Building federated data partitions...")
    node_loaders, val_loader, test_loader = build_federated_dataloaders(
        dataset_name     = dataset_name,
        cfg              = cfg,
        n_nodes          = n_nodes,
        batch_size       = batch_size,
        dirichlet_alpha  = alpha,
        lognormal_sigma2 = sigma2_qty,
        random_state     = seed,
    )

    # Determine n_features and n_classes from first batch
    sample_X, sample_y = next(iter(node_loaders[0]))
    n_features = sample_X.shape[1] if sample_X.ndim == 2 else None
    n_classes  = cfg["datasets"][dataset_name].get("n_classes", None)
    if n_classes is None:
        if dataset_name in ("mimic3", "physionet2012", "uci_heart"):
            n_classes = 2
        elif dataset_name == "chestxray14":
            n_classes = 14
        elif dataset_name == "wisdm":
            n_classes = 18

    # Per-node sample counts (for DP sensitivity and weighted aggregation)
    n_local_per_node = {nid: len(loader.dataset) for nid, loader in node_loaders.items()}

    # ── Phase 2: Build global model with MPS ─────────────────────────────────
    logger.info("[Phase 2] Building model with MPS compression...")
    global_model = build_model(dataset_name, cfg, n_features, n_classes).to(device)

    # Log compression stats
    comp_stats = compute_mps_compression_stats(global_model)
    comm_stats = compute_communication_metrics(global_model)
    logger.info(
        "MPS Compression: %.1fx overall, %.1f%% reduction, TX~%.1f KB/update",
        comp_stats.get("overall_ratio", 1.0),
        comp_stats.get("parameter_reduction_pct", 0.0),
        comm_stats.get("tx_size_kb", 0.0),
    )

    # Loss function
    loss_fn = get_loss_fn(dataset_name).to(device)

    # ── Phase 3: Context Controller + Privacy Manager ─────────────────────────
    logger.info("[Phase 3] Initializing Context Controller and Privacy Manager...")
    context_ctrl = ContextController(cfg, n_nodes, random_state=seed)
    privacy_mgr  = FederatedPrivacyManager(cfg, n_nodes)

    # Dataset assignment for all nodes
    node_dataset_map = {nid: dataset_name for nid in range(n_nodes)}

    # ── Phase 4a: Secure Aggregator setup ─────────────────────────────────────
    logger.info("[Phase 4] Setting up HE Secure Aggregator...")
    aggregator = SecureAggregator(cfg)
    ctx_bytes  = aggregator.setup()   # Generate CKKS keys
    if ctx_bytes is not None:
        logger.info("HE: CKKS context ready (poly_degree=%d)",
                    cfg.get("he", {}).get("poly_modulus_degree", 8192))
    else:
        logger.warning("HE: Running in PLAINTEXT mode (TenSEAL unavailable)")

    # ── Initialize federated clients ──────────────────────────────────────────
    logger.info("Initializing %d federated clients...", n_nodes)
    clients: Dict[int, FederatedClient] = {}
    for node_id in range(n_nodes):
        # Each client gets its own model copy (initialized from global)
        client_model = build_model(dataset_name, cfg, n_features, n_classes).to(device)
        clients[node_id] = FederatedClient(
            node_id      = node_id,
            model        = client_model,
            dataloader   = node_loaders[node_id],
            dataset_name = dataset_name,
            cfg          = cfg,
            privacy_mgr  = privacy_mgr,
            device       = device,
        )

    # ── SHAP Global Aggregator ────────────────────────────────────────────────
    shap_agg  = GlobalSHAPAggregator(n_features=n_features)
    E_global  = None   # Initially None; computed after first round

    # ── Metrics Tracker ───────────────────────────────────────────────────────
    tracker = FederatedMetricsTracker(n_nodes, dataset_name)

    # ── Initial global MPS state ──────────────────────────────────────────────
    global_mps_state = get_mps_state(global_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Federated Learning Loop
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Starting federated training: %d rounds, %.0f%% active clients",
                global_rounds, active_frac * 100)
    t_train_start = time.time()

    for round_idx in range(1, global_rounds + 1):
        t_round_start = time.time()

        # ── Step 1: Context update + active node selection ────────────────────
        exhausted = privacy_mgr.get_exhausted_nodes()
        contexts, active_ids = context_ctrl.update_for_round(
            round_idx        = round_idx,
            node_dataset_map = node_dataset_map,
            n_local_samples  = n_local_per_node,
            active_fraction  = active_frac,
            budget_exhausted = exhausted,
        )

        if not active_ids:
            logger.warning("Round %d: No active nodes (all budgets exhausted?)", round_idx)
            break

        # ── Step 2: Client training ───────────────────────────────────────────
        client_updates: List[ClientUpdate] = []
        node_train_metrics: Dict[int, dict] = {}

        for node_id in active_ids:
            ctx = contexts[node_id]
            try:
                update = clients[node_id].train_one_round(
                    context    = ctx,
                    round_idx  = round_idx,
                    global_mps = global_mps_state,
                    E_global   = E_global,
                )
                client_updates.append(update)
                node_train_metrics[node_id] = {
                    "accuracy": update.train_acc,
                    "loss":     update.train_loss,
                }
            except Exception as e:
                logger.error("Client %d training failed: %s", node_id, e, exc_info=True)
                continue

        if not client_updates:
            logger.warning("Round %d: No successful client updates.", round_idx)
            continue

        # ── Step 3: Update global SHAP vector ────────────────────────────────
        if round_idx % update_shap_every == 0 or round_idx == 1:
            for upd in client_updates:
                if upd.shap_E_local is not None:
                    shap_agg.submit_local(upd.node_id, upd.shap_E_local, upd.n_local)
            if shap_agg._round_buffer:
                E_global = shap_agg.aggregate()
                logger.debug("Global SHAP updated (top feature idx=%s)",
                             np.argsort(E_global)[-3:][::-1].tolist()
                             if E_global is not None else "N/A")

        # ── Step 4: HE Encryption + Aggregation ──────────────────────────────
        aggregator.reset()
        for upd in client_updates:
            enc_update = encrypt_client_update(
                node_id   = upd.node_id,
                n_local   = upd.n_local,
                mps_cores = upd.mps_cores,
                ctx_bytes = ctx_bytes,
            )
            aggregator.add_client_update(enc_update)

        # Aggregate in encrypted domain → decrypt → get global MPS cores
        global_mps_state = aggregator.aggregate()

        # Update global model for evaluation
        set_mps_state(global_model, global_mps_state)

        # ── Step 5: Evaluation on validation set ─────────────────────────────
        global_val_metrics = evaluate_model(
            global_model, val_loader, dataset_name, device, loss_fn)

        # XAI loss (mean over active nodes this round)
        xai_losses = []
        if E_global is not None:
            for upd in client_updates:
                if upd.shap_E_local is not None:
                    diff = upd.shap_E_local - E_global[:len(upd.shap_E_local)]
                    xai_losses.append(float(np.sum(diff ** 2)))
        mean_xai_loss = float(np.mean(xai_losses)) if xai_losses else 0.0

        # ── Step 6: Record metrics ────────────────────────────────────────────
        privacy_summary = privacy_mgr.get_epsilon_summary()
        tracker.record_round(
            round_idx       = round_idx,
            global_metrics  = global_val_metrics,
            node_metrics    = node_train_metrics,
            privacy_summary = privacy_summary,
            xai_loss        = mean_xai_loss,
            compression     = comm_stats,
        )

        t_round = time.time() - t_round_start
        n_exhausted = sum(1 for v in exhausted.values() if v)

        logger.info(
            "[Round %d/%d] val_acc=%.4f | PAM=%.4f | eps_mean=%.3f | "
            "L_XAI=%.4f | active=%d/%d | exhausted=%d | t=%.1fs",
            round_idx, global_rounds,
            global_val_metrics.get("accuracy", 0.0),
            tracker.history[-1]["pam"],
            privacy_summary.get("mean", 0.0),
            mean_xai_loss,
            len(active_ids), n_nodes, n_exhausted,
            t_round,
        )

        # ── Periodic checkpoint save ──────────────────────────────────────────
        if round_idx % save_every == 0:
            ckpt_path = os.path.join(
                output_dir, f"ckpt_{dataset_name}_round{round_idx:03d}.pt")
            torch.save({
                "round":           round_idx,
                "global_model":    global_model.state_dict(),
                "global_mps":      global_mps_state,
                "E_global":        E_global,
                "privacy_report":  privacy_mgr.get_privacy_report(),
            }, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Final Evaluation on Test Set
    # ══════════════════════════════════════════════════════════════════════════
    t_total = time.time() - t_train_start
    logger.info(
        "\n=====================================================\n"
        "  TRAINING COMPLETE - %s\n"
        "  Total time: %.1f min\n"
        "=====================================================",
        dataset_name, t_total / 60,
    )

    test_metrics = evaluate_model(global_model, test_loader, dataset_name, device, loss_fn)
    logger.info("Final Test Metrics: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

    # ── Summary table ─────────────────────────────────────────────────────────
    tracker.print_summary()

    # ── Save results ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, f"metrics_{dataset_name}.json")
    tracker.save(metrics_path)

    # ── Generate plots ────────────────────────────────────────────────────────
    if cfg.get("output", {}).get("plot_metrics", True):
        tracker.plot_metrics(output_dir)
        if tracker.history:
            tracker.plot_pam_radar(round_idx=-1, output_dir=output_dir)

    # ── Final model save ──────────────────────────────────────────────────────
    final_ckpt = os.path.join(output_dir, f"final_{dataset_name}.pt")
    torch.save({
        "global_model":   global_model.state_dict(),
        "global_mps":     global_mps_state,
        "test_metrics":   test_metrics,
        "E_global":       E_global,
        "compression":    comp_stats,
        "privacy_final":  privacy_mgr.get_privacy_report(),
        "config":         cfg,
    }, final_ckpt)
    logger.info("Final model saved: %s", final_ckpt)

    return tracker


# ═══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test with synthetic MIMIC-III data:
  python main_train.py --dataset mimic3 --smoke-test

  # Full training on UCI Heart Disease:
  python main_train.py --dataset uci_heart --nodes 20 --rounds 50

  # High-privacy mode with more nodes:
  python main_train.py --dataset physionet2012 --nodes 50 --epsilon 0.5

  # Train all datasets sequentially:
  python main_train.py --dataset all --nodes 20 --rounds 30
        """,
    )
    parser.add_argument(
        "--dataset", type=str, default="mimic3",
        choices=list(DATASET_REGISTRY.keys()) + ["all", "synthetic"],
        help="Dataset to train on (default: mimic3)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to dataset root directory (default: from config.yaml)",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--nodes", type=int, default=None,
        help="Number of federated nodes (overrides config)",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Number of global rounds (overrides config)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Dirichlet alpha for non-IID label skew (overrides config)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Base privacy budget ε (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device (auto = CUDA if available, else CPU)",
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Output directory for results and checkpoints",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a fast smoke test (5 nodes, 3 rounds, synthetic data)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_logging(level=args.log_level)

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── Config loading ────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Apply data path override if specified
    if args.data_path and args.dataset != "all":
        cfg.setdefault("datasets", {}).setdefault(args.dataset, {})["path"] = args.data_path

    # ── Run training ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    datasets_to_run = (
        list(DATASET_REGISTRY.keys())
        if args.dataset in ("all",)
        else [args.dataset if args.dataset != "synthetic" else "mimic3"]
    )

    # Set synthetic flag for "synthetic" mode or smoke test
    if args.dataset == "synthetic" or args.smoke_test:
        for ds in datasets_to_run:
            cfg.setdefault("datasets", {}).setdefault(ds, {})["use_synthetic"] = True

    all_trackers = {}
    for ds_name in datasets_to_run:
        logger.info("\n%s\n  Starting: %s\n%s",
                    "=" * 60, ds_name.upper(), "=" * 60)
        try:
            tracker = run_federated_training(
                dataset_name = ds_name,
                cfg          = cfg,
                device       = device,
                output_dir   = os.path.join(args.output, ds_name),
            )
            all_trackers[ds_name] = tracker
        except Exception as e:
            logger.error("Training failed for %s: %s", ds_name, e, exc_info=True)

    # ── Multi-dataset summary ─────────────────────────────────────────────────
    if len(all_trackers) > 1:
        logger.info("\n%s\n  MULTI-DATASET SUMMARY\n%s", "=" * 60, "=" * 60)
        for ds_name, tracker in all_trackers.items():
            best = tracker.get_best_round("pam")
            logger.info(
                "  %s: best_PAM=%.4f @ round %d, best_acc=%.4f",
                ds_name,
                best.get("pam", 0.0),
                best.get("round", 0),
                best.get("global", {}).get("accuracy", 0.0),
            )

    logger.info("\nQI-CAF-EI training complete. Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
