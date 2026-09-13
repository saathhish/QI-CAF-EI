# QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: Research](https://img.shields.io/badge/License-Research-green.svg)]()

A research-grade, privacy-preserving federated learning framework for healthcare edge intelligence. Implements the full five-layer composite system $S = (D, E, Q, F, G)$ with mathematical fidelity to the HOD PAPER specifications.

---

## System Architecture

```
QI-CAF-EI/
├── config/
│   └── config.yaml              # All hyperparameters (edit this!)
├── src/
│   ├── data_loaders.py          # Phase 1: 5 datasets + Dirichlet/Log-Normal non-IID
│   ├── quantum_mps.py           # Phase 2: MPS/Tensor-Train compression
│   ├── backbones.py             # Phase 2: MLP, BiLSTM, ResNet-50, CNN1D + MPS layers
│   ├── context_controller.py    # Phase 3: Context vector c_i + dynamic ε_i assignment
│   ├── privacy_accountant.py    # Phase 3: RDP accountant + noise calibration
│   ├── federated_client.py      # Phase 4: L_total = L_acc + L_XAI training loop
│   ├── secure_aggregator.py     # Phase 4: TenSEAL CKKS homomorphic aggregation
│   └── explainability.py        # Phase 4: SHAP + federated L_XAI consistency
├── utils/
│   ├── metrics.py               # PAM, AUC, F1, fairness, communication metrics
│   ├── news_score.py            # NEWS2 clinical urgency scoring
│   └── synthetic_data.py        # Schema-faithful synthetic data generators
├── main_train.py                # Federated training orchestrator
├── download_datasets.py         # Dataset download helper
└── requirements.txt
```

---

## Mathematical Foundations

### 1. Context Vector (HOD PAPER §1)
$$c_i = [s_i, d_i, l_i, b_i, u_i]^\top \quad \text{(sensitivity, device, latency, bandwidth, urgency)}$$
$$c_{i,k} = \frac{c_{i,k} - c_k^{\min}}{c_k^{\max} - c_k^{\min}}$$

### 2. MPS Tensor Compression (HOD PAPER §2)
$$W \approx \sum_{k=1}^{K} G_k^i, \quad G_k^i \in \mathbb{R}^{r_{k-1} \times I_k \times r_k}$$
Complexity: $\mathcal{O}(d) \to \mathcal{O}(r^2 \log d)$, bond rank $r=4$ (tabular), $r=6$ (vision).

### 3. Differential Privacy (HOD PAPER §3)
$$\sigma_i = \frac{\Delta S \sqrt{2 \ln(1.25/\delta)}}{\epsilon_i}, \quad \Delta S \leq \frac{2C}{n_i}$$
$$\tilde{g}_i = g_i + \mathcal{N}(0, \sigma_i^2 I)$$

### 4. Homomorphic Encryption Aggregation (HOD PAPER §4)
$$C_{global} = \sum_{i=1}^{N} \frac{n_i}{n} \odot \text{Enc}_{pk}(G_k^i)$$

### 5. Federated XAI Loss (HOD PAPER §5)
$$L_{XAI} = \sum_{i=1}^{N} \|E_i - E_{global}\|_2^2$$
$$L_{total} = L_{acc} + \lambda_{XAI} \cdot L_{XAI}$$

### 6. Polygon Area Metric
$$\text{PAM} = \frac{1}{2}\sum_{k=1}^{K} m_k \cdot m_{k+1} \cdot \sin\left(\frac{2\pi}{K}\right)$$

---

## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

For Homomorphic Encryption support (optional):
```bash
# Windows: use pre-built wheel
python -m pip install tenseal
```

### 2. Check Dataset Status

```bash
python download_datasets.py --status
```

### 3. Smoke Test (No Datasets Required)

```bash
python main_train.py --dataset mimic3 --smoke-test
```

This runs 3 federated rounds with 5 nodes using synthetic MIMIC-III data.
Takes ~30 seconds on CPU. Validates the entire pipeline.

### 4. Full Training

```bash
# MIMIC-III with synthetic data (20 nodes, 50 rounds):
python main_train.py --dataset mimic3 --nodes 20 --rounds 50

# UCI Heart Disease (fast, real data from Kaggle):
python main_train.py --dataset uci_heart --nodes 20 --rounds 50

# WISDM IoMT (automatic download):
python main_train.py --dataset wisdm --nodes 20 --rounds 50

# High-privacy mode (ε=0.5):
python main_train.py --dataset mimic3 --nodes 50 --epsilon 0.5

# All datasets sequentially:
python main_train.py --dataset all --nodes 20 --rounds 30
```

### 5. Download Datasets

```bash
# Free datasets (Kaggle API required):
python download_datasets.py --dataset uci_heart     # <1 MB
python download_datasets.py --dataset wisdm          # ~2 GB (auto)
python download_datasets.py --dataset chestxray14   # ~42 GB

# PhysioNet 2012 (free account required):
python download_datasets.py --dataset physionet2012

# MIMIC-III instructions (DUA required):
python download_datasets.py --dataset mimic3
```

---

## Configuration

Edit [`config/config.yaml`](config/config.yaml) to change:

| Section | Key | Default | Description |
|---|---|---|---|
| `federated` | `n_nodes` | 20 | Number of federated nodes |
| `federated` | `global_rounds` | 50 | Total training rounds |
| `federated` | `active_fraction` | 0.80 | Active nodes per round (q) |
| `partition` | `dirichlet_alpha` | 0.5 | Non-IID label skew (↓ = more skew) |
| `privacy` | `epsilon_base` | 1.0 | Base privacy budget ε |
| `privacy` | `epsilon_max` | 10.0 | Lifetime budget per node |
| `privacy` | `clip_norm` | 1.0 | Gradient clipping bound C |
| `mps` | `bond_rank_tabular` | 4 | MPS rank for tabular data |
| `mps` | `bond_rank_vision` | 6 | MPS rank for image data |
| `xai` | `lambda_xai` | 0.1 | Weight of L_XAI in total loss |
| `he` | `enabled` | true | Enable HE encryption |

---

## Dataset Mapping

| Dataset | Task | Model | n_features | Sensitivity |
|---|---|---|---|---|
| MIMIC-III | Binary mortality | MLP-MPS | 17 | HIGH (DUA) |
| PhysioNet 2012 | ICU mortality | BiLSTM-MPS | 221 | HIGH (PhysioNet) |
| ChestX-ray14 | 14-class multi-label | ResNet50-MPS | 2048→14 | MEDIUM |
| UCI Heart | Binary cardiac | MLP-MPS | ~22 | LOW (public) |
| WISDM IoMT | 18-class HAR | CNN1D-MPS | 200 | LOW (public) |

---

## Key Features

- **>90% Parameter Reduction**: MPS/Tensor-Train compression with TT-SVD initialization
- **Dynamic Privacy Budgets**: Context-aware ε_i assignment per node based on clinical urgency
- **Encrypted Aggregation**: TenSEAL CKKS FHE — server never sees raw gradients
- **Federated SHAP**: Consistency loss penalizes divergent feature attributions across hospitals
- **Non-IID Simulation**: Combined Dirichlet (label) + Log-Normal (quantity) skew
- **Multi-Objective PAM**: Single metric summarizing accuracy, privacy, compression, fairness, XAI

---

## Output Files

After training, `results/{dataset}/` contains:
- `metrics_{dataset}.json` — Full per-round metrics history
- `metrics_{dataset}.png` — Training curves (accuracy, PAM, ε, L_XAI)
- `pam_radar_{dataset}_r{N}.png` — Radar chart of PAM axes
- `final_{dataset}.pt` — Final global model + MPS state + SHAP vector
- `ckpt_{dataset}_round{N:03d}.pt` — Periodic checkpoints

---

## Verification

```bash
# 1. MPS compression test:
python -c "
from src.quantum_mps import MPSLinear
m = MPSLinear(256, 256, bond_rank=4, n_cores=4)
print(f'Dense: {m.dense_param_count():,} | MPS: {m.mps_param_count():,} | Ratio: {m.compression_ratio():.1f}×')
"

# 2. DP noise calibration test:
python -c "
from src.privacy_accountant import calibrate_noise_sigma
sigma = calibrate_noise_sigma(epsilon=1.0, delta=1e-5, clip_norm=1.0, n_local=500)
print(f'σ = {sigma:.4f} for ε=1.0, δ=1e-5, C=1.0, n=500')
"

# 3. NEWS score test:
python -c "
from utils.news_score import compute_news_score, normalize_news
score = compute_news_score(resp_rate=22, spo2=95, on_o2=False, systolic_bp=100, hr=110, consciousness='A', temp=37.5)
print(f'NEWS score = {score}, urgency u_i = {normalize_news(score):.3f}')
"

# 4. Full smoke test:
python main_train.py --dataset mimic3 --smoke-test
```

---

## Citation

If using this framework in research, please cite:
```
QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence
HOD PAPER, 2024
```

---

## License

Research use only. Dataset-specific terms apply (see individual dataset licenses).
MIMIC-III: PhysioNet Credentialed Access required.
