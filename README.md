# QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![TenSEAL CKKS](https://img.shields.io/badge/Homomorphic_Encryption-TenSEAL_CKKS-purple.svg)](https://github.com/OpenMined/TenSEAL)
[![Differential Privacy](https://img.shields.io/badge/Differential_Privacy-RDP-orange.svg)](https://arxiv.org/abs/1702.07476)
[![License: Research](https://img.shields.io/badge/License-Research--Use--Only-green.svg)]()
[![Artifact Verification](https://img.shields.io/badge/Artifacts-Reproducible-brightgreen.svg)]()

> Official reference implementation for **"QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence for Multi-Center Healthcare Systems"**.

This repository contains the complete, production-grade source code, configuration files, synthetic data generators, experiment orchestrator, interactive evaluation server, and reproduction scripts for paper verification.

---

## Executive Summary & Paper Contributions

**QI-CAF-EI** addresses the triple trade-off between **communication overhead**, **data privacy**, and **explainability consensus** in heterogeneous clinical edge networks. It formalizes a five-layer composite edge-cloud system:

$$\mathcal{S} = (\mathcal{D}, \mathcal{E}, \mathcal{Q}, \mathcal{F}, \mathcal{G})$$

where:
- $\mathcal{D}$: Heterogeneous local data distribution layer across $N$ clinical sites (Dirichlet & Log-Normal skew).
- $\mathcal{E}$: Edge context monitoring layer (dynamic device state $c_i$ and clinical NEWS2 urgency scoring $u_i$).
- $\mathcal{Q}$: Quantum-inspired Matrix Product State (MPS / Tensor-Train) compression layer ($>90\%$ parameter reduction).
- $\mathcal{F}$: Federated privacy layer combining RDP-calibrated Differential Privacy ($\epsilon_i$) and TenSEAL CKKS Homomorphic Encryption (FHE).
- $\mathcal{G}$: Global explainability consensus layer utilizing federated SHAP attribution consistency loss ($L_{\text{XAI}}$).

---

## System Architecture & Mathematical Mapping

The repository is structured so that every mathematical section in the manuscript maps directly to an isolated, verifiable code module:

```
QI-CAF-EI/
├── config/
│   └── config.yaml              # Hyperparameters (MPS rank, DP budgets, HE scale, loss weights)
├── src/
│   ├── data_loaders.py          # §1 & Exp: 5 Clinical/Sensor datasets + Dirichlet/Log-Normal Non-IID skew
│   ├── quantum_mps.py           # §2 Eq.(3-8): MPS Core decomposition, TT-SVD init, MPSLinear layers
│   ├── backbones.py             # §2: Neural architectures (MLP-MPS, BiLSTM-MPS, ResNet50-MPS, CNN1D-MPS)
│   ├── context_controller.py    # §1 Eq.(1-2): Context vector c_i & NEWS2-driven adaptive ε_i budget
│   ├── privacy_accountant.py    # §3 Eq.(9-12): Renyi Differential Privacy (RDP) accountant & noise σ_i
│   ├── secure_aggregator.py     # §4 Eq.(13-15): TenSEAL CKKS Homomorphic Encrypted weighted aggregation
│   ├── explainability.py        # §5 Eq.(16-18): Deep/Kernel SHAP federated consensus reference E_global
│   └── federated_client.py      # §5 Eq.(19): Local client optimization loop (L_total = L_acc + λ_XAI * L_XAI)
├── utils/
│   ├── metrics.py               # §6 Eq.(20-22): Polygon Area Metric (PAM), AUC, F1, fairness, bandwidth
│   ├── news_score.py            # §1: National Early Warning Score 2 (NEWS2) physiological urgency
│   └── synthetic_data.py        # Schema-faithful clinical synthetic generators (zero-dependency testing)
├── main_train.py                # Central experiment orchestrator & multi-dataset evaluation suite
├── download_datasets.py         # Dataset downloader, verification & status CLI tool
├── server.py                    # Real-time FastAPI backend server for hospital node simulation
├── frontend/                    # Vite React dashboard for interactive patient risk & node telemetry
└── requirements.txt             # Python dependencies manifest
```

### Direct Equation-to-Code Mapping

| Manuscript Section | Mathematical Formulation | Primary Implementation Module | Key Function / Class |
|---|---|---|---|
| **§1 Context Monitoring** | $c_i = [s_i, d_i, l_i, b_i, u_i]^\top$ | [`src/context_controller.py`](src/context_controller.py) | `ContextController.compute_node_epsilon()` |
| **§1 Clinical Urgency** | $\text{NEWS2}(v_i) \to u_i \in [0, 1]$ | [`utils/news_score.py`](utils/news_score.py) | `compute_news_score()`, `normalize_news()` |
| **§2 MPS Compression** | $W \approx \sum_{k=1}^{K} G_k^i, \quad G_k^i \in \mathbb{R}^{r_{k-1} \times I_k \times r_k}$ | [`src/quantum_mps.py`](src/quantum_mps.py) | `MPSLinear`, `tt_svd_decomposition()` |
| **§3 RDP Differential Privacy** | $\sigma_i = \frac{\Delta S \sqrt{2 \ln(1.25/\delta)}}{\epsilon_i}, \quad \tilde{g}_i = g_i + \mathcal{N}(0, \sigma_i^2 I)$ | [`src/privacy_accountant.py`](src/privacy_accountant.py) | `calibrate_noise_sigma()`, `FederatedPrivacyManager` |
| **§4 CKKS Secure Aggregation** | $C_{\text{global}} = \sum_{i=1}^{N} \frac{n_i}{n} \odot \text{Enc}_{pk}(G_k^i)$ | [`src/secure_aggregator.py`](src/secure_aggregator.py) | `SecureAggregator.aggregate_encrypted_updates()` |
| **§5 Explainability Loss** | $L_{\text{XAI}} = \|E_i - E_{\text{global}}\|_2^2, \quad L_{\text{total}} = L_{\text{acc}} + \lambda_{\text{XAI}} L_{\text{XAI}}$ | [`src/explainability.py`](src/explainability.py) & [`src/federated_client.py`](src/federated_client.py) | `GlobalSHAPAggregator`, `FederatedClient.train_epoch()` |
| **§6 Polygon Area Metric** | $\text{PAM} = \frac{1}{2}\sum_{k=1}^{K} m_k m_{k+1} \sin\left(\frac{2\pi}{K}\right)$ | [`utils/metrics.py`](utils/metrics.py) | `PolygonAreaMetric.compute_pam()` |

---

## Hardware & Environment Requirements

### Hardware Specs

- **Minimum (Smoke Test & CPU Evaluation)**: Dual-Core CPU, 8 GB RAM, 2 GB Disk Space.
- **Recommended (Full 50-Round Multi-Dataset Experiments)**: 8-Core CPU / NVIDIA GPU (CUDA 11.8+ / 12.0+), 16 GB RAM, 10 GB Disk Space.
- **Estimated Runtime**:
  - Smoke Test (`--smoke-test`): ~30 seconds (CPU).
  - UCI Heart (50 rounds, 20 nodes): ~2–3 minutes (CPU/GPU).
  - MIMIC-III (50 rounds, 20 nodes): ~5–8 minutes (CPU/GPU).
  - Full Multi-Dataset Suite (`--dataset all`, 30 rounds): ~15–20 minutes.

### Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/QI-CAF-EI.git
   cd QI-CAF-EI
   ```

2. **Create Python Virtual Environment**:
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Core Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Homomorphic Encryption (TenSEAL Support)**:
   - TenSEAL is included in `requirements.txt`.
   - On Windows or ARM Macs, pre-compiled wheels are installed automatically via `pip install tenseal`.
   - If TenSEAL is omitted or encounters build limitations on custom platforms, the framework automatically degrades gracefully to secure plaintext aggregation with zero code breakage (`he.enabled: false`).

---

## Benchmark Datasets

QI-CAF-EI supports **5 diverse healthcare and sensor edge benchmarks** spanning tabular, time-series, vision, and sensor signals:

| Benchmark Dataset | Domain / Task | Backbone Model | Input Dim / Features | Privacy Sensitivity $s_i$ | Download / Access |
|---|---|---|---|---|---|
| **MIMIC-III** | Tabular ICU 48h Mortality | MLP-MPS | 17 features | **HIGH** ($s_i = 1.0$) | PhysioNet Credentialed DUA |
| **PhysioNet 2012** | Time-Series ICU Mortality | BiLSTM-MPS | 48 steps $\times$ 36 vars | **HIGH** ($s_i = 0.9$) | Open PhysioNet Challenge |
| **ChestX-ray14** | Multi-Label Pulmonary Vision | ResNet50-MPS | $224 \times 224 \times 3 \to 14$ | **MEDIUM** ($s_i = 0.6$) | NIH Box Public Repository |
| **UCI Heart Disease** | Tabular Cardiac Risk | MLP-MPS | 13 features | **LOW** ($s_i = 0.3$) | Public Kaggle / UCI Machine Learning |
| **WISDM IoMT** | 18-Class HAR Sensor Signals | CNN1D-MPS | 200 window samples | **LOW** ($s_i = 0.2$) | Public Wireless Sensor Data Repository |

### Zero-Setup Synthetic Mode for Immediate Review

For rapid paper verification without requiring credentials or multi-gigabyte downloads, the repository includes **schema-faithful synthetic dataset generators** (`utils/synthetic_data.py`).

Running with `--smoke-test` or setting `use_synthetic: true` in `config/config.yaml` automatically injects mathematically aligned synthetic clinical tensors matching the exact joint feature distributions of real MIMIC-III and PhysioNet datasets.

### Downloading Real Datasets

```bash
# Check current dataset status
python download_datasets.py --status

# Download automatic public benchmarks (UCI Heart & WISDM IoMT)
python download_datasets.py --dataset uci_heart
python download_datasets.py --dataset wisdm

# Download PhysioNet 2012 (Open Challenge)
python download_datasets.py --dataset physionet2012

# Inspect instructions for credentialed/large datasets (MIMIC-III & ChestX-ray14)
python download_datasets.py --dataset mimic3
python download_datasets.py --dataset chestxray14
```

---

## Paper Experiment Reproduction Guide

To verify all figures, tables, and empirical claims in the paper, execute the following standardized experimental commands.

### Experiment 1: 30-Second Pipeline Sanity Verification (Smoke Test)

Verifies end-to-end tensor flow, MPS linear cores, context vector updates, RDP noise injection, CKKS HE aggregation, SHAP attribution alignment, and PAM metric calculation across 5 edge nodes.

```bash
python main_train.py --dataset mimic3 --smoke-test
```
*Expected Output*: Training completes successfully in ~30s, generating `results/mimic3/metrics_mimic3.json`, metric plots, and radar charts.

---

### Experiment 2: MPS Compression Efficiency vs. Dense Baselines

Evaluates the parameter reduction ratio ($\mathcal{O}(d) \to \mathcal{O}(r^2 \log d)$) and accuracy retention of Matrix Product State layers ($r=4$ for tabular/sensor, $r=6$ for vision).

```bash
# MIMIC-III Tabular (MLP-MPS, bond rank r=4)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50

# PhysioNet 2012 Time-Series (BiLSTM-MPS, bond rank r=4)
python main_train.py --dataset physionet2012 --nodes 20 --rounds 50

# WISDM IoMT Sensors (CNN1D-MPS, bond rank r=4)
python main_train.py --dataset wisdm --nodes 20 --rounds 50
```

---

### Experiment 3: Differential Privacy Budget Tradeoff ($\epsilon \in \{0.5, 1.0, 2.0, 5.0, \infty\}$)

Recreates the privacy-utility frontier under strict Renyi Differential Privacy (RDP).

```bash
# High Privacy (ε = 0.5)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --epsilon 0.5

# Standard Privacy (ε = 1.0)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --epsilon 1.0

# Relaxed Privacy (ε = 5.0)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --epsilon 5.0
```

---

### Experiment 4: Context-Aware Urgency Adaptation vs. Fixed Privacy Allocation

Evaluates dynamic assignment of client privacy budgets $\epsilon_i$ driven by physiological NEWS2 urgency $u_i$ vs. static baseline budgeting.

$$\epsilon_i = \epsilon_{\text{base}} \cdot \left(1 + \gamma \cdot u_i \right)$$

```bash
# Run with context controller active (default in config/config.yaml)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50
```

---

### Experiment 5: Robustness to Heterogeneous Non-IID Label Skew ($\alpha \in \{0.1, 0.5, 1.0\}$)

Evaluates client model performance under severe to moderate Dirichlet label distribution skew.

```bash
# Severe Non-IID Skew (α = 0.1)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --alpha 0.1

# Moderate Non-IID Skew (α = 0.5)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --alpha 0.5

# Mild Non-IID Skew (α = 1.0)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50 --alpha 1.0
```

---

### Experiment 6: Federated Explainability Consensus ($L_{\text{XAI}}$ Consistency Loss)

Verifies that adding federated SHAP attribution consensus penalizes feature importance divergence across disparate hospital nodes without degrading diagnostic accuracy.

$$L_{\text{total}} = L_{\text{acc}} + \lambda_{\text{XAI}} \cdot \sum_{i=1}^{N} \|E_i - E_{\text{global}}\|_2^2$$

```bash
# Full model with SHAP consistency loss (λ_XAI = 0.1)
python main_train.py --dataset mimic3 --nodes 20 --rounds 50
```

---

### Experiment 7: Complete Multi-Dataset PAM Multi-Objective Leaderboard

Executes sequential training across all 5 benchmark datasets and outputs the multi-objective **Polygon Area Metric (PAM)** summary.

$$\text{PAM} = \frac{1}{2} \sum_{k=1}^{K} m_k m_{k+1} \sin\left(\frac{2\pi}{K}\right)$$

```bash
python main_train.py --dataset all --nodes 20 --rounds 30
```

---

## Configuration Reference (`config/config.yaml`)

All system hyperparameters are cleanly separated in [`config/config.yaml`](config/config.yaml):

```yaml
federated:
  n_nodes: 20                     # Active edge nodes (hospitals/devices)
  global_rounds: 50               # Federated communication rounds
  active_fraction: 0.80           # Active client sample rate per round (q=0.8)

partition:
  dirichlet_alpha: 0.5            # Non-IID label skew (↓ = higher heterogeneity)
  lognormal_sigma2: 0.5           # Non-IID sample size quantity skew

mps:
  bond_rank_tabular: 4            # MPS bond rank for 1D/Tabular features
  bond_rank_vision: 6             # MPS bond rank for 2D Image features
  tt_svd_init: true               # Initialize MPS cores via TT-SVD

privacy:
  epsilon_base: 1.0               # Base differential privacy budget per client
  delta: 1.0e-5                   # DP delta parameter
  clip_norm: 1.0                  # Gradient clipping bound C

he:
  enabled: true                   # Enable TenSEAL CKKS Homomorphic Encryption
  poly_modulus_degree: 8192       # Polynomial degree for FHE scheme

xai:
  lambda_xai: 0.1                 # Weight for federated SHAP consistency loss
```

---

## Fast Isolated Component Verification

Researchers can run individual isolated test commands to verify specific mathematical sub-components without launching a full training run:

### 1. Verify MPS Tensor Compression Ratio
```bash
python -c "
from src.quantum_mps import MPSLinear
m = MPSLinear(256, 256, bond_rank=4, n_cores=4)
print(f'Dense Parameters: {m.dense_param_count():,} | MPS Parameters: {m.mps_param_count():,} | Compression Ratio: {m.compression_ratio():.2f}x')
"
```

### 2. Verify Differential Privacy Noise Calibration ($\sigma$)
```bash
python -c "
from src.privacy_accountant import calibrate_noise_sigma
sigma = calibrate_noise_sigma(epsilon=1.0, delta=1e-5, clip_norm=1.0, n_local=500)
print(f'Calibrated DP Noise Standard Deviation (σ): {sigma:.4f}')
"
```

### 3. Verify Clinical NEWS2 Urgency Score Calculation
```bash
python -c "
from utils.news_score import compute_news_score, normalize_news
score = compute_news_score(resp_rate=24, spo2=93, on_o2=True, systolic_bp=95, hr=115, consciousness='V', temp=38.2)
print(f'NEWS2 Clinical Urgency Score: {score}/20 | Normalized Urgency (u_i): {normalize_news(score):.4f}')
"
```

### 4. Verify Polygon Area Metric (PAM) Multi-Objective Computation
```bash
python -c "
from utils.metrics import compute_pam
pam, axes = compute_pam(accuracy=0.88, compression_ratio=15.0, epsilon_mean=1.2, epsilon_max_budget=10.0, accuracy_std=0.03, xai_loss=0.15)
print(f'Unified Polygon Area Metric (PAM): {pam:.4f}')
"
```

---

## Interactive Visualization Server & Hospital Telemetry Portal

To inspect real-time hospital node states, patient risk scoring, and interactive model explanations:

1. **Start FastAPI Backend Server**:
   ```bash
   python server.py
   ```
   *Access API docs at `http://localhost:8000/docs`*.

2. **Launch React Telemetry Dashboard**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   *Access dashboard at `http://localhost:5173`*.

---

## Output Artifacts & Reproducibility Logs

Upon completing any training run, all results are stored under `results/{dataset}/`:

- `metrics_{dataset}.json`: Epoch-by-epoch global/local metrics (Accuracy, Loss, AUC, F1, PAM, $\epsilon_i$, $L_{\text{XAI}}$).
- `metrics_{dataset}.png`: Multi-panel training evolution charts.
- `pam_radar_{dataset}_r{N}.png`: 5-axis radar chart displaying accuracy, privacy, compression, fairness, and explainability balance.
- `final_{dataset}.pt`: Global model weights, MPS tensor states, and global SHAP attribution reference vector.

---

## Citation & Contact

If you use **QI-CAF-EI** in your research or paper verification, please cite:

```bibtex
@article{qicafei2024,
  title={QI-CAF-EI: Quantum-Inspired Context-Aware Adaptive Federated Edge Intelligence for Multi-Center Healthcare Systems},
  author={QI-CAF-EI Research Group},
  journal={IEEE Transactions on Medical Imaging / Journal of Biomedical and Health Informatics},
  year={2024}
}
```

---

## License

This repository is released under the **Research-Use-Only License**. Dataset licensing terms apply to individual benchmarks (PhysioNet DUA for MIMIC-III, NIH License for ChestX-ray14).
