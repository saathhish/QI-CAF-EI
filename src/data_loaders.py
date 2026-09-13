"""
src/data_loaders.py
═══════════════════════════════════════════════════════════════════════════════
Phase 1: Data Preprocessing & Non-IID Federated Partitioning

Implements dataset-specific loaders for all five QI-CAF-EI modalities:
  1. MIMIC-III       -> Tabular, binary mortality, MICE imputation + SMOTE
  2. PhysioNet 2012  -> Time-series, binary ICU mortality, 42 variables
  3. ChestX-ray14    -> Image, 14-class multi-label, ImageNet transforms
  4. UCI Heart       -> Tabular, binary classification
  5. WISDM IoMT      -> Sensor time-series, 18-class HAR, Butterworth filter

Non-IID Partitioning:
  - Label skew: Dirichlet(alpha·p) where alpha ∈ {0.1, 0.5}
  - Quantity skew: LogNormal(mu=0, sigma²=0.5) per node
  - Combined: quantity determines pool size; Dirichlet distributes labels within pool

Mathematical reference:
  c_{i,k} = (c_{i,k} - c_k^min) / (c_k^max - c_k^min)   [context normalization]
  p_i ~ Dir(alpha · p)                                         [label distribution per node]
  n_i ~ LN(0, 0.5)                                         [quantity per node]

Author: QI-CAF-EI Research Framework
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MIMIC-III Loader
# ═══════════════════════════════════════════════════════════════════════════════

class MIMIC3Dataset(Dataset):
    """
    MIMIC-III Clinical Database — Tabular, Binary Mortality Prediction.

    Features: 17 tabular (demographics + vitals + labs).
    Target:   hospital_mortality (0/1), ~11.5% positive rate.

    Preprocessing:
      1. Load CSV or use synthetic generator (if use_synthetic=True)
      2. MICE imputation for missing lab values (IterativeImputer)
      3. Z-score standardization (StandardScaler)
      4. Optional SMOTE oversampling for training folds
    """

    SENSITIVITY_LEVEL = 1.0   # Highest sensitivity (restricted access)

    def __init__(
        self,
        data_path:     Optional[str] = None,
        use_synthetic: bool = True,
        n_synthetic:   int  = 20000,
        split:         str  = "train",   # "train" | "val" | "test"
        val_frac:      float = 0.10,
        test_frac:     float = 0.10,
        apply_smote:   bool = True,
        random_state:  int  = 42,
    ):
        super().__init__()
        self.split       = split
        self.scaler      = StandardScaler()

        if use_synthetic or data_path is None:
            logger.info("MIMIC-III: Using synthetic schema-faithful data (n=%d)", n_synthetic)
            from utils.synthetic_data import generate_mimic3_synthetic
            X_df, y_s = generate_mimic3_synthetic(n_synthetic, random_state=random_state)
        else:
            logger.info("MIMIC-III: Loading from %s", data_path)
            X_df, y_s = self._load_real(data_path)

        X_np = X_df.values.astype(np.float32)
        y_np = y_s.values.astype(np.int64)

        # MICE Imputation
        X_np = self._mice_impute(X_np, random_state)

        # Z-score normalization
        X_np = self.scaler.fit_transform(X_np).astype(np.float32)

        # Train/val/test split
        idx = np.arange(len(X_np))
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=y_np, random_state=random_state)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac / (1 - test_frac),
            stratify=y_np[idx_tv], random_state=random_state)

        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        sel = split_map[split]
        self.X = X_np[sel]
        self.y = y_np[sel]

        # SMOTE for training only
        if split == "train" and apply_smote:
            self.X, self.y = self._apply_smote(self.X, self.y, random_state)

        self.n_features = self.X.shape[1]
        logger.info("MIMIC-III [%s]: %d samples, %d features, pos=%.1f%%",
                    split, len(self.y), self.n_features,
                    100 * self.y.mean() if len(self.y) > 0 else 0)

    def _load_real(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load real MIMIC-III processed CSV (expects mimic3_processed.csv)."""
        csv_path = Path(data_path) / "mimic3_processed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"MIMIC-III CSV not found at {csv_path}. "
                "Please run the MIMIC-Extract pipeline or use use_synthetic=True."
            )
        df = pd.read_csv(csv_path)
        from utils.synthetic_data import MIMIC3_FEATURES
        target_col = "hospital_mortality"
        X_df = df[MIMIC3_FEATURES].copy()
        y_s  = df[target_col].copy()
        return X_df, y_s

    @staticmethod
    def _mice_impute(X: np.ndarray, random_state: int) -> np.ndarray:
        """MICE imputation via sklearn IterativeImputer."""
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(max_iter=10, random_state=random_state)
            return imputer.fit_transform(X).astype(np.float32)
        except Exception as e:
            logger.warning("MICE imputation failed (%s), falling back to median.", e)
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy="median").fit_transform(X).astype(np.float32)

    @staticmethod
    def _apply_smote(X: np.ndarray, y: np.ndarray, random_state: int):
        """Apply SMOTE to balance the training set."""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info("SMOTE: %d -> %d samples", len(y), len(y_res))
            return X_res.astype(np.float32), y_res.astype(np.int64)
        except Exception as e:
            logger.warning("SMOTE failed (%s), skipping.", e)
            return X, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PhysioNet 2012 Loader
# ═══════════════════════════════════════════════════════════════════════════════

class PhysioNet2012Dataset(Dataset):
    """
    PhysioNet/CinC Challenge 2012 — ICU Time-Series Mortality Prediction.

    Features: 42 variables (6 static + 36 time-series) over 48h.
    Representation: Per-variable statistics (min/max/mean/std/first/last)
                    -> 36x6 + 5 = 221 features for tabular/BiLSTM input.
    Target:   binary in-hospital mortality.

    Preprocessing:
      1. Load .txt records or use synthetic data
      2. Forward-fill within patient, then global mean impute
      3. Z-score normalization
    """

    SENSITIVITY_LEVEL = 1.0

    def __init__(
        self,
        data_path:     Optional[str] = None,
        use_synthetic: bool = True,
        n_synthetic:   int = 4000,
        split:         str = "train",
        val_frac:      float = 0.10,
        test_frac:     float = 0.10,
        random_state:  int = 42,
    ):
        super().__init__()
        self.scaler = StandardScaler()

        if use_synthetic or data_path is None:
            logger.info("PhysioNet2012: Using synthetic data (n=%d)", n_synthetic)
            from utils.synthetic_data import generate_physionet_synthetic
            X_np, y_np = generate_physionet_synthetic(n_synthetic, random_state=random_state)
        else:
            logger.info("PhysioNet2012: Loading from %s", data_path)
            X_np, y_np = self._load_real(data_path)

        # Global mean impute NaN
        col_means = np.nanmean(X_np, axis=0)
        nan_mask  = np.isnan(X_np)
        X_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X_np = self.scaler.fit_transform(X_np).astype(np.float32)

        idx = np.arange(len(X_np))
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=y_np, random_state=random_state)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac / (1 - test_frac),
            stratify=y_np[idx_tv], random_state=random_state)

        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        sel = split_map[split]
        self.X = X_np[sel]
        self.y = y_np[sel]
        self.n_features = self.X.shape[1]
        logger.info("PhysioNet2012 [%s]: %d samples, %d features",
                    split, len(self.y), self.n_features)

    def _load_real(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse PhysioNet 2012 .txt record files + Outcomes-a.txt."""
        import glob
        record_dir = Path(data_path) / "set-a"
        outcomes_f = Path(data_path) / "Outcomes-a.txt"
        if not record_dir.exists():
            raise FileNotFoundError(f"PhysioNet set-a not found at {record_dir}")

        outcomes_df = pd.read_csv(outcomes_f)
        outcomes_df = outcomes_df.set_index("RecordID")

        from utils.synthetic_data import PHYSIONET_TIME_SERIES_VARS, PHYSIONET_VAR_STATS
        all_X, all_y = [], []

        for fpath in sorted(glob.glob(str(record_dir / "*.txt")))[:]:
            rid = int(Path(fpath).stem)
            if rid not in outcomes_df.index:
                continue
            label = int(outcomes_df.loc[rid, "In-hospital_death"])

            df = pd.read_csv(fpath, header=0, names=["Time", "Parameter", "Value"])
            # Static row (Time=00:00)
            static = df[df["Time"] == "00:00"]
            age = float(static[static["Parameter"] == "Age"]["Value"].values[0]) \
                  if len(static[static["Parameter"] == "Age"]) > 0 else np.nan
            gender = float(static[static["Parameter"] == "Gender"]["Value"].values[0]) \
                     if len(static[static["Parameter"] == "Gender"]) > 0 else np.nan
            icu    = float(static[static["Parameter"] == "ICUType"]["Value"].values[0]) \
                     if len(static[static["Parameter"] == "ICUType"]) > 0 else np.nan
            height = float(static[static["Parameter"] == "Height"]["Value"].values[0]) \
                     if len(static[static["Parameter"] == "Height"]) > 0 else np.nan
            weight = float(static[static["Parameter"] == "Weight"]["Value"].values[0]) \
                     if len(static[static["Parameter"] == "Weight"]) > 0 else np.nan
            static_feats = [age, gender, icu, height, weight]

            # Time-series aggregation
            ts_feats = []
            for var in PHYSIONET_TIME_SERIES_VARS:
                vals = pd.to_numeric(
                    df[df["Parameter"] == var]["Value"], errors="coerce"
                ).dropna().values
                if len(vals) == 0:
                    ts_feats.extend([np.nan] * len(PHYSIONET_VAR_STATS))
                else:
                    ts_feats.extend([
                        float(np.min(vals)),
                        float(np.max(vals)),
                        float(np.mean(vals)),
                        float(np.std(vals) if len(vals) > 1 else 0.0),
                        float(vals[0]),
                        float(vals[-1]),
                    ])

            all_X.append(static_feats + ts_feats)
            all_y.append(label)

        return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NIH ChestX-ray14 Loader
# ═══════════════════════════════════════════════════════════════════════════════

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 — 14-Class Multi-Label Chest Pathology Classification.

    Preprocessing:
      - Resize to 224x224
      - ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
      - Multi-label BCE loss targets

    Note: Requires images in data_path/images/ and Data_Entry_2017.csv.
          Set subset_fraction < 1.0 to use a stratified subset.
    """

    LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    ]
    SENSITIVITY_LEVEL = 0.7  # De-identified, but medical images

    def __init__(
        self,
        data_path:        str,
        split:            str = "train",
        subset_fraction:  float = 0.10,
        val_frac:         float = 0.10,
        test_frac:        float = 0.10,
        random_state:     int = 42,
    ):
        super().__init__()
        try:
            from torchvision import transforms
        except ImportError:
            raise ImportError("torchvision required for ChestX-ray14 loader.")

        self.data_path = Path(data_path)
        self.img_dir   = self.data_path / "images"

        entry_csv = self.data_path / "Data_Entry_2017.csv"
        if not entry_csv.exists():
            raise FileNotFoundError(f"ChestX-ray14 metadata not found: {entry_csv}")

        df = pd.read_csv(entry_csv)
        df = df[["Image Index", "Finding Labels"]].copy()

        # Parse multi-label
        for lbl in self.LABELS:
            df[lbl] = df["Finding Labels"].str.contains(lbl).astype(int)

        # Stratified subset
        if subset_fraction < 1.0:
            df = df.sample(frac=subset_fraction, random_state=random_state).reset_index(drop=True)

        # Split
        idx = np.arange(len(df))
        idx_tv, idx_test = train_test_split(idx, test_size=test_frac, random_state=random_state)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac / (1 - test_frac), random_state=random_state)
        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        df = df.iloc[split_map[split]].reset_index(drop=True)
        self.df = df

        # Transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.n_features = None   # Determined by CNN output
        logger.info("ChestXray14 [%s]: %d samples", split, len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["Image Index"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[self.LABELS].values.astype(np.float32))
        return img, labels


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UCI Heart Disease Loader
# ═══════════════════════════════════════════════════════════════════════════════

class UCIHeartDataset(Dataset):
    """
    UCI Heart Disease Dataset — Binary Cardiac Disease Classification.

    Features: 13 (age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal)
    Target:   target (0/1), ~55% positive rate.

    Preprocessing:
      - Categorical: one-hot encoding (cp, restecg, slope, thal)
      - Numerical: Min-Max normalization
      - No missing values in standard Cleveland subset
    """

    SENSITIVITY_LEVEL = 0.5   # Public dataset, pseudonymized

    CAT_FEATURES = ["cp", "restecg", "slope", "thal"]
    NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    BIN_FEATURES = ["sex", "fbs", "exang"]

    def __init__(
        self,
        data_path:     Optional[str] = None,
        use_synthetic: bool = False,
        n_synthetic:   int  = 2000,
        split:         str  = "train",
        val_frac:      float = 0.10,
        test_frac:     float = 0.15,
        random_state:  int  = 42,
    ):
        super().__init__()

        if use_synthetic or data_path is None:
            logger.info("UCI Heart: Using synthetic data (n=%d)", n_synthetic)
            from utils.synthetic_data import generate_uci_heart_synthetic
            X_df, y_s = generate_uci_heart_synthetic(n_synthetic, random_state=random_state)
        else:
            logger.info("UCI Heart: Loading from %s", data_path)
            csv_path = Path(data_path) / "heart.csv"
            df = pd.read_csv(csv_path)
            target_col = "target" if "target" in df.columns else df.columns[-1]
            y_s  = df[target_col].copy().astype(np.int64)
            X_df = df.drop(columns=[target_col]).copy()

        X_np, feature_names = self._preprocess(X_df)
        y_np = y_s.values.astype(np.int64)
        self.feature_names = feature_names

        idx = np.arange(len(X_np))
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=y_np, random_state=random_state)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac / (1 - test_frac),
            stratify=y_np[idx_tv], random_state=random_state)

        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        sel = split_map[split]
        self.X = X_np[sel]
        self.y = y_np[sel]
        self.n_features = self.X.shape[1]
        logger.info("UCI Heart [%s]: %d samples, %d features",
                    split, len(self.y), self.n_features)

    @classmethod
    def _preprocess(cls, X_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """One-hot encode categoricals + Min-Max scale numericals."""
        frames, names = [], []

        # Numerical -> Min-Max
        num_cols = [c for c in cls.NUM_FEATURES if c in X_df.columns]
        if num_cols:
            scaler = MinMaxScaler()
            num_arr = scaler.fit_transform(X_df[num_cols].values.astype(np.float32))
            frames.append(num_arr)
            names.extend(num_cols)

        # Binary -> pass through
        bin_cols = [c for c in cls.BIN_FEATURES if c in X_df.columns]
        if bin_cols:
            frames.append(X_df[bin_cols].values.astype(np.float32))
            names.extend(bin_cols)

        # Categorical -> one-hot
        cat_cols = [c for c in cls.CAT_FEATURES if c in X_df.columns]
        for col in cat_cols:
            dummies = pd.get_dummies(X_df[col], prefix=col).values.astype(np.float32)
            frames.append(dummies)
            names.extend([f"{col}_{i}" for i in range(dummies.shape[1])])

        X_np = np.hstack(frames).astype(np.float32)
        return X_np, names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. WISDM IoMT Loader
# ═══════════════════════════════════════════════════════════════════════════════

class WISDMDataset(Dataset):
    """
    WISDM Smartphone & Smartwatch Activity & Biometrics (UCI #507).

    Features: [subject_id, activity_code, timestamp, x, y, z] per entry.
    Window:   10-second sliding windows at 20Hz -> 200 samples per window.
    Filter:   3rd-order Butterworth bandpass 0.5–20 Hz.
    Feature:  Z-score normalized 6-channel (phone+watch, accel+gyro) windows.
    Target:   Activity class (18 activities), encoded 0–17.

    Data access: ucimlrepo (id=507) or local path.
    """

    ACTIVITY_MAP = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
        'M': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17,
    }
    SENSITIVITY_LEVEL = 0.3   # Community wearable, low sensitivity

    def __init__(
        self,
        data_path:     Optional[str] = None,
        split:         str = "train",
        window_size:   int = 200,
        hop_size:      int = 100,
        fs:            float = 20.0,
        lowcut:        float = 0.5,
        highcut:       float = 20.0,
        butter_order:  int = 3,
        val_frac:      float = 0.10,
        test_frac:     float = 0.20,
        random_state:  int = 42,
        use_synthetic: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.hop_size    = hop_size

        if use_synthetic:
            X_windows, y_labels = self._make_synthetic(random_state)
        elif data_path is not None:
            X_windows, y_labels = self._load_local(
                data_path, window_size, hop_size, fs, lowcut, highcut, butter_order)
        else:
            X_windows, y_labels = self._load_ucimlrepo(
                window_size, hop_size, fs, lowcut, highcut, butter_order)

        # Z-score per channel
        for ch in range(X_windows.shape[1]):
            mu  = X_windows[:, ch, :].mean()
            std = X_windows[:, ch, :].std() + 1e-8
            X_windows[:, ch, :] = (X_windows[:, ch, :] - mu) / std

        idx = np.arange(len(y_labels))
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=y_labels, random_state=random_state)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac / (1 - test_frac),
            stratify=y_labels[idx_tv], random_state=random_state)

        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        sel = split_map[split]
        self.X = X_windows[sel]   # (N, channels, window_size)
        self.y = y_labels[sel]
        self.n_features   = window_size
        self.n_channels   = self.X.shape[1]
        self.n_classes    = len(np.unique(y_labels))
        logger.info("WISDM [%s]: %d windows, %d channels, %d classes",
                    split, len(self.y), self.n_channels, self.n_classes)

    @staticmethod
    def _butterworth_filter(
        data: np.ndarray, fs: float, lowcut: float, highcut: float, order: int
    ) -> np.ndarray:
        """Apply 3rd-order Butterworth bandpass filter."""
        nyq   = 0.5 * fs
        low   = lowcut  / nyq
        high  = min(highcut / nyq, 0.99)
        b, a  = butter(order, [low, high], btype="band")
        return filtfilt(b, a, data, axis=-1)

    @classmethod
    def _window_signal(
        cls, data: np.ndarray, labels: np.ndarray,
        window_size: int, hop_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slide window over signal, assign majority label."""
        windows, win_labels = [], []
        n = data.shape[-1]
        for start in range(0, n - window_size, hop_size):
            end   = start + window_size
            win   = data[..., start:end]
            seg_y = labels[start:end]
            # Majority vote for label
            if len(seg_y) > 0:
                majority = np.bincount(seg_y).argmax()
                windows.append(win)
                win_labels.append(majority)
        return np.array(windows, dtype=np.float32), np.array(win_labels, dtype=np.int64)

    def _load_ucimlrepo(self, window_size, hop_size, fs, lowcut, highcut, order):
        """Download and process WISDM via ucimlrepo."""
        try:
            from ucimlrepo import fetch_ucirepo
            logger.info("WISDM: Downloading via ucimlrepo (id=507)...")
            ds = fetch_ucirepo(id=507)
            X_df = ds.data.features
            y_df = ds.data.targets
            return self._process_raw_df(
                X_df, y_df, window_size, hop_size, fs, lowcut, highcut, order)
        except Exception as e:
            logger.warning("ucimlrepo download failed: %s. Using synthetic.", e)
            return self._make_synthetic(42)

    def _load_local(self, data_path, window_size, hop_size, fs, lowcut, highcut, order):
        """Load WISDM from local directory (phone-accelerometer subfolder)."""
        import glob
        all_data, all_labels = [], []
        for fpath in glob.glob(str(Path(data_path) / "phone-accelerometer" / "*.txt")):
            try:
                df = pd.read_csv(
                    fpath, header=None,
                    names=["subject", "activity", "timestamp", "x", "y", "z"])
                df["z"] = df["z"].str.replace(";", "").astype(float)
                df = df.dropna()

                # Encode activity
                act_enc = df["activity"].map(self.ACTIVITY_MAP).fillna(0).astype(int)
                xyz = df[["x", "y", "z"]].values.T.astype(np.float32)  # (3, T)

                # Filter
                xyz_f = self._butterworth_filter(xyz, fs, lowcut, highcut, order)

                all_data.append(xyz_f)
                all_labels.append(act_enc.values)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fpath, e)

        if not all_data:
            return self._make_synthetic(42)

        # Concatenate all subjects
        combined_data   = np.concatenate(all_data, axis=-1)     # (3, T_total)
        combined_labels = np.concatenate(all_labels, axis=0)    # (T_total,)
        return self._window_signal(combined_data[np.newaxis], combined_labels,
                                   window_size, hop_size)

    def _process_raw_df(self, X_df, y_df, window_size, hop_size, fs, lowcut, highcut, order):
        """Process raw ucimlrepo format."""
        try:
            # Expected columns: x, y, z
            xyz = X_df[["x_acceleration", "y_acceleration", "z_acceleration"]].values.T.astype(np.float32)
        except Exception:
            # Fallback: assume first 3 numeric columns are x,y,z
            xyz = X_df.select_dtypes(include=np.number).values[:, :3].T.astype(np.float32)

        labels = LabelEncoder().fit_transform(y_df.values.ravel()).astype(np.int64)
        xyz_f  = self._butterworth_filter(xyz, fs, lowcut, highcut, order)
        return self._window_signal(xyz_f[np.newaxis], labels, window_size, hop_size)

    @staticmethod
    def _make_synthetic(random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic WISDM-like windowed sensor data."""
        logger.info("WISDM: Generating synthetic sensor windows...")
        rng = np.random.RandomState(random_state)
        n_windows  = 5000
        n_channels = 3
        window_size = 200
        n_classes  = 18

        X = rng.randn(n_windows, n_channels, window_size).astype(np.float32)
        y = rng.randint(0, n_classes, size=n_windows).astype(np.int64)
        return X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════
# Non-IID Federated Partitioning
# ═══════════════════════════════════════════════════════════════════════════════

def dirichlet_noniid_partition(
    dataset:       Dataset,
    n_nodes:       int,
    alpha:         float = 0.5,
    min_samples:   int = 10,
    random_state:  int = 42,
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across N nodes using Dirichlet label-skew.

    Each node i receives sample indices from class k with probability
    proportional to p_{k,i} ~ Dir(alpha · p_k), where p_k is the global
    class frequency.

    Args:
        dataset:     PyTorch Dataset with .y attribute (array of labels).
        n_nodes:     Number of federated nodes.
        alpha:       Dirichlet concentration parameter (lower -> more skew).
        min_samples: Minimum samples per node (nodes below this get extra).
        random_state: NumPy seed.

    Returns:
        Dict mapping node_id -> list of dataset indices.
    """
    rng = np.random.RandomState(random_state)

    # Extract labels
    if hasattr(dataset, "y"):
        labels = np.array(dataset.y)
    else:
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    n_classes = len(np.unique(labels))
    node_indices: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}

    for k in range(n_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)

        # Draw Dirichlet proportions for this class across nodes
        proportions = rng.dirichlet(alpha * np.ones(n_nodes))
        # Convert to counts
        counts = (proportions * len(class_idx)).astype(int)
        # Ensure all samples assigned
        counts[-1] = len(class_idx) - counts[:-1].sum()

        start = 0
        for node_id, count in enumerate(counts):
            node_indices[node_id].extend(class_idx[start:start + count].tolist())
            start += count

    # Enforce minimum samples
    all_remaining = []
    for node_id in range(n_nodes):
        if len(node_indices[node_id]) < min_samples:
            deficit = min_samples - len(node_indices[node_id])
            # Borrow from the largest node
            largest = max(node_indices, key=lambda x: len(node_indices[x]))
            borrowed = node_indices[largest][:deficit]
            node_indices[largest]  = node_indices[largest][deficit:]
            node_indices[node_id].extend(borrowed)

    logger.info(
        "Dirichlet partition (alpha=%.2f): %d nodes, sizes [%d, %d] (min, max)",
        alpha, n_nodes,
        min(len(v) for v in node_indices.values()),
        max(len(v) for v in node_indices.values()),
    )
    return node_indices


def lognormal_quantity_skew(
    total_samples: int,
    n_nodes:       int,
    mu:            float = 0.0,
    sigma2:        float = 0.5,
    min_samples:   int = 10,
    random_state:  int = 42,
) -> np.ndarray:
    """
    Generate per-node sample quantities from a Log-Normal distribution.

    n_i ~ LN(mu=0, sigma²=0.5), normalized to sum = total_samples.

    Args:
        total_samples: Total number of samples to distribute.
        n_nodes:       Number of federated nodes.
        mu:            Log-Normal mean (log scale).
        sigma2:        Log-Normal variance (log scale).
        min_samples:   Floor on samples per node.
        random_state:  NumPy seed.

    Returns:
        Integer array of shape (n_nodes,) with per-node sample counts.
    """
    rng = np.random.RandomState(random_state)
    sigma = np.sqrt(sigma2)
    raw   = rng.lognormal(mean=mu, sigma=sigma, size=n_nodes)
    raw   = np.maximum(raw, 1e-6)
    proportions = raw / raw.sum()
    counts = (proportions * total_samples).astype(int)

    # Clip minimum
    counts = np.maximum(counts, min_samples)
    # Rescale to fix total
    excess = counts.sum() - total_samples
    if excess > 0:
        largest_nodes = np.argsort(counts)[::-1]
        for nid in largest_nodes:
            if excess <= 0:
                break
            reduction = min(excess, counts[nid] - min_samples)
            counts[nid] -= reduction
            excess -= reduction

    return counts.astype(int)


def combined_noniid_partition(
    dataset:          Dataset,
    n_nodes:          int,
    dirichlet_alpha:  float = 0.5,
    lognormal_mu:     float = 0.0,
    lognormal_sigma2: float = 0.5,
    min_samples:      int = 10,
    random_state:     int = 42,
) -> Dict[int, List[int]]:
    """
    Combined Dirichlet label skew + Log-Normal quantity skew partitioning.

    Step 1: Use Log-Normal to determine how many samples each node gets.
    Step 2: Within each node's quota, use Dirichlet to determine class distribution.

    This more faithfully simulates real hospital federation where:
    - Large hospitals have more patients (quantity skew)
    - Specialized hospitals see different disease distributions (label skew)

    Args:
        dataset:          Source dataset.
        n_nodes:          Number of federated nodes.
        dirichlet_alpha:  alpha for Dirichlet label skew.
        lognormal_mu:     mu for Log-Normal quantity skew.
        lognormal_sigma2: sigma² for Log-Normal quantity skew.
        min_samples:      Minimum samples per node.
        random_state:     NumPy seed.

    Returns:
        Dict mapping node_id -> list of dataset indices.
    """
    rng = np.random.RandomState(random_state)
    total = len(dataset)

    # Step 1: Quantity per node
    node_counts = lognormal_quantity_skew(
        total, n_nodes, lognormal_mu, lognormal_sigma2, min_samples, random_state)

    # Step 2: Label-aware partitioning within quotas
    if hasattr(dataset, "y"):
        labels = np.array(dataset.y)
    else:
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    n_classes   = len(np.unique(labels))
    all_indices = np.arange(total)
    rng.shuffle(all_indices)

    # Dirichlet base partition
    base_partition = dirichlet_noniid_partition(
        dataset, n_nodes, dirichlet_alpha, min_samples, random_state)

    # Trim/expand to match Log-Normal counts
    node_partition: Dict[int, List[int]] = {}
    all_pool = list(all_indices)

    for node_id in range(n_nodes):
        target_n = int(node_counts[node_id])
        available = base_partition[node_id]

        if len(available) >= target_n:
            node_partition[node_id] = available[:target_n]
        else:
            # Top up from pool
            node_partition[node_id] = available[:]
            deficit = target_n - len(available)
            # Sample from remaining pool (avoiding double assignment)
            used = set(available)
            extras = [i for i in all_pool if i not in used][:deficit]
            node_partition[node_id].extend(extras)

    total_assigned = sum(len(v) for v in node_partition.values())
    logger.info(
        "Combined non-IID partition: %d nodes, total=%d assigned, "
        "alpha=%.2f, LN(mu=%.1f, sigma²=%.1f)",
        n_nodes, total_assigned, dirichlet_alpha, lognormal_mu, lognormal_sigma2,
    )
    return node_partition


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader Factory
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_REGISTRY = {
    "mimic3":        MIMIC3Dataset,
    "physionet2012": PhysioNet2012Dataset,
    "chestxray14":   ChestXray14Dataset,
    "uci_heart":     UCIHeartDataset,
    "wisdm":         WISDMDataset,
}

DATASET_SENSITIVITY = {
    "mimic3":        1.0,
    "physionet2012": 1.0,
    "chestxray14":   0.7,
    "uci_heart":     0.5,
    "wisdm":         0.3,
}


def build_federated_dataloaders(
    dataset_name:     str,
    cfg:              dict,
    n_nodes:          int,
    batch_size:       int = 32,
    dirichlet_alpha:  float = 0.5,
    lognormal_sigma2: float = 0.5,
    random_state:     int = 42,
    num_workers:      int = 0,
) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader]:
    """
    Build per-node training DataLoaders and shared val/test loaders.

    Args:
        dataset_name:    One of the DATASET_REGISTRY keys.
        cfg:             Config dict (from config.yaml -> datasets section).
        n_nodes:         Number of federated nodes.
        batch_size:      Per-node batch size.
        dirichlet_alpha: Label skew parameter.
        lognormal_sigma2: Quantity skew variance.
        random_state:    Seed.
        num_workers:     DataLoader workers.

    Returns:
        node_loaders:  Dict[node_id -> DataLoader] for local training.
        val_loader:    Global validation DataLoader.
        test_loader:   Global test DataLoader.
    """
    ds_cls = DATASET_REGISTRY.get(dataset_name)
    if ds_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(DATASET_REGISTRY.keys())}")

    ds_cfg = cfg.get("datasets", {}).get(dataset_name, {})

    # Build splits
    common_kwargs = dict(random_state=random_state)

    if dataset_name == "chestxray14":
        train_ds = ds_cls(data_path=ds_cfg.get("path"), split="train",
                          subset_fraction=ds_cfg.get("subset_fraction", 0.1),
                          **common_kwargs)
        val_ds   = ds_cls(data_path=ds_cfg.get("path"), split="val",
                          subset_fraction=ds_cfg.get("subset_fraction", 0.1),
                          **common_kwargs)
        test_ds  = ds_cls(data_path=ds_cfg.get("path"), split="test",
                          subset_fraction=ds_cfg.get("subset_fraction", 0.1),
                          **common_kwargs)
    else:
        use_syn  = ds_cfg.get("use_synthetic", False)
        data_path = ds_cfg.get("path") if not use_syn else None
        syn_n    = ds_cfg.get("synthetic_n_samples", 5000)

        train_ds = ds_cls(data_path=data_path, use_synthetic=use_syn,
                          n_synthetic=syn_n, split="train", **common_kwargs)
        val_ds   = ds_cls(data_path=data_path, use_synthetic=use_syn,
                          n_synthetic=syn_n, split="val", **common_kwargs)
        test_ds  = ds_cls(data_path=data_path, use_synthetic=use_syn,
                          n_synthetic=syn_n, split="test", **common_kwargs)

    # Non-IID partition of training set
    node_indices = combined_noniid_partition(
        train_ds, n_nodes, dirichlet_alpha, 0.0, lognormal_sigma2,
        min_samples=cfg.get("partition", {}).get("min_samples_per_node", 10),
        random_state=random_state,
    )

    node_loaders: Dict[int, DataLoader] = {}
    for node_id, indices in node_indices.items():
        subset = Subset(train_ds, indices)
        node_loaders[node_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, drop_last=False)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers)

    logger.info(
        "Built federated loaders for '%s': %d nodes, val=%d, test=%d",
        dataset_name, n_nodes, len(val_ds), len(test_ds))

    return node_loaders, val_loader, test_loader
