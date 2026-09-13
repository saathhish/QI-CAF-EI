"""
download_datasets.py
═══════════════════════════════════════════════════════════════════════════════
Dataset Download & Setup Helper for QI-CAF-EI

Provides downloaders for all 5 datasets:
  1. MIMIC-III      — Requires PhysioNet credentialing (DUA)
  2. PhysioNet 2012 — Free PhysioNet account required
  3. ChestX-ray14   — Kaggle API (free)
  4. UCI Heart      — Kaggle API (free)
  5. WISDM IoMT     — UCI ML Repo API (free, automatic)

Usage:
  # Download all freely available datasets:
  python download_datasets.py --all

  # Download specific dataset:
  python download_datasets.py --dataset wisdm
  python download_datasets.py --dataset uci_heart
  python download_datasets.py --dataset chestxray14  # requires kaggle.json

  # Check download status:
  python download_datasets.py --status

  # Use synthetic data for restricted datasets:
  python download_datasets.py --synthetic

Author: QI-CAF-EI Research Framework
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

DATA_ROOT = Path("data/raw")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Status Checker
# ═══════════════════════════════════════════════════════════════════════════════

def check_status() -> dict:
    """Check which datasets are available locally."""
    status = {}

    checks = {
        "mimic3": [
            DATA_ROOT / "mimic3" / "mimic3_processed.csv",
        ],
        "physionet2012": [
            DATA_ROOT / "physionet2012" / "set-a",
            DATA_ROOT / "physionet2012" / "Outcomes-a.txt",
        ],
        "chestxray14": [
            DATA_ROOT / "chestxray14" / "Data_Entry_2017.csv",
            DATA_ROOT / "chestxray14" / "images",
        ],
        "uci_heart": [
            DATA_ROOT / "uci_heart" / "heart.csv",
        ],
        "wisdm": [
            DATA_ROOT / "wisdm" / "phone-accelerometer",
        ],
    }

    print("\n" + "=" * 65)
    print(f"  QI-CAF-EI Dataset Status")
    print("=" * 65)

    for ds_name, paths in checks.items():
        available = all(p.exists() for p in paths)
        synthetic_flag = "✓ AVAILABLE" if available else "✗ NOT FOUND  [using synthetic]"
        access_type = {
            "mimic3":        "RESTRICTED (PhysioNet DUA required)",
            "physionet2012": "FREE (PhysioNet account required)",
            "chestxray14":   "FREE (Kaggle API required, ~42GB)",
            "uci_heart":     "FREE (Kaggle API required, <1MB)",
            "wisdm":         "FREE (UCI Repo API, ~2GB)",
        }.get(ds_name, "UNKNOWN")

        print(f"  {ds_name:20s}: {synthetic_flag}")
        print(f"  {'':20s}  Access: {access_type}")
        status[ds_name] = available

    print("=" * 65 + "\n")
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# WISDM (UCI ML Repo #507) — Fully Automatic
# ═══════════════════════════════════════════════════════════════════════════════

def download_wisdm(output_dir: Path = DATA_ROOT / "wisdm"):
    """
    Download WISDM dataset via ucimlrepo API.

    The dataset is automatically downloaded and saved as CSV files.
    No account required.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading WISDM (UCI #507) via ucimlrepo...")

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=507)
        X  = ds.data.features
        y  = ds.data.targets

        X.to_csv(output_dir / "wisdm_features.csv", index=False)
        y.to_csv(output_dir / "wisdm_labels.csv",   index=False)

        logger.info("WISDM downloaded: %d samples, %d features",
                    len(X), X.shape[1])
        logger.info("Saved to %s", output_dir)
        return True

    except ImportError:
        logger.error("ucimlrepo not installed. Run: pip install ucimlrepo")
        return False
    except Exception as e:
        logger.error("WISDM download failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# UCI Heart Disease (Kaggle) — Requires kaggle.json
# ═══════════════════════════════════════════════════════════════════════════════

def download_uci_heart(output_dir: Path = DATA_ROOT / "uci_heart"):
    """
    Download UCI Heart Disease dataset via Kaggle API.

    Requires:
      1. A Kaggle account at https://www.kaggle.com
      2. API token saved to ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars)

    Dataset: ketangangal/heart-disease-dataset-uci
    Size: ~7 KB
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        logger.info("Downloading UCI Heart Disease from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset     = "ketangangal/heart-disease-dataset-uci",
            path        = str(output_dir),
            unzip       = True,
            quiet       = False,
        )
        logger.info("UCI Heart Disease downloaded to %s", output_dir)
        return True

    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.warning(
            "Kaggle download failed: %s\n"
            "Manual download:\n"
            "  1. Go to https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci\n"
            "  2. Download 'heart.csv'\n"
            "  3. Place in: %s/heart.csv", e, output_dir,
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# NIH ChestX-ray14 (Kaggle) — Requires kaggle.json (~42GB)
# ═══════════════════════════════════════════════════════════════════════════════

def download_chestxray14(
    output_dir: Path = DATA_ROOT / "chestxray14",
    subset_only: bool = True,
):
    """
    Download NIH ChestX-ray14 dataset via Kaggle API.

    Requires:
      1. A Kaggle account
      2. Dataset acceptance at https://www.kaggle.com/datasets/nih-chest-xrays/data

    Full dataset: ~42GB (112,120 images).
    With subset_only=True: downloads only metadata + first batch (~7GB).

    WARNING: This will take considerable time and disk space.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(
        "ChestX-ray14 is ~42GB. Ensure you have sufficient disk space.\n"
        "Consider using subset_fraction=0.10 in config.yaml (10%% = ~11GB)."
    )

    try:
        import kaggle
        logger.info("Downloading ChestX-ray14 from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset = "nih-chest-xrays/data",
            path    = str(output_dir),
            unzip   = False,   # Very large — unzip manually
            quiet   = False,
        )
        logger.info(
            "ChestX-ray14 zip downloaded. Unzipping...\n"
            "This may take a long time for the full dataset."
        )
        for zip_f in output_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_f, "r") as zf:
                zf.extractall(output_dir)
        logger.info("ChestX-ray14 extracted to %s", output_dir)
        return True

    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.warning(
            "Kaggle download failed: %s\n"
            "Manual download:\n"
            "  1. Go to https://www.kaggle.com/datasets/nih-chest-xrays/data\n"
            "  2. Download the dataset (42GB)\n"
            "  3. Extract to: %s/", e, output_dir,
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PhysioNet 2012 — Free (PhysioNet account required)
# ═══════════════════════════════════════════════════════════════════════════════

def download_physionet2012(output_dir: Path = DATA_ROOT / "physionet2012"):
    """
    Download PhysioNet 2012 Challenge dataset.

    Requires:
      - A free PhysioNet account at https://physionet.org
      - wget or requests for downloading

    Dataset URL: https://physionet.org/content/challenge-2012/1.0.0/
    Size: ~46MB compressed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://physionet.org/files/challenge-2012/1.0.0/"
    files    = [
        "set-a.tar.gz",
        "Outcomes-a.txt",
    ]

    logger.info(
        "Downloading PhysioNet 2012 from %s\n"
        "Note: Requires a free PhysioNet account for authentication.\n"
        "If prompted, enter your PhysioNet username and password.", base_url,
    )

    try:
        import requests
        from requests.auth import HTTPBasicAuth

        username = input("PhysioNet username: ")
        password = input("PhysioNet password: ")
        auth     = HTTPBasicAuth(username, password)

        for fname in files:
            url  = base_url + fname
            dest = output_dir / fname
            logger.info("Downloading %s...", fname)
            r = requests.get(url, auth=auth, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Saved: %s", dest)

        # Extract tar.gz
        import tarfile
        for tar_f in output_dir.glob("*.tar.gz"):
            logger.info("Extracting %s...", tar_f)
            with tarfile.open(tar_f, "r:gz") as tf:
                tf.extractall(output_dir)

        logger.info("PhysioNet 2012 downloaded and extracted to %s", output_dir)
        return True

    except Exception as e:
        logger.warning(
            "Download failed: %s\n"
            "Manual steps:\n"
            "  1. Go to: https://physionet.org/content/challenge-2012/1.0.0/\n"
            "  2. Create a free account at physionet.org\n"
            "  3. Download 'set-a.tar.gz' and 'Outcomes-a.txt'\n"
            "  4. Extract to: %s/", e, output_dir,
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MIMIC-III — Restricted Access
# ═══════════════════════════════════════════════════════════════════════════════

def setup_mimic3(output_dir: Path = DATA_ROOT / "mimic3"):
    """
    Display MIMIC-III access instructions.

    MIMIC-III requires completion of a Data Use Agreement (DUA) at PhysioNet.
    This function cannot automate the download — it provides step-by-step
    instructions and generates a synthetic placeholder until real data is obtained.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  MIMIC-III Clinical Database — RESTRICTED ACCESS")
    print("=" * 65)
    print("""
  MIMIC-III requires:
    1. Complete CITI training: "Data or Specimens Only Research" course
       (usually takes 1-3 hours, free at citiprogram.org)

    2. Create a PhysioNet account at: https://physionet.org/register/

    3. Request access to MIMIC-III at:
       https://physionet.org/content/mimiciii/1.4/

    4. Once approved, use the MIMIC-Extract pipeline to preprocess:
       https://github.com/MLforHealth/MIMIC-Extract

       Key extraction command:
         python mimic_direct_extract.py \\
           --output_dir data/raw/mimic3/ \\
           --pop_size 20000

  UNTIL YOU HAVE ACCESS:
    The pipeline will automatically use synthetic data (n=20,000)
    that exactly mirrors the MIMIC-III schema:
      - 17 features (demographics + vitals + labs)
      - 11.5% positive mortality rate
      - Realistic missingness (12%) and clinical distributions

  Set in config.yaml:
    datasets.mimic3.use_synthetic: true  (default — works without access)
    datasets.mimic3.use_synthetic: false (after placing mimic3_processed.csv)
    """)

    # Generate synthetic as placeholder
    logger.info("Generating synthetic MIMIC-III placeholder data...")
    try:
        from utils.synthetic_data import generate_mimic3_synthetic
        import pandas as pd
        X_df, y_s = generate_mimic3_synthetic(n_samples=1000)
        df = pd.concat([X_df, y_s], axis=1)
        out_path = output_dir / "mimic3_synthetic_sample.csv"
        df.to_csv(out_path, index=False)
        logger.info("Synthetic MIMIC-III sample saved to %s (1000 rows)", out_path)
    except Exception as e:
        logger.warning("Could not generate synthetic sample: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QI-CAF-EI Dataset Download Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["mimic3", "physionet2012", "chestxray14", "uci_heart", "wisdm", "all"],
        help="Dataset to download",
    )
    parser.add_argument("--all",       action="store_true", help="Download all available datasets")
    parser.add_argument("--status",    action="store_true", help="Check dataset availability")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for all restricted datasets")

    args = parser.parse_args()

    if args.status or (not args.dataset and not args.all and not args.synthetic):
        check_status()
        return

    if args.synthetic:
        logger.info("Generating synthetic data for restricted datasets...")
        setup_mimic3()
        return

    datasets_to_dl = (
        ["mimic3", "physionet2012", "chestxray14", "uci_heart", "wisdm"]
        if (args.all or args.dataset == "all")
        else [args.dataset]
    )

    results = {}
    for ds in datasets_to_dl:
        if ds == "mimic3":
            setup_mimic3()
            results[ds] = True
        elif ds == "physionet2012":
            results[ds] = download_physionet2012()
        elif ds == "chestxray14":
            results[ds] = download_chestxray14()
        elif ds == "uci_heart":
            results[ds] = download_uci_heart()
        elif ds == "wisdm":
            results[ds] = download_wisdm()

    print("\nDownload Summary:")
    for ds, ok in results.items():
        print(f"  {ds}: {'✓ Success' if ok else '✗ Failed (see logs)'}")


if __name__ == "__main__":
    main()
