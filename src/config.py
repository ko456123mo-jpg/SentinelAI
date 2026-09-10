"""
SentinelAI - AI Cybersecurity Assistant
=======================================
Central configuration: paths, constants, random seeds and logging helpers.

Every module imports this file first so that the whole pipeline is
deterministic (reproducible results) and paths stay consistent.
"""

import os
import json
import random
import time

# ----------------------------------------------------------------------
# 0. Environment tweaks (MUST run before tensorflow / matplotlib import)
# ----------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")        # hide TF info logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")       # deterministic ops
os.environ.setdefault("MPLBACKEND", "Agg")               # headless plotting

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------
# 1. Reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# 2. Project paths
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(ROOT, "models")

RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")

DOCS_DIR = os.path.join(ROOT, "docs")


def ensure_dirs():
    """Create every directory required by the project structure."""
    for d in (RAW_DIR, PROC_DIR, TEST_DIR, MODEL_DIR,
              FIG_DIR, PRED_DIR, REPORT_DIR, DOCS_DIR):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------
# 3. Dataset constants
# ----------------------------------------------------------------------
# ---- 3.1 Synthetic network-traffic flows (generated, documented) ----
N_FLOWS = 25000                    # total generated flows
FLOW_CLASSES = ["Benign", "DDoS", "PortScan", "BruteForce", "Botnet"]
FLOW_CLASS_WEIGHTS = [0.45, 0.20, 0.15, 0.12, 0.08]   # imbalanced on purpose
MISSING_RATE = 0.015               # injected missing values (byte_std, iat)
DUPLICATE_RATE = 0.008             # injected duplicate rows

# ---- 3.2 Real NLP dataset (UCI SMS Spam Collection) ----
SMS_RAW_FILE = os.path.join(RAW_DIR, "SMSSpamCollection")
SMS_CSV = os.path.join(RAW_DIR, "sms_spam.csv")

# ---- 3.3 Malware byte-plot images (REAL Malimg corpus preferred) ----
# The four families are taken from the Malimg dataset (Nataraj et al. 2011)
# so real byte-plots are used when the corpus can be fetched; a documented
# synthetic generator remains as the offline fallback.
# NOTE: Allaple.A and C2LOP.P were paired with Lolyda.AA2 and Alueron.gen!J
# because the figshare Malimg mirror's Autorun.K/Yuner.A folders contain
# duplicated (pixel-identical) images that cannot be classified honestly.
MALWARE_FAMILIES = ["Allaple_A", "C2LOP_P", "Lolyda_AA2", "Alueron_genJ"]
MALWARE_IMG_DIR = os.path.join(RAW_DIR, "malware_images")
MALWARE_IMG_SIZE = 48               # 48 x 48 grayscale (CNN input)
MALWARE_PER_FAMILY = 180            # synthetic fallback: 180 * 4 = 720 images


# ----------------------------------------------------------------------
# 3.4 REAL dataset sources (auto-downloaded; generated data = fallback)
# ----------------------------------------------------------------------
# Network flows  -> CICIDS2017 (Canadian Institute for Cybersecurity),
#                   full 85-column CSV as parquet, hosted on Hugging Face.
CICIDS2017_PARQUET_BASE = (
    "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/"
    "traffic_labels/")
CICIDS2017_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet",      # FTP/SSH-Patator + BENIGN
    "Wednesday-workingHours.pcap_ISCX.csv.parquet",    # DoS/DDoS floods + BENIGN
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",   # DDoS LOIC
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",          # Bot + BENIGN
]
CICIDS_CACHE_DIR = os.path.join(RAW_DIR, "cicids_cache")   # gitignored

# Malware images -> Malimg corpus (Nataraj et al. 2011) via figshare mirror.
MALIMG_URL = "https://ndownloader.figshare.com/files/42443904"
MALIMG_ZIP = os.path.join(RAW_DIR, "malimg.zip")            # gitignored

# CICIDS2017 raw label -> SentinelAI class (DoS/DDoS floods collapse into DDoS,
# the two Patator families are password-guessing -> BruteForce).
CICIDS_LABEL_MAP = {
    "BENIGN": "Benign",
    "DDoS": "DDoS", "DoS Hulk": "DDoS", "DoS GoldenEye": "DDoS",
    "DoS slowloris": "DDoS", "DoS Slowhttptest": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Botnet",
}
# Per-class sampling caps (keeps the dataset tractable and honest about the
# real CICIDS2017 class imbalance; Bot has only 1,966 flows in total).
REAL_FLOW_CAP = {"Benign": 14000, "DDoS": 6000, "PortScan": 5500,
                 "BruteForce": 5000, "Botnet": 1966}

# Malimg folder name -> project family name.
MALIMG_FAMILY_MAP = {"Allaple.A": "Allaple_A", "C2LOP.P": "C2LOP_P",
                     "Lolyda.AA2": "Lolyda_AA2",
                     "Alueron.gen!J": "Alueron_genJ"}
# Per-family caps for the real corpus: every Malimg image (train+validation,
# both mirrors) is pooled and capped so the four classes stay balanced
# (C2LOP.P is kept whole at 171 images).
REAL_MALWARE_CAP = {"Allaple_A": 200, "C2LOP_P": 171, "Lolyda_AA2": 200,
                    "Alueron_genJ": 200}

# ----------------------------------------------------------------------
# 4. Shared ML constants
# ----------------------------------------------------------------------
TEST_SIZE = 0.20                    # 80/20 stratified train/test split
CV_FOLDS = 5                        # cross-validation folds (documentation)
SVM_TRAIN_CAP = 10000               # SVM trained on a capped subsample
K_RANGE = range(2, 9)               # k values tested for K-Means
SELECT_K_FEATURES = 12              # features kept after feature selection

# ----------------------------------------------------------------------
# 5. Reinforcement-learning constants
# ----------------------------------------------------------------------
THREAT_LEVELS = ["Benign", "Suspicious", "Malicious"]
CONF_LEVELS = ["Low", "Medium", "High"]
CRITICALITY_LEVELS = ["Low", "High"]
RL_ACTIONS = ["Monitor", "Alert_Analyst", "Block_IP", "Isolate_Host"]
RL_EPISODES = 6000
RL_ALPHA = 0.10                     # learning rate
RL_GAMMA = 0.90                     # discount factor
RL_EPS_START, RL_EPS_END, RL_EPS_DECAY = 1.0, 0.05, 0.9995

# ----------------------------------------------------------------------
# 6. Small utilities used by every stage
# ----------------------------------------------------------------------
_STAGE_TIMING = {}


class StageTimer:
    """Context manager that prints stage banners and measures duration."""

    def __init__(self, stage_no: int, name: str):
        self.stage_no = stage_no
        self.name = name

    def __enter__(self):
        print("\n" + "=" * 74)
        print(f"  STAGE {self.stage_no:>2} | {self.name}")
        print("=" * 74)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        _STAGE_TIMING[self.name] = round(dt, 2)
        print(f"  --> {self.name} finished in {dt:.1f}s")
        return False


def save_json(path: str, obj: dict) -> str:
    """Save a dict as pretty JSON (creates parent folders)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
