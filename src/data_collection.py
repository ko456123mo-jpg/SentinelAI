"""
SentinelAI - Stage 2: Data Collection
=====================================
Collects / generates the three data sources used by the system:

1. NETWORK TRAFFIC FLOWS  (generated - statistically modelled, documented)
   25,000 bidirectional flow records with 16 features and 5 classes
   (Benign / DDoS / PortScan / BruteForce / Botnet).
   Generation is fully seeded => reproducible. Using generated data is
   explicitly allowed by the project requirements ("generated data"),
   and every distribution used is documented in the code below.

2. SPAM / PHISHING TEXTS  (REAL public dataset - UCI Machine Learning
   Repository, SMS Spam Collection, 5,574 real messages).
   Downloaded from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
   If the raw file is missing (offline run) a small documented fallback
   corpus is generated instead.

3. MALWARE VISUALIZATION IMAGES (generated).
   Simulates the classical "malware-to-image" approach (Nataraj et al. 2011):
   each malware binary is rendered as a 48x48 grayscale byte-plot.
   Four synthetic families with distinct visual signatures are produced.

Outputs -> data/raw/
    network_flows.csv, sms_spam.csv, malware_images/<family>/*.png,
    dataset_documentation.json
"""

import os

import numpy as np
import pandas as pd

from src import config
from src.config import StageTimer, save_json

RNG = np.random.default_rng(config.SEED)


# ======================================================================
# 1. NETWORK TRAFFIC FLOWS  (generated)
# ======================================================================
def _gen_benign(n):
    """Normal user traffic - a MIXTURE so it overlaps attack signatures:

      62% web/dns/mail   - ordinary browsing, lookups, messaging
       9% IT/SSH admin   - legit remote work on port 22/3389  (looks like
                           BruteForce!)
       8% beacon apps    - push notifications / NTP, small packets, long
                           regular gaps (looks like Botnet C2!)
       7% monitoring     - legit short probes on many ports (looks like
                           PortScan!)
      14% flash crowd    - big downloads / video bursts (looks like DDoS!)
    """
    w = [0.62, 0.09, 0.08, 0.07, 0.14]
    parts = RNG.multinomial(n, w)

    frames = [pd.DataFrame({                       # ordinary web traffic
        "duration": RNG.lognormal(0.1, 0.9, parts[0]),
        "protocol": RNG.choice(["TCP", "UDP", "ICMP"], parts[0],
                               p=[0.62, 0.33, 0.05]),
        "src_port": RNG.integers(32768, 61000, parts[0]),
        "dst_port": RNG.choice([53, 80, 443, 993, 5228], parts[0]),
        "src_bytes": RNG.lognormal(7.2, 1.1, parts[0]),
        "dst_bytes": RNG.lognormal(7.8, 1.2, parts[0]),
        "src_pkts": RNG.integers(4, 40, parts[0]),
        "dst_pkts": RNG.integers(4, 45, parts[0]),
        "syn_rate": RNG.beta(2.5, 9.0, parts[0]),
        "ack_rate": RNG.beta(9.0, 2.0, parts[0]),
        "psh_rate": RNG.beta(2.0, 6.0, parts[0]),
        "avg_pkt_size": RNG.normal(720, 160, parts[0]),
        "byte_std": RNG.lognormal(5.5, 0.6, parts[0]),
        "flow_iat_mean": RNG.lognormal(4.8, 0.7, parts[0]),
        "active_duration": RNG.lognormal(-0.5, 0.9, parts[0]),
        "is_land": np.zeros(parts[0], dtype=int),
    })]

    frames.append(pd.DataFrame({                   # IT / SSH admin work
        "duration": RNG.uniform(1.0, 25.0, parts[1]),
        "protocol": "TCP",
        "src_port": RNG.integers(32768, 61000, parts[1]),
        "dst_port": RNG.choice([22, 3389, 21], parts[1]),
        "src_bytes": RNG.lognormal(6.0, 1.0, parts[1]),
        "dst_bytes": RNG.lognormal(6.4, 1.0, parts[1]),
        "src_pkts": RNG.integers(15, 90, parts[1]),
        "dst_pkts": RNG.integers(15, 90, parts[1]),
        "syn_rate": RNG.uniform(0.25, 0.55, parts[1]),
        "ack_rate": RNG.uniform(0.40, 0.80, parts[1]),
        "psh_rate": RNG.uniform(0.30, 0.70, parts[1]),
        "avg_pkt_size": RNG.uniform(200, 700, parts[1]),
        "byte_std": RNG.lognormal(4.2, 0.6, parts[1]),
        "flow_iat_mean": RNG.uniform(20, 800, parts[1]),
        "active_duration": RNG.uniform(1.0, 22.0, parts[1]),
        "is_land": np.zeros(parts[1], dtype=int),
    }))

    frames.append(pd.DataFrame({                   # beacon-like apps
        "duration": RNG.uniform(0.2, 2.5, parts[2]),
        "protocol": RNG.choice(["TCP", "UDP"], parts[2], p=[0.6, 0.4]),
        "src_port": RNG.integers(1024, 65535, parts[2]),
        "dst_port": RNG.choice([5228, 8080, 123, 443], parts[2]),
        "src_bytes": RNG.uniform(50, 600, parts[2]),
        "dst_bytes": RNG.uniform(50, 900, parts[2]),
        "src_pkts": RNG.integers(2, 18, parts[2]),
        "dst_pkts": RNG.integers(2, 18, parts[2]),
        "syn_rate": RNG.uniform(0.05, 0.35, parts[2]),
        "ack_rate": RNG.uniform(0.40, 0.90, parts[2]),
        "psh_rate": RNG.uniform(0.05, 0.40, parts[2]),
        "avg_pkt_size": RNG.uniform(70, 200, parts[2]),
        "byte_std": RNG.uniform(5, 120, parts[2]),
        "flow_iat_mean": RNG.uniform(4000, 55000, parts[2]),
        "active_duration": RNG.uniform(0.1, 2.2, parts[2]),
        "is_land": np.zeros(parts[2], dtype=int),
    }))

    frames.append(pd.DataFrame({                   # monitoring probes
        "duration": RNG.uniform(0.005, 0.12, parts[3]),
        "protocol": "TCP",
        "src_port": RNG.integers(40000, 65000, parts[3]),
        "dst_port": RNG.integers(1, 65535, parts[3]),
        "src_bytes": RNG.uniform(0, 120, parts[3]),
        "dst_bytes": RNG.uniform(0, 120, parts[3]),
        "src_pkts": RNG.integers(1, 6, parts[3]),
        "dst_pkts": RNG.integers(0, 4, parts[3]),
        "syn_rate": RNG.uniform(0.30, 0.85, parts[3]),
        "ack_rate": RNG.uniform(0.00, 0.35, parts[3]),
        "psh_rate": RNG.uniform(0.00, 0.20, parts[3]),
        "avg_pkt_size": RNG.uniform(20, 140, parts[3]),
        "byte_std": RNG.uniform(0, 60, parts[3]),
        "flow_iat_mean": RNG.uniform(0.05, 1.5, parts[3]),
        "active_duration": RNG.uniform(0.005, 0.1, parts[3]),
        "is_land": np.zeros(parts[3], dtype=int),
    }))

    frames.append(pd.DataFrame({                   # flash crowd / downloads
        "duration": RNG.uniform(2.0, 15.0, parts[4]),
        "protocol": RNG.choice(["TCP", "UDP"], parts[4], p=[0.8, 0.2]),
        "src_port": RNG.integers(32768, 61000, parts[4]),
        "dst_port": RNG.choice([80, 443], parts[4]),
        "src_bytes": RNG.lognormal(4.5, 0.7, parts[4]),
        "dst_bytes": RNG.lognormal(10.5, 1.0, parts[4]),   # big downloads
        "src_pkts": RNG.integers(200, 1800, parts[4]),     # heavy streams
        "dst_pkts": RNG.integers(200, 2200, parts[4]),
        "syn_rate": RNG.uniform(0.02, 0.45, parts[4]),
        "ack_rate": RNG.uniform(0.50, 0.98, parts[4]),
        "psh_rate": RNG.uniform(0.05, 0.45, parts[4]),
        "avg_pkt_size": RNG.uniform(500, 1400, parts[4]),
        "byte_std": RNG.lognormal(4.5, 0.6, parts[4]),
        "flow_iat_mean": RNG.uniform(0.5, 25.0, parts[4]),
        "active_duration": RNG.uniform(1.0, 14.0, parts[4]),
        "is_land": np.zeros(parts[4], dtype=int),
    }))

    return pd.concat(frames, ignore_index=True)


def _gen_ddos(n):
    """TCP SYN flood: thousands of small packets, almost no payload back."""
    return pd.DataFrame({
        "duration": RNG.uniform(1.0, 10.0, n),
        "protocol": RNG.choice(["TCP", "UDP"], n, p=[0.85, 0.15]),
        "src_port": RNG.integers(1024, 65535, n),
        "dst_port": RNG.choice([80, 443, 53], n),
        "src_bytes": RNG.lognormal(4.0, 0.5, n),           # tiny payloads
        "dst_bytes": RNG.uniform(0, 40, n),                # almost nothing back
        "src_pkts": RNG.integers(500, 5000, n),            # packet flood
        "dst_pkts": RNG.integers(0, 6, n),
        "syn_rate": RNG.uniform(0.80, 1.00, n),            # SYN storm
        "ack_rate": RNG.uniform(0.00, 0.15, n),
        "psh_rate": RNG.uniform(0.00, 0.20, n),
        "avg_pkt_size": RNG.uniform(40, 220, n),
        "byte_std": RNG.lognormal(2.0, 0.5, n),
        "flow_iat_mean": RNG.uniform(0.05, 2.0, n),
        "active_duration": RNG.uniform(0.5, 9.0, n),
        "is_land": np.zeros(n, dtype=int),
    })


def _gen_portscan(n):
    """Very short probes against sequential ports - zero payload."""
    return pd.DataFrame({
        "duration": RNG.uniform(0.001, 0.05, n),           # ultra short
        "protocol": "TCP",
        "src_port": RNG.integers(40000, 65000, n),
        "dst_port": RNG.integers(1, 65535, n),             # scans anything
        "src_bytes": RNG.uniform(0, 60, n),
        "dst_bytes": RNG.uniform(0, 40, n),
        "src_pkts": RNG.integers(1, 4, n),
        "dst_pkts": RNG.integers(0, 2, n),
        "syn_rate": RNG.uniform(0.55, 1.00, n),
        "ack_rate": RNG.uniform(0.00, 0.20, n),
        "psh_rate": RNG.uniform(0.00, 0.10, n),
        "avg_pkt_size": RNG.uniform(20, 80, n),
        "byte_std": RNG.uniform(0, 30, n),
        "flow_iat_mean": RNG.uniform(0.01, 0.30, n),
        "active_duration": RNG.uniform(0.001, 0.05, n),
        "is_land": np.zeros(n, dtype=int),
    })


def _gen_bruteforce(n):
    """Password-guessing against SSH/RDP/FTP/Telnet/MySQL."""
    return pd.DataFrame({
        "duration": RNG.uniform(0.5, 8.0, n),
        "protocol": RNG.choice(["TCP"], n),
        "src_port": RNG.integers(30000, 62000, n),
        "dst_port": RNG.choice([21, 22, 23, 3389, 3306], n),
        "src_bytes": RNG.lognormal(5.0, 0.6, n),           # login attempts
        "dst_bytes": RNG.uniform(0, 2200, n),              # banners/errors
        "src_pkts": RNG.integers(10, 60, n),
        "dst_pkts": RNG.integers(8, 50, n),
        "syn_rate": RNG.uniform(0.25, 0.60, n),
        "ack_rate": RNG.uniform(0.35, 0.75, n),
        "psh_rate": RNG.uniform(0.30, 0.70, n),
        "avg_pkt_size": RNG.uniform(90, 420, n),
        "byte_std": RNG.lognormal(4.0, 0.5, n),
        "flow_iat_mean": RNG.uniform(20, 400, n),
        "active_duration": RNG.uniform(0.3, 6.0, n),
        "is_land": (RNG.random(n) < 0.03).astype(int),     # rare loopback bug
    })


def _gen_botnet(n):
    """C2 beaconing: silent, regular, small, consistent packets."""
    return pd.DataFrame({
        "duration": RNG.uniform(0.2, 3.0, n),
        "protocol": RNG.choice(["TCP", "UDP"], n, p=[0.7, 0.3]),
        "src_port": RNG.integers(1024, 65535, n),
        "dst_port": RNG.choice([6667, 444, 1080, 8080], n),  # classic C2 ports
        "src_bytes": RNG.uniform(40, 300, n),
        "dst_bytes": RNG.uniform(40, 500, n),
        "src_pkts": RNG.integers(2, 15, n),
        "dst_pkts": RNG.integers(2, 15, n),
        "syn_rate": RNG.uniform(0.05, 0.35, n),
        "ack_rate": RNG.uniform(0.40, 0.90, n),
        "psh_rate": RNG.uniform(0.05, 0.40, n),
        "avg_pkt_size": RNG.uniform(60, 150, n),           # small & tight
        "byte_std": RNG.uniform(5, 60, n),                 # low variability
        "flow_iat_mean": RNG.uniform(8000, 65000, n),      # long regular gaps
        "active_duration": RNG.uniform(0.1, 2.0, n),
        "is_land": np.zeros(n, dtype=int),
    })


_FLOW_GENERATORS = {
    "Benign": _gen_benign,
    "DDoS": _gen_ddos,
    "PortScan": _gen_portscan,
    "BruteForce": _gen_bruteforce,
    "Botnet": _gen_botnet,
}


def generate_network_flows() -> str:
    """Build the labelled flow dataset and inject realistic imperfections."""
    counts = RNG.multinomial(config.N_FLOWS, config.FLOW_CLASS_WEIGHTS)
    frames = []
    for cls, n in zip(config.FLOW_CLASSES, counts):
        df = _FLOW_GENERATORS[cls](int(n))
        df["label"] = cls
        frames.append(df)

    data = (pd.concat(frames, ignore_index=True)
            .sample(frac=1.0, random_state=config.SEED)
            .reset_index(drop=True))

    # --- realistic sensor/exporter measurement noise ----------------------
    # rates get +-6% jitter, byte/pkt counters get ~15% multiplicative noise
    for col in ("syn_rate", "ack_rate", "psh_rate"):
        data[col] = (data[col] + RNG.normal(0, 0.06, len(data))).clip(0, 1)
    for col in ("src_bytes", "dst_bytes"):
        data[col] = data[col] * RNG.lognormal(0, 0.15, len(data))
    data["avg_pkt_size"] = (data["avg_pkt_size"] *
                            RNG.lognormal(0, 0.12, len(data))).clip(lower=0)

    # --- inject imperfections so preprocessing has real work to do -------
    n_rows = len(data)
    # (a) duplicates (double-counted exporter logs)
    dup_idx = RNG.choice(n_rows, int(n_rows * config.DUPLICATE_RATE),
                         replace=False)
    data = pd.concat([data, data.iloc[dup_idx]], ignore_index=True)
    # (b) missing values (sensor/exporter glitches) in 2 numeric columns
    for col in ("byte_std", "flow_iat_mean"):
        holes = RNG.choice(len(data), int(len(data) * config.MISSING_RATE),
                           replace=False)
        data.loc[holes, col] = np.nan
    # (c) impossible negative glitches in avg_pkt_size (0.3%)
    neg = RNG.choice(len(data), int(len(data) * 0.003), replace=False)
    data.loc[neg, "avg_pkt_size"] = -np.abs(data.loc[neg, "avg_pkt_size"])

    data = data.sample(frac=1.0, random_state=config.SEED).reset_index(
        drop=True)
    path = os.path.join(config.RAW_DIR, "network_flows.csv")
    data.to_csv(path, index=False)
    print(f"    [data] network_flows.csv -> {data.shape[0]:,} rows x "
          f"{data.shape[1]} cols  (after injecting duplicates/missing)")
    return path


# ======================================================================
# 2. SPAM TEXTS  (real UCI dataset with documented fallback)
# ======================================================================
_FALLBACK_SPAM = [
    ("spam", "URGENT! You have won a 1 week FREE membership in our £100,000 "
             "Prize Jackpot! Txt the word: CLAIM to 81025 T&Cs 08712405020"),
    ("spam", "WINNER!! As a valued network customer you have been selected "
             "to receive a £900 prize reward! To claim text CODE 87121"),
    ("spam", "FreeMsg Hey there darling it's been 3 week's now and no word "
             "back! I'd like some fun you up for it? Txt me now"),
    ("spam", "Congratulations ur awarded 500 of CD vouchers or 4 passes to "
             "a theme park! To claim txt VODAFONE to 87121"),
    ("spam", "You are a winner U have been specially selected 2 receive "
             "£1000 cash or a 4* holiday txt SA to 87121"),
    ("spam", "URGENT Your mobile number has won £5000, claim now by "
             "replying WIN to 80086"),
    ("spam", " sextext me now for free pics on this premium rate line"),
    ("spam", "PRIVATE! Your 2003 Account Statement shows 786 unredeemed "
             "points. Call 08719898213 to claim reward"),
    ("ham", "Hey are you free for lunch tomorrow? I found a nice place",
    ),
    ("ham", "I'll be home in 20 minutes, start the rice please"),
    ("ham", "Thanks for the birthday wishes! Really appreciate it"),
    ("ham", "Meeting moved to 3pm in the same room, see you there"),
    ("ham", "Can you pick up milk on your way home? Thanks"),
    ("ham", "The report looks good, just fix the last chart and send it"),
    ("ham", "Happy new year! Wishing you all the best in 2025"),
    ("ham", "Don't forget the gym at 6, I booked the court"),
]


def load_sms_dataset() -> pd.DataFrame:
    """Read the real UCI corpus, or build the documented fallback corpus."""
    if os.path.exists(config.SMS_RAW_FILE):
        df = pd.read_csv(config.SMS_RAW_FILE, sep="\t", header=None,
                         names=["label", "text"], encoding="utf-8")
        source = ("REAL dataset - UCI SMS Spam Collection (5,574 real "
                  "messages). https://archive.ics.uci.edu/ml/datasets/"
                  "SMS+Spam+Collection")
    else:
        rows = _FALLBACK_SPAM * 12            # documented synthetic fallback
        df = pd.DataFrame(rows, columns=["label", "text"])
        source = ("Generated fallback corpus (offline run) - documented "
                  "synthetic spam/ham messages")
    df = df.dropna().reset_index(drop=True)
    df.to_csv(config.SMS_CSV, index=False)
    print(f"    [data] sms_spam.csv -> {len(df):,} messages "
          f"({(df.label == 'spam').sum():,} spam / {(df.label == 'ham').sum():,} ham)")
    print(f"           source: {source}")
    return df


# ======================================================================
# 3. MALWARE BYTE-PLOT IMAGES  (generated, 4 synthetic families)
# ======================================================================
def _family_pattern(family: str) -> np.ndarray:
    """Visual signature of each synthetic malware family (48x48 grayscale).

    Mimics how real families look in byte-plot visualization:
      Allaple_A : high-entropy noise with faint horizontal bands
      Autorun_K : strong vertical stripes (repeating code sections)
      C2LOP_P   : sharp rectangular blocks (packed sections)
      Yuner_A   : smooth low-entropy gradient (unpacked interpret code)
    """
    s = config.MALWARE_IMG_SIZE
    yy, xx = np.mgrid[0:s, 0:s]
    brightness = RNG.uniform(-18, 18)        # per-sample intensity shift
    noise = {"Allaple_A": 22, "Autorun_K": 26, "C2LOP_P": 24,
             "Yuner_A": 20}[family]

    if family == "Allaple_A":
        img = RNG.integers(0, 256, (s, s)).astype(float)
        img += 60 * np.sin(2 * np.pi * yy / 6.0)          # faint bands
    elif family == "Autorun_K":
        img = 90 + 70 * np.sin(2 * np.pi * xx / 4.5)      # vertical stripes
    elif family == "C2LOP_P":
        img = np.full((s, s), 35.0)
        for _ in range(RNG.integers(5, 9)):               # random blocks
            y0, x0 = RNG.integers(0, s - 14, 2)
            h, w = RNG.integers(8, 15), RNG.integers(8, 15)
            img[y0:y0 + h, x0:x0 + w] = RNG.uniform(150, 255)
    else:  # Yuner_A
        img = 40 + 170 * (xx + yy) / (2.0 * s)            # smooth gradient
    img += RNG.normal(0, noise, (s, s))
    img += brightness
    # random occlusions (packing/obfuscation artefacts)
    for _ in range(RNG.integers(2, 5)):
        y0, x0 = RNG.integers(0, s - 6, 2)
        img[y0:y0 + RNG.integers(3, 7), x0:x0 + RNG.integers(3, 7)] = \
            RNG.uniform(0, 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_malware_images() -> str:
    """Render PNG byte-plots for every family into data/raw/malware_images."""
    try:
        from PIL import Image
    except ImportError:                                    # pragma: no cover
        print("    [skip] Pillow missing - malware images not generated")
        return config.MALWARE_IMG_DIR

    for fam in config.MALWARE_FAMILIES:
        fam_dir = os.path.join(config.MALWARE_IMG_DIR, fam)
        os.makedirs(fam_dir, exist_ok=True)
        for i in range(config.MALWARE_PER_FAMILY):
            Image.fromarray(_family_pattern(fam)).save(
                os.path.join(fam_dir, f"{fam}_{i:04d}.png"))
    total = config.MALWARE_PER_FAMILY * len(config.MALWARE_FAMILIES)
    print(f"    [data] malware_images -> {total} PNG byte-plots "
          f"({len(config.MALWARE_FAMILIES)} families x "
          f"{config.MALWARE_PER_FAMILY})")
    return config.MALWARE_IMG_DIR


# ======================================================================
# 4. THREAT-INTELLIGENCE FEED  (web / network programming)
# ======================================================================
FEED_URL = ("https://raw.githubusercontent.com/stamparm/blackbook/"
            "master/blackbook.txt")
# fallback list used only when the sandbox/host has no internet access
_FALLBACK_DOMAINS = [
    "malware-example-01.com", "phishing-example-02.ru",
    "c2-beacon-example-03.net", "fake-login-example-04.xyz",
    "driveby-download-example-05.info", "trojan-drop-example-06.top",
]


def download_threat_feed() -> pd.DataFrame:
    """Fetch a real public malware-domain blocklist over HTTP(S).

    Demonstrates legitimate web data collection (allowed by Stage 2:
    'public web/API data'). Falls back to a documented offline list.
    """
    import urllib.request

    source = FEED_URL
    try:
        req = urllib.request.Request(FEED_URL,
                                     headers={"User-Agent": "SentinelAI/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        domains = sorted({ln.strip().lower() for ln in text.splitlines()
                          if ln.strip() and not ln.startswith("#")})
        real = True
    except Exception as exc:                     # offline -> documented fallback
        print(f"    [warn] threat feed unreachable ({exc}) -> fallback list")
        domains = list(_FALLBACK_DOMAINS)
        real = False
        source = "documented offline fallback list"

    df = pd.DataFrame({"domain": domains})
    path = os.path.join(config.RAW_DIR, "threat_feed_domains.csv")
    df.to_csv(path, index=False)
    print(f"    [data] threat_feed_domains.csv -> {len(df):,} malicious "
          f"domains ({'REAL public feed: stamparm/blackbook' if real else 'fallback'})")
    return df


# ======================================================================
# Stage runner
# =======================================================================
def run(save_doc: bool = True) -> dict:
    """Execute the full data-collection stage."""
    with StageTimer(2, "Data Collection"):
        config.ensure_dirs()
        flows_path = generate_network_flows()
        sms_df = load_sms_dataset()
        imgs_path = generate_malware_images()
        feed_df = download_threat_feed()

        doc = {
            "network_flows": {
                "path": os.path.relpath(flows_path, config.ROOT),
                "rows": int(pd.read_csv(flows_path).shape[0]),
                "source": "Generated (seeded statistical models per class, "
                          "distributions documented in src/data_collection.py)",
                "classes": config.FLOW_CLASSES,
                "features": 16,
                "imperfections_injected": {
                    "duplicate_rows": config.DUPLICATE_RATE,
                    "missing_values": config.MISSING_RATE,
                    "negative_glitches": 0.003,
                },
            },
            "sms_spam": {
                "path": os.path.relpath(config.SMS_CSV, config.ROOT),
                "rows": int(len(sms_df)),
                "source": ("REAL: UCI ML Repository - SMS Spam Collection "
                           "(Almeida & Hidalgo)") if os.path.exists(
                    config.SMS_RAW_FILE) else "generated fallback corpus",
                "classes": ["ham", "spam"],
            },
            "malware_images": {
                "path": os.path.relpath(imgs_path, config.ROOT),
                "rows": int(config.MALWARE_PER_FAMILY *
                            len(config.MALWARE_FAMILIES)),
                "source": "Generated byte-plot visualizations inspired by "
                          "Nataraj et al. (2011) malware imaging",
                "classes": config.MALWARE_FAMILIES,
                "image_size": f"{config.MALWARE_IMG_SIZE}x"
                              f"{config.MALWARE_IMG_SIZE} grayscale",
            },
            "threat_feed": {
                "path": "data/raw/threat_feed_domains.csv",
                "rows": int(len(feed_df)),
                "source": ("REAL public web feed (HTTP download, network "
                           "programming): stamparm/blackbook malicious-"
                           "domain blocklist") if len(feed_df) > 100 else
                          "documented offline fallback list",
            },
        }
        if save_doc:
            save_json(os.path.join(config.RAW_DIR,
                                   "dataset_documentation.json"), doc)
    return doc


if __name__ == "__main__":
    run()
