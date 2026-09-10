#!/usr/bin/env bash
# ============================================================
#  SentinelAI - AI Cybersecurity Assistant
#  Developer: Mohammed Moneer Al-absi (2023050086)
#  One-command launcher for Linux / Kali
#  Usage:   bash run_kali.sh
# ============================================================
cd "$(dirname "$0")" || exit 1

echo "=================================================="
echo "  SentinelAI - AI Cybersecurity Assistant"
echo "  Developer: Mohammed Moneer Al-absi"
echo "=================================================="

PY=""

# --- 1) Use the conda env "sentinel" if it exists (miniforge) ---
if [ -z "$PY" ] && [ -f "$HOME/miniforge/bin/activate" ]; then
    if source "$HOME/miniforge/bin/activate" sentinel 2>/dev/null; then
        PY=python
        echo "[+] Using conda environment: sentinel"
    fi
fi

# --- 2) Use an existing local .venv if present ---
if [ -z "$PY" ] && [ -f ".venv/bin/python" ]; then
    if source .venv/bin/activate; then
        PY=python
        echo "[+] Using local environment: .venv"
    fi
fi

# --- 3) Otherwise create a local .venv (avoids Kali's system-Python restrictions) ---
if [ -z "$PY" ]; then
    echo "[*] Creating local Python environment (.venv) - first run only..."
    if ! python3 -m venv .venv; then
        echo "[!] Could not create .venv"
        echo "    FIX: sudo apt install -y python3-venv python3-pip"
        exit 1
    fi
    source .venv/bin/activate && PY=python
    echo "[+] Created local environment: .venv"
fi

# --- Check Python version (TensorFlow needs 3.10 - 3.13) ---
if ! "$PY" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>/dev/null; then
    echo "[!] Incompatible Python version: $("$PY" --version 2>&1)"
    echo "    TensorFlow requires Python 3.10 - 3.13."
    echo "    FIX: create an env with Python 3.12/3.13, e.g.:"
    echo "         conda create -n sentinel python=3.12 -y && conda activate sentinel"
    exit 1
fi

# --- Install requirements on first run ---
if ! "$PY" -c "import flask" 2>/dev/null; then
    echo "[*] First run: installing requirements (needs internet, ~600 MB, 5-10 min)..."
    "$PY" -m pip install --quiet --upgrade pip
    if ! "$PY" -m pip install -r requirements.txt; then
        echo "[!] Installation failed. Check:"
        echo "    - Internet connection"
        echo "    - Disk space:  df -h   (need ~2 GB free)"
        exit 1
    fi
fi

echo "[*] Starting web console at: http://localhost:7860"
echo "    (browser opens automatically - keep this terminal open)"
echo "    (to stop: Ctrl+C)"
exec "$PY" -m src.webapp
