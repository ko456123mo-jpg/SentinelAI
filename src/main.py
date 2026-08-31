"""
SentinelAI - Pipeline orchestrator (main.py)
============================================
Runs every stage in the order required by the project outline:

    Stage  1  Problem definition      (see README / docs report)
    Stage  2  Data collection
    Stage  3  Data preprocessing
    Stage  4  Supervised learning
    Stage  5  Model evaluation       (inside supervised module)
    Stage  6  Unsupervised learning
    Stage  7  Deep learning (MLP)
    Stage  8  NLP (spam/phishing)
    Stage  9  Computer vision (CNN)
    Stage 10  Reinforcement learning (Q-Learning)
    Stage 11  AI agent orchestration

Usage:
    python -m src.main                # run everything
    python -m src.main --stages collect,preprocess,supervised
    python -m src.main --skip rl,agent
    python -m src.main --list
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                     # noqa: E402
from src.config import StageTimer, save_json, load_json    # noqa: E402

STAGES = [
    ("collect",     "src.data_collection", "run"),
    ("preprocess",  "src.preprocessing", "run"),
    ("supervised",  "src.supervised_model", "run"),
    ("unsupervised", "src.unsupervised_model", "run"),
    ("dl",          "src.deep_learning", "run"),
    ("nlp",         "src.nlp", "run"),
    ("cv",          "src.computer_vision", "run"),
    ("rl",          "src.reinforcement_learning", "run"),
    ("agent",       "src.agent", "run"),
]
STAGE_RESULT_FILES = {
    "collect": "dataset_documentation.json",
    "preprocess": "preprocessing_report.json",
    "supervised": "supervised_results.json",
    "unsupervised": "unsupervised_results.json",
    "dl": "deep_learning_results.json",
    "nlp": "nlp_results.json",
    "cv": "cv_results.json",
    "rl": "rl_results.json",
    "agent": "agent_results.json",
}


def run_stages(selected):
    import importlib
    for key, module_name, fn in STAGES:
        if key not in selected:
            print(f"\n-- stage '{key}' skipped")
            continue
        mod = importlib.import_module(module_name)
        getattr(mod, fn)()


def main():
    parser = argparse.ArgumentParser(description="SentinelAI pipeline")
    parser.add_argument("--stages", type=str, default="all",
                        help="comma list or 'all'")
    parser.add_argument("--skip", type=str, default="",
                        help="stages to skip")
    parser.add_argument("--list", action="store_true",
                        help="list available stages")
    args = parser.parse_args()

    if args.list:
        print("stages:", ", ".join(s[0] for s in STAGES))
        return

    names = [s[0] for s in STAGES]
    selected = set(names) if args.stages == "all" else {
        s.strip() for s in args.stages.split(",") if s.strip()}
    selected -= {s.strip() for s in args.skip.split(",") if s.strip()}

    config.ensure_dirs()
    t0 = time.time()

    with StageTimer(1, "Problem Definition (recap)"):
        print(
            "  SentinelAI - unified threat detection & response.\n"
            "  Problem : SOC teams face floods of heterogeneous events\n"
            "            (traffic, messages, files) and slow manual triage.\n"
            "  Users   : security analysts / network operators.\n"
            "  System  : predicts the threat class, flags anomalies,\n"
            "            and recommends a response decision automatically.")

    run_stages(selected)

    # ---------------- aggregate summary ---------------------------------
    summary = {"total_runtime_sec": round(time.time() - t0, 1)}
    for key, fname in STAGE_RESULT_FILES.items():
        p = os.path.join(config.REPORT_DIR, fname)
        if os.path.exists(p):
            summary[key] = load_json(p)
    out = save_json(os.path.join(config.REPORT_DIR,
                                 "pipeline_summary.json"), summary)

    print("\n" + "=" * 74)
    print(f"  PIPELINE COMPLETE in {time.time() - t0:.1f}s")
    print(f"  summary -> {os.path.relpath(out, config.ROOT)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
