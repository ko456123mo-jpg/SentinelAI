"""Integration tests - run AFTER the pipeline has been trained once
(they verify saved artifacts + a live end-to-end agent decision).

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                   # noqa: E402

MODELS_OK = os.path.exists(os.path.join(
    config.MODEL_DIR, "preprocess_pipeline.joblib"))


@unittest.skipUnless(MODELS_OK, "trained artifacts not found - "
                                "run `python -m src.main` first")
class TestTrainedPipeline(unittest.TestCase):

    def test_supervised_accuracy_above_threshold(self):
        import json
        with open(os.path.join(config.REPORT_DIR,
                               "supervised_results.json")) as fh:
            res = json.load(fh)
        best = res["metrics_per_model"][res["best_model"]]
        self.assertGreater(best["accuracy"], 0.90)
        self.assertGreater(best["f1_macro"], 0.85)

    def test_agent_produces_valid_incident(self):
        import pandas as pd
        from src.agent import SentinelAgent
        agent = SentinelAgent().load()
        raw = pd.read_csv(os.path.join(config.RAW_DIR,
                                       "network_flows.csv"))
        row = raw.iloc[0]
        incident = agent.analyze_flow(row.drop(labels=["label"]).to_dict())
        for key in ("prediction", "confidence", "risk_score",
                    "response_action", "rl_q_values"):
            self.assertIn(key, incident)
        self.assertIn(incident["response_action"], config.RL_ACTIONS)

    def test_agent_catches_obvious_ddos(self):
        """A hand-crafted flow matching the REAL CICIDS2017 DDoS (LOIC)
        signature - a long-lived HTTP flood on port 80 with large
        packet-size variance - must be classified as DDoS and escalated."""
        from src.agent import SentinelAgent
        agent = SentinelAgent().load()
        incident = agent.analyze_flow({
            "duration": 100.0, "protocol": "TCP", "src_port": 46707,
            "dst_port": 80, "src_bytes": 336.0, "dst_bytes": 11595.0,
            "src_pkts": 6, "dst_pkts": 6, "syn_rate": 0.0,
            "ack_rate": 0.07, "psh_rate": 0.0, "avg_pkt_size": 852.0,
            "byte_std": 1497.0, "flow_iat_mean": 6.6,
            "active_duration": 0.0, "is_land": 0, "criticality": 1,
        })
        self.assertEqual(incident["prediction"], "DDoS")
        self.assertIn(incident["response_action"],
                      ("Block_IP", "Isolate_Host", "Alert_Analyst"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
