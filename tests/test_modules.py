"""Unit tests for SentinelAI modules (fast, no full retraining).

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config                                   # noqa: E402
from src.reinforcement_learning import (QLearningAgent,   # noqa: E402
                                        SecurityResponseEnv)
from src.preprocessing import clean                      # noqa: E402
from src.nlp import clean_text                           # noqa: E402


class TestRLComponents(unittest.TestCase):

    def test_state_encode_decode_roundtrip(self):
        for s in range(18):
            t, c, cr = SecurityResponseEnv.decode(s)
            self.assertEqual(SecurityResponseEnv.encode(t, c, cr), s)

    def test_q_learning_updates_q_toward_reward(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        env = SecurityResponseEnv()
        s = SecurityResponseEnv.encode(2, 2, 1)          # malicious/high
        env.state = s
        before = float(agent.Q[s, 3])
        _, r = env.step(3)                               # isolate -> +5
        agent.learn(s, 3, r, 0)
        self.assertGreater(float(agent.Q[s, 3]), before)

    def test_env_punishes_missing_malicious_traffic(self):
        env = SecurityResponseEnv()
        env.state = SecurityResponseEnv.encode(2, 2, 1)  # malicious, high
        _, r_monitor = env.step(0)
        _, r_isolate = env.step(3)
        self.assertLess(r_monitor, 0)
        self.assertGreater(r_isolate, r_monitor)

    def test_env_punishes_blocking_benign(self):
        env = SecurityResponseEnv()
        env.state = 0                                     # benign
        _, r = env.step(2)                                # block
        self.assertEqual(r, -3)


class TestNlpCleaning(unittest.TestCase):

    def test_masks_urls_and_numbers(self):
        out = clean_text("Visit http://x.com now or call 0871 234567")
        self.assertIn("url", out)
        self.assertIn("num", out)
        self.assertNotIn("http", out)

    def test_lowercase_and_punctuation(self):
        out = clean_text("WINNER!!! Claim your PRIZE...")
        self.assertEqual(out, out.lower())
        self.assertNotIn("!", out)


class TestPreprocessing(unittest.TestCase):

    def test_clean_removes_duplicates_and_missing(self):
        df = pd.DataFrame({
            "avg_pkt_size": [100.0, -5.0, 100.0],
            "byte_std": [np.nan, 2.0, np.nan],
        })
        # duplicates are only detected on identical FULL rows -> craft one
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = {}
        out = clean(df, report)
        self.assertEqual(report["missing_after"], 0)
        self.assertGreaterEqual(report["duplicates_removed"], 1)
        self.assertEqual((out["avg_pkt_size"] < 0).sum(), 0)


class TestDataGeneration(unittest.TestCase):

    def test_malware_pattern_shape_and_range(self):
        from src.data_collection import _family_pattern
        for fam in config.MALWARE_FAMILIES:
            img = _family_pattern(fam)
            self.assertEqual(img.shape,
                             (config.MALWARE_IMG_SIZE, config.MALWARE_IMG_SIZE))
            self.assertTrue((img >= 0).all() and (img <= 255).all())

    def test_agent_state_routing(self):
        from src.agent import SentinelAgent
        ag = SentinelAgent()
        self.assertEqual(ag.route({"dst_port": 22}), "flow")
        self.assertEqual(ag.route({"text": "hi"}), "message")
        self.assertEqual(ag.route({"image_path": "x.png"}), "image")
        with self.assertRaises(ValueError):
            ag.route({"unknown": 1})


class TestThreatIntel(unittest.TestCase):

    def test_domain_extraction_and_ioc(self):
        from src.agent import _DOMAIN_RE
        text = "Visit http://evil.example.com or mail us at x.y.org"
        found = {d.lower() for d in _DOMAIN_RE.findall(text)}
        self.assertIn("evil.example.com", found)
        self.assertIn("x.y.org", found)

    def test_ioc_check_against_fake_blacklist(self):
        from src.agent import SentinelAgent
        ag = SentinelAgent()
        ag.artifacts["blacklist"] = {"bad-domain.xyz"}
        self.assertEqual(ag.ioc_check("see bad-domain.xyz now"),
                         ["bad-domain.xyz"])
        self.assertEqual(ag.ioc_check("clean message"), [])

    def test_hog_features_shape(self):
        from skimage.feature import hog
        img = np.random.default_rng(0).random((48, 48))
        f = hog(img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys")
        self.assertEqual(f.ndim, 1)
        self.assertGreater(len(f), 100)



class TestRealDataMapping(unittest.TestCase):
    """The CICIDS2017 / Malimg -> SentinelAI schema mapping (no download)."""

    def test_cicids_label_map_covers_the_five_classes(self):
        import pandas as pd
        from src import config
        raw = ["BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye",
               "DoS slowloris", "DoS Slowhttptest", "PortScan",
               "FTP-Patator", "SSH-Patator", "Bot", "Heartbleed"]
        mapped = pd.Series(raw).map(config.CICIDS_LABEL_MAP)
        self.assertEqual(mapped[0], "Benign")
        self.assertEqual(mapped[1], "DDoS")
        self.assertEqual(mapped[6], "PortScan")
        self.assertEqual(mapped[7], "BruteForce")
        self.assertEqual(mapped[9], "Botnet")
        # Heartbleed (11 rows) is intentionally out of scope -> dropped
        self.assertTrue(pd.isna(mapped[10]))

    def test_real_flow_caps_cover_all_classes(self):
        from src import config
        self.assertEqual(set(config.REAL_FLOW_CAP.keys()),
                         set(config.FLOW_CLASSES))
        self.assertTrue(all(v > 0 for v in config.REAL_FLOW_CAP.values()))

    def test_malimg_family_map_matches_project_families(self):
        from src import config
        self.assertEqual(set(config.MALIMG_FAMILY_MAP.values()),
                         set(config.MALWARE_FAMILIES))
        self.assertIn("Allaple.A", config.MALIMG_FAMILY_MAP)

    def test_cicids_feature_derivation(self):
        """syn/ack/psh rates are flag-counts normalised by total packets."""
        import pandas as pd
        total = pd.Series([4.0, 0.0]).clip(lower=1.0)
        syn = (pd.Series([4, 0]) / total).clip(0, 1)
        self.assertAlmostEqual(syn[0], 1.0)
        self.assertAlmostEqual(syn[1], 0.0)


class TestUnsupervisedExtras(unittest.TestCase):

    def test_hierarchical_clustering_recovers_blobs(self):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score
        X, y = make_blobs(n_samples=300,
                          centers=[(-8, -8), (0, 0), (8, 8)],
                          cluster_std=0.6, random_state=0)
        labels = AgglomerativeClustering(n_clusters=3,
                                         linkage="ward").fit_predict(X)
        self.assertGreater(adjusted_rand_score(y, labels), 0.9)

    def test_dbscan_flags_uniform_data_as_noise(self):
        from sklearn.cluster import DBSCAN
        X = np.random.default_rng(1).random((300, 2))
        labels = DBSCAN(eps=0.02, min_samples=5).fit_predict(X)
        self.assertGreater(float((labels == -1).mean()), 0.5)


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
