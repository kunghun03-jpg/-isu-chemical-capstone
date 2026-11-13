"""Web environment guards and real loader tests using synthetic model bundles."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch

from scripts import setup_web_env, verify_web_env
from vibration.inference_bundle import manifest_id, sha256_file
from vibration.model import Autoencoder
from vibration.preprocessing import PREPROCESSING_CONFIG


def synthetic_bundles(root):
    """Random/minimal fixtures only, never real trained models or performance data."""
    from scripts.verify_web_env import installed_versions

    versions = installed_versions()
    for rpm in (900, 1200, 1800):
        folder = Path(root) / f"{rpm}rpm"
        folder.mkdir(parents=True)
        fingerprint = "synthetic-web-loading-test"
        with torch.random.fork_rng():
            torch.manual_seed(42)
            torch.save({"model_state_dict": Autoencoder().state_dict(),
                        "data_fingerprint": fingerprint}, folder / "autoencoder.pt")
        np.savez(folder / "normalization.npz", mean=np.zeros((1, 1, 3201)), std=np.ones((1, 1, 3201)))
        features = np.random.default_rng(42).normal(size=(8, 3201))
        models = {"autoencoder": {"artifact": "autoencoder.pt",
                                   "score_method": "double_precision_reconstruction_MSE",
                                   "file_threshold": 0., "segment_threshold": 0.}}
        for name, model in (("ocsvm", OneClassSVM()), ("iforest", IsolationForest(n_estimators=2, random_state=42))):
            model.fit(features)
            joblib.dump({"model": model, "config": {"data_fingerprint": fingerprint},
                         "thresholds": {"file": 0., "segment": 0.}}, folder / f"{name}.joblib")
            models[name] = {"artifact": f"{name}.joblib", "score_method": "negative_score_samples",
                            "file_threshold": 0., "segment_threshold": 0.}
        manifest = {
            "schema_version": 1, "rpm": rpm, "preprocessing": deepcopy(PREPROCESSING_CONFIG),
            "normalization": {"mode": "per_frequency", "artifact": "normalization.npz"},
            "file_aggregation": "arithmetic_mean_of_8_segment_scores", "decision_rule": "score > threshold",
            "versions": {k: versions[k] for k in ("numpy", "torch", "scikit_learn")},
            "data_fingerprint": fingerprint, "models": models,
            "artifacts": {name: sha256_file(folder / name) for name in
                          ("normalization.npz", "autoencoder.pt", "ocsvm.joblib", "iforest.joblib")},
        }
        manifest["bundle_id"] = manifest_id(manifest)
        (folder / "manifest.json").write_text(json.dumps(manifest))


class WebEnvTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="web_env_test_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "checkout"
        self.project.mkdir()
        self.home = self.root / "home"
        (self.home / "capstone/.venv").mkdir(parents=True)
        (self.home / "capstone/.venv/sentinel").write_text("preserve")

    def test_protected_original_env_and_checkout_rejected(self):
        old = self.home / "capstone/.venv"
        for target in (old, old / "child", old.parent, self.project / "web-env"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                setup_web_env.validate_target(target, self.project, self.home)
        self.assertEqual((old / "sentinel").read_text(), "preserve")

    def test_only_owned_external_environment_can_be_reused(self):
        target = self.home / "web-env"
        self.assertEqual(setup_web_env.validate_target(target, self.project, self.home), target)
        target.mkdir()
        with self.assertRaises(ValueError):
            setup_web_env.validate_target(target, self.project, self.home)
        (target / "pyvenv.cfg").write_text("home = synthetic\n")
        (target / setup_web_env.MARKER).write_text(json.dumps({"project": str(self.project)}))
        self.assertEqual(setup_web_env.validate_target(target, self.project, self.home), target)
        link = self.home / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            setup_web_env.validate_target(link, self.project, self.home)

    def test_version_mismatch_blocks_model_loading(self):
        versions = dict(verify_web_env.PINNED, numpy="2.5.4")
        with patch.object(verify_web_env, "installed_versions", return_value=versions):
            report = verify_web_env.verify(self.root / "absent")
        self.assertFalse(report["ready"])
        self.assertEqual(report["bundles"], [])

    def test_missing_real_artifacts_never_reports_success(self):
        with patch.object(verify_web_env, "installed_versions", return_value=verify_web_env.PINNED.copy()):
            report = verify_web_env.verify(self.root / "absent")
        self.assertFalse(report["ready"])
        self.assertEqual(len(report["bundles"]), 3)
        self.assertTrue(all("FileNotFoundError" in item["error"] for item in report["bundles"]))

    def test_load_synthetic_three_model_bundles_and_reject_tampering(self):
        bundles = self.root / "bundles"
        synthetic_bundles(bundles)
        with patch.object(verify_web_env, "installed_versions", return_value=verify_web_env.PINNED.copy()):
            report = verify_web_env.verify(bundles)
            self.assertTrue(report["ready"])
            self.assertTrue(all(item["models"] == ["autoencoder", "ocsvm", "iforest"] for item in report["bundles"]))
            self.assertFalse(report["actual_mat_inference_validated"])
            with (bundles / "900rpm/autoencoder.pt").open("ab") as stream:
                stream.write(b"tamper")
            report = verify_web_env.verify(bundles)
        self.assertFalse(report["ready"])
        self.assertFalse(report["bundles"][0]["loaded"])
        self.assertTrue(report["bundles"][1]["loaded"])


if __name__ == "__main__":
    unittest.main()
