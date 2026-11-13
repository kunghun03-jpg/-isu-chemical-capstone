"""Offline evaluator tests with random weights and synthetic MAT/NPZ only."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from scripts.evaluate_dae_wsl import evaluate, inspect_datasets
from tests import test_dae as synthetic
from vibration.dae_preprocessing import normalize_spectrum, preprocess_mat_bytes
from vibration.dae_inference import SavedDAEInference


class DAEEvaluationTests(unittest.TestCase):
    def setUp(self):
        synthetic.DAEInferenceTests.setUp(self)
        temp = tempfile.TemporaryDirectory(prefix="dae_evaluation_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data"
        self.data.mkdir()

    def mat(self, rpm, condition, name="sample", date="20260919", seed=1):
        folder = self.data / date / f"{date}_{rpm}RPM" / condition / "Vib"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.mat"
        savemat(path, {"signals": np.random.default_rng(seed).normal(size=(6, 65536)), "fs": 25600})
        return path

    def test_rpm_metrics_per_condition_and_no_support_or_file_changes(self):
        # Zero threshold gives deterministic alarms for these random-weight fixtures.
        np.savez(self.folder / "normal_threshold.npz", threshold=0.0)
        for i, rpm in enumerate((900, 1200, 1800)):
            self.mat(rpm, "Normal1", seed=i * 3 + 1)
            self.mat(rpm, "IR", seed=i * 3 + 2)
            self.mat(rpm, "Looeness0.1", seed=i * 3 + 3)
        np.savez(self.folder / "fault_dataset.npz", X=np.zeros((1, 800)), labels=["IR"],
                 groups=["20260919_1200RPM"], file_names=["20260919_1200RPM/IR/Vib/sample.mat"])
        report = evaluate(self.folder, self.data, self.root / "results")
        self.assertTrue(report["source_unchanged"])
        self.assertIsNone(report["supported_rpms"])
        self.assertFalse(report["web_activation_authorized"])
        self.assertEqual(report["evaluation_label"], "독립성 미확인 예비 평가")
        self.assertEqual(report["counts"], {"discovered": 9, "evaluated": 9, "excluded": 0})
        self.assertEqual(report["leakage_and_provenance"]["fault"]["rpm_clues"], [1200])
        for item in report["metrics"]:
            self.assertEqual(item["rate"], 1.0)
            self.assertEqual(item["dates"], ["20260919"])
            if item["condition"] == "ALL":
                self.assertEqual(item["evaluated"], 1 if item["class"] == "normal" else 2)
        self.assertTrue(any(i["condition"] == "Looseness0.1" for i in report["metrics"]))
        self.assertFalse((self.folder / "manifest.json").exists())

    def test_exclusions_duplicates_bad_files_unknown_conditions_and_unavailable_rate(self):
        first = self.mat(900, "Normal1")
        self.mat(900, "Normal1", "copy").write_bytes(first.read_bytes())
        self.mat(900, "IR", seed=2)
        bad = self.mat(900, "IR", "broken", seed=3); bad.write_bytes(b"bad MAT")
        self.mat(900, "Unknown", seed=4)
        report = evaluate(self.folder, self.data, self.root / "results")
        self.assertEqual(report["counts"], {"discovered": 5, "evaluated": 2, "excluded": 3})
        self.assertEqual(report["exclusion_reasons"]["duplicate_ch3_signal"], 1)
        self.assertEqual(report["exclusion_reasons"]["condition_not_selected_or_unknown"], 1)
        missing = [i for i in report["metrics"] if i["rpm"] == 1200]
        self.assertTrue(all(i["rate"] is None for i in missing))
        with (self.root / "results/files.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any("MAT" in r["excluded_reason"] for r in rows))

    def test_metadata_overlap_and_known_training_members_are_excluded(self):
        path = self.mat(900, "Normal1", "train", date="20260912")
        self.mat(1200, "IR", "heldout", seed=2)
        values = normalize_spectrum(preprocess_mat_bytes(path.read_bytes()), self.mean, self.std).astype(np.float32)
        np.savez(self.folder / "normal_dataset_normalized.npz", mean=self.mean, std=self.std,
                 X_train=values[None], X_val=values[None], X_test=values[None],
                 train_files=[str(path)], val_files=[str(path)], test_files=[str(path)],
                 train_groups=["20260912_900RPM"], train_run_ids=["run-A"])
        audit, _, _ = inspect_datasets(self.folder)
        self.assertEqual(audit["normal"]["split_evidence"]["train"]["metadata_rpm_clues"], [900])
        self.assertEqual(audit["split_overlaps"][0]["equal_float32_input_count"], 1)
        self.assertTrue(audit["split_overlaps"][0]["file_ids"])
        report = evaluate(self.folder, self.data, self.root / "results")
        self.assertEqual(report["counts"]["evaluated"], 1)
        self.assertTrue(any("known_train" in k for k in report["exclusion_reasons"]))

    def test_matching_training_features_without_file_metadata_are_excluded(self):
        path = self.mat(900, "Normal1")
        self.mat(1200, "IR", seed=2)
        values = normalize_spectrum(preprocess_mat_bytes(path.read_bytes()), self.mean, self.std)
        np.savez(self.folder / "normal_dataset_normalized.npz", mean=self.mean, std=self.std,
                 X_train=values[None])
        report = evaluate(self.folder, self.data, self.root / "results")
        self.assertTrue(any("matching_train_or_validation_input" in k for k in report["exclusion_reasons"]))

    def test_no_results_path_guards_date_filter_and_no_overwrite(self):
        self.mat(900, "Normal1")
        with self.assertRaises(ValueError):
            evaluate(self.folder, self.data, self.data / "results")
        with self.assertRaises(ValueError):
            evaluate(self.folder, self.data, self.root / "filtered", dates=["20260912"])
        report = json.loads((self.root / "filtered/report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["evaluated"], 0)
        with self.assertRaises(FileExistsError):
            evaluate(self.folder, self.data, self.root / "filtered")
        with self.assertRaises(FileNotFoundError):
            evaluate(self.folder / "missing", self.data, self.root / "missing")

    def test_conflicting_signal_labels_and_object_metadata_do_not_claim_independence(self):
        normal = self.mat(900, "Normal1")
        self.mat(900, "IR").write_bytes(normal.read_bytes())
        self.mat(1200, "Normal2", seed=9)
        np.savez(self.folder / "fault_dataset.npz", X=np.zeros((1, 800)),
                 groups=np.array([{"rpm": 900}], dtype=object))
        report = evaluate(self.folder, self.data, self.root / "results")
        self.assertEqual(report["exclusion_reasons"]["duplicate_signal_conflicting_metadata"], 2)
        self.assertEqual(report["leakage_and_provenance"]["run_independence"], "unconfirmed")
        self.assertTrue(any("object" in n for n in report["leakage_and_provenance"]["notes"]))

    def test_nontrivial_fixed_threshold_rates_match_individual_legacy_adapter(self):
        paths = [self.mat(900, condition, str(seed), seed=seed)
                 for condition, seed in (("Normal1", 1), ("Normal2", 2), ("IR", 3), ("OR", 4))]
        engine = SavedDAEInference(self.folder)
        scores = [engine.predict_mat(path)["mse"] for path in paths]
        threshold = (sorted(scores)[1] + sorted(scores)[2]) / 2
        np.savez(self.folder / "normal_threshold.npz", threshold=threshold)
        report = evaluate(self.folder, self.data, self.root / "results")
        groups = {m["class"]: m for m in report["metrics"] if m["rpm"] == 900 and m["condition"] == "ALL"}
        self.assertEqual(groups["normal"]["rate"], sum(x > threshold for x in scores[:2]) / 2)
        self.assertEqual(groups["fault"]["rate"], sum(x > threshold for x in scores[2:]) / 2)
        self.assertEqual(groups["normal"]["alarms"] + groups["fault"]["alarms"], 2)
        self.assertEqual(report["threshold"], threshold)


if __name__ == "__main__":
    unittest.main()
