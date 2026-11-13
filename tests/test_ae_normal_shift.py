"""Synthetic NPZ/checkpoint diagnostics; not measured real AE performance."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.analyze_ae_normal_shift import analyze
from scripts.analyze_date_score_distributions import write_csv
from tests.test_web_env import synthetic_bundles
from vibration.comparison_metrics import score_rows, file_scores
from vibration.dataset import write_rows
from vibration.experiment import load_experiment
from vibration.inference_bundle import manifest_id, sha256_file
from vibration.model import Autoencoder
from vibration.preprocessing import apply_normalization, frequency_bins


class AENormalShiftTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ae_normal_shift_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.outputs = self.root / "inputs"
        self.name = "date_prelim_20261106"
        self.bundles = self.outputs / "inference_bundles" / self.name
        synthetic_bundles(self.bundles)
        rng = np.random.default_rng(8)
        self.thresholds = {}
        for rpm in (900, 1200, 1800):
            source = self.outputs / f"{rpm}rpm" / self.name
            source.mkdir(parents=True)
            rows, arrays, meta = [], {}, {}
            for split, date in (("train", "20260912"), ("validation", "20260916"),
                                ("test_normal", "20260919"), ("test_anomaly", "20260919")):
                conditions = ("IR",) if split == "test_anomaly" else ("Normal1", "Normal2")
                records = []
                for condition in conditions:
                    record = {"file": f"data/{date}/{date}_{rpm}RPM/{condition}/Vib/{split}.mat",
                              "date": date, "rpm": str(rpm), "condition": condition,
                              "run_id": "", "split": split, "date_group": f"date:{date}"}
                    records.append(record); rows.append(record)
                array = rng.normal(.3, .05, size=(len(records) * 8, 1, 3201)).astype(np.float32)
                array[:, :, :5] = .2 if split != "test_normal" else .2002
                if split == "test_normal":
                    array[:, :, 5:] += .1
                arrays[split] = array
                meta[split] = {key: np.repeat([r[key] for r in records], 8)
                               for key in ("file", "date", "rpm", "condition", "run_id", "date_group")}
                meta[split]["source_file"] = meta[split].pop("file")
                meta[split]["segment_id"] = np.tile(np.arange(8), len(records))
            mean = arrays["train"].astype(np.float64).mean(axis=0, keepdims=True)
            std = np.maximum(arrays["train"].astype(np.float64).std(axis=0, keepdims=True), 1e-6)
            for split, values in arrays.items():
                np.savez(source / f"{split}_spectra.npz", X=values, frequency=frequency_bins(), **meta[split])
                np.savez(source / f"{split}_normalized.npz", X=apply_normalization(values, mean, std), **meta[split])
            write_rows(source / f"split_{rpm}rpm.csv", rows, fields=list(rows[0]))
            (source / f"split_config_{rpm}rpm.json").write_text(json.dumps({"mode": "date_preliminary", "group_type": "date_not_actual_run"}))
            data, fingerprint = load_experiment(source, rpm)
            with torch.random.fork_rng():
                torch.manual_seed(42)
                model = Autoencoder().eval()
            all_rows = []
            with torch.no_grad():
                for split, values in data.items():
                    x = torch.from_numpy(values["X"]).float()
                    scores = (model(x).double() - x.double()).square().mean(dim=(1, 2)).numpy()
                    all_rows.extend(score_rows(values, split, scores))
            files = file_scores(all_rows)
            threshold = float(np.percentile([r["score"] for r in files if r["split"] == "validation"], 95))
            self.thresholds[rpm] = threshold
            for row in files:
                row.update(threshold=threshold, predicted_anomaly=int(row["score"] > threshold))
            write_csv(source / f"evaluation_file_{rpm}rpm.csv", [r for r in files if r["split"].startswith("test_")])
            checkpoint = {"model_state_dict": model.state_dict(), "data_fingerprint": fingerprint, "epoch": 30}
            torch.save(checkpoint, source / f"autoencoder_{rpm}rpm.pt")
            folder = self.bundles / f"{rpm}rpm"
            torch.save(checkpoint, folder / "autoencoder.pt")
            np.savez(folder / "normalization.npz", mean=mean, std=std)
            path = folder / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["data_fingerprint"] = fingerprint
            manifest["models"]["autoencoder"]["file_threshold"] = threshold
            manifest["artifacts"] = {name: sha256_file(folder / name) for name in manifest["artifacts"]}
            manifest["bundle_id"] = manifest_id(manifest)
            path.write_text(json.dumps(manifest))

    def test_complete_diagnostics_reproduce_fpr_preserve_thresholds_and_separate_hypotheses(self):
        output = self.root / "results"
        with patch("torch.Tensor.backward", side_effect=AssertionError("training")), patch(
            "torch.optim.Adam", side_effect=AssertionError("optimizer")
        ):
            report = analyze(self.outputs, self.bundles, output)
        self.assertEqual(report["status"], "completed_diagnostics")
        self.assertTrue(report["source_unchanged"])
        self.assertFalse(report["retraining"])
        self.assertFalse(report["threshold_changed"])
        self.assertTrue(report["hypotheses_not_confirmed"])
        for item in report["observations"]:
            rpm = item["rpm"]
            self.assertEqual(item["stored_file_threshold"], self.thresholds[rpm])
            self.assertGreaterEqual(item["std_floor_bins"], 5)
            self.assertEqual(item["fpr_comparison"][0]["recomputed_fp"], item["fpr_comparison"][0]["original_fp"])
            test = next(d for d in item["distributions"] if d["split"] == "test_normal" and d["condition"] == "ALL")
            train = next(d for d in item["distributions"] if d["split"] == "train" and d["condition"] == "ALL")
            self.assertGreater(test["median"], train["median"])
            for suffix in ("scores.png", "spectra.png", "normalization.png", "band_errors.csv", "band_shifts.csv", "profiles.npz"):
                self.assertTrue((output / f"{rpm}_{suffix}").is_file())
        with self.assertRaises(FileExistsError):
            analyze(self.outputs, self.bundles, output)

    def test_missing_npz_fails_with_report_without_invented_observations(self):
        (self.outputs / "900rpm" / self.name / "validation_spectra.npz").unlink()
        output = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            analyze(self.outputs, self.bundles, output)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["source_unchanged"])
        self.assertEqual(report["observations"], [])

    def test_corrupt_original_scores_are_not_silently_used_as_new_results(self):
        path = self.outputs / "900rpm" / self.name / "evaluation_file_900rpm.csv"
        from scripts.analyze_date_score_distributions import read_csv
        rows = read_csv(path)
        for row in rows:
            if row["split"] == "test_normal":
                row["score"] = str(float(row["score"]) + 1)
        write_csv(path, rows)
        with self.assertRaises(AssertionError):
            analyze(self.outputs, self.bundles, self.root / "bad_scores")

    def test_output_inside_inputs_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze(self.outputs, self.bundles, self.outputs / "new-results")
