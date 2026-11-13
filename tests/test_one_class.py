"""가상 CH3 배열만 사용합니다. 실제 MAT 학습·최종 성능 평가가 아닙니다."""

from pathlib import Path
import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vibration import comparison_metrics, evaluate, file_evaluate, iforest, ocsvm, train
from vibration.comparison_metrics import file_scores, predictions, q95, score_rows
from vibration.dataset import read_rows, write_rows
from vibration.experiment import load_experiment
from vibration.model import Autoencoder
from vibration.one_class import (
    anomaly_scores,
    features,
    load_model,
    make_model,
    run_model,
)


class OneClassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="one_class_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        self.output.mkdir()
        self.rows = []
        rng = np.random.default_rng(42)
        self.arrays = {}
        for index, name in enumerate(
            ("train", "validation", "test_normal", "test_anomaly")
        ):
            n_files = 3 if name == "train" else 2
            condition = "IR" if name == "test_anomaly" else "Normal1"
            data = rng.normal(0, 0.5, size=(n_files * 8, 1, 3201))
            if name == "test_anomaly":
                data += 6
            self.arrays[name] = data
            for file_index in range(n_files):
                self.rows.append(
                    {
                        "file": f"data/20260916_1800RPM/{condition}/Vib/{name}_{file_index}.mat",
                        "date": "20260916",
                        "rpm": "1800",
                        "condition": condition,
                        "run_id": f"SYNTHETIC_ONLY_{index}",
                        "split": name,
                    }
                )
        mean = self.arrays["train"].mean(axis=0, keepdims=True)
        std = self.arrays["train"].std(axis=0, keepdims=True)
        for name, data in self.arrays.items():
            self.arrays[name] = ((data - mean) / std).astype(np.float32)
            records = [r for r in self.rows if r["split"] == name]
            np.savez_compressed(
                self.output / f"{name}_normalized.npz",
                X=self.arrays[name],
                source_file=np.repeat([r["file"] for r in records], 8),
                condition=np.repeat([r["condition"] for r in records], 8),
                date=np.repeat([r["date"] for r in records], 8),
                rpm=np.repeat([r["rpm"] for r in records], 8),
                run_id=np.repeat([r["run_id"] for r in records], 8),
                segment_id=np.tile(np.arange(8), len(records)),
            )
        write_rows(self.output / "split_1800rpm.csv", self.rows)
        (self.output / "split_config_1800rpm.json").write_text(
            json.dumps({"mode": "run"}), encoding="utf-8"
        )

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_both_models_fit_train_only_score_q95_and_reload(self):
        datasets, fingerprint = load_experiment(self.output, 1800)
        for name in ("ocsvm", "iforest"):
            with self.subTest(model=name):
                model = make_model(name)
                with patch(
                    "vibration.one_class.make_model", return_value=model
                ), patch.object(
                    type(model), "fit", autospec=True, side_effect=type(model).fit
                ) as fit:
                    self.quiet(run_model, self.output, 1800, name)
                    self.assertEqual(fit.call_count, 1)
                    np.testing.assert_array_equal(
                        fit.call_args.args[1], features(datasets["train"]["X"])
                    )
                base = self.output / "comparisons" / name
                restored = load_model(base / "model.joblib", fingerprint=fingerprint)
                scores = anomaly_scores(restored["model"], datasets["validation"]["X"])
                np.testing.assert_allclose(
                    scores, -model.score_samples(features(datasets["validation"]["X"]))
                )
                np.testing.assert_allclose(
                    scores, anomaly_scores(model, datasets["validation"]["X"])
                )
                validation = score_rows(datasets["validation"], "validation", scores)
                self.assertEqual(
                    restored["thresholds"]["segment"], float(np.percentile(scores, 95))
                )
                self.assertEqual(
                    restored["thresholds"]["file"], q95(file_scores(validation))
                )
                with (base / "scores_segment.csv").open(encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), sum(len(d["X"]) for d in datasets.values()))
                summary = json.loads((base / "metrics.json").read_text())
                for level in ("segment", "file"):
                    self.assertTrue(0 <= summary[level]["auroc"] <= 1)
                    self.assertTrue(0 <= summary[level]["auprc_average_precision"] <= 1)
                with self.assertRaises(ValueError):
                    load_model(
                        base / "model.joblib", fingerprint="different_experiment"
                    )

    def test_q95_ignores_faults_and_equality_is_normal(self):
        normal = [
            {"split": "validation", "score": value} for value in (1.0, 2.0, 3.0, 4.0)
        ]
        threshold = q95(normal)
        self.assertEqual(
            threshold, q95(normal + [{"split": "test_anomaly", "score": 1e20}])
        )
        self.assertEqual(
            predictions([{"score": threshold}], threshold)[0]["predicted_anomaly"], 0
        )

    def test_unknown_runs_and_file_mode_block_all_models(self):
        cases = []
        blank = [dict(r) for r in self.rows]
        blank[0]["run_id"] = ""
        cases.append((blank, "run"))
        overlap = [dict(r) for r in self.rows]
        overlap[3]["run_id"] = overlap[0]["run_id"]
        cases.append((overlap, "run"))
        cases.append((self.rows, "file"))
        for rows, mode in cases:
            write_rows(self.output / "split_1800rpm.csv", rows)
            (self.output / "split_config_1800rpm.json").write_text(
                json.dumps({"mode": mode})
            )
            for module in (ocsvm, iforest, train, evaluate, file_evaluate):
                with self.subTest(mode=mode, module=module.__name__), self.assertRaises(
                    ValueError
                ):
                    module.run(self.root, self.output, 1800)
        self.assertFalse((self.output / "comparisons").exists())

    def test_real_unconfirmed_inventory_is_blocked_without_training(self):
        project = Path(__file__).resolve().parents[1]
        # 실제 미확정 목록은 읽기만 하고 테스트 출력은 항상 임시 폴더에 한정합니다.
        rows = read_rows(project / "manifests" / "run_inventory_unconfirmed.csv")
        rows = [r for r in rows if r["rpm"] == "1800"]
        self.assertTrue(rows)
        self.assertTrue(all(not r["run_id"] and not r["split"] for r in rows))
        write_rows(self.output / "split_1800rpm.csv", rows)
        for module in (ocsvm, iforest, train, evaluate, file_evaluate):
            with self.subTest(module=module.__name__), patch(
                "vibration.one_class.make_model"
            ) as factory:
                with self.assertRaises(ValueError):
                    module.run(project, self.output, 1800)
                factory.assert_not_called()

    def test_mismatched_npz_run_metadata_is_rejected(self):
        path = self.output / "validation_normalized.npz"
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["run_id"] = np.repeat("wrong_run", len(arrays["X"]))
        np.savez_compressed(path, **arrays)
        with self.assertRaises(ValueError):
            load_experiment(self.output, 1800)

    def test_three_model_comparison_and_wrong_test_set_rejection(self):
        for module in (ocsvm, iforest):
            self.quiet(module.run, self.root, self.output, 1800)
        _, fingerprint = load_experiment(self.output, 1800)
        # AE 학습은 하지 않습니다. 가상 배열에서 초기 가중치의 추론·기존 평가 연결만 검사합니다.
        torch.manual_seed(42)
        torch.save(
            {
                "model_state_dict": Autoencoder().state_dict(),
                "epoch": 0,
                "data_fingerprint": fingerprint,
            },
            self.output / "autoencoder_1800rpm.pt",
        )
        self.quiet(evaluate.run, self.root, self.output, 1800)
        self.quiet(file_evaluate.run, self.root, self.output, 1800)
        self.quiet(comparison_metrics.run, self.root, self.output, 1800)
        summary = json.loads((self.output / "comparison_summary.json").read_text())
        self.assertEqual(set(summary["models"]), {"autoencoder", "ocsvm", "iforest"})
        self.assertEqual(summary["models"]["ocsvm"]["file"]["normal_count"], 2)
        path = self.output / "comparisons" / "ocsvm" / "scores_file.csv"
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row["file"] != self.rows[-1]["file"]]
        comparison_metrics.write_csv(path, rows)
        with self.assertRaises(ValueError):
            self.quiet(comparison_metrics.run, self.root, self.output, 1800)


if __name__ == "__main__":
    unittest.main()
