"""새 날짜 경로·상태 표기·Normal3 제외 정책과 품질검사 검증."""

from pathlib import Path
import contextlib
import io
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from vibration import check, split, spectra, normalize
from vibration.dataset import (
    discover_files,
    read_rows,
    validate_assignments,
    validate_file_metadata,
    write_inventory,
    write_rows,
)


class DataLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="data_layout_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        self.output.mkdir()

    def add(self, folder, condition, name="valid", **changes):
        path = self.root / "data" / folder / condition / "Vib" / f"{name}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"signals": np.ones((6, 65536)), "fs": np.array([[25600.0]])}
        data.update(changes)
        savemat(path, data)
        return path

    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def test_flat_nested_alias_and_inventory(self):
        self.add("20260916_1800RPM", "Normal1")
        self.add("20260919/20260919_1800RPM", "Looeness0.05")
        self.add("20260912/20260912_1800RPM", "Normal3")
        rows = discover_files(self.root, 1800)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {r["condition"] for r in rows}, {"Normal1", "Normal3", "Looseness0.05"}
        )
        for row in rows:
            validate_file_metadata(self.root, row, 1800)
        path = self.root / "inventory.csv"
        write_inventory(path, rows)
        self.assertTrue(
            all(not r["run_id"] and not r["split"] for r in read_rows(path))
        )
        self.assertIn("Looeness0.05", path.read_text(encoding="utf-8-sig"))
        self.assertEqual(len(discover_files(self.root, 1800, ["20260919"])), 1)

    def test_normal3_counted_but_not_split(self):
        for index in range(10):
            self.add("20260919/20260919_1800RPM", "Normal1", str(index))
        self.add("20260919/20260919_1800RPM", "Normal3")
        details = self.quiet(check.run, self.root, self.output, 1800)
        summary = check.summarize(details)
        self.assertEqual(summary["normal_total"], 11)
        self.assertEqual(summary["normal3_total"], 1)
        self.assertEqual(summary["default_normal_total"], 10)
        self.quiet(split.run, self.root, self.output, 1800, mode="file")
        self.assertEqual(len(read_rows(self.output / "split_1800rpm.csv")), 10)
        row = next(
            r for r in discover_files(self.root, 1800) if r["condition"] == "Normal3"
        )
        with self.assertRaisesRegex(ValueError, "Normal3"):
            validate_assignments([{**row, "run_id": "test", "split": "test_anomaly"}])

    def test_quality_issues_and_sources_unchanged(self):
        base = "20260919/20260919_1800RPM"
        self.add(base, "Normal1", "good")
        self.add(base, "Normal1", "wrong_fs", fs=np.array([[12800.0]]))
        self.add(base, "Normal1", "wrong_shape", signals=np.ones((1, 100)))
        signal = np.ones((6, 65536))
        signal[0, 0] = np.nan
        signal[2, 0] = np.inf
        self.add(base, "Normal1", "nonfinite", signals=signal)
        signal = np.ones((6, 65536))
        signal[2, 0] = 1e101
        self.add(base, "Normal1", "extreme", signals=signal)
        invalid = self.root / "data" / base / "Normal1/Vib/unreadable.mat"
        invalid.write_bytes(b"invalid mat data")
        before = {p: p.read_bytes() for p in (self.root / "data").rglob("*.mat")}
        details = self.quiet(check.run, self.root, self.output, 1800)
        self.assertEqual(len(details), 6)
        self.assertEqual(sum(r["status"] == "fail" for r in details), 5)
        self.assertEqual(check.summarize(details)["reason_counts"]["read_error"], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue(all(r["source_stat_unchanged"] for r in details))

    def test_mismatched_parent_date_rejected(self):
        self.add("20260912/20260919_1800RPM", "Normal1")
        with self.assertRaises(ValueError):
            discover_files(self.root, 1800)

    def test_nested_run_split_preprocessing_and_normal3_exclusion(self):
        assignments = [
            ("20260912", "Normal1", "train"),
            ("20260916", "Normal2", "validation"),
            ("20260919", "Normal1", "test_normal"),
            ("20260919", "Looeness0.1", "test_anomaly"),
        ]
        for date, condition, _ in assignments:
            self.add(f"{date}/{date}_1800RPM", condition)
        self.add("20260919/20260919_1800RPM", "Normal3")
        self.quiet(check.run, self.root, self.output, 1800)
        rows = discover_files(self.root, 1800)
        for row in rows:
            if row["condition"] == "Normal3":
                continue
            target = next(
                s
                for d, c, s in assignments
                if d == row["date"]
                and (
                    c == row["condition"]
                    or c.replace("Looeness", "Looseness") == row["condition"]
                )
            )
            row.update(run_id=f"TEST_ONLY_{target}", split=target)
        manifest = self.root / "test_only_runs.csv"
        write_rows(manifest, rows)
        self.quiet(split.run, self.root, self.output, 1800, manifest=manifest)
        selected = read_rows(self.output / "split_1800rpm.csv")
        self.assertEqual(len(selected), 4)
        self.assertFalse(any(r["condition"] == "Normal3" for r in selected))
        self.quiet(spectra.run, self.root, self.output, 1800)
        self.quiet(normalize.run, self.root, self.output, 1800)
        with np.load(self.output / "test_anomaly_normalized.npz") as data:
            self.assertEqual(data["X"].shape, (8, 1, 3201))
            self.assertEqual(set(data["condition"]), {"Looseness0.1"})
            self.assertEqual(set(data["date"]), {"20260919"})

    def test_date_preliminary_keeps_run_unknown_and_checks_dates(self):
        from vibration.experiment import load_experiment
        from vibration.dataset import validate_date_groups

        for date, condition in [
            ("20260912", "Normal1"),
            ("20260916", "Normal2"),
            ("20260919", "Normal1"),
            ("20260919", "Looeness0.1"),
            ("20260912", "IR"),
            ("20260919", "Normal3"),
        ]:
            self.add(f"{date}/{date}_1800RPM", condition)
        self.quiet(check.run, self.root, self.output, 1800)
        self.quiet(split.run, self.root, self.output, 1800, mode="date_preliminary")
        rows = read_rows(self.output / "split_1800rpm.csv")
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            all(
                not r["run_id"] and r["date_group"] == f"date:{r['date']}" for r in rows
            )
        )
        validate_date_groups(rows)
        self.quiet(spectra.run, self.root, self.output, 1800)
        self.quiet(normalize.run, self.root, self.output, 1800)
        data, _ = load_experiment(self.output, 1800)
        self.assertEqual(set(data["train"]["date_group"]), {"date:20260912"})
        with self.assertRaises(ValueError):
            validate_date_groups([{**rows[0], "run_id": "fake_actual_run"}, *rows[1:]])
        with self.assertRaises(ValueError):
            validate_date_groups(
                [{**rows[0], "date_group": "date:20260919"}, *rows[1:]]
            )


if __name__ == "__main__":
    unittest.main()
