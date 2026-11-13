"""임시 데이터로 Run 충돌·메타데이터·날짜 추가를 검사합니다. 학습은 하지 않습니다."""

from pathlib import Path
import contextlib
import csv
import io
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from vibration import check, normalize, spectra, split
from vibration.dataset import discover_files, read_rows, write_rows


class RunSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="run_split_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        self.output.mkdir()
        self.manifest = self.root / "confirmed.csv"
        self.rows = []
        # 테스트에서만 만든 가상의 Run입니다. 실제 데이터에는 이 ID를 적용하지 않습니다.
        for index, (date, condition, assigned) in enumerate(
            [
                ("20260912", "Normal1", "train"),
                ("20260916", "Normal2", "validation"),
                ("20260919", "Normal1", "test_normal"),
                ("20260919", "IR", "test_anomaly"),
            ]
        ):
            path = (
                self.root
                / "data"
                / f"{date}_1800RPM"
                / condition
                / "Vib"
                / f"{index}.mat"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            savemat(
                path,
                {
                    "signals": np.random.default_rng(index).normal(size=(6, 65536)),
                    "fs": np.array([[25600.0]]),
                },
            )
            self.rows.append(
                {
                    "file": path.relative_to(self.root).as_posix(),
                    "date": date,
                    "rpm": "1800",
                    "condition": condition,
                    "run_id": f"TEST_ONLY_{index}",
                    "split": assigned,
                }
            )
        write_rows(self.manifest, self.rows)
        self.quiet(check.run, self.root, self.output, 1800)

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def apply(self):
        self.quiet(split.run, self.root, self.output, 1800, manifest=self.manifest)

    def test_multidate_run_and_metadata_pipeline(self):
        self.apply()
        rows = read_rows(self.output / "split_1800rpm.csv")
        self.assertEqual(rows, self.rows)
        self.quiet(spectra.run, self.root, self.output, 1800)
        self.quiet(normalize.run, self.root, self.output, 1800)
        for row in self.rows:
            with np.load(self.output / f"{row['split']}_normalized.npz") as data:
                self.assertEqual(data["X"].shape, (8, 1, 3201))
                for key in ("date", "rpm", "run_id"):
                    self.assertEqual(set(data[key]), {row[key]})

    def test_same_run_across_dates_rejected_without_overwriting(self):
        self.apply()
        target = self.output / "split_1800rpm.csv"
        before = target.read_bytes()
        self.rows[1]["run_id"] = self.rows[0]["run_id"]
        write_rows(self.manifest, self.rows)
        with self.assertRaisesRegex(ValueError, "동일 Run"):
            self.apply()
        self.assertEqual(target.read_bytes(), before)

    def test_preprocessing_rechecks_run_conflicts(self):
        self.apply()
        self.rows[1]["run_id"] = self.rows[0]["run_id"]
        write_rows(self.output / "split_1800rpm.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "동일 Run"):
            self.quiet(spectra.run, self.root, self.output, 1800)
        self.assertFalse((self.output / "train_spectra.npz").exists())

    def test_unchecked_new_file_is_rejected(self):
        source = self.root / self.rows[0]["file"]
        added = source.with_name("added.mat")
        added.write_bytes(source.read_bytes())
        self.rows.append(
            {**self.rows[0], "file": added.relative_to(self.root).as_posix()}
        )
        write_rows(self.manifest, self.rows)
        with self.assertRaisesRegex(ValueError, "품질검사"):
            self.apply()
        # 검사한 뒤에는 같은 Run의 여러 파일을 같은 split에 배정할 수 있습니다.
        self.quiet(check.run, self.root, self.output, 1800)
        self.apply()
        self.assertEqual(len(read_rows(self.output / "split_1800rpm.csv")), 5)

    def test_unknown_duplicate_missing_and_wrong_metadata(self):
        original = [dict(row) for row in self.rows]
        variants = []
        unknown = [dict(row) for row in original]
        unknown[0]["run_id"] = ""
        variants.append(unknown)
        variants.append(original + [dict(original[0])])
        variants.append(original[1:])
        wrong = [dict(row) for row in original]
        wrong[0]["date"] = "20260919"
        variants.append(wrong)
        for rows in variants:
            with self.subTest(rows=rows):
                write_rows(self.manifest, rows)
                with self.assertRaises(ValueError):
                    self.apply()
        self.assertFalse((self.output / "split_1800rpm.csv").exists())

    def test_inventory_never_invents_runs(self):
        path = self.quiet(split.inventory, self.root, self.output, 1800)
        rows = read_rows(path)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(not r["run_id"] and not r["split"] for r in rows))
        self.assertEqual(len(discover_files(self.root, 1800, ["20260912"])), 1)
        with self.assertRaises(ValueError):
            discover_files(self.root, 1800, ["20260920"])

    def test_missing_partition_and_unchecked_files_rejected(self):
        self.rows[2]["split"] = "validation"
        write_rows(self.manifest, self.rows)
        with self.assertRaisesRegex(ValueError, "test_normal"):
            self.apply()
        (self.output / "quality_checked_1800rpm.csv").unlink()
        with self.assertRaisesRegex(ValueError, "check"):
            self.apply()

    def test_file_mode_and_cli(self):
        # 같은 날짜의 한 상태에 충분한 파일을 추가하여 종전 60/20/20을 확인합니다.
        path = self.root / self.rows[0]["file"]
        for index in range(1, 10):
            path.with_name(f"extra{index}.mat").write_bytes(path.read_bytes())
        self.quiet(
            split.run, self.root, self.output, 1800, mode="file", dates=["20260912"]
        )
        rows = read_rows(self.output / "split_1800rpm.csv")
        self.assertEqual(
            [
                sum(r["split"] == s for r in rows)
                for s in ("train", "validation", "test_normal")
            ],
            [6, 2, 2],
        )
        self.assertTrue(all(r["run_id"] == "" for r in rows))
        script = Path(__file__).resolve().parents[1] / "run_project.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(self.root),
                "--rpm",
                "1800",
                "--stage",
                "inventory",
                "--run-name",
                "setup",
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(self.root),
                "--rpm",
                "1800",
                "--stage",
                "split",
                "--run-name",
                "missing",
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
