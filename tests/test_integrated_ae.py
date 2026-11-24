import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import infer_integrated_ae
from scripts.compare_autoencoders import compare

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "outputs/experimental_integrated_ae"
RPMS = (900, 1200, 1800)


class IntegratedAETests(unittest.TestCase):

    def reference(self, rpm):
        folder = BUNDLES / f"{rpm}rpm"

        with (folder / "validation_scores.csv").open(
            encoding="utf-8-sig", newline=""
        ) as f:
            rows = list(csv.DictReader(f))

        row = next(
            r for r in rows
            if r["condition"] == "Normal1"
        )

        path = ROOT / row["file"]

        if not path.is_file():
            self.skipTest(f"원본 MAT 필요: {path}")

        return path, row

    def test_bundle_integrity(self):
        for rpm in RPMS:
            with self.subTest(rpm=rpm):
                folder = BUNDLES / f"{rpm}rpm"
                manifest = json.loads(
                    (folder / "manifest.json").read_text()
                )

                self.assertEqual(manifest["rpm"], rpm)
                self.assertEqual(
                    manifest["status"], "experimental_only"
                )
                self.assertIs(
                    manifest["approved_for_production"], False
                )
                self.assertIs(
                    manifest["defect_recall_verified"], False
                )

                for name, expected in (
                    manifest["artifacts_sha256"].items()
                ):
                    actual = hashlib.sha256(
                        (folder / name).read_bytes()
                    ).hexdigest()

                    self.assertEqual(actual, expected)

    def test_inference_reproducibility(self):
        for rpm in RPMS:
            with self.subTest(rpm=rpm):
                path, row = self.reference(rpm)
                result = infer_integrated_ae.predict(path, rpm)

                np.testing.assert_allclose(
                    result["score"],
                    float(row["score"]),
                    rtol=1e-6,
                    atol=1e-8,
                )

                self.assertEqual(
                    result["threshold"],
                    float(row["threshold"]),
                )

                self.assertEqual(
                    result["predicted_anomaly"],
                    int(row["predicted_anomaly"]),
                )

    def test_autoencoder_comparison(self):
        for rpm in RPMS:
            with self.subTest(rpm=rpm):
                path, _ = self.reference(rpm)
                result = compare(path, rpm)

                self.assertEqual(
                    result["comparison_status"],
                    "experimental_only",
                )

                for name in ("operating_ae", "integrated_ae"):
                    item = result[name]

                    self.assertTrue(
                        np.isfinite(item["score"])
                    )
                    self.assertTrue(
                        np.isfinite(item["threshold"])
                    )
                    self.assertEqual(
                        item["predicted_anomaly"],
                        int(item["score"] > item["threshold"]),
                    )

                self.assertEqual(
                    result["decisions_match"],
                    result["operating_ae"]["predicted_anomaly"]
                    == result["integrated_ae"]["predicted_anomaly"],
                )

                self.assertIs(
                    result["production_model_modified"], False
                )

    def test_unapproved_policy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "900rpm"

            shutil.copytree(BUNDLES / "900rpm", folder)

            path = folder / "manifest.json"
            manifest = json.loads(path.read_text())

            manifest["approved_for_production"] = True
            path.write_text(json.dumps(manifest))

            with patch.object(
                infer_integrated_ae, "BUNDLE_ROOT", root
            ):
                with self.assertRaisesRegex(
                    ValueError, "실험용 번들 정책 불일치"
                ):
                    infer_integrated_ae.predict(
                        "unused.mat", 900
                    )


if __name__ == "__main__":
    unittest.main()
