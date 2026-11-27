"""P3 독립 추론 모듈 자동 검증."""

import json
import unittest
import tempfile

from pathlib import Path

import numpy as np

from scipy.io import savemat

from vibration.physical_features import (
    FEATURE_NAMES,
    extract_physical_features,
)

from vibration.physical_inference import (
    METHOD,
    FEATURE_FLOOR,
    SCALE_FLOOR,
    PhysicalInferenceError,
    load_physical_reference,
    analyze_physical_signals,
    analyze_physical_mat,
)


class TestPhysicalInference(unittest.TestCase):

    DATE = "20260919"
    RPM = 1200
    FS = 25600

    def setUp(self):

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        self.folder = (
            self.root
            / self.DATE
            / f"{self.RPM}rpm"
        )

        self.folder.mkdir(parents=True)

        t = np.arange(65536) / self.FS

        self.signals = np.zeros(
            (6, 65536),
            dtype=np.float64,
        )

        self.signals[2] = (
            np.sin(2 * np.pi * 20 * t)
            + 0.1 * np.sin(2 * np.pi * 40 * t)
        )

        features = np.median(
            extract_physical_features(
                self.signals,
                self.FS,
                self.RPM,
            ),
            axis=0,
        )

        self.center = np.log(
            np.maximum(features, FEATURE_FLOOR)
        )

        self.scale = np.full(10, 0.1)

        self.reference_path = (
            self.folder
            / "C_mixed_per_frequency_reference.npz"
        )

        self.metadata_path = (
            self.folder
            / "experiment_metadata.json"
        )

        self.meta = {
            "date": self.DATE,
            "rpm": self.RPM,
            "feature_names": list(FEATURE_NAMES),
            "feature_transform": "log",
            "file_aggregation":
                "median_of_8_segments",
            "normal_reference": "median_mad",
            "feature_floor": FEATURE_FLOOR,
            "scale_floor": SCALE_FLOOR,
            "threshold_source":
                "normal_validation_q95",
            "thresholds": {METHOD: 1.0},
            "evaluation_status":
                "exploratory_only",
            "approved_for_production": False,
            "run_independence_verified": False,
            "test_previously_examined": True,
        }

        self.write_reference()
        self.write_metadata()

    def write_reference(self, names=FEATURE_NAMES):

        np.savez_compressed(
            self.reference_path,
            center=self.center,
            scale=self.scale,
            feature_names=np.array(names),
        )

    def write_metadata(self):

        self.metadata_path.write_text(
            json.dumps(self.meta),
            encoding="utf-8",
        )

    def analyze(self, signals=None):

        return analyze_physical_signals(
            self.signals if signals is None
            else signals,
            self.FS,
            self.DATE,
            self.RPM,
            reference_root=self.root,
        )

    def test_reference_loading(self):

        ref = load_physical_reference(
            self.DATE,
            self.RPM,
            self.root,
        )

        self.assertEqual(ref.threshold, 1.0)

        np.testing.assert_array_equal(
            ref.center, self.center
        )

        np.testing.assert_array_equal(
            ref.scale, self.scale
        )

        self.assertFalse(ref.center.flags.writeable)
        self.assertFalse(ref.scale.flags.writeable)

    def test_normal_decision(self):

        result = self.analyze()

        self.assertAlmostEqual(
            result["score"], 0.0, places=10
        )

        self.assertEqual(
            result["experimental_decision"],
            "Normal",
        )

        self.assertFalse(
            result["predicted_anomaly"]
        )

    def test_abnormal_decision(self):

        result = self.analyze(
            self.signals * 2
        )

        self.assertGreater(
            result["score"],
            result["threshold"],
        )

        self.assertEqual(
            result["experimental_decision"],
            "Abnormal",
        )

    def test_score_formula(self):

        result = self.analyze(
            self.signals * 2
        )

        features = np.array([
            result["features"][name]
            for name in FEATURE_NAMES
        ])

        expected = np.abs(
            (
                np.log(
                    np.maximum(
                        features,
                        FEATURE_FLOOR,
                    )
                )
                - self.center
            )
            / self.scale
        )

        actual = np.array([
            result["feature_deviations"][name]
            for name in FEATURE_NAMES
        ])

        np.testing.assert_allclose(
            actual, expected
        )

        self.assertAlmostEqual(
            result["score"],
            expected.mean(),
        )

        self.assertEqual(
            result["dominant_feature"],
            FEATURE_NAMES[int(expected.argmax())],
        )

        self.assertAlmostEqual(
            result["score_ratio"],
            result["score"] / result["threshold"],
        )

    def test_mat_signal_parity(self):

        path = self.root / "sample.mat"

        savemat(
            path,
            {
                "signals": self.signals,
                "fs": self.FS,
            },
        )

        actual = analyze_physical_mat(
            path,
            self.DATE,
            self.RPM,
            reference_root=self.root,
        )

        expected = self.analyze()

        self.assertEqual(actual, expected)

    def test_experimental_status(self):

        result = self.analyze()

        self.assertEqual(
            result["evaluation_status"],
            "exploratory_only",
        )

        self.assertFalse(
            result["approved_for_production"]
        )

        self.assertFalse(
            result["run_independence_verified"]
        )

        self.assertFalse(
            result["measurement_metadata_verified"]
        )

    def test_unsupported_date(self):

        with self.assertRaises(
            PhysicalInferenceError
        ):

            load_physical_reference(
                "20260920",
                self.RPM,
                self.root,
            )

    def test_unsupported_rpm(self):

        with self.assertRaises(
            PhysicalInferenceError
        ):

            load_physical_reference(
                self.DATE,
                1500,
                self.root,
            )

    def test_missing_reference(self):

        self.reference_path.unlink()

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_invalid_feature_order(self):

        self.write_reference(
            tuple(reversed(FEATURE_NAMES))
        )

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_invalid_scale(self):

        self.scale[0] = 0
        self.write_reference()

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_invalid_threshold(self):

        self.meta["thresholds"][METHOD] = -1
        self.write_metadata()

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_metadata_mismatch(self):

        self.meta["rpm"] = 900
        self.write_metadata()

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_production_status_rejection(self):

        self.meta["approved_for_production"] = True
        self.write_metadata()

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze()

    def test_invalid_signal(self):

        signals = self.signals.copy()
        signals[2, 0] = np.nan

        with self.assertRaises(
            PhysicalInferenceError
        ):
            self.analyze(signals)

    def test_invalid_mat(self):

        path = self.root / "invalid.mat"

        savemat(path, {"fs": self.FS})

        with self.assertRaises(
            PhysicalInferenceError
        ):

            analyze_physical_mat(
                path,
                self.DATE,
                self.RPM,
                reference_root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
