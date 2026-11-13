"""저장 모델 추론 회귀 검사: 실제 데이터 학습은 수행하지 않습니다."""

from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from vibration.inference import SavedModelInference
from vibration.preprocessing import ch3_log_spectra, apply_normalization

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/inference_bundles/date_prelim_20261106/900rpm"


class PreprocessingTests(unittest.TestCase):
    def test_legacy_fft_equivalence_and_ch3_selection(self):
        raw = np.random.default_rng(42).normal(size=(6, 65536))
        original = raw.copy()
        segments = raw.reshape(6, 8, 8192).transpose(1, 0, 2)
        segments = segments - segments.mean(axis=2, keepdims=True)
        window = np.hanning(8192)
        amplitude = np.abs(np.fft.rfft(segments * window, axis=2)) / window.sum()
        amplitude[:, :, 1:-1] *= 2
        mask = np.fft.rfftfreq(8192, d=1 / 25600) <= 10000
        expected = np.log1p(amplitude[:, 2:3, mask]).astype(np.float32)
        np.testing.assert_array_equal(ch3_log_spectra(raw, 25600), expected)
        np.testing.assert_array_equal(raw, original)
        raw[0] *= 100
        np.testing.assert_array_equal(ch3_log_spectra(raw, 25600), expected)

    def test_reject_invalid_input_and_global_statistics(self):
        raw = np.zeros((6, 65536))
        for bad, fs in [
            (raw[:, :-1], 25600),
            (raw, 12800),
            (raw.astype(complex), 25600),
        ]:
            with self.assertRaises(ValueError):
                ch3_log_spectra(bad, fs)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ch3_log_spectra(raw, 25600)
        with self.assertRaises(ValueError):
            apply_normalization(
                np.zeros((8, 1, 3201)), np.zeros((1, 1, 1)), np.ones((1, 1, 1))
            )


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "기존 모델 패키지 필요")
class SavedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SavedModelInference(BUNDLE)
        stream = BytesIO()
        savemat(
            stream,
            {
                "signals": np.random.default_rng(7).normal(size=(6, 65536)),
                "fs": 25600.0,
            },
        )
        cls.raw = stream.getvalue()

    def test_no_fitting_or_threshold_recalculation(self):
        with patch(
            "sklearn.svm.OneClassSVM.fit", side_effect=AssertionError("fit forbidden")
        ), patch(
            "sklearn.ensemble.IsolationForest.fit",
            side_effect=AssertionError("fit forbidden"),
        ), patch(
            "torch.Tensor.backward", side_effect=AssertionError("backward forbidden")
        ), patch(
            "numpy.quantile", side_effect=AssertionError("Q95 forbidden")
        ):
            result = self.engine.predict_bytes(self.raw, rpm=900)
        for name, actual in result["models"].items():
            self.assertEqual(
                actual["file_score"], np.mean([s["score"] for s in actual["segments"]])
            )
            self.assertEqual(
                actual["file_threshold"],
                self.engine.manifest["models"][name]["file_threshold"],
            )
            self.assertEqual(
                actual["predicted_anomaly"],
                int(actual["file_score"] > actual["file_threshold"]),
            )

    def test_standalone_bundle_and_mat_path_api(self):
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copied)
            path = Path(folder) / "uploaded.mat"
            path.write_bytes(self.raw)
            standalone = SavedModelInference(copied)
            actual = standalone.predict_mat(path, rpm=900)
            expected = self.engine.predict_bytes(self.raw, rpm=900)
            self.assertEqual(actual["models"], expected["models"])

    def test_rpm_bad_mat_dae_and_tampering_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.predict_bytes(self.raw, rpm=1200)
        with self.assertRaises(ValueError):
            self.engine.predict_bytes(b"invalid MAT", rpm=900)
        with self.assertRaises(ValueError):
            SavedModelInference(BUNDLE, models=("dae",))
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copied)
            with (copied / "normalization.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                SavedModelInference(copied)


if __name__ == "__main__":
    unittest.main()
