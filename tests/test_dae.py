"""Synthetic compatibility/error tests, not trained DAE performance validation."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from scipy.signal import welch
import torch

from vibration.dae_model import ConvAutoencoder
from vibration.dae_preprocessing import (
    dae_log_spectrum, normalize_spectrum, preprocess_mat_bytes, preprocess_uploaded_mat,
)
from vibration.dae_inference import SavedDAEInference


def mat_bytes(signals, fs=25600):
    stream = BytesIO()
    savemat(stream, {"signals": signals, "fs": fs})
    return stream.getvalue()


class DAEPreprocessingTests(unittest.TestCase):
    def test_exact_legacy_welch_formula_and_no_mutation(self):
        signals = np.random.default_rng(42).normal(size=(6, 65536))
        original = signals.copy()
        frequency, psd = welch(
            signals[2].astype(np.float64), fs=25600.0, window="hann",
            nperseg=8192, noverlap=4096, nfft=8192,
        )
        selected = np.sqrt(psd)[(frequency >= 0) & (frequency < 10000)]
        self.assertEqual(len(selected), 3200)
        expected = np.log10(selected.reshape(800, 4).mean(axis=1) + 1e-12).astype(np.float32)
        actual = preprocess_mat_bytes(mat_bytes(signals))
        self.assertEqual(actual.shape, (800,))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(signals, original)
        np.testing.assert_array_equal(preprocess_uploaded_mat(BytesIO(mat_bytes(signals))), expected)
        # Legacy uses CH3 only, including its quality checks.
        signals[0] = np.nan
        np.testing.assert_array_equal(dae_log_spectrum(signals, 25600), expected)

    def test_zero_signal_and_saved_normalization_arithmetic(self):
        spectrum = dae_log_spectrum(np.zeros((6, 65536)), 25600)
        np.testing.assert_array_equal(spectrum, np.full(800, -12, dtype=np.float32))
        mean = np.linspace(-3, 2, 800, dtype=np.float64)
        std = np.linspace(1e-9, 2, 800, dtype=np.float64)
        for m, s in (
            (mean, std), (mean[None], std[None]), (0.0, 2.0),
            (np.array([0.0]), np.array([2.0])),
        ):
            np.testing.assert_array_equal(normalize_spectrum(spectrum, m, s), (spectrum - m) / s)

    def test_reject_invalid_mat_and_signal(self):
        for raw in (b"", b"broken", "not bytes"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                preprocess_mat_bytes(raw)
        missing = BytesIO()
        savemat(missing, {"other": 1})
        with self.assertRaises(ValueError):
            preprocess_mat_bytes(missing.getvalue())
        signals = np.zeros((6, 65536))
        for bad, fs in (
            (signals[:, :-1], 25600), (signals, 12800), (signals, [25600, 25600]),
            (signals.astype(complex), 25600),
        ):
            with self.subTest(shape=bad.shape, fs=fs), self.assertRaises(ValueError):
                dae_log_spectrum(bad, fs)
        for value in (np.nan, np.inf):
            signals[2, 0] = value
            with self.assertRaises(ValueError):
                dae_log_spectrum(signals, 25600)

    def test_reject_invalid_statistics(self):
        spectrum = np.zeros(800, dtype=np.float32)
        for mean, std in (
            (np.zeros(799), np.ones(800)), (np.zeros((800, 1)), np.ones(800)),
            (np.nan, 1), (0, np.inf), (0, 0), (0, -1), (0, 1j),
        ):
            with self.subTest(mean=np.shape(mean), std=np.shape(std)), self.assertRaises(ValueError):
                normalize_spectrum(spectrum, mean, std)


class DAEInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dae_synthetic_")
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.model = ConvAutoencoder().eval()
        torch.save(self.model.state_dict(), self.folder / "autoencoder_model.pth")
        self.mean = np.linspace(-3, 1, 800)
        self.std = np.linspace(0.1, 2, 800)
        # No training arrays are required to load inference statistics.
        np.savez(self.folder / "normal_dataset_normalized.npz", mean=self.mean, std=self.std)
        np.savez(self.folder / "normal_threshold.npz", threshold=1.0)
        self.raw = mat_bytes(np.random.default_rng(7).normal(size=(6, 65536)))

    def test_model_shapes_and_legacy_checkpoint_layout(self):
        x = torch.zeros(2, 1, 800)
        self.assertEqual(tuple(self.model.encoder(x).shape), (2, 2, 50))
        self.assertEqual(tuple(self.model(x).shape), (2, 1, 800))
        expected = {
            "encoder.0.weight": (16, 1, 3), "encoder.2.weight": (32, 16, 3),
            "encoder.4.weight": (64, 32, 3), "encoder.6.weight": (2, 64, 3),
            "decoder.0.weight": (2, 64, 4), "decoder.2.weight": (64, 32, 4),
            "decoder.4.weight": (32, 16, 4), "decoder.6.weight": (16, 1, 4),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(self.model.state_dict()[name].shape), shape)
        for shape in ((1, 800), (1, 6, 800), (1, 1, 799), (1, 1, 801)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.model(torch.zeros(shape))

    def test_exact_float32_mse_strict_threshold_and_reused_model(self):
        normalized = (preprocess_mat_bytes(self.raw) - self.mean) / self.std
        tensor = torch.tensor(normalized, dtype=torch.float32).reshape(1, 1, 800)
        with torch.no_grad():
            reconstructed = self.model(tensor)
            expected_mse = torch.mean((tensor - reconstructed) ** 2, dim=(1, 2)).item()
        for threshold in (expected_mse, expected_mse / 2):
            np.savez(self.folder / "normal_threshold.npz", threshold=threshold)
            engine = SavedDAEInference(self.folder)
            self.assertFalse(engine.model.training)
            self.assertTrue(all(not p.requires_grad for p in engine.model.parameters()))
            with patch("torch.load", side_effect=AssertionError("reload")), patch(
                "numpy.percentile", side_effect=AssertionError("threshold fitting")
            ), patch("torch.randn_like", side_effect=AssertionError("inference noise")):
                result = engine.predict_bytes(self.raw)
                again = engine.predict_bytes(self.raw)
            self.assertEqual(result["mse"], expected_mse)
            self.assertEqual(again["mse"], expected_mse)
            self.assertEqual(result["prediction"], "이상" if expected_mse > threshold else "정상")
            np.testing.assert_array_equal(result["input_spectrum"], normalized)
            np.testing.assert_array_equal(result["reconstructed_spectrum"], reconstructed.squeeze().numpy())
        path = self.folder / "synthetic.mat"
        path.write_bytes(self.raw)
        self.assertEqual(engine.predict_mat(path)["mse"], expected_mse)

    def test_missing_invalid_artifacts_and_nonfinite_output(self):
        with self.assertRaises(FileNotFoundError):
            SavedDAEInference(self.folder / "missing")
        for value in (np.nan, np.inf, -1, [1, 2]):
            np.savez(self.folder / "normal_threshold.npz", threshold=value)
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                SavedDAEInference(self.folder)
        np.savez(self.folder / "normal_threshold.npz", threshold=1)
        state = self.model.state_dict()
        state["encoder.0.weight"][0, 0, 0] = float("nan")
        torch.save(state, self.folder / "autoencoder_model.pth")
        with self.assertRaises(ValueError):
            SavedDAEInference(self.folder)
        torch.save({}, self.folder / "autoencoder_model.pth")
        with self.assertRaises(RuntimeError):
            SavedDAEInference(self.folder)
        torch.save(ConvAutoencoder().state_dict(), self.folder / "autoencoder_model.pth")
        engine = SavedDAEInference(self.folder)
        with patch.object(engine.model, "forward", return_value=torch.full((1, 1, 800), float("nan"))):
            with self.assertRaises(ValueError):
                engine.predict_bytes(self.raw)


if __name__ == "__main__":
    unittest.main()
