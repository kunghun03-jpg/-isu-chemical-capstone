"""물리적 진동 특징 추출 모듈 검증."""

import unittest

import numpy as np

from vibration.physical_features import (
    FEATURE_NAMES,
    extract_physical_features,
)


class TestPhysicalFeatures(unittest.TestCase):

    FS = 25600
    SAMPLES = 65536

    def make_signal(self, rpm=900, amplitude=1.0):

        t = np.arange(self.SAMPLES) / self.FS

        signals = np.zeros(
            (6, self.SAMPLES),
            dtype=np.float64,
        )

        signals[2] = amplitude * np.sin(
            2 * np.pi * (rpm / 60) * t
        )

        return signals

    def test_feature_names(self):

        self.assertEqual(
            len(FEATURE_NAMES), 10
        )

        self.assertEqual(
            len(set(FEATURE_NAMES)), 10
        )

    def test_supported_rpm(self):

        for rpm in (900, 1200, 1800):

            with self.subTest(rpm=rpm):

                features = extract_physical_features(
                    self.make_signal(rpm),
                    self.FS,
                    rpm,
                )

                self.assertEqual(
                    features.shape, (8, 10)
                )

                self.assertTrue(
                    np.isfinite(features).all()
                )

                self.assertTrue(
                    (features >= 0).all()
                )

                np.testing.assert_allclose(
                    features[:, 0],
                    1 / np.sqrt(2),
                    atol=0.03,
                )

                np.testing.assert_allclose(
                    features[:, 2],
                    1.5,
                    atol=0.15,
                )

                self.assertTrue(
                    (features[:, 4] > 0.8).all()
                )

                self.assertTrue(
                    (
                        features[:, 4]
                        > 10 * features[:, 5]
                    ).all()
                )

                self.assertTrue(
                    (
                        features[:, 4]
                        > 10 * features[:, 6]
                    ).all()
                )

                np.testing.assert_allclose(
                    features[:, 7],
                    0.5,
                    atol=0.03,
                )

    def test_ch3_selection(self):

        signals = self.make_signal()

        expected = extract_physical_features(
            signals, self.FS, 900
        )

        signals[0] = 100
        signals[1] = -200
        signals[3] = 300
        signals[4] = -400
        signals[5] = 500

        actual = extract_physical_features(
            signals, self.FS, 900
        )

        np.testing.assert_array_equal(
            actual, expected
        )

    def test_amplitude_scaling(self):

        original = extract_physical_features(
            self.make_signal(amplitude=1),
            self.FS,
            900,
        )

        doubled = extract_physical_features(
            self.make_signal(amplitude=2),
            self.FS,
            900,
        )

        # RMS, Peak, harmonic amplitude
        for index in (0, 1, 4, 5, 6):

            np.testing.assert_allclose(
                doubled[:, index],
                original[:, index] * 2,
                rtol=1e-8,
                atol=1e-12,
            )

        # Kurtosis, Crest Factor
        for index in (2, 3):

            np.testing.assert_allclose(
                doubled[:, index],
                original[:, index],
                rtol=1e-8,
                atol=1e-12,
            )

        # Spectral energy
        for index in (7, 8, 9):

            np.testing.assert_allclose(
                doubled[:, index],
                original[:, index] * 4,
                rtol=1e-8,
                atol=1e-12,
            )

    def test_dc_offset_removal(self):

        signals = self.make_signal()

        expected = extract_physical_features(
            signals, self.FS, 900
        )

        signals[2] += 10

        actual = extract_physical_features(
            signals, self.FS, 900
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-7,
            atol=1e-10,
        )

    def test_input_not_modified(self):

        signals = self.make_signal()
        before = signals.copy()

        extract_physical_features(
            signals, self.FS, 900
        )

        np.testing.assert_array_equal(
            signals, before
        )

    def test_invalid_shape(self):

        with self.assertRaises(ValueError):

            extract_physical_features(
                np.zeros((3, 65536)),
                self.FS,
                900,
            )

    def test_invalid_fs(self):

        with self.assertRaises(ValueError):

            extract_physical_features(
                self.make_signal(),
                12800,
                900,
            )

    def test_invalid_rpm(self):

        with self.assertRaises(ValueError):

            extract_physical_features(
                self.make_signal(),
                self.FS,
                1500,
            )

    def test_nonfinite_raw_channel(self):

        signals = self.make_signal()
        signals[0, 0] = np.nan

        with self.assertRaises(ValueError):

            extract_physical_features(
                signals, self.FS, 900
            )

    def test_zero_energy(self):

        with self.assertRaises(ValueError):

            extract_physical_features(
                np.zeros((6, 65536)),
                self.FS,
                900,
            )


if __name__ == "__main__":
    unittest.main()
