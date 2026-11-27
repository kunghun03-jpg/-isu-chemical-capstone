"""회전체 이상 탐지를 위한 실험용 물리적 진동 특징."""

import numpy as np

from vibration.channel import select_ch3


FEATURE_NAMES = (
    "rms",
    "peak",
    "kurtosis",
    "crest_factor",
    "amplitude_1x",
    "amplitude_2x",
    "amplitude_3x",
    "energy_low",
    "energy_mid",
    "energy_high",
)


def extract_physical_features(signals, fs, rpm):
    """8개 진동 구간에서 물리적 특징 10개씩 추출."""

    if (
        not isinstance(signals, np.ndarray)
        or signals.shape != (6, 65536)
    ):
        raise ValueError(
            "signals must have shape (6, 65536)"
        )

    if (
        signals.dtype.kind not in "fiu"
        or not np.isfinite(signals).all()
        or (np.abs(signals) > 1e100).any()
    ):
        raise ValueError("Invalid raw signals")

    fs_value = np.asarray(fs)

    if (
        fs_value.size != 1
        or fs_value.dtype.kind not in "fiu"
        or float(fs_value.item()) != 25600.0
    ):
        raise ValueError("Expected fs=25600 Hz")

    if rpm not in (900, 1200, 1800):
        raise ValueError("Unsupported RPM")

    fs = float(fs_value.item())

    x = np.asarray(
        select_ch3(signals),
        dtype=np.float64,
    ).reshape(8, 8192)

    # 구간별 DC 성분 제거
    x = x - x.mean(
        axis=1, keepdims=True
    )

    rms = np.sqrt(
        np.mean(x ** 2, axis=1)
    )

    if (rms <= 0).any():
        raise ValueError(
            "Zero-energy vibration segment"
        )

    peak = np.max(
        np.abs(x), axis=1
    )

    crest_factor = peak / rms

    # Pearson kurtosis: Gaussian 기준 3
    standardized = x / rms[:, None]

    kurtosis = np.mean(
        standardized ** 4, axis=1
    )

    window = np.hanning(8192)

    spectrum = np.fft.rfft(
        x * window, axis=1
    )

    freq = np.fft.rfftfreq(
        8192, d=1 / fs
    )

    df = fs / 8192

    # Hann window의 coherent gain 보정
    amplitude = (
        np.abs(spectrum) / window.sum()
    )

    amplitude[:, 1:-1] *= 2

    # One-sided power spectral density
    psd = (
        np.abs(spectrum) ** 2
        / (fs * np.sum(window ** 2))
    )

    psd[:, 1:-1] *= 2

    rotation_hz = rpm / 60.0

    def harmonic_amplitude(order):

        center = rotation_hz * order

        mask = (
            np.abs(freq - center) <= df
        )

        if not mask.any():
            raise ValueError(
                "Empty harmonic band"
            )

        return amplitude[:, mask].max(
            axis=1
        )

    def band_energy(low, high, inclusive=False):

        upper = (
            freq <= high
            if inclusive
            else freq < high
        )

        mask = (freq >= low) & upper

        return (
            psd[:, mask].sum(axis=1) * df
        )

    features = np.column_stack([
        rms,
        peak,
        kurtosis,
        crest_factor,
        harmonic_amplitude(1),
        harmonic_amplitude(2),
        harmonic_amplitude(3),
        band_energy(0, 100),
        band_energy(100, 1000),
        band_energy(1000, 10000, True),
    ])

    if features.shape != (8, 10):
        raise ValueError(
            "Unexpected feature shape"
        )

    if (
        not np.isfinite(features).all()
        or (features < 0).any()
    ):
        raise ValueError(
            "Invalid physical features"
        )

    return features
