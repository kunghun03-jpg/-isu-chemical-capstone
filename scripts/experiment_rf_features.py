"""Experimental rotating-harmonic features; does not change production inference."""

import numpy as np

from scripts.analyze_fault_signatures import fft_amplitude, peak_component


EXTRA_FEATURES = (
    "three_x_amplitude",
    "four_x_amplitude",
    "five_x_amplitude",
    "ratio_2x_1x",
    "ratio_3x_1x",
    "harmonic_energy_fraction",
)


def extract_extra_features(signal, fs, rpm, tolerance=1.0):
    frequency, amplitude = fft_amplitude(signal, fs)
    fundamental = rpm / 60.0

    harmonics = {}

    for order in range(1, 6):
        _, value = peak_component(
            frequency,
            amplitude,
            order * fundamental,
            tolerance,
        )
        harmonics[order] = value

    one_x = harmonics[1]
    epsilon = 1e-12

    # FFT 진폭 제곱의 상대적 비율. 물리적 PSD 에너지와는 구분한다.
    total = float(np.sum(amplitude[frequency > 0] ** 2))
    harmonic_sum = float(sum(value ** 2 for value in harmonics.values()))

    return {
        "three_x_amplitude": harmonics[3],
        "four_x_amplitude": harmonics[4],
        "five_x_amplitude": harmonics[5],
        "ratio_2x_1x": harmonics[2] / max(one_x, epsilon),
        "ratio_3x_1x": harmonics[3] / max(one_x, epsilon),
        "harmonic_energy_fraction": harmonic_sum / total if total > 0 else 0.0,
    }


if __name__ == "__main__":
    fs = 25600
    rpm = 1200
    t = np.arange(65536) / fs

    # 1200RPM의 1X=20Hz, 2X=40Hz, 3X=60Hz를 가진 테스트 신호
    signal = (
        np.sin(2 * np.pi * 20 * t)
        + 0.5 * np.sin(2 * np.pi * 40 * t)
        + 0.25 * np.sin(2 * np.pi * 60 * t)
    )

    features = extract_extra_features(signal, fs, rpm)

    for name, value in features.items():
        print(f"{name}: {value:.6f}")
