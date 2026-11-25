"""Reuse verified AE reconstruction arrays for bandwise analysis."""

import numpy as np

from vibration.bandwise_ae import build_frequency_bands
from vibration.bandwise_analysis import summarize_bandwise_result


def analyze_reconstruction_data(data, result, rpm):

    frequencies, masks = build_frequency_bands(rpm)

    frequency = np.asarray(
        data["frequency"], dtype=np.float64
    )

    inputs = np.asarray(
        data["input"], dtype=np.float64
    )

    reconstructed = np.asarray(
        data["reconstruction"], dtype=np.float64
    )

    stored_scores = np.asarray(
        data["scores"], dtype=np.float64
    )

    if frequency.shape != (3201,):
        raise ValueError("Invalid frequency shape")

    if not np.allclose(
        frequency,
        frequencies,
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Frequency grid mismatch")

    if inputs.shape != (8, 3201):
        raise ValueError("Invalid input shape")

    if reconstructed.shape != inputs.shape:
        raise ValueError("Invalid reconstruction shape")

    if stored_scores.shape != (8,):
        raise ValueError("Invalid segment score shape")

    for array in (
        inputs,
        reconstructed,
        stored_scores,
    ):
        if not np.isfinite(array).all():
            raise ValueError("Non-finite reconstruction data")

    if result["input"]["rpm"] != rpm:
        raise ValueError("RPM mismatch")

    error = np.square(
        reconstructed - inputs
    )

    if not np.isfinite(error).all():
        raise ValueError("Invalid reconstruction error")

    segment_scores = error.mean(axis=1)

    item = result["models"]["autoencoder"]

    expected_scores = np.asarray(
        [
            segment["score"]
            for segment in sorted(
                item["segments"],
                key=lambda s: s["segment_id"],
            )
        ],
        dtype=np.float64,
    )

    if expected_scores.shape != (8,):
        raise ValueError("Invalid reference segment scores")

    np.testing.assert_allclose(
        segment_scores,
        stored_scores,
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        segment_scores,
        expected_scores,
        rtol=1e-10,
        atol=1e-12,
    )

    overall = float(error.mean())

    np.testing.assert_allclose(
        overall,
        float(item["file_score"]),
        rtol=1e-10,
        atol=1e-12,
    )

    total_error = float(error.sum())

    bandwise = {
        "rpm": rpm,
        "bundle_id": result["bundle_id"],
        "overall_mse": overall,
        "bands": {},
    }

    for name, mask in masks.items():

        selected = error[:, mask]
        band_sum = float(selected.sum())

        bandwise["bands"][name] = {
            "mse": float(selected.mean()),
            "bin_count": int(mask.sum()),
            "min_hz": float(frequencies[mask].min()),
            "max_hz": float(frequencies[mask].max()),
            "error_fraction": (
                band_sum / total_error
                if total_error > 0
                else 0.0
            ),
            "global_contribution": (
                band_sum / error.size
            ),
        }

    return summarize_bandwise_result(bandwise)
