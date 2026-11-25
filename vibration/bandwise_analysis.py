"""Supplementary frequency-band reconstruction analysis."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from vibration.bandwise_ae import (
    BandwiseAE,
    SUPPORTED_RPMS,
)


BROAD_BANDS = (
    "0_100Hz",
    "100_1000Hz",
    "1000_10000Hz",
)

HARMONIC_BANDS = (
    "1X",
    "2X",
)

BAND_LABELS = {
    "1X": "1X rotation frequency",
    "2X": "2X rotation frequency",
    "0_100Hz": "Low frequency",
    "100_1000Hz": "Mid frequency",
    "1000_10000Hz": "High frequency",
}


@lru_cache(maxsize=3)
def get_bandwise_model(rpm):

    if rpm not in SUPPORTED_RPMS:
        raise ValueError("Unsupported RPM")

    return BandwiseAE(rpm)


def summarize_bandwise_result(result):

    overall = float(result["overall_mse"])

    if not np.isfinite(overall) or overall < 0:
        raise ValueError("Invalid overall MSE")

    bands = []

    for name in BROAD_BANDS + HARMONIC_BANDS:

        source = result["bands"][name]

        mse = float(source["mse"])
        fraction = float(source["error_fraction"])
        contribution = float(
            source["global_contribution"]
        )

        if not np.isfinite(
            [mse, fraction, contribution]
        ).all():
            raise ValueError("Invalid band statistics")

        if mse < 0 or contribution < 0:
            raise ValueError("Negative reconstruction error")

        if not 0 <= fraction <= 1:
            raise ValueError("Invalid error fraction")

        bands.append({
            "band": name,
            "label": BAND_LABELS[name],
            "band_type": (
                "broad"
                if name in BROAD_BANDS
                else "harmonic"
            ),
            "min_hz": float(source["min_hz"]),
            "max_hz": float(source["max_hz"]),
            "bin_count": int(source["bin_count"]),
            "mse": mse,
            "error_fraction": fraction,
            "error_percentage": fraction * 100,
            "global_contribution": contribution,
        })

    broad = [
        band for band in bands
        if band["band_type"] == "broad"
    ]

    combined = sum(
        band["global_contribution"]
        for band in broad
    )

    if not np.isclose(
        combined,
        overall,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Band contribution mismatch")

    if overall > 0:

        highest_mse = max(
            broad,
            key=lambda band: band["mse"],
        )["band"]

        highest_contribution = max(
            broad,
            key=lambda band: band["global_contribution"],
        )["band"]

    else:
        highest_mse = None
        highest_contribution = None

    return {
        "schema_version": 1,
        "rpm": result["rpm"],
        "bundle_id": result["bundle_id"],
        "overall_mse": overall,
        "bands": bands,
        "highest_mse_band": highest_mse,
        "highest_contribution_band": (
            highest_contribution
        ),
        "analysis_status": "supplementary_only",
        "anomaly_decision_provided": False,
        "fault_classification_provided": False,
        "approved_for_production": False,
        "operational_model_changed": False,
        "harmonic_bands_overlap_broad_bands": True,
    }


def analyze_bandwise_bytes(raw, rpm):

    model = get_bandwise_model(rpm)

    result = model.score_bytes(raw)

    return summarize_bandwise_result(result)


def analyze_bandwise_file(path, rpm):

    raw = Path(path).read_bytes()

    return analyze_bandwise_bytes(raw, rpm)
