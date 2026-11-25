"""Experimental validation-based bandwise AE calibration."""

import numpy as np


SUPPORTED_RPMS = (900, 1200, 1800)

PRIMARY_BANDS = (
    "0_100Hz",
    "100_1000Hz",
    "1000_10000Hz",
)

AUXILIARY_BANDS = (
    "1X",
    "2X",
)

ALL_BANDS = PRIMARY_BANDS + AUXILIARY_BANDS


def fit_band_statistics(rpm, rows):

    if rpm not in SUPPORTED_RPMS:
        raise ValueError("Unsupported RPM")

    rows = list(rows)

    if len(rows) < 2:
        raise ValueError(
            "Insufficient validation samples"
        )

    seen = set()
    bundle_ids = set()

    for row in rows:

        if int(row["rpm"]) != rpm:
            raise ValueError("RPM mismatch")

        if row["split"] != "validation":
            raise ValueError(
                "Only Validation data allowed"
            )

        if row["condition"] not in (
            "Normal1", "Normal2"
        ):
            raise ValueError(
                "Only Normal data allowed"
            )

        if row["date"] != "20260916":
            raise ValueError(
                "Unexpected validation date"
            )

        file = row["file"]

        if file in seen:
            raise ValueError("Duplicate file")

        seen.add(file)
        bundle_ids.add(row["bundle_id"])

    if len(bundle_ids) != 1:
        raise ValueError("Mixed model bundles")

    statistics = {}

    for band in ALL_BANDS:

        values = np.asarray(
            [
                float(row[f"{band}_mse"])
                for row in rows
            ],
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid MSE: {band}"
            )

        if (values < 0).any():
            raise ValueError(
                f"Negative MSE: {band}"
            )

        mean = float(values.mean())
        std = float(values.std(ddof=0))

        if not np.isfinite(mean):
            raise ValueError(
                f"Invalid mean: {band}"
            )

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid standard deviation: {band}"
            )

        statistics[band] = {
            "mean": mean,
            "std": std,
            "count": len(values),
        }

    return {
        "rpm": rpm,
        "bundle_id": next(iter(bundle_ids)),
        "sample_count": len(rows),
        "statistics": statistics,
        "primary_bands": list(PRIMARY_BANDS),
        "auxiliary_bands": list(AUXILIARY_BANDS),
        "standardization": "zscore",
        "ddof": 0,
        "aggregation": "max",
        "calibration_split": "normal_validation",
        "split_mode": "date_preliminary",
        "run_independence_verified": False,
        "evaluation_status": "experimental_only",
        "approved_for_production": False,
    }


def calculate_bandwise_score(
    band_mse,
    calibration,
    *,
    rpm,
    bundle_id,
):

    if rpm != calibration["rpm"]:
        raise ValueError("Calibration RPM mismatch")

    if bundle_id != calibration["bundle_id"]:
        raise ValueError("Calibration bundle mismatch")

    if tuple(
        calibration["primary_bands"]
    ) != PRIMARY_BANDS:
        raise ValueError("Primary band mismatch")

    z_scores = {}

    for band in ALL_BANDS:

        mse = float(band_mse[band])

        stats = calibration["statistics"][band]

        mean = float(stats["mean"])
        std = float(stats["std"])

        if not np.isfinite(mse) or mse < 0:
            raise ValueError(
                f"Invalid MSE: {band}"
            )

        if not np.isfinite(mean) or mean < 0:
            raise ValueError(
                f"Invalid mean: {band}"
            )

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid standard deviation: {band}"
            )

        z = (mse - mean) / std

        if not np.isfinite(z):
            raise ValueError(
                f"Invalid z-score: {band}"
            )

        z_scores[band] = float(z)

    dominant_band = max(
        PRIMARY_BANDS,
        key=lambda band: z_scores[band],
    )

    return {
        "rpm": rpm,
        "bundle_id": bundle_id,
        "band_z_scores": z_scores,
        "bandwise_score": z_scores[dominant_band],
        "dominant_band": dominant_band,
        "evaluation_status": "experimental_only",
        "approved_for_production": False,
        "operational_model_changed": False,
    }
