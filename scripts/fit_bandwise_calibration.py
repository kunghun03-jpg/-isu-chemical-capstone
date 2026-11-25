"""Fit and save experimental bandwise AE calibration."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vibration.bandwise_calibration import (
    ALL_BANDS,
    PRIMARY_BANDS,
    SUPPORTED_RPMS,
    fit_band_statistics,
    calculate_bandwise_score,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "outputs/experimental_bandwise_ae"
)

INPUT_PATH = (
    OUTPUT_DIR / "validation_band_scores.csv"
)

CALIBRATION_PATH = (
    OUTPUT_DIR / "bandwise_calibration.json"
)

SCORES_PATH = (
    OUTPUT_DIR / "validation_calibrated_scores.csv"
)

EXPECTED_COUNTS = {
    900: 108,
    1200: 94,
    1800: 103,
}

EXPECTED_CONDITIONS = {
    (900, "Normal1"): 50,
    (900, "Normal2"): 58,
    (1200, "Normal1"): 48,
    (1200, "Normal2"): 46,
    (1800, "Normal1"): 52,
    (1800, "Normal2"): 51,
}


def load_validation_rows():

    with INPUT_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 305:
        raise ValueError("Unexpected file count")

    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate validation files")

    counts = Counter(
        int(row["rpm"]) for row in rows
    )

    if counts != EXPECTED_COUNTS:
        raise ValueError("RPM count mismatch")

    conditions = Counter(
        (int(row["rpm"]), row["condition"])
        for row in rows
    )

    if conditions != EXPECTED_CONDITIONS:
        raise ValueError("Condition count mismatch")

    for row in rows:

        if int(row["segment_count"]) != 8:
            raise ValueError("Segment count mismatch")

        if row["evaluation_status"] != "experimental_only":
            raise ValueError("Invalid evaluation status")

    return rows


def main():

    rows = load_validation_rows()

    artifact = {
        "schema_version": 1,
        "method": "bandwise_zscore_max",
        "source_version": "date_prelim_20261106",
        "source_sha256": hashlib.sha256(
            INPUT_PATH.read_bytes()
        ).hexdigest(),
        "validation_file_count": len(rows),
        "calibration_split": "normal_validation",
        "primary_bands": list(PRIMARY_BANDS),
        "aggregation": "max",
        "threshold_configured": False,
        "run_independence_verified": False,
        "evaluation_status": "experimental_only",
        "approved_for_production": False,
        "operational_model_changed": False,
        "rpms": {},
    }

    calibrated_rows = []

    print("===== Fit bandwise calibration =====")

    for rpm in SUPPORTED_RPMS:

        subset = [
            row for row in rows
            if int(row["rpm"]) == rpm
        ]

        calibration = fit_band_statistics(
            rpm, subset
        )

        artifact["rpms"][str(rpm)] = calibration

        scores = []

        for row in subset:

            band_mse = {
                band: float(row[f"{band}_mse"])
                for band in ALL_BANDS
            }

            result = calculate_bandwise_score(
                band_mse,
                calibration,
                rpm=rpm,
                bundle_id=row["bundle_id"],
            )

            output = dict(row)

            for band in ALL_BANDS:
                output[f"{band}_zscore"] = (
                    result["band_z_scores"][band]
                )

            output["bandwise_score"] = (
                result["bandwise_score"]
            )

            output["dominant_band"] = (
                result["dominant_band"]
            )

            calibrated_rows.append(output)

            scores.append(
                result["bandwise_score"]
            )

        scores = np.asarray(
            scores, dtype=np.float64
        )

        if not np.isfinite(scores).all():
            raise ValueError("Invalid combined scores")

        print(
            f"{rpm} RPM: "
            f"{len(scores)} files, "
            f"mean={scores.mean():.8f}, "
            f"std={scores.std():.8f}"
        )

    if len(calibrated_rows) != len(rows):
        raise ValueError("Missing calibrated rows")

    json_tmp = CALIBRATION_PATH.with_suffix(
        ".json.tmp"
    )

    csv_tmp = SCORES_PATH.with_suffix(
        ".csv.tmp"
    )

    json_tmp.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    with csv_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(calibrated_rows[0]),
        )

        writer.writeheader()
        writer.writerows(calibrated_rows)

    # Verify serialized statistics and scores.
    restored = json.loads(
        json_tmp.read_text(encoding="utf-8")
    )

    with csv_tmp.open(
        encoding="utf-8", newline=""
    ) as f:
        restored_rows = list(csv.DictReader(f))

    if restored != artifact:
        raise ValueError("Calibration reload mismatch")

    if len(restored_rows) != 305:
        raise ValueError("Saved score count mismatch")

    for row in restored_rows:

        rpm = int(row["rpm"])

        calibration = restored["rpms"][str(rpm)]

        band_mse = {
            band: float(row[f"{band}_mse"])
            for band in ALL_BANDS
        }

        result = calculate_bandwise_score(
            band_mse,
            calibration,
            rpm=rpm,
            bundle_id=row["bundle_id"],
        )

        np.testing.assert_allclose(
            result["bandwise_score"],
            float(row["bandwise_score"]),
            rtol=1e-12,
            atol=1e-12,
        )

        for band in ALL_BANDS:

            np.testing.assert_allclose(
                result["band_z_scores"][band],
                float(row[f"{band}_zscore"]),
                rtol=1e-12,
                atol=1e-12,
            )

        if (
            result["dominant_band"]
            != row["dominant_band"]
        ):
            raise ValueError("Dominant band mismatch")

    json_tmp.replace(CALIBRATION_PATH)
    csv_tmp.replace(SCORES_PATH)

    print("\n===== Calibration saved =====")
    print("Calibration:", CALIBRATION_PATH.name)
    print("Scores:", SCORES_PATH.name)
    print("Total files:", len(restored_rows))
    print("JSON reload: PASS")
    print("Saved score parity: 305/305 PASS")
    print("Threshold configured: False")
    print("Operational model changed: False")


if __name__ == "__main__":
    main()
