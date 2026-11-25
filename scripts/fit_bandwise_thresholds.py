"""Fit experimental bandwise AE thresholds using Normal Validation only."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from vibration.bandwise_calibration import (
    ALL_BANDS,
    PRIMARY_BANDS,
    SUPPORTED_RPMS,
    calculate_bandwise_score,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "outputs/experimental_bandwise_ae"
)

CALIBRATION_PATH = (
    OUTPUT_DIR / "bandwise_calibration.json"
)

SCORES_PATH = (
    OUTPUT_DIR / "validation_calibrated_scores.csv"
)

SOURCE_PATH = (
    OUTPUT_DIR / "validation_band_scores.csv"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "bandwise_thresholds.json"
)

EXPECTED_COUNTS = {
    900: 108,
    1200: 94,
    1800: 103,
}

QUANTILES = {
    "Q95": 0.95,
    "Q97_5": 0.975,
    "Q99": 0.99,
}

DEFAULT_THRESHOLD = "Q95"


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_inputs():

    calibration = json.loads(
        CALIBRATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    with SCORES_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if calibration["method"] != "bandwise_zscore_max":
        raise ValueError("Unexpected calibration method")

    if calibration["source_sha256"] != sha256(SOURCE_PATH):
        raise ValueError("Calibration source mismatch")

    if calibration["calibration_split"] != "normal_validation":
        raise ValueError("Invalid calibration split")

    if calibration["approved_for_production"]:
        raise ValueError("Unexpected production approval")

    if tuple(calibration["primary_bands"]) != PRIMARY_BANDS:
        raise ValueError("Primary band mismatch")

    if set(calibration["rpms"]) != {
        str(rpm) for rpm in SUPPORTED_RPMS
    }:
        raise ValueError("Calibration RPM mismatch")

    if len(rows) != 305:
        raise ValueError("Unexpected Validation count")

    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate Validation files")

    with SOURCE_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        source_rows = list(csv.DictReader(f))

    source_map = {
        row["file"]: row for row in source_rows
    }

    if (
        len(source_rows) != 305
        or len(source_map) != 305
        or set(source_map) != {
            row["file"] for row in rows
        }
    ):
        raise ValueError("Source file mismatch")

    for row in rows:

        if row["split"] != "validation":
            raise ValueError("Non-validation data")

        if row["condition"] not in (
            "Normal1", "Normal2"
        ):
            raise ValueError("Non-normal data")

        if row["date"] != "20260916":
            raise ValueError("Unexpected date")

        if int(row["rpm"]) not in SUPPORTED_RPMS:
            raise ValueError("Unsupported RPM")

        if int(row["segment_count"]) != 8:
            raise ValueError("Invalid segment count")

        original = source_map[row["file"]]

        if any(
            row[key] != value
            for key, value in original.items()
        ):
            raise ValueError("Original score mismatch")

    return calibration, rows


def main():

    calibration, rows = load_inputs()

    artifact = {
        "schema_version": 1,
        "method": "bandwise_zscore_max",
        "source_version": "date_prelim_20261106",
        "calibration_sha256": sha256(
            CALIBRATION_PATH
        ),
        "validation_scores_sha256": sha256(
            SCORES_PATH
        ),
        "validation_file_count": len(rows),
        "threshold_source": "normal_validation",
        "default_threshold": DEFAULT_THRESHOLD,
        "quantile_method": "linear",
        "primary_bands": list(PRIMARY_BANDS),
        "decision_rule": "score > threshold",
        "split_mode": "date_preliminary",
        "run_independence_verified": False,
        "validation_fpr_is_independent": False,
        "evaluation_status": "experimental_only",
        "approved_for_production": False,
        "operational_model_changed": False,
        "rpms": {},
    }

    print("===== Fit bandwise thresholds =====")

    total = 0

    for rpm in SUPPORTED_RPMS:

        subset = [
            row for row in rows
            if int(row["rpm"]) == rpm
        ]

        if len(subset) != EXPECTED_COUNTS[rpm]:
            raise ValueError("RPM count mismatch")

        rpm_calibration = calibration[
            "rpms"
        ][str(rpm)]

        if rpm_calibration["sample_count"] != len(subset):
            raise ValueError("Calibration count mismatch")

        verified_scores = []

        for row in subset:

            band_mse = {
                band: float(row[f"{band}_mse"])
                for band in ALL_BANDS
            }

            result = calculate_bandwise_score(
                band_mse,
                rpm_calibration,
                rpm=rpm,
                bundle_id=row["bundle_id"],
            )

            score = result["bandwise_score"]

            np.testing.assert_allclose(
                score,
                float(row["bandwise_score"]),
                rtol=1e-12,
                atol=1e-12,
            )

            verified_scores.append(score)

        scores = np.asarray(
            verified_scores,
            dtype=np.float64,
        )

        if not np.isfinite(scores).all():
            raise ValueError("Invalid score")

        thresholds = {}

        print(f"\n===== {rpm} RPM =====")
        print("Validation files:", len(scores))

        for name, quantile in QUANTILES.items():

            threshold = float(
                np.quantile(
                    scores,
                    quantile,
                    method="linear",
                )
            )

            if not np.isfinite(threshold):
                raise ValueError("Invalid threshold")

            flags = scores > threshold

            fp = int(flags.sum())
            tn = len(scores) - fp
            fpr = fp / len(scores)

            condition_results = {}

            for condition in ("Normal1", "Normal2"):

                mask = np.asarray([
                    row["condition"] == condition
                    for row in subset
                ])

                count = int(mask.sum())

                if count == 0:
                    raise ValueError(
                        "Missing normal condition"
                    )

                condition_fp = int(
                    flags[mask].sum()
                )

                condition_results[condition] = {
                    "files": count,
                    "FP": condition_fp,
                    "TN": count - condition_fp,
                    "FPR": condition_fp / count,
                }

            thresholds[name] = {
                "quantile": quantile,
                "threshold": threshold,
                "validation_files": len(scores),
                "validation_FP": fp,
                "validation_TN": tn,
                "validation_FPR": fpr,
                "conditions": condition_results,
            }

            print(
                f"{name:7s} "
                f"threshold={threshold:.10f} "
                f"FP={fp}/{len(scores)} "
                f"FPR={fpr:.2%}"
            )

            for condition, stats in (
                condition_results.items()
            ):
                print(
                    f"  {condition}: "
                    f"FP={stats['FP']}/{stats['files']} "
                    f"FPR={stats['FPR']:.2%}"
                )

        if not (
            thresholds["Q95"]["threshold"]
            <= thresholds["Q97_5"]["threshold"]
            <= thresholds["Q99"]["threshold"]
        ):
            raise ValueError("Threshold order mismatch")

        if not (
            thresholds["Q95"]["validation_FP"]
            >= thresholds["Q97_5"]["validation_FP"]
            >= thresholds["Q99"]["validation_FP"]
        ):
            raise ValueError("False positive order mismatch")

        artifact["rpms"][str(rpm)] = {
            "rpm": rpm,
            "bundle_id": rpm_calibration["bundle_id"],
            "sample_count": len(scores),
            "thresholds": thresholds,
            "default_threshold": thresholds[
                DEFAULT_THRESHOLD
            ]["threshold"],
        }

        total += len(scores)

        print("Score consistency: PASS")
        print("Threshold ordering: PASS")

    if total != 305:
        raise ValueError("Missing Validation scores")

    temporary = OUTPUT_PATH.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    restored = json.loads(
        temporary.read_text(encoding="utf-8")
    )

    if restored != artifact:
        raise ValueError("Threshold reload mismatch")

    temporary.replace(OUTPUT_PATH)

    print("\n===== Threshold configuration completed =====")
    print("Validation files:", total)
    print("Default threshold: Q95")
    print("RPM-specific thresholds: 9/9")
    print("Saved score consistency: 305/305 PASS")
    print("Threshold reload: PASS")
    print("Validation FPR is independent: False")
    print("Operational model changed: False")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
