"""Freeze bandwise AE experimental configuration before evaluation."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "outputs/experimental_bandwise_ae"
)

BUNDLE_ROOT = (
    ROOT
    / "outputs/inference_bundles/date_prelim_20261106"
)

RPMS = (900, 1200, 1800)

PRIMARY_BANDS = [
    "0_100Hz",
    "100_1000Hz",
    "1000_10000Hz",
]

EXPECTED_THRESHOLDS = {
    900: 2.4203923674,
    1200: 2.1956653014,
    1800: 2.1865503520,
}

EXPECTED_TEST_COUNTS = {
    "900": {
        "normal": 100,
        "anomaly": 398,
    },
    "1200": {
        "normal": 99,
        "anomaly": 393,
    },
    "1800": {
        "normal": 102,
        "anomaly": 392,
    },
}


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():

    output = OUTPUT_DIR / "experiment_config.json"

    if output.exists():
        raise FileExistsError(
            "Experiment already frozen. "
            "Do not overwrite the configuration."
        )

    calibration_path = (
        OUTPUT_DIR / "bandwise_calibration.json"
    )

    thresholds_path = (
        OUTPUT_DIR / "bandwise_thresholds.json"
    )

    calibration = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = json.loads(
        thresholds_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        thresholds["calibration_sha256"]
        != sha256(calibration_path)
    ):
        raise ValueError(
            "Calibration hash mismatch"
        )

    if thresholds["default_threshold"] != "Q95":
        raise ValueError(
            "Default threshold must be Q95"
        )

    if thresholds["threshold_source"] != "normal_validation":
        raise ValueError(
            "Invalid threshold source"
        )

    if calibration["primary_bands"] != PRIMARY_BANDS:
        raise ValueError(
            "Calibration band mismatch"
        )

    if thresholds["primary_bands"] != PRIMARY_BANDS:
        raise ValueError(
            "Threshold band mismatch"
        )

    if calibration["validation_file_count"] != 305:
        raise ValueError(
            "Invalid calibration sample count"
        )

    if thresholds["validation_file_count"] != 305:
        raise ValueError(
            "Invalid threshold sample count"
        )

    if (
        calibration["approved_for_production"]
        or thresholds["approved_for_production"]
    ):
        raise ValueError(
            "Unexpected production approval"
        )

    frozen = {
        "schema_version": 1,
        "experiment_id": (
            "bandwise_ae_q95_date_preliminary"
        ),
        "method": "bandwise_zscore_max",
        "source_version": "date_prelim_20261106",
        "calibration_source": "normal_validation",
        "validation_files": 305,
        "primary_bands": PRIMARY_BANDS,
        "auxiliary_bands": ["1X", "2X"],
        "aggregation": "max",
        "standardization": "zscore",
        "ddof": 0,
        "default_threshold": "Q95",
        "comparison_thresholds": [
            "Q97_5", "Q99"
        ],
        "decision_rule": "score > threshold",
        "test_date": "20260919",
        "expected_test_counts": EXPECTED_TEST_COUNTS,
        "expected_total_files": 1484,
        "evaluation_status": "exploratory_only",
        "test_previously_used": True,
        "run_independence_verified": False,
        "post_test_tuning_allowed": False,
        "approved_for_production": False,
        "operational_model_changed": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "artifact_hashes": {},
        "rpms": {},
    }

    artifacts = [
        calibration_path,
        thresholds_path,
        OUTPUT_DIR / "validation_band_scores.csv",
        OUTPUT_DIR / "validation_calibrated_scores.csv",
        ROOT / "vibration/bandwise_ae.py",
        ROOT / "vibration/bandwise_calibration.py",
        ROOT / "vibration/inference.py",
        ROOT / "vibration/preprocessing.py",
        ROOT / "vibration/model.py",
    ]

    for rpm in RPMS:

        key = str(rpm)

        rpm_calibration = calibration["rpms"][key]
        rpm_thresholds = thresholds["rpms"][key]

        if (
            rpm_calibration["bundle_id"]
            != rpm_thresholds["bundle_id"]
        ):
            raise ValueError(
                f"Bundle mismatch: {rpm} RPM"
            )

        q95 = float(
            rpm_thresholds["default_threshold"]
        )

        np.testing.assert_allclose(
            q95,
            EXPECTED_THRESHOLDS[rpm],
            rtol=0,
            atol=1e-9,
        )

        np.testing.assert_allclose(
            q95,
            rpm_thresholds["thresholds"]["Q95"][
                "threshold"
            ],
            rtol=0,
            atol=1e-12,
        )

        bundle = BUNDLE_ROOT / f"{rpm}rpm"

        for name in (
            "autoencoder.pt",
            "normalization.npz",
            "manifest.json",
        ):
            artifacts.append(bundle / name)

        frozen["rpms"][key] = {
            "bundle_id": rpm_calibration["bundle_id"],
            "threshold": q95,
        }

        print(
            f"{rpm} RPM: "
            f"Q95={q95:.10f} PASS"
        )

    for path in artifacts:

        if not path.is_file():
            raise FileNotFoundError(path)

        relative = path.relative_to(ROOT).as_posix()

        frozen["artifact_hashes"][relative] = (
            sha256(path)
        )

    temporary = output.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            frozen,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    restored = json.loads(
        temporary.read_text(encoding="utf-8")
    )

    if restored != frozen:
        raise ValueError(
            "Configuration reload mismatch"
        )

    for relative, expected in (
        restored["artifact_hashes"].items()
    ):

        if sha256(ROOT / relative) != expected:
            raise ValueError(
                f"Artifact changed: {relative}"
            )

    temporary.replace(output)

    print("\n===== Experiment configuration frozen =====")
    print("Default threshold: Q95")
    print("Expected test files: 1484")
    print(
        "Frozen artifacts:",
        len(frozen["artifact_hashes"])
    )
    print("Artifact hash verification: PASS")
    print("Configuration reload: PASS")
    print("Evaluation status: exploratory_only")
    print("Post-test tuning allowed: False")
    print("Operational model changed: False")
    print("Saved:", output)


if __name__ == "__main__":
    main()
