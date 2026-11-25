"""Paired exploratory evaluation of AE and Bandwise AE."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from vibration.bandwise_ae import BandwiseAE
from vibration.bandwise_calibration import (
    ALL_BANDS,
    calculate_bandwise_score,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = "date_prelim_20261106"

OUTPUT_DIR = (
    ROOT / "outputs/experimental_bandwise_ae"
)

CONFIG_PATH = OUTPUT_DIR / "experiment_config.json"

RPMS = (900, 1200, 1800)

OLD_THRESHOLDS = {
    900: 0.324213973,
    1200: 0.184915447,
    1800: 0.445469778,
}

EXPECTED_COUNTS = {
    900: (100, 398),
    1200: (99, 393),
    1800: (102, 392),
}

NORMAL_CONDITIONS = ("Normal1", "Normal2")

FAULT_GROUPS = {
    "IR": "Bearing",
    "OR": "Bearing",
    "Roller": "Bearing",
    "Looseness0.05": "Looseness",
    "Looseness0.1": "Looseness",
    "LoosenessFull": "Looseness",
    "Unbalance_-1": "Unbalance",
    "Unbalance_-2": "Unbalance",
}

THRESHOLD_NAMES = ("Q95", "Q97_5", "Q99")


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_csv(path):

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:

        return list(csv.DictReader(f))


def path_key(value):

    return str(value).strip().replace("\\", "/")


def parse_bool(value):

    value = str(value).strip().lower()

    if value in ("true", "1", "1.0"):
        return True

    if value in ("false", "0", "0.0"):
        return False

    raise ValueError(
        f"Unknown prediction value: {value}"
    )


def verify_frozen_config(config):

    if config["default_threshold"] != "Q95":
        raise ValueError("Default threshold changed")

    if config["evaluation_status"] != "exploratory_only":
        raise ValueError("Invalid evaluation status")

    if config["post_test_tuning_allowed"] is not False:
        raise ValueError("Post-test tuning not allowed")

    if config["approved_for_production"]:
        raise ValueError("Unexpected production approval")

    if config["expected_total_files"] != 1484:
        raise ValueError("Test count mismatch")

    hashes = config["artifact_hashes"]

    if len(hashes) != 18:
        raise ValueError("Frozen artifact count mismatch")

    for relative, expected in hashes.items():

        if sha256(ROOT / relative) != expected:

            raise ValueError(
                f"Frozen artifact changed: {relative}"
            )

    print("Frozen artifact verification: PASS")


def load_test_files(rpm):

    directory = ROOT / f"outputs/{rpm}rpm" / VERSION

    split_rows = read_csv(
        directory / f"split_{rpm}rpm.csv"
    )

    evaluation_rows = read_csv(
        directory / f"evaluation_file_{rpm}rpm.csv"
    )

    split_map = {}

    for row in split_rows:

        if row["split"] not in (
            "test_normal", "test_anomaly"
        ):
            continue

        key = path_key(row["file"])

        if key in split_map:
            raise ValueError("Duplicate test split file")

        if row["date"] != "20260919":
            raise ValueError("Unexpected test date")

        if int(row["rpm"]) != rpm:
            raise ValueError("Test RPM mismatch")

        condition = row["condition"]

        if row["split"] == "test_normal":

            if condition not in NORMAL_CONDITIONS:
                raise ValueError("Invalid normal condition")

        elif condition not in FAULT_GROUPS:

            raise ValueError(
                f"Unknown fault condition: {condition}"
            )

        split_map[key] = row

    evaluation_map = {}

    for row in evaluation_rows:

        key = path_key(row["file"])

        if key in evaluation_map:
            raise ValueError("Duplicate evaluation file")

        evaluation_map[key] = row

    if set(split_map) != set(evaluation_map):
        raise ValueError("Test file list mismatch")

    files = []

    for key, row in sorted(split_map.items()):

        reference = evaluation_map[key]

        if reference["split"] != row["split"]:
            raise ValueError("Reference split mismatch")

        if reference["condition"] != row["condition"]:
            raise ValueError("Reference condition mismatch")

        source = Path(key)

        if not source.is_absolute():
            source = ROOT / source

        if not source.is_file():
            raise FileNotFoundError(source)

        old_score = float(reference["score"])
        old_threshold = float(reference["threshold"])

        if not np.isfinite(old_score) or old_score < 0:
            raise ValueError("Invalid old score")

        np.testing.assert_allclose(
            old_threshold,
            OLD_THRESHOLDS[rpm],
            rtol=1e-6,
            atol=1e-8,
        )

        old_prediction = parse_bool(
            reference["predicted_anomaly"]
        )

        if old_prediction != (
            old_score > old_threshold
        ):
            raise ValueError("Old decision mismatch")

        files.append({
            "file": key,
            "path": source,
            "condition": row["condition"],
            "split": row["split"],
            "actual_anomaly": (
                row["split"] == "test_anomaly"
            ),
            "old_score": old_score,
            "old_threshold": old_threshold,
            "old_prediction": old_prediction,
        })

    counts = Counter(
        item["split"] for item in files
    )

    normal, anomaly = EXPECTED_COUNTS[rpm]

    if counts != {
        "test_normal": normal,
        "test_anomaly": anomaly,
    }:
        raise ValueError("Test count mismatch")

    print(
        f"{rpm} RPM: "
        f"{normal} normal, "
        f"{anomaly} anomaly"
    )

    return files


def evaluate_rpm(
    rpm,
    files,
    config,
    calibration,
    thresholds,
):

    model = BandwiseAE(rpm)

    rpm_cal = calibration["rpms"][str(rpm)]
    rpm_thresholds = thresholds["rpms"][str(rpm)]

    expected_bundle = config["rpms"][str(rpm)]["bundle_id"]

    if rpm_cal["bundle_id"] != expected_bundle:
        raise ValueError("Calibration bundle mismatch")

    if rpm_thresholds["bundle_id"] != expected_bundle:
        raise ValueError("Threshold bundle mismatch")

    q95 = float(
        rpm_thresholds["thresholds"]["Q95"]["threshold"]
    )

    if q95 != config["rpms"][str(rpm)]["threshold"]:
        raise ValueError("Frozen threshold mismatch")

    rows = []

    for index, item in enumerate(files, start=1):

        result = model.score_file(item["path"])

        if result["bundle_id"] != expected_bundle:
            raise ValueError("Inference bundle mismatch")

        overall = float(result["overall_mse"])

        np.testing.assert_allclose(
            overall,
            item["old_score"],
            rtol=1e-5,
            atol=1e-7,
        )

        old_prediction = (
            overall > item["old_threshold"]
        )

        if old_prediction != item["old_prediction"]:
            raise ValueError(
                "Stored AE prediction mismatch"
            )

        band_mse = {
            band: result["bands"][band]["mse"]
            for band in ALL_BANDS
        }

        calibrated = calculate_bandwise_score(
            band_mse,
            rpm_cal,
            rpm=rpm,
            bundle_id=result["bundle_id"],
        )

        new_score = calibrated["bandwise_score"]

        row = {
            "rpm": rpm,
            "file": item["file"],
            "date": "20260919",
            "condition": item["condition"],
            "split": item["split"],
            "actual_anomaly": int(
                item["actual_anomaly"]
            ),
            "ae_score": overall,
            "ae_reference_score": item["old_score"],
            "ae_threshold": item["old_threshold"],
            "ae_prediction": int(old_prediction),
            "bandwise_score": new_score,
            "dominant_band": calibrated["dominant_band"],
            "bundle_id": result["bundle_id"],
            "evaluation_status": "exploratory_only",
        }

        for band in ALL_BANDS:

            row[f"{band}_mse"] = band_mse[band]

            row[f"{band}_zscore"] = calibrated[
                "band_z_scores"
            ][band]

        for name in THRESHOLD_NAMES:

            threshold = float(
                rpm_thresholds["thresholds"][name][
                    "threshold"
                ]
            )

            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("Invalid threshold")

            row[f"{name}_threshold"] = threshold

            row[f"{name}_prediction"] = int(
                new_score > threshold
            )

        rows.append(row)

        if index % 50 == 0 or index == len(files):

            print(
                f"[{rpm} RPM] "
                f"{index}/{len(files)} evaluated",
                flush=True,
            )

    return rows


def metrics(rows, model):

    y = np.asarray([
        row["actual_anomaly"]
        for row in rows
    ], dtype=int)

    if model == "AE":

        predictions = np.asarray([
            row["ae_prediction"]
            for row in rows
        ], dtype=int)

        scores = np.asarray([
            row["ae_score"] / row["ae_threshold"]
            for row in rows
        ], dtype=float)

    else:

        name = model.removeprefix("Bandwise_")

        predictions = np.asarray([
            row[f"{name}_prediction"]
            for row in rows
        ], dtype=int)

        scores = np.asarray([
            row["bandwise_score"]
            / row[f"{name}_threshold"]
            for row in rows
        ], dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Invalid evaluation score")

    tp = int(((y == 1) & (predictions == 1)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())

    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "model": model,
        "files": len(rows),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "FPR": fpr,
        "Recall": recall,
        "Specificity": specificity,
        "Balanced_Accuracy": (
            recall + specificity
        ) / 2,
        "AUROC": float(
            roc_auc_score(y, scores)
        ),
        "AUPRC": float(
            average_precision_score(y, scores)
        ),
        "ranking_score": "score_divided_by_threshold",
        "evaluation_status": "exploratory_only",
    }


def build_group_results(rows):

    result = []

    models = {
        "AE": "ae_prediction",
        "Bandwise_Q95": "Q95_prediction",
        "Bandwise_Q97_5": "Q97_5_prediction",
        "Bandwise_Q99": "Q99_prediction",
    }

    groups = [
        ("condition", condition)
        for condition in (
            *NORMAL_CONDITIONS,
            *FAULT_GROUPS,
        )
    ]

    groups += [
        ("fault_group", group)
        for group in (
            "Bearing",
            "Looseness",
            "Unbalance",
        )
    ]

    for rpm in ("ALL", *RPMS):

        base = [
            row for row in rows
            if rpm == "ALL" or row["rpm"] == rpm
        ]

        for group_type, group in groups:

            if group_type == "condition":

                subset = [
                    row for row in base
                    if row["condition"] == group
                ]

            else:

                subset = [
                    row for row in base
                    if FAULT_GROUPS.get(
                        row["condition"]
                    ) == group
                ]

            if not subset:
                continue

            labels = {
                row["actual_anomaly"]
                for row in subset
            }

            if len(labels) != 1:
                raise ValueError("Mixed group labels")

            actual = next(iter(labels))

            for model, column in models.items():

                detected = sum(
                    row[column] for row in subset
                )

                result.append({
                    "rpm": rpm,
                    "group_type": group_type,
                    "group": group,
                    "model": model,
                    "files": len(subset),
                    "actual_anomaly": actual,
                    "predicted_anomaly": detected,
                    "rate_type": (
                        "Recall" if actual else "FPR"
                    ),
                    "rate": detected / len(subset),
                    "evaluation_status": "exploratory_only",
                })

    return result


def save_csv(path, rows):

    temporary = path.with_suffix(".csv.tmp")

    with temporary.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def main():

    config_hash = sha256(CONFIG_PATH)

    config = read_json(CONFIG_PATH)

    verify_frozen_config(config)

    calibration = read_json(
        OUTPUT_DIR / "bandwise_calibration.json"
    )

    thresholds = read_json(
        OUTPUT_DIR / "bandwise_thresholds.json"
    )

    print("\n===== Test preflight =====")

    test_sets = {
        rpm: load_test_files(rpm)
        for rpm in RPMS
    }

    paths = [
        item["path"].resolve()
        for files in test_sets.values()
        for item in files
    ]

    if len(paths) != 1484:
        raise ValueError("Unexpected total file count")

    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source files")

    print("\n===== Paired evaluation =====")

    rows = []

    for rpm in RPMS:

        rows.extend(
            evaluate_rpm(
                rpm,
                test_sets[rpm],
                config,
                calibration,
                thresholds,
            )
        )

    if len(rows) != 1484:
        raise ValueError("Missing evaluation rows")

    models = (
        "AE",
        "Bandwise_Q95",
        "Bandwise_Q97_5",
        "Bandwise_Q99",
    )

    summary = []

    for rpm in ("ALL", *RPMS):

        subset = [
            row for row in rows
            if rpm == "ALL" or row["rpm"] == rpm
        ]

        for model in models:

            result = metrics(subset, model)
            result["rpm"] = rpm

            summary.append(result)

    baseline = summary[0]

    if (
        baseline["FP"] != 224
        or baseline["FN"] != 142
    ):
        raise ValueError(
            "Historical AE baseline mismatch"
        )

    group_results = build_group_results(rows)

    expected_group_counts = {
        "Bearing": 442,
        "Looseness": 452,
        "Unbalance": 289,
    }

    actual_group_counts = Counter(
        FAULT_GROUPS[row["condition"]]
        for row in rows
        if row["actual_anomaly"]
    )

    if actual_group_counts != expected_group_counts:
        raise ValueError("Fault group count mismatch")

    verify_frozen_config(config)

    if sha256(CONFIG_PATH) != config_hash:
        raise ValueError("Experiment config changed")

    save_csv(
        OUTPUT_DIR / "test_bandwise_predictions.csv",
        rows,
    )

    save_csv(
        OUTPUT_DIR / "test_bandwise_metrics.csv",
        summary,
    )

    save_csv(
        OUTPUT_DIR / "test_bandwise_group_metrics.csv",
        group_results,
    )

    print("\n===== Overall comparison =====")

    for row in summary:

        if row["rpm"] != "ALL":
            continue

        print(
            f"{row['model']:18s} "
            f"FP={row['FP']:3d} "
            f"FN={row['FN']:3d} "
            f"FPR={row['FPR']:.2%} "
            f"Recall={row['Recall']:.2%} "
            f"BA={row['Balanced_Accuracy']:.2%} "
            f"AUROC={row['AUROC']:.6f} "
            f"AUPRC={row['AUPRC']:.6f}"
        )

    print("\n===== Primary Q95 group comparison =====")

    for row in group_results:

        if (
            row["rpm"] == "ALL"
            and row["model"] in (
                "AE", "Bandwise_Q95"
            )
            and (
                row["group_type"] == "fault_group"
                or row["group"] in NORMAL_CONDITIONS
            )
        ):

            print(
                f"{row['group']:12s} "
                f"{row['model']:14s} "
                f"{row['rate_type']}="
                f"{row['rate']:.2%} "
                f"({row['predicted_anomaly']}"
                f"/{row['files']})"
            )

    print("\n===== Evaluation completed =====")
    print("Evaluated files: 1484")
    print("AE score consistency: 1484/1484 PASS")
    print("Historical baseline: PASS")
    print("Default threshold: Q95")
    print("Evaluation status: exploratory_only")
    print("Post-test tuning allowed: False")
    print("Operational model changed: False")
    print("Saved directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
