"""정상 물리적 진동 특징 기반 이상 탐지 비교 실험."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.io import loadmat

from vibration.physical_features import (
    FEATURE_NAMES,
    extract_physical_features,
)

from scripts.evaluate_normal1_dataset_ae import (
    require,
    metrics,
    sha256,
)


ROOT = Path("outputs/experimental_normal1_training")
OUTPUT = Path("outputs/experimental_physical_features")

FEATURE_FLOOR = 1e-12
SCALE_FLOOR = 0.05


def extract_files(rows, rpm):

    results = []

    for index, row in enumerate(
        rows.itertuples(index=False), 1
    ):

        raw = loadmat(row.file)

        features = extract_physical_features(
            raw["signals"],
            raw["fs"],
            rpm,
        )

        # 8개 구간의 특징별 중앙값을
        # 파일 대표 특징으로 사용
        values = np.median(features, axis=0)

        require(
            values.shape == (10,)
            and np.isfinite(values).all()
            and (values >= 0).all(),
            f"Invalid features: {row.file}",
        )

        item = {
            "file": row.file,
            "condition": row.condition,
            "rpm": rpm,
        }

        item.update(
            dict(zip(FEATURE_NAMES, values))
        )

        results.append(item)

        if index % 100 == 0:
            print(
                f"Features: {index}/{len(rows)}",
                flush=True,
            )

    return pd.DataFrame(results)


def transform(frame):

    values = frame[
        list(FEATURE_NAMES)
    ].to_numpy(dtype=np.float64)

    return np.log(
        np.maximum(values, FEATURE_FLOOR)
    )


def fit_reference(frame):

    X = transform(frame)

    center = np.median(X, axis=0)

    mad = np.median(
        np.abs(X - center),
        axis=0,
    )

    scale = np.maximum(
        1.4826 * mad,
        SCALE_FLOOR,
    )

    require(
        np.isfinite(center).all()
        and np.isfinite(scale).all()
        and (scale > 0).all(),
        "Invalid normal reference",
    )

    return center, scale


def deviations(frame, center, scale):

    values = transform(frame)

    z = np.abs(
        (values - center) / scale
    )

    require(
        np.isfinite(z).all(),
        "Invalid deviation scores",
    )

    return z


def score_values(z, mode):

    if mode == "mean":
        return z.mean(axis=1)

    if mode == "max":
        return z.max(axis=1)

    raise ValueError(mode)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        choices=[
            "20260912",
            "20260916",
            "20260919",
        ],
    )

    parser.add_argument(
        "--rpm",
        required=True,
        type=int,
        choices=[900, 1200, 1800],
    )

    args = parser.parse_args()

    source = (
        ROOT / args.date / f"{args.rpm}rpm"
    )

    out = (
        OUTPUT / args.date / f"{args.rpm}rpm"
    )

    require(
        not out.exists(),
        f"Output already exists: {out}",
    )

    source_paths = [
        source / name
        for name in (
            "split.csv",
            "test_scores.csv",
            "metrics_comparison.csv",
        )
    ]

    hashes = {
        str(p): sha256(p)
        for p in source_paths
    }

    split = pd.read_csv(
        source / "split.csv",
        dtype={"date": str},
    )

    configurations = {
        "P1_normal1_mean": (
            "A_normal1_per_frequency",
            "mean",
        ),

        "P2_normal1_max": (
            "A_normal1_per_frequency",
            "max",
        ),

        "P3_mixed_mean": (
            "C_mixed_per_frequency",
            "mean",
        ),

        "P4_mixed_max": (
            "C_mixed_per_frequency",
            "max",
        ),
    }

    reference_names = [
        "A_normal1_per_frequency",
        "C_mixed_per_frequency",
    ]

    partitions = {}
    common_files = None

    print("\n===== Split verification =====")

    for reference in reference_names:

        rows = split[
            split["method"] == reference
        ].copy()

        require(
            rows["file"].is_unique
            and rows["date"].eq(args.date).all()
            and rows["rpm"].eq(args.rpm).all(),
            "Invalid split metadata",
        )

        train = rows[
            rows["split"] == "train"
        ]

        val = rows[
            rows["split"] == "validation"
        ]

        test = rows[
            rows["split"] == "test"
        ]

        train_files = set(train["file"])
        val_files = set(val["file"])
        test_files = set(test["file"])

        require(
            len(train_files) == 20
            and len(val_files) == 20
            and len(test_files) > 0
            and len(train) + len(val) + len(test)
                == len(rows)
            and not train_files & val_files
            and not train_files & test_files
            and not val_files & test_files,
            "Invalid data partitions",
        )

        expected = (
            {"Normal1": 20}
            if reference.startswith("A_")
            else {"Normal1": 10, "Normal2": 10}
        )

        require(
            train["condition"].value_counts().to_dict()
            == expected
            and val["condition"].value_counts().to_dict()
            == expected,
            "Invalid normal development data",
        )

        if common_files is None:
            common_files = test_files

        require(
            test_files == common_files,
            "Common test files mismatch",
        )

        partitions[reference] = (
            train, val, test
        )

        print(reference)
        print("Train:", len(train))
        print("Validation:", len(val))
        print("Test:", len(test))

    # 정상 개발 데이터만 먼저 처리

    prepared = {}

    for reference in reference_names:

        train, val, _ = partitions[reference]

        print(f"\n===== {reference} =====")

        train_features = extract_files(
            train, args.rpm
        )

        val_features = extract_files(
            val, args.rpm
        )

        center, scale = fit_reference(
            train_features
        )

        prepared[reference] = {
            "center": center,
            "scale": scale,
            "validation": val_features,
            "validation_z": deviations(
                val_features, center, scale
            ),
        }

        print(
            pd.DataFrame({
                "feature": FEATURE_NAMES,
                "log_center": center,
                "robust_scale": scale,
            }).to_string(index=False)
        )

    thresholds = {}
    validation_results = []

    print("\n===== Frozen thresholds =====")

    for method, (reference, mode) in (
        configurations.items()
    ):

        item = prepared[reference]

        score = score_values(
            item["validation_z"],
            mode,
        )

        threshold = float(
            np.percentile(score, 95)
        )

        require(
            np.isfinite(threshold)
            and threshold > 0,
            "Invalid threshold",
        )

        thresholds[method] = threshold

        frame = item["validation"][
            ["file", "condition", "rpm"]
        ].copy()

        frame["method"] = method
        frame["score"] = score
        frame["threshold"] = threshold

        validation_results.append(frame)

        print(method, threshold)

    # 모든 정상 기준과 임계값 확정 후 시험

    print("\n===== Test feature extraction =====")

    test = partitions[reference_names[0]][2]

    test_features = extract_files(
        test, args.rpm
    )

    require(
        test_features["file"].is_unique
        and set(test_features["file"]) == common_files,
        "Test feature file mismatch",
    )

    results = []
    predictions = []
    groups = []
    feature_diagnostics = []

    for method, (reference, mode) in (
        configurations.items()
    ):

        item = prepared[reference]

        z = deviations(
            test_features,
            item["center"],
            item["scale"],
        )

        frame = test_features[
            ["file", "condition", "rpm"]
        ].copy()

        frame["date"] = args.date
        frame["method"] = method
        frame["score"] = score_values(z, mode)
        frame["threshold"] = thresholds[method]

        frame["predicted_anomaly"] = (
            frame["score"] > frame["threshold"]
        ).astype(int)

        frame["evaluation_status"] = (
            "exploratory_only"
        )

        frame["dominant_feature"] = [
            FEATURE_NAMES[i]
            for i in z.argmax(axis=1)
        ]

        for i, name in enumerate(FEATURE_NAMES):
            frame[f"deviation_{name}"] = z[:, i]

        results.append({
            "date": args.date,
            "rpm": args.rpm,
            **metrics(frame, method),
        })

        predictions.append(frame)

        for condition, group in frame.groupby(
            "condition"
        ):

            groups.append({
                "method": method,
                "condition": condition,
                "files": len(group),
                "metric": (
                    "FPR"
                    if condition in (
                        "Normal1", "Normal2"
                    )
                    else "Recall"
                ),
                "rate_pct": (
                    group["predicted_anomaly"]
                    .mean() * 100
                ),
            })

            feature_diagnostics.append({
                "method": method,
                "condition": condition,
                **{
                    name: group[
                        f"deviation_{name}"
                    ].median()
                    for name in FEATURE_NAMES
                },
            })

    # 기존 AE와 동일 파일 비교

    saved = pd.read_csv(
        source / "test_scores.csv"
    )

    expected = test_features.set_index("file")

    for method in reference_names:

        frame = saved[
            saved["method"] == method
        ].copy()

        require(
            frame["file"].is_unique
            and set(frame["file"]) == common_files,
            "AE comparison files mismatch",
        )

        current = frame.set_index("file")

        require(
            np.array_equal(
                current.loc[
                    expected.index, "condition"
                ].to_numpy(),
                expected["condition"].to_numpy(),
            ),
            "AE condition mismatch",
        )

        results.append({
            "date": args.date,
            "rpm": args.rpm,
            **metrics(frame, method),
        })

        for condition, group in frame.groupby(
            "condition"
        ):

            groups.append({
                "method": method,
                "condition": condition,
                "files": len(group),
                "metric": (
                    "FPR"
                    if condition in (
                        "Normal1", "Normal2"
                    )
                    else "Recall"
                ),
                "rate_pct": (
                    group["predicted_anomaly"]
                    .mean() * 100
                ),
            })

    require(
        all(
            sha256(p) == hashes[str(p)]
            for p in source_paths
        ),
        "Original sources changed",
    )

    result_df = pd.DataFrame(results)
    group_df = pd.DataFrame(groups)
    diagnostic_df = pd.DataFrame(
        feature_diagnostics
    )

    out.mkdir(parents=True, exist_ok=False)

    result_df.to_csv(
        out / "metrics_comparison.csv",
        index=False,
    )

    group_df.to_csv(
        out / "condition_metrics.csv",
        index=False,
    )

    diagnostic_df.to_csv(
        out / "feature_diagnostics.csv",
        index=False,
    )

    test_features.to_csv(
        out / "test_features.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        out / "test_scores.csv",
        index=False,
    )

    pd.concat(
        validation_results,
        ignore_index=True,
    ).to_csv(
        out / "validation_scores.csv",
        index=False,
    )

    for reference, item in prepared.items():

        np.savez_compressed(
            out / f"{reference}_reference.npz",
            center=item["center"],
            scale=item["scale"],
            feature_names=np.array(FEATURE_NAMES),
        )

    metadata = {
        "date": args.date,
        "rpm": args.rpm,
        "feature_names": list(FEATURE_NAMES),
        "feature_transform": "log",
        "file_aggregation": "median_of_8_segments",
        "normal_reference": "median_mad",
        "feature_floor": FEATURE_FLOOR,
        "scale_floor": SCALE_FLOOR,
        "thresholds": thresholds,
        "threshold_source": "normal_validation_q95",
        "common_test_files": len(common_files),
        "source_sha256": hashes,
        "evaluation_status": "exploratory_only",
        "test_previously_examined": True,
        "run_independence_verified": False,
        "approved_for_production": False,
    }

    (out / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n===== Overall comparison =====")

    print(
        result_df.round(6).to_string(index=False)
    )

    print("\n===== Condition comparison (%) =====")

    print(
        group_df.pivot(
            index=["condition", "files", "metric"],
            columns="method",
            values="rate_pct",
        ).round(2).to_string()
    )

    print("\n===== Physical feature deviations =====")

    print(
        diagnostic_df[
            diagnostic_df["method"].isin([
                "P2_normal1_max",
                "P4_mixed_max",
            ])
        ].round(3).to_string(index=False)
    )

    print("\n===== Final verification =====")
    print("Common test files:", len(common_files))
    print("Fault training: NONE")
    print("Normal reference: Train only")
    print("Threshold: Validation Q95 only")
    print("AE retraining: NONE")
    print("Original source hashes: PASS")
    print("Production model changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", out)
    print("Physical feature experiment: PASS")


if __name__ == "__main__":
    main()
