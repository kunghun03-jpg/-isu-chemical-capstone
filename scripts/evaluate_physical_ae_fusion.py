"""AE와 물리적 특징 점수의 실험용 결합."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_normal1_dataset_ae import (
    require,
    metrics,
    sha256,
)

AE_ROOT = Path(
    "outputs/experimental_normal1_training"
)

P_ROOT = Path(
    "outputs/experimental_physical_features"
)

OUT = Path(
    "outputs/experimental_physical_ae_fusion"
)

AE_METHOD = "C_mixed_per_frequency"
P_METHOD = "P3_mixed_mean"

NORMAL = ("Normal1", "Normal2")

METHODS = {
    "F1_mean": lambda a, p: (a + p) / 2,
    "F2_max": np.maximum,
    "F3_min": np.minimum,
}


def align(ae, physical):

    ae = ae.set_index("file")
    physical = physical.set_index("file")

    require(
        ae.index.is_unique
        and physical.index.is_unique
        and set(ae.index) == set(physical.index),
        "File alignment mismatch",
    )

    physical = physical.loc[ae.index]

    require(
        np.array_equal(
            ae["condition"],
            physical["condition"],
        ),
        "Condition alignment mismatch",
    )

    for frame in (ae, physical):

        score = frame["score"].to_numpy(
            dtype=np.float64
        )

        require(
            np.isfinite(score).all()
            and (score >= 0).all(),
            "Invalid score",
        )

    return ae, physical


def threshold_from_test(frame):

    values = frame["threshold"].unique()

    require(
        len(values) == 1
        and np.isfinite(values[0])
        and values[0] > 0,
        "Invalid source threshold",
    )

    return float(values[0])


def evaluate(date, rpm):

    ae_dir = (
        AE_ROOT / date / f"{rpm}rpm"
    )

    p_dir = (
        P_ROOT / date / f"{rpm}rpm"
    )

    source_paths = [
        ae_dir / "split.csv",
        ae_dir / AE_METHOD
            / "validation_scores.csv",
        ae_dir / "test_scores.csv",
        p_dir / "validation_scores.csv",
        p_dir / "test_scores.csv",
        ae_dir / "experiment_metadata.json",
        p_dir / "experiment_metadata.json",
    ]

    hashes = {
        str(p): sha256(p)
        for p in source_paths
    }

    split = pd.read_csv(
        source_paths[0]
    )

    split = split[
        split["method"] == AE_METHOD
    ]

    require(
        split["file"].is_unique,
        "Duplicate split files",
    )

    expected_val = split[
        split["split"] == "validation"
    ]

    expected_test = split[
        split["split"] == "test"
    ]

    expected_train = split[
        split["split"] == "train"
    ]

    require(
        len(expected_train) == 20
        and len(expected_val) == 20
        and expected_train["condition"]
            .isin(NORMAL).all()
        and expected_val["condition"]
            .isin(NORMAL).all(),
        "Invalid development split",
    )

    require(
        not set(expected_train["file"])
            & set(expected_val["file"])
        and not set(expected_train["file"])
            & set(expected_test["file"])
        and not set(expected_val["file"])
            & set(expected_test["file"]),
        "Development/test overlap",
    )

    # Validation 먼저 처리

    ae_val = pd.read_csv(
        source_paths[1]
    )

    p_val = pd.read_csv(
        source_paths[3]
    )

    p_val = p_val[
        p_val["method"] == P_METHOD
    ]

    ae_val, p_val = align(
        ae_val, p_val
    )

    require(
        len(ae_val) == 20
        and set(ae_val.index)
            == set(expected_val["file"])
        and ae_val["condition"]
            .isin(NORMAL).all(),
        "Validation files mismatch",
    )

    val_conditions = (
        expected_val.set_index("file")
        .loc[ae_val.index, "condition"]
    )

    require(
        np.array_equal(
            val_conditions,
            ae_val["condition"],
        ),
        "Validation condition mismatch",
    )

    ae_meta = json.loads(
        source_paths[5].read_text(
            encoding="utf-8"
        )
    )

    p_meta = json.loads(
        source_paths[6].read_text(
            encoding="utf-8"
        )
    )

    ae_threshold = float(
        ae_meta["thresholds"][AE_METHOD]
    )

    p_threshold = float(
        p_meta["thresholds"][P_METHOD]
    )

    for frame, threshold in (
        (ae_val, ae_threshold),
        (p_val, p_threshold),
    ):

        require(
            np.isfinite(threshold)
            and threshold > 0
            and np.isclose(
                np.percentile(
                    frame["score"], 95
                ),
                threshold,
                rtol=1e-6,
                atol=1e-10,
            ),
            "Source validation Q95 mismatch",
        )

    a = (
        ae_val["score"].to_numpy()
        / ae_threshold
    )

    p = (
        p_val["score"].to_numpy()
        / p_threshold
    )

    thresholds = {}
    validation_rows = []

    # 결합 임계값을 Test 평가 전에 확정

    for name, combine in METHODS.items():

        scores = combine(a, p)

        threshold = float(
            np.percentile(scores, 95)
        )

        require(
            np.isfinite(threshold)
            and threshold > 0,
            "Invalid fusion threshold",
        )

        thresholds[name] = threshold

        frame = ae_val[
            ["condition"]
        ].reset_index()

        frame["method"] = name
        frame["score"] = scores
        frame["threshold"] = threshold

        validation_rows.append(frame)

    print(
        f"\n===== {date} / {rpm} RPM ====="
    )

    print("Frozen thresholds:", thresholds)

    # 임계값 확정 후 Test 읽기

    ae_test = pd.read_csv(
        source_paths[2]
    )

    ae_test = ae_test[
        ae_test["method"] == AE_METHOD
    ]

    p_test = pd.read_csv(
        source_paths[4]
    )

    p_test = p_test[
        p_test["method"] == P_METHOD
    ]

    ae_test, p_test = align(
        ae_test, p_test
    )

    require(
        set(ae_test.index)
        == set(expected_test["file"]),
        "Test files mismatch",
    )

    test_conditions = (
        expected_test.set_index("file")
        .loc[ae_test.index, "condition"]
    )

    require(
        np.array_equal(
            test_conditions,
            ae_test["condition"],
        ),
        "Test condition mismatch",
    )

    for frame, threshold in (
        (ae_test, ae_threshold),
        (p_test, p_threshold),
    ):

        require(
            np.isclose(
                threshold_from_test(frame),
                threshold,
                rtol=1e-6,
                atol=1e-10,
            ),
            "Test/source threshold mismatch",
        )

        require(
            np.array_equal(
                (frame["score"] > threshold)
                    .astype(int),
                frame["predicted_anomaly"],
            ),
            "Original prediction mismatch",
        )

    a = (
        ae_test["score"].to_numpy()
        / ae_threshold
    )

    p = (
        p_test["score"].to_numpy()
        / p_threshold
    )

    predictions = [
        ae_test.reset_index(),
        p_test.reset_index(),
    ]

    for name, combine in METHODS.items():

        frame = ae_test[
            ["condition"]
        ].reset_index()

        frame["date"] = date
        frame["rpm"] = rpm
        frame["method"] = name
        frame["score"] = combine(a, p)
        frame["threshold"] = thresholds[name]

        frame["predicted_anomaly"] = (
            frame["score"] > frame["threshold"]
        ).astype(int)

        frame["evaluation_status"] = (
            "exploratory_only"
        )

        frame["ae_ratio"] = a
        frame["physical_ratio"] = p

        predictions.append(frame)

    results = []

    for frame in predictions:

        method = frame["method"].iloc[0]

        results.append({
            "date": date,
            "rpm": rpm,
            **metrics(frame, method),
        })

    require(
        all(
            sha256(path) == hashes[str(path)]
            for path in source_paths
        ),
        "Original source changed",
    )

    print("Common validation files: 20")
    print("Common test files:", len(ae_test))
    print("Source hashes: PASS")

    return (
        pd.DataFrame(results),
        pd.concat(
            predictions, ignore_index=True
        ),
        pd.concat(
            validation_rows, ignore_index=True
        ).assign(date=date, rpm=rpm),
        {
            "date": date,
            "rpm": rpm,
            "thresholds": thresholds,
            "source_sha256": hashes,
        },
    )


def main():

    require(
        not OUT.exists(),
        f"Output already exists: {OUT}",
    )

    results = []
    predictions = []
    validations = []
    records = []

    for date in (
        "20260912",
        "20260916",
        "20260919",
    ):

        for rpm in (900, 1200, 1800):

            m, p, v, record = evaluate(
                date, rpm
            )

            results.append(m)
            predictions.append(p)
            validations.append(v)
            records.append(record)

    result = pd.concat(
        results, ignore_index=True
    )

    pred = pd.concat(
        predictions, ignore_index=True
    )

    val = pd.concat(
        validations, ignore_index=True
    )

    require(
        len(result) == 45
        and len(pred) == 3932 * 5
        and len(val) == 9 * 20 * 3,
        "Unexpected result size",
    )

    overall = result.groupby(
        "method"
    ).agg(
        files=("files", "sum"),
        TP=("TP", "sum"),
        TN=("TN", "sum"),
        FP=("FP", "sum"),
        FN=("FN", "sum"),
        AUROC=("AUROC", "mean"),
        AUPRC=("AUPRC_AP", "mean"),
    )

    overall["FPR_pct"] = (
        overall["FP"]
        / (overall["FP"] + overall["TN"])
        * 100
    )

    overall["Recall_pct"] = (
        overall["TP"]
        / (overall["TP"] + overall["FN"])
        * 100
    )

    overall["BA_pct"] = (
        100 - overall["FPR_pct"]
        + overall["Recall_pct"]
    ) / 2

    require(
        (overall["files"] == 3932).all()
        and (
            overall["TP"] + overall["FN"]
            == 3564
        ).all()
        and (
            overall["TN"] + overall["FP"]
            == 368
        ).all(),
        "Unexpected evaluation cohort",
    )

    conditions = (
        pred.groupby(
            ["condition", "method"]
        )["predicted_anomaly"]
        .mean()
        .unstack()
        * 100
    )

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    result.to_csv(
        OUT / "dataset_metrics.csv",
        index=False,
    )

    overall.to_csv(
        OUT / "overall_metrics.csv"
    )

    conditions.to_csv(
        OUT / "condition_metrics.csv"
    )

    pred.to_csv(
        OUT / "test_scores.csv",
        index=False,
    )

    val.to_csv(
        OUT / "validation_scores.csv",
        index=False,
    )

    metadata = {
        "fusion_methods": list(METHODS),
        "threshold_source":
            "common_normal_validation_q95",
        "weights": [0.5, 0.5],
        "datasets": records,
        "evaluation_status":
            "exploratory_only",
        "test_previously_examined": True,
        "run_independence_verified": False,
        "approved_for_production": False,
    }

    (OUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n===== Overall comparison =====")
    print(overall.round(6).to_string())

    print("\n===== Condition comparison (%) =====")
    print(conditions.round(2).to_string())

    print("\n===== Dataset comparison =====")
    print(
        result[
            [
                "date", "rpm", "method",
                "FPR", "Recall",
                "Balanced_Accuracy", "AUROC",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n===== Final verification =====")
    print("Datasets: 9/9")
    print("Common test files: 3932")
    print("Common validation alignment: PASS")
    print("Fusion threshold: Validation Q95")
    print("Fault training: NONE")
    print("Model retraining: NONE")
    print("Production changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", OUT)
    print("Fusion experiment: PASS")


if __name__ == "__main__":
    main()
