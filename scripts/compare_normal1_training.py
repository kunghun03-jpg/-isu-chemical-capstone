"""정상 학습 기준과 정규화 방식의 비교 실험."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat

from vibration.preprocessing import (
    ch3_log_spectra,
    apply_normalization,
)

from scripts.evaluate_normal1_dataset_ae import (
    require,
    sha256,
    train_ae,
    ae_scores,
    metrics,
)


ROOT = Path("outputs/experimental_normal1_dataset_ae")
OUTPUT = Path("outputs/experimental_normal1_training")


def load_spectra(frame):

    arrays = []
    metadata = []

    for row in frame.itertuples(index=False):

        raw = loadmat(row.file)

        spectra = ch3_log_spectra(
            raw["signals"],
            raw["fs"],
        )

        require(
            spectra.shape == (8, 1, 3201)
            and np.isfinite(spectra).all()
            and (spectra >= 0).all(),
            f"스펙트럼 오류: {row.file}",
        )

        arrays.append(spectra)

        for segment_id in range(8):

            metadata.append({
                "file": row.file,
                "condition": row.condition,
                "rpm": int(row.rpm),
                "segment_id": segment_id,
            })

    require(bool(arrays), "빈 데이터 분할")

    return (
        np.concatenate(arrays, axis=0),
        pd.DataFrame(metadata),
    )


def fit_normalization(X, mode):

    values = X.astype(np.float64)

    if mode == "per_frequency":

        mean = values.mean(
            axis=0, keepdims=True
        )

        std = values.std(
            axis=0, keepdims=True
        )

    elif mode == "global_train":

        mean = np.full(
            (1, 1, 3201),
            values.mean(),
            dtype=np.float64,
        )

        std = np.full(
            (1, 1, 3201),
            values.std(),
            dtype=np.float64,
        )

    else:
        raise ValueError(mode)

    std = np.maximum(std, 1e-6)

    return mean, std


def chronological(frame):

    return frame.sort_values(
        "file",
        key=lambda col: col.map(
            lambda p: Path(p).name
        ),
    ).reset_index(drop=True)


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

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    folder = (
        ROOT / args.date / f"{args.rpm}rpm"
    )

    out = (
        OUTPUT / args.date / f"{args.rpm}rpm"
    )

    require(
        not out.exists(),
        f"기존 실험 결과 존재: {out}",
    )

    source_paths = [
        folder / name
        for name in (
            "split.csv",
            "normalization.npz",
            "autoencoder.pt",
            "validation_scores.csv",
            "test_scores.csv",
        )
    ]

    source_hashes = {
        str(path): sha256(path)
        for path in source_paths
    }

    split = pd.read_csv(
        folder / "split.csv",
        dtype={"date": str},
    )

    require(
        split["file"].is_unique
        and split["date"].eq(args.date).all()
        and split["rpm"].eq(args.rpm).all(),
        "기존 데이터 분할 오류",
    )

    n1_train = chronological(
        split[split["split"] == "train"]
    )

    n1_val = chronological(
        split[split["split"] == "validation"]
    )

    old_test = split[
        split["split"].isin([
            "test",
            "test_normal",
            "test_anomaly",
        ])
    ].copy()

    require(
        len(n1_train) == 20
        and len(n1_val) == 20
        and n1_train["condition"].eq("Normal1").all()
        and n1_val["condition"].eq("Normal1").all(),
        "기존 Normal1 분할 오류",
    )

    require(
        len(n1_train) + len(n1_val) + len(old_test)
        == len(split),
        "분할 누락",
    )

    n2 = chronological(
        old_test[
            old_test["condition"] == "Normal2"
        ]
    )

    require(
        len(n2) > 20,
        "Normal2 데이터 부족",
    )

    require(
        len({Path(p).name for p in n2["file"]})
        == len(n2),
        "Normal2 파일명 중복",
    )

    n2_train = n2.iloc[:10].copy()
    n2_val = n2.iloc[10:20].copy()

    reserved = set(n2_train["file"]) | set(n2_val["file"])

    test = old_test[
        ~old_test["file"].isin(reserved)
    ].copy()

    mixed_train = pd.concat(
        [n1_train.iloc[:10], n2_train],
        ignore_index=True,
    )

    mixed_val = pd.concat(
        [n1_val.iloc[:10], n2_val],
        ignore_index=True,
    )

    experiments = {
        "A_normal1_per_frequency": (
            n1_train,
            n1_val,
            "per_frequency",
        ),

        "B_normal1_global": (
            n1_train,
            n1_val,
            "global_train",
        ),

        "C_mixed_per_frequency": (
            mixed_train,
            mixed_val,
            "per_frequency",
        ),

        "D_mixed_global": (
            mixed_train,
            mixed_val,
            "global_train",
        ),
    }

    test_files = set(test["file"])

    print("\n===== Common test files =====")

    print(
        test.groupby("condition")
        .size().to_string()
    )

    print("Total:", len(test))

    # 모든 실험의 분할을 학습 전에 검증

    for method, (train, val, mode) in (
        experiments.items()
    ):

        train_files = set(train["file"])
        val_files = set(val["file"])

        require(
            len(train_files) == 20
            and len(val_files) == 20
            and not train_files & val_files
            and not train_files & test_files
            and not val_files & test_files,
            f"파일 분할 오류: {method}",
        )

        require(
            train["condition"].isin(
                ["Normal1", "Normal2"]
            ).all()
            and val["condition"].isin(
                ["Normal1", "Normal2"]
            ).all(),
            "학습·검증 결함 데이터 유입",
        )

    print("\n===== Load development spectra =====")

    datasets = {
        "normal1": (
            load_spectra(n1_train),
            load_spectra(n1_val),
        ),
        "mixed": (
            load_spectra(mixed_train),
            load_spectra(mixed_val),
        ),
    }

    prepared = {}

    # 모든 모델의 학습 및 임계값 확정을
    # 시험 데이터 평가보다 먼저 수행

    for method, (train, val, mode) in (
        experiments.items()
    ):

        print(f"\n{'=' * 65}")
        print(method)
        print("=" * 65)

        print("Train conditions:")
        print(train["condition"].value_counts())

        print("Validation conditions:")
        print(val["condition"].value_counts())

        key = (
            "mixed"
            if method.startswith(("C_", "D_"))
            else "normal1"
        )

        (
            (train_raw, train_meta),
            (val_raw, val_meta),
        ) = datasets[key]

        mean, std = fit_normalization(
            train_raw, mode
        )

        train_X = apply_normalization(
            train_raw, mean, std
        )

        val_X = apply_normalization(
            val_raw, mean, std
        )

        model, history, epoch, loss = train_ae(
            train_X, val_X
        )

        validation_scores = ae_scores(
            model, val_X, val_meta
        )

        threshold = float(
            np.percentile(
                validation_scores["score"], 95
            )
        )

        require(
            np.isfinite(threshold)
            and threshold > 0,
            "임계값 오류",
        )

        require(
            set(validation_scores["file"])
            == set(val["file"]),
            "검증 파일 불일치",
        )

        print("Best epoch:", epoch)
        print("Frozen threshold:", threshold)

        prepared[method] = {
            "model": model,
            "mean": mean,
            "std": std,
            "history": history,
            "epoch": epoch,
            "loss": loss,
            "threshold": threshold,
            "validation_scores": validation_scores,
            "train": train,
            "val": val,
            "normalization": mode,
        }

    print("\n===== All thresholds frozen =====")

    # 공통 시험 데이터는 모든 학습 종료 후 평가

    test_raw, test_meta = load_spectra(test)

    results = []
    predictions = []
    groups = []
    split_rows = []

    for method, item in prepared.items():

        test_X = apply_normalization(
            test_raw,
            item["mean"],
            item["std"],
        )

        frame = ae_scores(
            item["model"],
            test_X,
            test_meta,
        )

        require(
            frame["file"].is_unique
            and set(frame["file"]) == test_files,
            "시험 파일 불일치",
        )

        frame["threshold"] = item["threshold"]

        frame["predicted_anomaly"] = (
            frame["score"] > frame["threshold"]
        ).astype(int)

        frame["method"] = method
        frame["date"] = args.date

        frame["evaluation_status"] = (
            "exploratory_only"
        )

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

        for name, rows in (
            ("train", item["train"]),
            ("validation", item["val"]),
            ("test", test),
        ):

            selected = rows.copy()

            selected["split"] = name
            selected["method"] = method

            split_rows.append(selected)

    # 기존 AE와 재학습 A의 점수 재현성 확인

    saved = pd.read_csv(
        folder / "test_scores.csv"
    )

    saved = saved[
        saved["method"] == "C_normal1_ae"
    ].set_index("file")

    current = predictions[0].set_index("file")

    require(
        saved.index.is_unique
        and set(current.index) <= set(saved.index),
        "기존 AE 비교 파일 오류",
    )

    baseline = saved.loc[current.index]

    parity = bool(
        np.allclose(
            baseline["score"],
            current["score"],
            rtol=1e-5,
            atol=1e-8,
        )
        and np.allclose(
            baseline["threshold"],
            current["threshold"],
            rtol=1e-5,
            atol=1e-8,
        )
        and np.array_equal(
            baseline["predicted_anomaly"],
            current["predicted_anomaly"],
        )
    )

    print("\n===== Baseline reproduction =====")

    print(
        "Original AE parity:",
        "PASS" if parity
        else "NOT MATCHED",
    )

    # 원래 저장된 AE도 공통 시험 파일에서 별도 비교

    original = baseline.reset_index().copy()

    original["method"] = "A_saved_reference"

    results.insert(0, {
        "date": args.date,
        "rpm": args.rpm,
        **metrics(
            original,
            "A_saved_reference",
        ),
    })

    predictions.insert(0, original)

    for condition, group in original.groupby(
        "condition"
    ):

        groups.append({
            "method": "A_saved_reference",
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
            sha256(path) == source_hashes[str(path)]
            for path in source_paths
        ),
        "기존 산출물 변경 감지",
    )

    out.mkdir(parents=True, exist_ok=False)

    for method, item in prepared.items():

        target = out / method
        target.mkdir()

        np.savez_compressed(
            target / "normalization.npz",
            mean=item["mean"],
            std=item["std"],
        )

        torch.save({
            "model_state_dict":
                item["model"].state_dict(),
            "best_epoch": item["epoch"],
            "validation_loss": item["loss"],
            "threshold": item["threshold"],
            "normalization":
                item["normalization"],
            "input_shape": [1, 3201],
            "evaluation_status":
                "exploratory_only",
            "approved_for_production": False,
        }, target / "autoencoder.pt")

        pd.DataFrame(
            item["history"]
        ).to_csv(
            target / "training_history.csv",
            index=False,
        )

        item["validation_scores"].to_csv(
            target / "validation_scores.csv",
            index=False,
        )

    result_df = pd.DataFrame(results)
    group_df = pd.DataFrame(groups)

    result_df.to_csv(
        out / "metrics_comparison.csv",
        index=False,
    )

    group_df.to_csv(
        out / "condition_metrics.csv",
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
        split_rows,
        ignore_index=True,
    ).to_csv(
        out / "split.csv",
        index=False,
    )

    metadata = {
        "date": args.date,
        "rpm": args.rpm,
        "train_files_per_model": 20,
        "validation_files_per_model": 20,
        "common_test_files": len(test_files),
        "seed": 42,
        "max_epochs": 30,
        "patience": 8,
        "batch_size": 32,
        "learning_rate": 0.001,
        "threshold_method": "Q95",
        "score_method": "normalized_mse",
        "baseline_reproduction": parity,
        "thresholds": {
            name: item["threshold"]
            for name, item in prepared.items()
        },
        "source_sha256": source_hashes,
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
        result_df.round(6).to_string(
            index=False
        )
    )

    print("\n===== Condition comparison (%) =====")

    print(
        group_df.pivot(
            index=["condition", "files", "metric"],
            columns="method",
            values="rate_pct",
        ).round(2).to_string()
    )

    print("\n===== Final verification =====")
    print("Common test files:", len(test_files))
    print("Training files per model: 20")
    print("Validation files per model: 20")
    print("Train / validation / test overlap: 0")
    print("Fault training: NONE")
    print("Original source hashes: PASS")
    print("Production model changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", out)
    print("Training comparison: PASS")


if __name__ == "__main__":
    main()
