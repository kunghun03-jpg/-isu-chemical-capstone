"""Normal1 기반 날짜별 확대 실험. 운영 모델 변경 없음."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat

from vibration.dataset import (
    discover_files,
    validate_assignments,
    validate_file_metadata,
)

from vibration.preprocessing import (
    ch3_log_spectra,
    apply_normalization,
    frequency_bins,
)

from scripts.evaluate_normal1_dataset_ae import (
    require,
    sha256,
    train_ae,
    ae_scores,
    distance_scores,
    metrics,
)


ROOT = Path.cwd()
OUTPUT = ROOT / "outputs"

FAULTS = {
    "IR",
    "OR",
    "Roller",
    "Looseness0.05",
    "Looseness0.1",
    "LoosenessFull",
    "Unbalance_-1",
    "Unbalance_-2",
}

NORMALS = {"Normal1", "Normal2"}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        choices=["20260912", "20260916"],
    )

    parser.add_argument(
        "--rpm",
        required=True,
        type=int,
        choices=[900, 1200, 1800],
    )

    args = parser.parse_args()
    date = args.date
    rpm = args.rpm

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    source = (
        OUTPUT / f"{rpm}rpm"
        / "date_prelim_20261106"
    )

    out = (
        OUTPUT
        / "experimental_normal1_dataset_ae"
        / date / f"{rpm}rpm"
    )

    require(
        not out.exists(),
        f"기존 실험 결과 존재: {out}",
    )

    print(f"===== {date} / {rpm} RPM =====")

    # 1. 기존 품질 검사 결과 적용

    checked_path = (
        source / f"quality_checked_{rpm}rpm.csv"
    )

    excluded_path = (
        source / f"quality_check_{rpm}rpm.csv"
    )

    checked = set(
        pd.read_csv(checked_path)["file"]
    )

    excluded = set(
        pd.read_csv(excluded_path)["file"]
    )

    records = discover_files(
        ROOT, rpm, [date]
    )

    require(
        all(r["file"] in checked for r in records),
        "품질 미검사 파일 발견",
    )

    records = [
        dict(r) for r in records
        if r["file"] not in excluded
        and r["condition"] != "Normal3"
    ]

    require(
        {r["condition"] for r in records}
        == NORMALS | FAULTS,
        "정상·결함 조건 구성 불일치",
    )

    require(
        len({r["file"] for r in records})
        == len(records),
        "중복 파일 발견",
    )

    # 2. 시간순 Normal1 분할

    normal1 = sorted(
        [
            r["file"] for r in records
            if r["condition"] == "Normal1"
        ],
        key=lambda p: Path(p).name,
    )

    require(len(normal1) > 40, "Normal1 부족")

    require(
        len({Path(p).name for p in normal1})
        == len(normal1),
        "Normal1 파일명 중복",
    )

    train_files = set(normal1[:20])
    val_files = set(normal1[20:40])

    for row in records:

        validate_file_metadata(ROOT, row, rpm)

        file = row["file"]

        if file in train_files:
            row["split"] = "train"

        elif file in val_files:
            row["split"] = "validation"

        elif row["condition"] in NORMALS:
            row["split"] = "test_normal"

        else:
            row["split"] = "test_anomaly"

    validate_assignments(
        records,
        require_run=False,
    )

    split = pd.DataFrame(records)

    print("\n===== Split =====")

    print(
        pd.crosstab(
            split["condition"],
            split["split"],
        ).to_string()
    )

    # 3. 기존 정상 스펙트럼 재사용

    old_split = (
        "train" if date == "20260912"
        else "validation"
    )

    normal_path = (
        source / f"{old_split}_spectra.npz"
    )

    source_paths = [
        normal_path,
        checked_path,
        excluded_path,
    ]

    source_hashes = {
        str(p): sha256(p)
        for p in source_paths
    }

    normal_spectra = {}

    with np.load(
        normal_path, allow_pickle=False
    ) as data:

        require(
            np.array_equal(
                data["frequency"],
                frequency_bins(),
            ),
            "주파수 구성 불일치",
        )

        X = data["X"]
        files = data["source_file"].astype(str)
        segments = data["segment_id"]
        conditions = data["condition"].astype(str)
        dates = data["date"].astype(str)
        speeds = data["rpm"].astype(str)

        require(
            len(X) == len(files) == len(segments)
            == len(conditions) == len(dates)
            == len(speeds),
            "정상 스펙트럼 메타데이터 길이 오류",
        )

        expected = {
            r["file"]: r["condition"]
            for r in records
            if r["condition"] in NORMALS
        }

        require(
            set(files) == set(expected),
            "정상 스펙트럼 파일 목록 불일치",
        )

        require(
            np.all(dates == date)
            and np.all(speeds == str(rpm)),
            "날짜·RPM 불일치",
        )

        for file in sorted(expected):

            mask = files == file

            ids = segments[mask]

            require(
                len(ids) == 8
                and set(ids) == set(range(8)),
                f"구간 구성 오류: {file}",
            )

            require(
                np.all(
                    conditions[mask] == expected[file]
                ),
                f"조건 불일치: {file}",
            )

            normal_spectra[file] = (
                X[mask][np.argsort(ids)].copy()
            )

    # 4. 데이터 분할별 배열 구성

    def collect(partitions):

        arrays = []
        metadata = []

        selected = [
            r for r in records
            if r["split"] in partitions
        ]

        selected.sort(key=lambda r: r["file"])

        for index, row in enumerate(selected, 1):

            file = row["file"]

            if row["condition"] in NORMALS:

                values = normal_spectra[file]

            else:

                raw = loadmat(ROOT / file)

                require(
                    {"signals", "fs"} <= set(raw),
                    f"MAT 구조 확인 필요: {file}",
                )

                values = ch3_log_spectra(
                    raw["signals"],
                    raw["fs"],
                )

                if index % 50 == 0:
                    print(
                        f"Spectra: {index}/{len(selected)}",
                        flush=True,
                    )

            require(
                values.shape == (8, 1, 3201)
                and np.isfinite(values).all()
                and (values >= 0).all(),
                f"스펙트럼 오류: {file}",
            )

            arrays.append(values)

            for segment in range(8):

                metadata.append({
                    **row,
                    "rpm": rpm,
                    "segment_id": segment,
                })

        require(bool(arrays), "빈 데이터 분할")

        return (
            np.concatenate(arrays, axis=0),
            pd.DataFrame(metadata),
        )

    train_raw, train_meta = collect(["train"])
    val_raw, val_meta = collect(["validation"])

    require(
        len(train_raw) == 160
        and len(val_raw) == 160,
        "학습·검증 구간 수 오류",
    )

    require(
        train_meta["condition"].eq("Normal1").all()
        and val_meta["condition"].eq("Normal1").all(),
        "Normal1 이외 학습 데이터 유입",
    )

    # 5. Train-only 정규화

    train64 = train_raw.astype(np.float64)

    mean = train64.mean(
        axis=0, keepdims=True
    )

    std = np.maximum(
        train64.std(axis=0, keepdims=True),
        1e-6,
    )

    train_X = apply_normalization(
        train_raw, mean, std
    )

    val_X = apply_normalization(
        val_raw, mean, std
    )

    # 6. 기존 AE 학습 설정 재사용

    print("\n===== Normal1 AE training =====")

    model, history, epoch, loss = train_ae(
        train_X, val_X
    )

    reference = train_X.mean(
        axis=0,
        dtype=np.float64,
    )

    val_ae = ae_scores(
        model, val_X, val_meta
    )

    val_distance = distance_scores(
        val_X, val_meta, reference
    )

    ae_threshold = float(
        np.percentile(val_ae["score"], 95)
    )

    distance_threshold = float(
        np.percentile(val_distance["score"], 95)
    )

    print("\n===== Frozen thresholds =====")
    print("Normal1 AE:", ae_threshold)
    print("Normal distance:", distance_threshold)

    # 7. 임계값 확정 이후 결함 스펙트럼 생성

    print("\n===== Test spectra =====")

    test_raw, test_meta = collect(
        ["test_normal", "test_anomaly"]
    )

    test_files = set(test_meta["file"])

    require(
        not train_files & val_files
        and not test_files & train_files
        and not test_files & val_files,
        "학습·검증·시험 파일 중복",
    )

    test_X = apply_normalization(
        test_raw, mean, std
    )

    test_ae = ae_scores(
        model, test_X, test_meta
    )

    test_distance = distance_scores(
        test_X, test_meta, reference
    )

    experiments = {
        "C_normal1_ae": (
            val_ae,
            test_ae,
            ae_threshold,
        ),
        "D_normal_distance": (
            val_distance,
            test_distance,
            distance_threshold,
        ),
    }

    results = []
    validation_rows = []
    test_rows = []
    groups = []

    # 8. 공통 시험 파일 성능 비교

    for method, (v, t, threshold) in (
        experiments.items()
    ):

        v = v.copy()
        t = t.copy()

        require(
            v["file"].is_unique
            and set(v["file"]) == val_files
            and t["file"].is_unique
            and set(t["file"]) == test_files,
            "평가 파일 목록 불일치",
        )

        for frame in (v, t):

            frame["method"] = method
            frame["date"] = date
            frame["threshold"] = threshold

            frame["predicted_anomaly"] = (
                frame["score"] > threshold
            ).astype(int)

            frame["evaluation_status"] = (
                "exploratory_only"
            )

        validation_rows.append(v)
        test_rows.append(t)

        results.append({
            "date": date,
            "rpm": rpm,
            **metrics(t, method),
        })

        for condition, group in t.groupby(
            "condition"
        ):

            groups.append({
                "date": date,
                "rpm": rpm,
                "method": method,
                "condition": condition,
                "files": len(group),
                "metric": (
                    "FPR" if condition in NORMALS
                    else "Recall"
                ),
                "rate_pct": (
                    group["predicted_anomaly"].mean()
                    * 100
                ),
            })

    require(
        all(
            sha256(p) == source_hashes[str(p)]
            for p in source_paths
        ),
        "기존 산출물 변경 감지",
    )

    # 9. 별도 실험 결과 저장

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    split.to_csv(
        out / "split.csv",
        index=False,
    )

    np.savez_compressed(
        out / "normalization.npz",
        mean=mean,
        std=std,
    )

    np.savez_compressed(
        out / "normal_reference.npz",
        reference=reference,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": epoch,
        "validation_loss": loss,
        "threshold": ae_threshold,
        "input_shape": [1, 3201],
        "evaluation_status": "exploratory_only",
        "approved_for_production": False,
    }, out / "autoencoder.pt")

    pd.DataFrame(history).to_csv(
        out / "training_history.csv",
        index=False,
    )

    pd.concat(
        validation_rows,
        ignore_index=True,
    ).to_csv(
        out / "validation_scores.csv",
        index=False,
    )

    pd.concat(
        test_rows,
        ignore_index=True,
    ).to_csv(
        out / "test_scores.csv",
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

    metadata = {
        "date": date,
        "rpm": rpm,
        "training_condition": "Normal1",
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "normalization": "per_frequency",
        "std_floor": 1e-6,
        "threshold_method": "Q95",
        "thresholds": {
            "C_normal1_ae": ae_threshold,
            "D_normal_distance": distance_threshold,
        },
        "seed": 42,
        "max_epochs": 30,
        "patience": 8,
        "batch_size": 32,
        "learning_rate": 0.001,
        "source_sha256": source_hashes,
        "evaluation_status": "exploratory_only",
        "run_independence_verified": False,
        "approved_for_production": False,
        "original_ae_comparison": "excluded",
    }

    (out / "experiment_metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
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

    print("\n===== Final verification =====")
    print("Best epoch:", epoch)
    print("Best validation loss:", loss)
    print("Common evaluation files:", len(test_files))
    print("Train / validation / test overlap: 0")
    print("Training condition: Normal1 only")
    print("Original source hashes: PASS")
    print("Production model changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", out)
    print("Expanded experiment: PASS")


if __name__ == "__main__":
    main()
