"""고정 Normal1 AE의 이상 점수 계산 방식 비교."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat
from vibration.model import Autoencoder
from vibration.preprocessing import (
    ch3_log_spectra,
    apply_normalization,
    frequency_bins,
)

from scripts.evaluate_normal1_dataset_ae import (
    require,
    metrics,
    sha256,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        choices=["20260912", "20260916", "20260919"],
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
        Path("outputs/experimental_normal1_dataset_ae")
        / args.date / f"{args.rpm}rpm"
    )

    out = (
        Path("outputs/experimental_normal1_scoring")
        / args.date / f"{args.rpm}rpm"
    )

    require(
        not out.exists(),
        f"기존 실험 결과 존재: {out}",
    )

    inputs = [
        folder / name
        for name in (
            "normalization.npz",
            "autoencoder.pt",
            "split.csv",
            "validation_scores.csv",
            "test_scores.csv",
        )
    ]

    hashes = {
        str(p): sha256(p)
        for p in inputs
    }

    with np.load(
        folder / "normalization.npz",
        allow_pickle=False,
    ) as data:

        mean = data["mean"]
        std = data["std"]

    checkpoint = torch.load(
        folder / "autoencoder.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = Autoencoder()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    split = pd.read_csv(
        folder / "split.csv",
        dtype={"date": str},
    )

    require(
        split["file"].is_unique,
        "중복 파일",
    )

    require(
        split["date"].eq(args.date).all()
        and split["rpm"].eq(args.rpm).all(),
        "데이터셋 불일치",
    )

    train = split[split["split"] == "train"]

    val = split[split["split"] == "validation"]

    test = split[
        split["split"].isin([
            "test",
            "test_normal",
            "test_anomaly",
        ])
    ]

    require(
        len(train) == 20
        and len(val) == 20
        and train["condition"].eq("Normal1").all()
        and val["condition"].eq("Normal1").all(),
        "정상 학습·검증 조건 오류",
    )

    require(
        len(train) + len(val) + len(test)
        == len(split),
        "분할 구성 오류",
    )

    freq = frequency_bins()

    masks = [
        freq < 100,
        (freq >= 100) & (freq < 1000),
        freq >= 1000,
    ]

    require(
        np.all(np.sum(masks, axis=0) == 1),
        "주파수 대역 구성 오류",
    )

    def calculate(rows, name):

        result = []

        print(f"\n===== {name} =====")

        for index, row in enumerate(
            rows.itertuples(index=False), 1
        ):

            raw = loadmat(row.file)

            spectra = ch3_log_spectra(
                raw["signals"],
                raw["fs"],
            )

            X = apply_normalization(
                spectra, mean, std
            )

            with torch.no_grad():

                restored = model(
                    torch.from_numpy(X).float()
                ).numpy()

            residual = (
                restored.astype(np.float64)
                - X.astype(np.float64)
            )

            error = residual ** 2

            item = {
                "file": row.file,
                "condition": row.condition,
                "date": args.date,
                "rpm": args.rpm,

                "A_original_mse": float(
                    error.mean()
                ),

                "B_log_spectrum_mse": float(
                    ((residual * std) ** 2).mean()
                ),
            }

            for i, mask in enumerate(masks):

                item[f"band_{i}"] = float(
                    error[:, :, mask].mean()
                )

            result.append(item)

            if index % 100 == 0:

                print(
                    f"{index}/{len(rows)}",
                    flush=True,
                )

        frame = pd.DataFrame(result)

        cols = [
            "A_original_mse",
            "B_log_spectrum_mse",
            "band_0",
            "band_1",
            "band_2",
        ]

        require(
            np.isfinite(
                frame[cols].to_numpy()
            ).all(),
            "비유한 점수",
        )

        return frame

    # Normal1 Train 기준 대역별 보정

    train_scores = calculate(train, "Train")

    band_cols = [
        "band_0",
        "band_1",
        "band_2",
    ]

    band_scale = (
        train_scores[band_cols]
        .median()
        .to_numpy()
    )

    require(
        np.isfinite(band_scale).all()
        and (band_scale > 0).all(),
        "대역별 보정 통계 오류",
    )

    def add_band_score(frame):

        frame["C_bandwise_mse"] = (
            frame[band_cols].to_numpy()
            / band_scale
        ).max(axis=1)

        return frame

    val_scores = add_band_score(
        calculate(val, "Validation")
    )

    methods = [
        "A_original_mse",
        "B_log_spectrum_mse",
        "C_bandwise_mse",
    ]

    # Normal1 Validation만 사용하여 Q95 결정

    thresholds = {
        name: float(
            np.percentile(val_scores[name], 95)
        )
        for name in methods
    }

    require(
        all(
            np.isfinite(v) and v > 0
            for v in thresholds.values()
        ),
        "임계값 오류",
    )

    require(
        np.isclose(
            thresholds["A_original_mse"],
            float(checkpoint["threshold"]),
            rtol=1e-5,
            atol=1e-8,
        ),
        "기존 Q95 불일치",
    )

    print("\n===== Frozen thresholds =====")

    for name, threshold in thresholds.items():

        print(name, threshold)

    # 임계값 확정 후 Test 평가

    test_scores = add_band_score(
        calculate(test, "Test")
    )

    # 기존 저장 점수와 일치 여부 검증

    for filename, calculated in (
        ("validation_scores.csv", val_scores),
        ("test_scores.csv", test_scores),
    ):

        saved = pd.read_csv(folder / filename)

        saved = saved[
            saved["method"] == "C_normal1_ae"
        ].set_index("file")

        current = calculated.set_index("file")

        require(
            saved.index.is_unique
            and current.index.is_unique
            and set(saved.index) == set(current.index),
            "기존 파일 목록 불일치",
        )

        saved = saved.loc[current.index]

        require(
            np.array_equal(
                saved["condition"],
                current["condition"],
            ),
            "기존 조건 불일치",
        )

        require(
            np.allclose(
                saved["score"],
                current["A_original_mse"],
                rtol=1e-5,
                atol=1e-8,
            ),
            "기존 파일 점수 불일치",
        )

        require(
            np.array_equal(
                saved["score"].to_numpy()
                > float(checkpoint["threshold"]),
                current["A_original_mse"].to_numpy()
                > thresholds["A_original_mse"],
            ),
            "기존 판정 결과 불일치",
        )

    results = []
    groups = []
    predictions = []

    for method in methods:

        frame = test_scores[
            ["file", "condition", "date", "rpm"]
        ].copy()

        frame["score"] = test_scores[method]
        frame["threshold"] = thresholds[method]

        frame["predicted_anomaly"] = (
            frame["score"] > frame["threshold"]
        ).astype(int)

        frame["method"] = method

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
                    if condition in ("Normal1", "Normal2")
                    else "Recall"
                ),
                "rate_pct": (
                    group["predicted_anomaly"].mean()
                    * 100
                ),
            })

    require(
        all(
            sha256(p) == hashes[str(p)]
            for p in inputs
        ),
        "기존 산출물 변경 감지",
    )

    result_df = pd.DataFrame(results)
    group_df = pd.DataFrame(groups)

    out.mkdir(parents=True, exist_ok=False)

    result_df.to_csv(
        out / "metrics_comparison.csv",
        index=False,
    )

    group_df.to_csv(
        out / "condition_metrics.csv",
        index=False,
    )

    val_scores.to_csv(
        out / "validation_scores.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        out / "test_scores.csv",
        index=False,
    )

    metadata = {
        "date": args.date,
        "rpm": args.rpm,
        "thresholds": thresholds,
        "band_scale": band_scale.tolist(),
        "band_scale_source": "Normal1_train",
        "bands_hz": [
            [0, 100],
            [100, 1000],
            [1000, 10000],
        ],
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

    print("\n===== Final verification =====")
    print("Original validation/test score parity: PASS")
    print("Band calibration: Normal1 Train only")
    print("Threshold calibration: Normal1 Validation only")
    print("Model retraining: NONE")
    print("Original source hashes: PASS")
    print("Production model changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", out)
    print("Scoring comparison: PASS")


if __name__ == "__main__":
    main()
