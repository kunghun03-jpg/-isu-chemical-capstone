"""900 RPM 불균형 결함의 주파수별 AE 복원 오차 분석."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from vibration.model import Autoencoder
from vibration.inference import SavedModelInference
from vibration.preprocessing import ch3_log_spectra

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEST = OUT / "defect_detection_analysis"

RPM = 900
FS = 25600
FREQ_STEP = 3.125

torch.set_num_threads(1)

# 1. 대표 파일 선정
features = pd.read_csv(
    DEST / "unbalance_900_signal_features.csv"
)

comparison = pd.read_csv(
    DEST / "all_models_file_comparison.csv"
)

assert features["file"].is_unique
assert comparison["file"].is_unique

groups = [
    "Normal1",
    "Normal2",
    "Detected_By_Any",
    "Common_Missed",
]

selected = []

for group in groups:

    candidates = features[
        features["group"] == group
    ].copy()

    if len(candidates) < 5:
        raise ValueError(f"파일 부족: {group}")

    if group == "Common_Missed":

        assert len(candidates) == 5
        chosen = candidates

    else:

        median = candidates["one_x_amplitude"].median()

        candidates["distance"] = (
            candidates["one_x_amplitude"] - median
        ).abs()

        chosen = candidates.sort_values(
            ["distance", "file"]
        ).head(5)

    selected.append(chosen)

selected = pd.concat(
    selected, ignore_index=True
)

selected = selected.merge(
    comparison[
        [
            "file", "file_key", "AE",
            "score_abs", "threshold_abs",
            "Integrated_AE",
        ]
    ],
    on="file",
    validate="one_to_one",
)

assert len(selected) == 20

print("===== 분석 대상 선정 =====")
print(selected["group"].value_counts().to_string())


# 2. 기존 AE 모델 로딩
operating = SavedModelInference(
    OUT / "inference_bundles/date_prelim_20261106/900rpm",
    models=("autoencoder",),
    selected_artifacts_only=True,
)

operating_model = operating._estimators["autoencoder"]
operating_model.eval()

archive = pd.read_csv(
    OUT / "900rpm/date_prelim_20261106"
    / "evaluation_file_900rpm.csv"
)

archive = archive[
    archive["split"].isin([
        "test_normal", "test_anomaly"
    ])
].copy()


def file_key(value):

    value = str(value).replace("\\", "/")

    if "/data/" in value:
        return "data/" + value.split("/data/", 1)[1]

    if value.startswith("data/"):
        return value

    if value.startswith("./data/"):
        return value[2:]

    raise ValueError(f"경로 오류: {value}")


archive["file_key"] = archive["file"].map(file_key)

assert archive["file_key"].is_unique

archive = archive.set_index("file_key")


# 3. 통합 AE 모델 로딩
bundle = OUT / "experimental_integrated_ae/900rpm"

manifest = json.loads(
    (bundle / "manifest.json").read_text(
        encoding="utf-8"
    )
)

assert manifest["rpm"] == RPM
assert manifest["model"] == "integrated_autoencoder"
assert manifest["status"] == "experimental_only"
assert manifest["approved_for_production"] is False
assert manifest["threshold_source"] == "normal_validation_only"
assert manifest["normalization_source"] == "normal_train_only"

for name, expected_hash in manifest["artifacts_sha256"].items():

    actual_hash = hashlib.sha256(
        (bundle / name).read_bytes()
    ).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(f"파일 무결성 오류: {name}")

checkpoint = torch.load(
    bundle / "model.pt",
    map_location="cpu",
    weights_only=True,
)

integrated_model = Autoencoder()

integrated_model.load_state_dict(
    checkpoint["model_state_dict"]
)

integrated_model.eval()

with np.load(bundle / "normalization.npz") as stats:
    mean = stats["mean"]
    std = stats["std"]

assert mean.shape == (1, 1, 3201)
assert std.shape == (1, 1, 3201)
assert np.isfinite(mean).all()
assert np.isfinite(std).all()
assert (std > 0).all()

integrated_threshold = float(manifest["threshold"])

assert np.isfinite(integrated_threshold)
assert integrated_threshold > 0

print("\n===== 두 AE 모델 로딩 완료 =====")


# 4. 주파수 대역 정의
frequencies = np.arange(3201) * FREQ_STEP

bands = {
    "1X": np.abs(frequencies - 15) <= FREQ_STEP,
    "2X": np.abs(frequencies - 30) <= FREQ_STEP,
    "0_100Hz": frequencies < 100,
    "100_1000Hz": (
        (frequencies >= 100)
        & (frequencies < 1000)
    ),
    "1000_10000Hz": frequencies >= 1000,
}

print("\n===== 주파수 대역 =====")

for name, mask in bands.items():

    print(
        f"{name}: "
        f"{frequencies[mask].min():.3f}"
        f"~{frequencies[mask].max():.3f} Hz "
        f"({mask.sum()} bins)"
    )


def reconstruct(model, normalized):

    normalized = np.asarray(
        normalized, dtype=np.float32
    )

    assert normalized.shape == (8, 1, 3201)
    assert np.isfinite(normalized).all()

    tensor = torch.from_numpy(normalized)

    with torch.inference_mode():

        reconstructed = model(
            tensor
        ).cpu().numpy()

    assert reconstructed.shape == normalized.shape
    assert np.isfinite(reconstructed).all()

    inputs = normalized[:, 0, :].astype(np.float64)

    outputs = reconstructed[:, 0, :].astype(np.float64)

    error = (inputs - outputs) ** 2

    return inputs, outputs, error


# 5. 파일별 복원 분석
records = []
spectra = {}

print("\n===== 파일별 복원 오차 분석 =====")

for i, row in selected.iterrows():

    path = Path(row["file"])

    if not path.is_file():
        path = ROOT / row["file_key"]

    if not path.is_file():
        raise FileNotFoundError(path)

    archived = archive.loc[row["file_key"]]

    _, operating_x = operating.preprocess(
        path.read_bytes()
    )

    mat = loadmat(path)

    fs = float(np.asarray(mat["fs"]).squeeze())

    assert fs == FS

    raw = np.asarray(
        ch3_log_spectra(
            mat["signals"],
            mat["fs"],
        ),
        dtype=np.float32,
    )

    integrated_x = (
        (raw - mean) / std
    ).astype(np.float32)

    cases = [
        (
            "AE",
            operating_model,
            operating_x,
            float(archived["score"]),
            float(archived["threshold"]),
            int(row["AE"]),
        ),
        (
            "Integrated_AE",
            integrated_model,
            integrated_x,
            float(row["score_abs"]),
            integrated_threshold,
            int(row["Integrated_AE"]),
        ),
    ]

    np.testing.assert_allclose(
        integrated_threshold,
        float(row["threshold_abs"]),
        rtol=1e-6,
        atol=1e-9,
    )

    for (
        name, model, normalized,
        saved_score, threshold, saved_prediction
    ) in cases:

        inputs, outputs, error = reconstruct(
            model, normalized
        )

        overall_mse = float(error.mean())

        # 기존 저장 점수와 동일한 추론인지 검증
        np.testing.assert_allclose(
            overall_mse,
            saved_score,
            rtol=1e-5,
            atol=1e-7,
        )

        prediction = int(
            overall_mse > threshold
        )

        assert prediction == saved_prediction

        result = {
            "file": row["file"],
            "group": row["group"],
            "model": name,
            "overall_mse": overall_mse,
            "threshold": threshold,
            "score_ratio": overall_mse / threshold,
            "prediction": prediction,
            "evaluation_status": "exploratory_only",
        }

        for band, mask in bands.items():

            band_error = error[:, mask]

            result[f"{band}_mse"] = float(
                band_error.mean()
            )

            result[f"{band}_error_fraction"] = (
                float(band_error.sum())
                / max(float(error.sum()), 1e-30)
            )

            result[f"{band}_global_contribution"] = (
                float(band_error.sum())
                / error.size
            )

        mask = bands["1X"]

        result["1X_input_abs_mean"] = float(
            np.abs(inputs[:, mask]).mean()
        )

        result["1X_reconstruction_abs_mean"] = float(
            np.abs(outputs[:, mask]).mean()
        )

        records.append(result)

        key = (name, row["group"])

        spectra.setdefault(key, []).append(
            (
                inputs.mean(axis=0),
                outputs.mean(axis=0),
                error.mean(axis=0),
            )
        )

    print(
        f"[{i+1:02d}/20] "
        f"{row['group']} / {path.name}"
    )


# 6. 결과 저장
results = pd.DataFrame(records)

assert len(results) == 40

results.to_csv(
    DEST / "unbalance_reconstruction_errors.csv",
    index=False,
)

summary_columns = [
    "overall_mse",
    "score_ratio",
    "1X_mse",
    "2X_mse",
    "0_100Hz_mse",
    "100_1000Hz_mse",
    "1000_10000Hz_mse",
    "1X_error_fraction",
    "1X_global_contribution",
    "0_100Hz_error_fraction",
]

summary = results.groupby(
    ["model", "group"]
)[summary_columns].median()

summary.to_csv(
    DEST / "unbalance_reconstruction_summary.csv"
)

print("\n===== 그룹별 복원 오차 중앙값 =====")

for name in ("AE", "Integrated_AE"):

    print(f"\n[{name}]")

    print(
        summary.loc[name].T.to_string(
            float_format=lambda x: f"{x:.8f}"
        )
    )


print("\n===== 공통 미탐 5개 상세 결과 =====")

missed = results[
    results["group"] == "Common_Missed"
].copy()

missed["filename"] = missed["file"].map(
    lambda x: Path(x).name
)

print(
    missed[
        [
            "filename",
            "model",
            "score_ratio",
            "1X_mse",
            "1X_error_fraction",
            "1X_global_contribution",
            "overall_mse",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# 7. 저주파 입력·복원 및 오차 시각화
low_mask = frequencies <= 100

for model in ("AE", "Integrated_AE"):

    fig, axes = plt.subplots(
        4, 2,
        figsize=(14, 12),
        constrained_layout=True,
    )

    for i, group in enumerate(groups):

        values = spectra[(model, group)]

        inputs = np.stack([v[0] for v in values])
        outputs = np.stack([v[1] for v in values])
        errors = np.stack([v[2] for v in values])

        ax = axes[i, 0]

        ax.plot(
            frequencies[low_mask],
            np.median(inputs, axis=0)[low_mask],
            label="Input",
        )

        ax.plot(
            frequencies[low_mask],
            np.median(outputs, axis=0)[low_mask],
            label="Reconstruction",
        )

        ax.set_title(f"{group} | Input vs Reconstruction")
        ax.set_ylabel("Normalized spectrum")
        ax.legend()

        ax = axes[i, 1]

        ax.plot(
            frequencies[low_mask],
            np.median(errors, axis=0)[low_mask],
        )

        ax.set_title(f"{group} | Squared error")
        ax.set_ylabel("Squared error")

        for ax in axes[i]:

            ax.axvline(
                15, linestyle="--",
                label="1X",
                alpha=0.7,
            )

            ax.axvline(
                30, linestyle=":",
                alpha=0.7,
            )

            ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.2)

    fig.suptitle(
        f"{model} | 900 RPM Unbalance Reconstruction\n"
        "Exploratory analysis; 5 files per group"
    )

    target = DEST / (
        f"unbalance_reconstruction_{model}.png"
    )

    fig.savefig(target, dpi=160)
    plt.close(fig)

    print("\n그래프 저장:", target)


print("\n===== 최종 복원 오차 분석 완료 =====")
print("분석 파일: 20개")
print("모델: 기존 AE, 통합 AE")
print("저장 평가 점수 및 판정 일치: 40/40")
print("추가 모델 학습 없음")
print("기존 임계값 변경 없음")
print("운영 모델 변경 없음")
print("탐색적 분석 결과")
