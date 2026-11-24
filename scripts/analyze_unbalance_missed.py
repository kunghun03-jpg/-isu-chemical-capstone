"""900 RPM Unbalance_-2 공통 미탐 원인 탐색."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]

DEST = ROOT / "outputs/defect_detection_analysis"

FS = 25600
SAMPLES = 65536

MODELS = [
    "AE", "OCSVM", "IF", "DAE",
    "Integrated_AE", "Relative_AE", "Hybrid_AE",
]

df = pd.read_csv(
    DEST / "all_models_file_comparison.csv"
)

df = df[
    (df["rpm"] == 900)
    & (
        (df["ground_truth"] == 0)
        | (df["condition"] == "Unbalance_-2")
    )
].copy()

assert len(df) == 150

df["group"] = np.where(
    df["ground_truth"] == 0,
    df["condition"],
    np.where(
        (df[MODELS] == 0).all(axis=1),
        "Common_Missed",
        "Detected_By_Any",
    ),
)

expected = {
    "Normal1": 49,
    "Normal2": 51,
    "Common_Missed": 5,
    "Detected_By_Any": 45,
}

counts = df["group"].value_counts().to_dict()

assert counts == expected, counts

print("===== 비교 대상 =====")
print(df["group"].value_counts().to_string())


def extract_features(path):

    signals = np.asarray(
        loadmat(path)["signals"]
    )

    if signals.shape != (6, SAMPLES):
        raise ValueError(
            f"신호 형태 불일치: {signals.shape}"
        )

    x = signals[2].astype(np.float64)

    if not np.isfinite(x).all():
        raise ValueError("비정상 신호값 발견")

    x = detrend(x, type="constant")

    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))

    window = np.hanning(len(x))

    fft = np.fft.rfft(x * window)

    freq = np.fft.rfftfreq(
        len(x), d=1 / FS
    )

    amplitude = (
        2 * np.abs(fft) / window.sum()
    )

    power = np.abs(fft) ** 2

    # 1X = 15 Hz, 2X = 30 Hz
    mask_1x = np.abs(freq - 15) <= 1
    mask_2x = np.abs(freq - 30) <= 1

    total_power = power[1:].sum()

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / max(rms, 1e-12),
        "one_x_amplitude": amplitude[mask_1x].max(),
        "two_x_amplitude": amplitude[mask_2x].max(),
        "one_x_energy_ratio": (
            power[mask_1x].sum()
            / max(total_power, 1e-30)
        ),
        "low_energy_ratio": (
            power[(freq > 0) & (freq <= 100)].sum()
            / max(total_power, 1e-30)
        ),
    }


rows = []

for _, row in df.iterrows():

    path = Path(row["file"])

    if not path.exists():
        path = ROOT / row["file_key"]

    features = extract_features(path)

    rows.append({
        "file": row["file"],
        "rpm": row["rpm"],
        "condition": row["condition"],
        "group": row["group"],
        **features,
    })

features = pd.DataFrame(rows)

columns = [
    "rms",
    "peak",
    "crest_factor",
    "one_x_amplitude",
    "two_x_amplitude",
    "one_x_energy_ratio",
    "low_energy_ratio",
]

print("\n===== 그룹별 진동 특징 중앙값 =====")

print(
    features.groupby("group")[columns]
    .median().T.to_string(
        float_format=lambda x: f"{x:.8f}"
    )
)

print("\n===== 그룹별 진동 특징 Q25 =====")

print(
    features.groupby("group")[columns]
    .quantile(0.25).T.to_string(
        float_format=lambda x: f"{x:.8f}"
    )
)

print("\n===== 그룹별 진동 특징 Q75 =====")

print(
    features.groupby("group")[columns]
    .quantile(0.75).T.to_string(
        float_format=lambda x: f"{x:.8f}"
    )
)

print("\n===== 공통 미탐 5개 파일의 특징 =====")

missed = features[
    features["group"] == "Common_Missed"
].copy()

missed["filename"] = missed["file"].map(
    lambda x: Path(x).name
)

print(
    missed[["filename", *columns]].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

print("\n===== 공통 미탐 파일의 기존 모델 점수 비율 =====")

score_columns = [
    ("Integrated_AE", "score_abs", "threshold_abs"),
    ("Relative_AE", "score_rel", "threshold_rel"),
    ("Hybrid_AE", "hybrid_score", "hybrid_threshold"),
]

missed_scores = df[
    df["group"] == "Common_Missed"
].copy()

missed_scores["filename"] = missed_scores["file"].map(
    lambda x: Path(x).name
)

for model, score, threshold in score_columns:

    missed_scores[model + "_score_ratio"] = (
        missed_scores[score]
        / missed_scores[threshold]
    )

ratio_columns = [
    m + "_score_ratio"
    for m, _, _ in score_columns
]

print(
    missed_scores[
        ["filename", *ratio_columns]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

features.to_csv(
    DEST / "unbalance_900_signal_features.csv",
    index=False,
)

print("\n===== 분석 완료 =====")
print("분석 파일: 150개")
print("추가 모델 학습 없음")
print("추가 모델 추론 없음")
print("기존 임계값 변경 없음")
print("탐색적 신호 특징 분석")
