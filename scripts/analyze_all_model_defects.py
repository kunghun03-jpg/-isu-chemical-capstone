"""7개 모델의 동일 시험 파일 기반 탐색적 결함 분석."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEST = OUT / "defect_detection_analysis"

DEST.mkdir(parents=True, exist_ok=True)

RPMS = (900, 1200, 1800)

MODELS = [
    "AE",
    "OCSVM",
    "IF",
    "DAE",
    "Integrated_AE",
    "Relative_AE",
    "Hybrid_AE",
]

DEFECT_MAP = {
    "IR": "Bearing",
    "OR": "Bearing",
    "Roller": "Bearing",
    "Looseness0.05": "Looseness",
    "Looseness0.1": "Looseness",
    "LoosenessFull": "Looseness",
    "Unbalance_-1": "Unbalance",
    "Unbalance_-2": "Unbalance",
}


def file_key(value):

    path = str(value).replace("\\", "/")

    if "/data/" in path:
        return "data/" + path.split("/data/", 1)[1]

    if path.startswith("data/"):
        return path

    if path.startswith("./data/"):
        return path[2:]

    raise ValueError(f"파일 경로 확인 필요: {value}")


base = pd.read_csv(
    OUT / "hybrid_ae_exploratory_evaluation"
    / "predictions.csv"
)

base["file_key"] = base["file"].map(file_key)

base = base.rename(columns={
    "predicted_anomaly_abs": "Integrated_AE",
    "predicted_anomaly_rel": "Relative_AE",
    "hybrid_prediction": "Hybrid_AE",
})

assert len(base) == 1484
assert base["file_key"].is_unique
assert base["ground_truth"].isin([0, 1]).all()

base_keys = set(base["file_key"])


def load_original(model):

    parts = []

    for rpm in RPMS:

        folder = OUT / f"{rpm}rpm/date_prelim_20261106"

        if model == "AE":

            path = folder / f"evaluation_file_{rpm}rpm.csv"

        elif model == "OCSVM":

            path = folder / "comparisons/ocsvm/scores_file.csv"

        else:

            path = folder / "comparisons/iforest/scores_file.csv"

        df = pd.read_csv(path)

        df = df[
            df["split"].isin([
                "test_normal",
                "test_anomaly",
            ])
        ].copy()

        if "rpm" in df.columns:

            if not df["rpm"].eq(rpm).all():
                raise ValueError(f"{model}: RPM 불일치")

        df["rpm"] = rpm

        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def attach_model(base, df, model):

    df = df.copy()

    df = df[
        df["split"].isin([
            "test_normal",
            "test_anomaly",
        ])
    ].copy()

    df["file_key"] = df["file"].map(file_key)

    if not df["file_key"].is_unique:
        raise ValueError(f"{model}: 중복 파일 발견")

    keys = set(df["file_key"])

    missing = base_keys - keys
    extra = keys - base_keys

    if missing or extra:

        raise ValueError(
            f"{model}: 평가 파일 불일치 / "
            f"누락 {len(missing)} / 추가 {len(extra)}"
        )

    df["truth_check"] = (
        df["split"] == "test_anomaly"
    ).astype(int)

    if not df["predicted_anomaly"].isin([0, 1]).all():
        raise ValueError(f"{model}: 판정값 오류")

    aligned = df.set_index("file_key").loc[
        base["file_key"]
    ]

    for col in ("rpm", "condition"):

        if not (
            aligned[col].to_numpy()
            == base[col].to_numpy()
        ).all():
            raise ValueError(
                f"{model}: {col} 불일치"
            )

    if not (
        aligned["truth_check"].to_numpy()
        == base["ground_truth"].to_numpy()
    ).all():

        raise ValueError(f"{model}: 정답 불일치")

    base[model] = aligned[
        "predicted_anomaly"
    ].to_numpy()

    print(f"{model}: 동일 평가 파일 1484개 확인")

    return base


print("===== 7개 모델 평가 파일 일치 검사 =====")

for model in ("AE", "OCSVM", "IF"):

    base = attach_model(
        base,
        load_original(model),
        model,
    )

dae = pd.read_csv(
    OUT / "dae_group_split_experiment_20261119"
    / "dae_test_scores.csv"
)

base = attach_model(base, dae, "DAE")

for model in MODELS:

    if not base[model].isin([0, 1]).all():
        raise ValueError(f"{model}: 판정값 오류")

faults = base[base["ground_truth"] == 1].copy()
normal = base[base["ground_truth"] == 0].copy()

assert len(faults) == 1183
assert len(normal) == 301

faults["defect_type"] = faults["condition"].map(
    DEFECT_MAP
)

assert faults["defect_type"].notna().all()

print("전체 평가 파일:", len(base))
print("정상:", len(normal))
print("결함:", len(faults))


def summarize(data, groups):

    rows = []

    for keys, group in data.groupby(groups):

        if not isinstance(keys, tuple):
            keys = (keys,)

        labels = dict(zip(groups, keys))

        for model in MODELS:

            count = len(group)
            tp = int(group[model].sum())

            rows.append({
                **labels,
                "model": model,
                "files": count,
                "TP": tp,
                "FN": count - tp,
                "recall": tp / count,
                "evaluation_status": "exploratory_only",
            })

    return pd.DataFrame(rows)


def display_summary(data, groups, title, filename):

    summary = summarize(data, groups)

    table = summary.pivot(
        index=groups,
        columns="model",
        values="recall",
    )

    print(f"\n===== {title} =====")

    print(table[MODELS].to_string(
        float_format=lambda x: f"{x:.2%}"
    ))

    summary.to_csv(
        DEST / filename,
        index=False,
    )


display_summary(
    faults,
    ["defect_type"],
    "7개 모델 대분류별 탐지율",
    "all_models_major_defects.csv",
)

display_summary(
    faults,
    ["condition"],
    "7개 모델 세부 결함별 탐지율",
    "all_models_detailed_defects.csv",
)

display_summary(
    faults,
    ["rpm", "condition"],
    "7개 모델 RPM별 세부 결함 탐지율",
    "all_models_rpm_defects.csv",
)


print("\n===== 전체 정상 오탐률 및 결함 탐지율 =====")

metrics = []

for model in MODELS:

    fp = int(normal[model].sum())
    tp = int(faults[model].sum())

    fpr = fp / len(normal)
    recall = tp / len(faults)

    balanced_accuracy = (
        recall + (1 - fpr)
    ) / 2

    metrics.append({
        "model": model,
        "FP": fp,
        "FN": len(faults) - tp,
        "FPR": fpr,
        "Recall": recall,
        "Balanced_Accuracy": balanced_accuracy,
        "evaluation_status": "exploratory_only",
    })

metrics = pd.DataFrame(metrics)

print(metrics.drop(
    columns="evaluation_status"
).to_string(
    index=False,
    float_format=lambda x: f"{x:.2%}",
))

metrics.to_csv(
    DEST / "all_models_metrics.csv",
    index=False,
)


print("\n===== RPM별 정상 오탐률 =====")

print(
    normal.groupby("rpm")[MODELS].mean().to_string(
        float_format=lambda x: f"{x:.2%}"
    )
)


print("\n===== Unbalance 탐지율 집중 비교 =====")

unbalance = faults[
    faults["defect_type"] == "Unbalance"
]

print(
    unbalance.groupby(
        ["rpm", "condition"]
    )[MODELS].mean().to_string(
        float_format=lambda x: f"{x:.2%}"
    )
)


experimental_models = [
    "Integrated_AE",
    "Relative_AE",
    "Hybrid_AE",
]

missed = faults[
    (faults[experimental_models] == 0).all(axis=1)
].copy()

print("\n===== 기존 공통 미탐 115개 추가 분석 =====")

assert len(missed) == 115

for model in ("AE", "OCSVM", "IF", "DAE"):

    recovered = int(missed[model].sum())

    print(
        f"{model}: {recovered}/115개 탐지 "
        f"({recovered/115:.2%})"
    )


print("\n===== 7개 모델 전체 공통 미탐 =====")

all_missed = faults[
    (faults[MODELS] == 0).all(axis=1)
].copy()

print(
    f"공통 미탐: {len(all_missed)}/1183 "
    f"({len(all_missed)/1183:.2%})"
)

if not all_missed.empty:

    print(pd.crosstab(
        all_missed["condition"],
        all_missed["rpm"],
        margins=True,
    ).to_string())


print("\n===== 7개 모델 전체 공통 정상 오탐 =====")

all_false_positive = normal[
    (normal[MODELS] == 1).all(axis=1)
].copy()

print(
    f"공통 오탐: {len(all_false_positive)}/301 "
    f"({len(all_false_positive)/301:.2%})"
)

if not all_false_positive.empty:

    print(pd.crosstab(
        all_false_positive["condition"],
        all_false_positive["rpm"],
        margins=True,
    ).to_string())


base.to_csv(
    DEST / "all_models_file_comparison.csv",
    index=False,
)

missed.to_csv(
    DEST / "experimental_missed_comparison.csv",
    index=False,
)

all_missed.to_csv(
    DEST / "all_models_common_missed.csv",
    index=False,
)

all_false_positive.to_csv(
    DEST / "all_models_common_false_positive.csv",
    index=False,
)

print("\n===== 통합 분석 완료 =====")
print("비교 모델: 7개")
print("평가 파일: 1484개")
print("결함 파일: 1183개")
print("저장 위치:", DEST)
print("추가 학습 없음")
print("추가 MAT 추론 없음")
print("임계값 변경 없음")
print("기존 시험 데이터 기반 탐색적 비교")
