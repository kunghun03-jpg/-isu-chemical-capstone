"""저장된 모델별 평가 결과를 활용한 결함 탐지율 분석."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs/hybrid_ae_exploratory_evaluation"
    / "predictions.csv"
)

DEST = ROOT / "outputs/defect_detection_analysis"

MODELS = {
    "integrated_ae": "predicted_anomaly_abs",
    "relative_ae": "predicted_anomaly_rel",
    "hybrid_ae": "hybrid_prediction",
}

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

EXPECTED = {
    "IR": 147,
    "OR": 147,
    "Roller": 148,
    "Looseness0.05": 151,
    "Looseness0.1": 151,
    "LoosenessFull": 150,
    "Unbalance_-1": 144,
    "Unbalance_-2": 145,
}

df = pd.read_csv(SOURCE)

if len(df) != 1484:
    raise ValueError("전체 평가 파일 수 불일치")

if df["file"].duplicated().any():
    raise ValueError("중복 파일 발견")

if not df["ground_truth"].isin([0, 1]).all():
    raise ValueError("정답 레이블 오류")

faults = df[df["ground_truth"] == 1].copy()

if len(faults) != 1183:
    raise ValueError("전체 결함 파일 수 불일치")

if faults["condition"].value_counts().to_dict() != EXPECTED:
    raise ValueError("결함별 파일 수 불일치")

for col in MODELS.values():

    if not faults[col].isin([0, 1]).all():
        raise ValueError(f"판정값 오류: {col}")

faults["defect_type"] = faults["condition"].map(
    DEFECT_MAP
)

if faults["defect_type"].isna().any():
    raise ValueError("미등록 결함 코드 발견")

DEST.mkdir(parents=True, exist_ok=True)

print("===== 평가 데이터 검증 완료 =====")
print("전체 결함:", len(faults))
print("대분류: 3종")
print("세부 결함: 8종")


def summarize(data, group_cols):

    rows = []

    for keys, group in data.groupby(group_cols):

        if not isinstance(keys, tuple):
            keys = (keys,)

        labels = dict(zip(group_cols, keys))

        for model, col in MODELS.items():

            total = len(group)
            tp = int(group[col].sum())
            fn = total - tp

            rows.append({
                **labels,
                "model": model,
                "files": total,
                "TP": tp,
                "FN": fn,
                "recall": tp / total,
                "fnr": fn / total,
                "evaluation_status": "exploratory_only",
            })

    return pd.DataFrame(rows)


def display_table(summary, index):

    table = summary.pivot(
        index=index,
        columns="model",
        values="recall",
    )

    table = table[list(MODELS)]

    print(
        table.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )


# 1. 대분류별 탐지율
major = summarize(
    faults, ["defect_type"]
)

print("\n===== 대분류별 이상 탐지율 =====")

display_table(
    major, ["defect_type"]
)

major.to_csv(
    DEST / "major_defect_summary.csv",
    index=False,
)


# 2. 세부 결함별 탐지율
detail = summarize(
    faults, ["condition"]
)

print("\n===== 세부 결함 8종 이상 탐지율 =====")

display_table(
    detail, ["condition"]
)

detail.to_csv(
    DEST / "detailed_defect_summary.csv",
    index=False,
)


# 3. RPM별 대분류 성능
rpm_major = summarize(
    faults, ["rpm", "defect_type"]
)

print("\n===== RPM·대분류별 이상 탐지율 =====")

display_table(
    rpm_major, ["rpm", "defect_type"]
)

rpm_major.to_csv(
    DEST / "rpm_major_defect_summary.csv",
    index=False,
)


# 4. RPM별 세부 결함 성능
rpm_detail = summarize(
    faults, ["rpm", "condition"]
)

print("\n===== RPM·세부 결함별 이상 탐지율 =====")

display_table(
    rpm_detail, ["rpm", "condition"]
)

rpm_detail.to_csv(
    DEST / "rpm_detailed_defect_summary.csv",
    index=False,
)


# 5. 공통 미탐 분석
faults["detected_model_count"] = (
    faults[list(MODELS.values())].sum(axis=1)
)

missed = faults[
    faults["detected_model_count"] == 0
].copy()

print("\n===== 세 모델 공통 미탐 =====")

print(
    f"공통 미탐: {len(missed)}/1183 "
    f"({len(missed)/1183:.2%})"
)

if not missed.empty:

    print(pd.crosstab(
        missed["condition"],
        missed["rpm"],
        margins=True,
    ).to_string())

missed.to_csv(
    DEST / "common_missed_faults.csv",
    index=False,
)


# 6. 모델 간 탐지 일치 정도
print("\n===== 결함별 탐지 성공 모델 수 =====")

for count in (0, 1, 2, 3):

    n = int(
        (faults["detected_model_count"] == count).sum()
    )

    print(
        f"{count}/3 모델 탐지: "
        f"{n}개 ({n/1183:.2%})"
    )


# 7. 모델별 가장 강한 조건과 취약한 조건
print("\n===== 모델별 강점·취약점 =====")

for model, group in rpm_detail.groupby("model"):

    ranked = group.sort_values(
        ["recall", "rpm", "condition"]
    )

    print(f"\n[{model}]")

    print("가장 취약한 조건 3개:")

    for _, row in ranked.head(3).iterrows():

        print(
            f"  {row['rpm']} RPM / "
            f"{row['condition']} / "
            f"Recall {row['recall']:.2%} / "
            f"미탐 {row['FN']}/{row['files']}"
        )

    print("가장 잘 탐지하는 조건 3개:")

    for _, row in ranked.tail(3).iloc[::-1].iterrows():

        print(
            f"  {row['rpm']} RPM / "
            f"{row['condition']} / "
            f"Recall {row['recall']:.2%} / "
            f"탐지 {row['TP']}/{row['files']}"
        )


# 8. 기존 AE·OCSVM·IF·DAE 자료 확인
print("\n===== 기존 모델 평가 자료 후보 =====")

for path in sorted((ROOT / "outputs").rglob("*.csv")):

    name = path.name.lower()

    if any(key in name for key in (
        "comparison_file",
        "predictions",
        "dae_rpm",
        "defect_summary",
    )):
        print(path.relative_to(ROOT))


print("\n===== 결함 탐지 분석 완료 =====")
print("저장 위치:", DEST)
print("추가 모델 학습 없음")
print("추가 MAT 추론 없음")
print("기존 임계값 변경 없음")
print("기존 Test 기반 탐색적 분석")
