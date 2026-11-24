"""동일 평가 파일 기반 4개 모델 탐색적 성능 비교 UI."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs/defect_detection_analysis"
)

MODELS = [
    "AE",
    "OCSVM",
    "DAE",
    "IF",
]

MODEL_NAMES = {
    "AE": "AE",
    "OCSVM": "OCSVM",
    "IF": "IF",
    "DAE": "DAE",
}

ORDER = [MODEL_NAMES[m] for m in MODELS]


def load_csv(name, required):

    path = ROOT / name

    frame = pd.read_csv(path)

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name}: 필수 열 누락 {sorted(missing)}"
        )

    frame = frame[
        frame["model"].isin(MODELS)
    ].copy()

    if set(frame["model"]) != set(MODELS):
        raise ValueError(f"{name}: 모델 구성 불일치")

    if not frame["evaluation_status"].eq(
        "exploratory_only"
    ).all():
        raise ValueError(f"{name}: 평가 범위 불일치")

    return frame


def defect_table(frame, column, expected_columns):

    if frame.duplicated(["model", column]).any():
        raise ValueError("결함별 평가 결과 중복")

    if not (frame["TP"] + frame["FN"]).eq(
        frame["files"]
    ).all():
        raise ValueError("결함별 평가 파일 수 불일치")

    pivot = frame.pivot(
        index="model",
        columns=column,
        values="recall",
    )

    pivot = pivot.reindex(
        index=MODELS,
        columns=expected_columns,
    )

    if pivot.isna().any().any():
        raise ValueError("결함별 평가 결과 누락")

    if not ((pivot >= 0) & (pivot <= 1)).all().all():
        raise ValueError("탐지율 범위 오류")

    pivot = (pivot * 100).round(2)

    pivot = pivot.rename(index=MODEL_NAMES)

    pivot.index.name = "모델"

    return pivot


def render_four_model_performance():

    st.subheader("4개 이상 탐지 모델 종합 성능 비교")

    st.caption(
        "2026-09-19 동일 평가 파일 기반 · "
        "날짜 기준 탐색적 평가"
    )

    cols = st.columns(4)

    cols[0].metric("비교 모델", "4개")
    cols[1].metric("정상 파일", "301개")
    cols[2].metric("결함 파일", "1,183개")
    cols[3].metric("전체 파일", "1,484개")

    st.warning(
        "물리적 Run 독립성이 확인되지 않았으며 "
        "기존 평가 데이터가 반복 활용되었습니다. "
        "독립적인 최종 성능 검증이나 "
        "산업 현장 적용 성능을 의미하지 않습니다. "
        "운영 판정 기준은 기존 AE로 유지합니다."
    )

    try:

        metrics = load_csv(
            "all_models_metrics.csv",
            [
                "model", "FP", "FN", "FPR",
                "Recall", "Balanced_Accuracy",
                "evaluation_status",
            ],
        )

        major = load_csv(
            "all_models_major_defects.csv",
            [
                "model", "defect_type", "files",
                "TP", "FN", "recall",
                "evaluation_status",
            ],
        )

        detailed = load_csv(
            "all_models_detailed_defects.csv",
            [
                "model", "condition", "files",
                "TP", "FN", "recall",
                "evaluation_status",
            ],
        )

        rpm_data = load_csv(
            "all_models_rpm_defects.csv",
            [
                "rpm", "model", "condition", "files",
                "TP", "FN", "recall",
                "evaluation_status",
            ],
        )

        if metrics["model"].duplicated().any():
            raise ValueError("종합 성능 결과 중복")

        metrics = metrics.set_index("model").loc[MODELS]

        for col in ["FPR", "Recall", "Balanced_Accuracy"]:
            if not metrics[col].between(0, 1).all():
                raise ValueError(f"{col}: 지표 범위 오류")

        if not (
            major.groupby("model")["files"].sum() == 1183
        ).all():
            raise ValueError("결함 평가 파일 수 불일치")

        display = pd.DataFrame(index=metrics.index)

        display["결함 탐지율 (%)"] = (
            metrics["Recall"] * 100
        ).round(2)

        display["정상 오탐률 (%)"] = (
            metrics["FPR"] * 100
        ).round(2)

        display["Balanced Accuracy (%)"] = (
            metrics["Balanced_Accuracy"] * 100
        ).round(2)

        display["오탐 수"] = metrics["FP"]
        display["미탐 수"] = metrics["FN"]

        display = display.rename(index=MODEL_NAMES)

        display.index.name = "모델"

        st.markdown("### 종합 성능 비교표")

        st.dataframe(
            display.reset_index(),
            hide_index=True,
            width="stretch",
        )

        st.markdown("### 결함 탐지율 · 정상 오탐률")

        st.bar_chart(
            display[
                ["결함 탐지율 (%)", "정상 오탐률 (%)"]
            ],
            y_label="비율 (%)",
            stack=False,
        )

        st.markdown("### Balanced Accuracy 비교")

        st.bar_chart(
            display[["Balanced Accuracy (%)"]],
            y_label="Balanced Accuracy (%)",
        )

        st.caption(
            "Balanced Accuracy = "
            "(결함 탐지율 + 정상 판별률) / 2. "
            "오탐률과 탐지율을 함께 고려해야 합니다."
        )

        st.divider()

        st.markdown("### 결함 종류별 탐지율")

        major_columns = [
            "Bearing",
            "Looseness",
            "Unbalance",
        ]

        major_pivot = defect_table(
            major, "defect_type", major_columns
        )

        st.dataframe(
            major_pivot.reset_index(),
            hide_index=True,
            width="stretch",
        )

        defect = st.selectbox(
            "결함 종류 선택",
            major_columns,
            key="four_model_major_defect",
        )

        st.bar_chart(
            major_pivot[[defect]],
            y_label="결함 탐지율 (%)",
        )

        st.caption(
            "Bearing 442개 · Looseness 452개 · "
            "Unbalance 289개. "
            "해당 결함을 이상으로 탐지한 비율이며 "
            "결함 종류 분류 정확도가 아닙니다."
        )

        st.divider()

        st.markdown("### 세부 결함별 탐지율")

        conditions = [
            "IR",
            "OR",
            "Roller",
            "Looseness0.05",
            "Looseness0.1",
            "LoosenessFull",
            "Unbalance_-1",
            "Unbalance_-2",
        ]

        detail_pivot = defect_table(
            detailed, "condition", conditions
        )

        st.dataframe(
            detail_pivot.reset_index(),
            hide_index=True,
            width="stretch",
        )

        condition = st.selectbox(
            "세부 결함 선택",
            conditions,
            key="four_model_detail_defect",
        )

        st.bar_chart(
            detail_pivot[[condition]],
            y_label="결함 탐지율 (%)",
        )

        st.divider()

        st.markdown("### RPM별 결함 탐지율")

        rpm = st.selectbox(
            "측정 RPM",
            [900, 1200, 1800],
            format_func=lambda value: f"{value} RPM",
            key="four_model_performance_rpm",
        )

        selected = rpm_data[
            rpm_data["rpm"] == rpm
        ].copy()

        rpm_pivot = defect_table(
            selected, "condition", conditions
        )

        st.dataframe(
            rpm_pivot.reset_index(),
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            rpm_pivot[[condition]],
            y_label="결함 탐지율 (%)",
        )

        st.caption(
            "모든 수치는 저장된 오프라인 평가 CSV를 "
            "기반으로 표시합니다. "
            "현재 업로드 파일의 추론 결과와는 별개입니다."
        )

    except Exception as error:
        st.error(
            "최신 4개 모델 성능 자료를 표시하지 "
            f"못했습니다: {error}"
        )
