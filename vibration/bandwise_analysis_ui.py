"""Read-only Streamlit UI for supplementary bandwise analysis."""

import altair as alt
import pandas as pd
import streamlit as st


BAND_NAMES = {
    "1X": "1X",
    "2X": "2X",
    "0_100Hz": "저주파",
    "100_1000Hz": "중주파",
    "1000_10000Hz": "고주파",
}

BAND_ORDER = [
    "1X",
    "2X",
    "저주파",
    "중주파",
    "고주파",
]

BROAD_ORDER = [
    "저주파",
    "중주파",
    "고주파",
]

BAND_RANGES = {
    "1X": "회전 기본 주파수 주변",
    "2X": "회전 2배 주파수 주변",
    "0_100Hz": "0–100 Hz",
    "100_1000Hz": "100–1,000 Hz",
    "1000_10000Hz": "1,000–10,000 Hz",
}

BAND_COLORS = [
    "#5793db",
    "#e8a34b",
    "#8b79c6",
]


def render_bandwise_analysis(analysis):

    st.subheader("주파수 대역별 복원 오차 분석")

    if analysis is None:
        st.info(
            "MAT 파일 분석 후 대역별 복원 오차를 "
            "확인할 수 있습니다."
        )
        return

    if (
        analysis.get("analysis_status")
        != "supplementary_only"
        or analysis.get("anomaly_decision_provided")
        is not False
        or analysis.get("fault_classification_provided")
        is not False
        or analysis.get("approved_for_production")
        is not False
        or analysis.get("operational_model_changed")
        is not False
    ):
        raise ValueError(
            "보조 분석 정책이 올바르지 않습니다."
        )

    st.caption(
        f"{analysis['rpm']} RPM · "
        "AE 복원 결과 기반 보조 분석 전용"
    )

    bands = {
        band["band"]: band
        for band in analysis["bands"]
    }

    if set(bands) != set(BAND_NAMES):
        raise ValueError("주파수 대역 구성 오류")

    highest_mse = analysis["highest_mse_band"]
    highest_contribution = analysis[
        "highest_contribution_band"
    ]

    left, right = st.columns(2)

    left.metric(
        "평균 복원 오차 최대 대역",
        BAND_NAMES.get(highest_mse, "없음"),
    )

    if highest_mse is not None:
        left.caption(
            f"{BAND_RANGES[highest_mse]} · "
            f"MSE {bands[highest_mse]['mse']:.6f}"
        )

    right.metric(
        "전체 오차 기여도 최대 대역",
        BAND_NAMES.get(
            highest_contribution, "없음"
        ),
    )

    if highest_contribution is not None:
        right.caption(
            f"{BAND_RANGES[highest_contribution]} · "
            f"기여 비중 "
            f"{bands[highest_contribution]['error_percentage']:.2f}%"
        )

    frame = pd.DataFrame([
        {
            "대역": BAND_NAMES[name],
            "구분": (
                "회전 고조파"
                if band["band_type"] == "harmonic"
                else "광대역"
            ),
            "최소 주파수 (Hz)": band["min_hz"],
            "최대 주파수 (Hz)": band["max_hz"],
            "Bin 수": band["bin_count"],
            "평균 MSE": band["mse"],
            "오차 비중 (%)": band["error_percentage"],
            "전체 MSE 기여도": band[
                "global_contribution"
            ],
        }
        for name, band in bands.items()
    ])

    frame["대역"] = pd.Categorical(
        frame["대역"],
        categories=BAND_ORDER,
        ordered=True,
    )

    frame = frame.sort_values("대역")

    st.markdown("**대역별 평균 복원 오차 (MSE)**")

    mse_chart = alt.Chart(frame).mark_bar(
        cornerRadiusEnd=3
    ).encode(
        x=alt.X(
            "평균 MSE:Q",
            title="평균 복원 오차",
            scale=alt.Scale(zero=True),
        ),
        y=alt.Y(
            "대역:N",
            sort=BAND_ORDER,
            title=None,
            axis=alt.Axis(
                labelOverlap=False,
                labelFontSize=13,
                labelPadding=10,
            ),
        ),
        color=alt.Color(
            "구분:N",
            scale=alt.Scale(
                domain=["회전 고조파", "광대역"],
                range=["#94a3b8", "#2563eb"],
            ),
            legend=alt.Legend(
                orient="top",
                title=None,
            ),
        ),
        tooltip=[
            "대역:N",
            "구분:N",
            alt.Tooltip(
                "평균 MSE:Q",
                format=".6f",
            ),
            alt.Tooltip(
                "최소 주파수 (Hz):Q",
                format=".3f",
            ),
            alt.Tooltip(
                "최대 주파수 (Hz):Q",
                format=".3f",
            ),
            "Bin 수:Q",
        ],
    ).properties(height=230)

    st.altair_chart(
        mse_chart,
        width="stretch",
            height=380,
    )

    st.markdown("**전체 복원 오차 기여 비중**")

    broad = frame[
        frame["구분"] == "광대역"
    ].copy()

    broad["순서"] = broad["대역"].map({
        "저주파": 0,
        "중주파": 1,
        "고주파": 2,
    }).astype(int)

    broad["전체"] = "복원 오차"

    if analysis["overall_mse"] > 0:

        share_chart = alt.Chart(
            broad
        ).mark_bar().encode(
            x=alt.X(
                "오차 비중 (%):Q",
                stack="zero",
                scale=alt.Scale(domain=[0, 100]),
                title="전체 제곱오차 비중 (%)",
            ),
            y=alt.Y(
                "전체:N",
                axis=None,
                title=None,
            ),
            color=alt.Color(
                "대역:N",
                scale=alt.Scale(
                    domain=BROAD_ORDER,
                    range=BAND_COLORS,
                ),
                legend=alt.Legend(
                    orient="top",
                    title=None,
                ),
            ),
            order=alt.Order("순서:Q"),
            tooltip=[
                "대역:N",
                alt.Tooltip(
                    "오차 비중 (%):Q",
                    format=".2f",
                ),
                alt.Tooltip(
                    "전체 MSE 기여도:Q",
                    format=".6f",
                ),
            ],
        ).properties(height=100)

        st.altair_chart(
            share_chart,
            width="stretch",
            height=180,
        )

    else:
        st.info("전체 복원 오차가 0입니다.")

    st.markdown("**대역별 상세 분석**")

    st.dataframe(
        frame.drop(columns=["구분"]),
        hide_index=True,
        width="stretch",
        column_config={
            "평균 MSE": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "오차 비중 (%)": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "전체 MSE 기여도": st.column_config.NumberColumn(
                format="%.6f"
            ),
        },
    )

    st.caption(
        "평균 MSE는 해당 대역 내 주파수 bin의 "
        "평균 제곱오차입니다. 전체 오차 기여도는 "
        "대역에 포함된 bin 개수도 반영합니다."
    )

    st.caption(
        "1X와 2X 대역은 광대역과 중첩되므로 "
        "전체 복원 오차 기여 비중 그래프에는 "
        "저·중·고주파만 표시합니다."
    )

    st.info(
        "본 결과는 주파수 대역별 복원 오차를 "
        "설명하기 위한 보조 분석입니다. "
        "기존 Normal/Abnormal 판정과 임계값에는 "
        "영향을 주지 않으며, 특정 결함의 발생을 "
        "확정하지 않습니다."
    )
