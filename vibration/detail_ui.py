"""기존 FFT 값과 추론 결과를 읽기만 하는 상세 그래프."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from vibration.web_inference import MODEL_LABELS


def spectrum_frame(preview):
    values = np.asarray(preview["log_spectra"])
    frequency = np.asarray(preview["frequency"])
    if (
        values.shape != (8, 1, 3201)
        or frequency.shape != (3201,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "CH3 log1p 스펙트럼 또는 주파수 값이 누락되었거나 잘못되었습니다."
        )
    return pd.DataFrame(
        {
            "주파수 (Hz)": frequency,
            **{f"구간 {i + 1}": values[i, 0] for i in range(8)},
            "8구간 log1p 평균": values[:, 0].mean(axis=0, dtype=np.float64),
        }
    )


def segment_frame(item):
    segments = item.get("segments")
    if (
        not isinstance(segments, list)
        or len(segments) != 8
        or {r.get("segment_id") for r in segments} != set(range(8))
    ):
        raise ValueError("8개 구간 점수 또는 구간 ID가 누락되었습니다.")
    if "segment_threshold" not in item:
        raise ValueError(
            "저장된 구간 임계값이 없습니다. 파일 임계값으로 대체하지 않습니다."
        )
    ordered = sorted(segments, key=lambda row: row["segment_id"])
    scores = np.array([row["score"] for row in ordered], dtype=float)
    threshold = float(item["segment_threshold"])
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("구간 점수·임계값이 유한하지 않습니다.")
    if not np.isclose(scores.mean(), item["file_score"], rtol=1e-12, atol=1e-12):
        raise ValueError("8구간 평균과 기존 파일 점수가 일치하지 않습니다.")
    return pd.DataFrame(
        {
            "구간": np.arange(1, 9),
            "시작 (초)": np.arange(8) * 0.32,
            "종료 (초)": np.arange(1, 9) * 0.32,
            "구간 점수": scores,
            "저장 구간 임계값": threshold,
        }
    )


def render_details(result, preview):
    st.subheader("상세 분석")
    st.write("CH3 주파수 스펙트럼 · 정규화 전 log1p 변환값")
    try:
        frame = spectrum_frame(preview)
        st.line_chart(frame, x="주파수 (Hz)", y="8구간 log1p 평균")
        st.caption(
            "기존 Hann FFT·단측 진폭 보정·0–10,000Hz·log1p 결과를 재사용합니다. 세로축은 원래 진폭이 아닌 log1p(진폭)이며, Train Z-score 적용 전입니다. 대표선은 8개 log1p 스펙트럼의 평균으로, 평균 진폭에 log1p를 취한 값과는 다릅니다."
        )
        with st.expander("8개 구간의 log1p 스펙트럼 보기"):
            st.line_chart(frame, x="주파수 (Hz)", y=[f"구간 {i}" for i in range(1, 9)])
        st.caption(
            "구간 1–8은 추론의 segment_id 0–7에 대응하며, 각 0.32초입니다. 센서의 물리 단위는 임의로 확정하지 않습니다."
        )
    except (KeyError, TypeError, ValueError) as error:
        st.warning(f"스펙트럼을 표시할 수 없습니다: {error}")
    for name, item in result["models"].items():
        label = MODEL_LABELS[name]
        st.write(f"{label} · 8개 구간 점수")
        try:
            data = segment_frame(item)
            line = (
                alt.Chart(data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("구간:O", sort=list(range(1, 9))),
                    y=alt.Y("구간 점수:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("구간:O"),
                        alt.Tooltip("시작 (초):Q", format=".2f"),
                        alt.Tooltip("종료 (초):Q", format=".2f"),
                        alt.Tooltip("구간 점수:Q", format=".12g"),
                        alt.Tooltip("저장 구간 임계값:Q", format=".12g"),
                    ],
                )
            )
            rule = (
                alt.Chart(
                    pd.DataFrame({"저장 구간 임계값": [item["segment_threshold"]]})
                )
                .mark_rule(color="red", strokeDash=[6, 4])
                .encode(y="저장 구간 임계값:Q")
            )
            st.altair_chart((line + rule).interactive(), width="stretch")
            st.caption(
                f"빨간 점선: 저장 구간 임계값 {item['segment_threshold']:.12g}. 구간 초과 {int((data['구간 점수'] > item['segment_threshold']).sum())}/8개."
            )
            st.caption(
                f"파일 요약 — 기존 8구간 평균 점수 {item['file_score']:.12g} / 저장 파일 임계값 {item['file_threshold']:.12g} / 판정: {'이상' if item['predicted_anomaly'] else '정상'}"
            )
        except (KeyError, TypeError, ValueError) as error:
            st.warning(f"{label} 구간 그래프를 표시할 수 없습니다: {error}")
    st.info(
        "구간 임계값과 파일 임계값은 다릅니다. 일부 구간이 임계값을 초과해도 파일 판정은 기존 8구간 평균 점수와 저장 파일 임계값으로 유지합니다."
    )
