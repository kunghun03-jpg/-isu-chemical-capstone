"""저장 모델로 MAT 한 개를 분석하는 최소 Streamlit 화면."""

from pathlib import Path
import hashlib
import json

import pandas as pd
import streamlit as st

from vibration.inference import SavedModelInference
from vibration.history_ui import save_result, render_history
from vibration.detail_ui import render_details
from vibration.web_inference import (
    MODEL_OPTIONS,
    MODEL_LABELS,
    preview_mat,
    upload_name,
    error_message,
)

ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT / "outputs/inference_bundles/date_prelim_20261106"


@st.cache_resource(show_spinner=False)
def load_engine(rpm, models):
    # 사용자 업로드는 캐시하지 않습니다. 신뢰하는 로컬 저장 모델만 재사용합니다.
    return SavedModelInference(BUNDLE_ROOT / f"{rpm}rpm", models=models)


def main():
    st.set_page_config(page_title="CH3 진동 분석", layout="wide")
    st.title("CH3 진동 분석")
    st.write("MAT 파일 하나를 업로드하고 저장된 모델로 정상·이상을 분석합니다.")
    st.warning(
        "성능 한계: 날짜가 바뀐 정상 데이터에서 오탐률이 높았습니다. 이상 판정만으로 실제 고장을 확정할 수 없습니다. 실제 독립 Run 검증은 아직 완료되지 않았습니다."
    )
    st.caption(
        "기존 주파수별 Train 정규화와 저장 임계값을 사용합니다. 분석 중 재학습이나 임계값 재계산은 없습니다."
    )
    rpm = st.selectbox("RPM", (900, 1200, 1800))
    choice = st.selectbox("모델", tuple(MODEL_OPTIONS))
    st.caption(
        "RPM은 파일의 실제 측정 조건에 맞게 선택하세요. 파일명이나 폴더의 상태 이름은 판정에 사용하지 않습니다."
    )
    uploaded = st.file_uploader(
        "MAT 파일 업로드", type=["mat"], accept_multiple_files=False
    )
    raw = uploaded.getvalue() if uploaded is not None else None
    identity = (
        (hashlib.sha256(raw).hexdigest(), rpm, choice, upload_name(uploaded.name))
        if raw is not None
        else None
    )
    if st.session_state.get("input_identity") != identity:
        st.session_state.pop("analysis_result", None)
        st.session_state["input_identity"] = identity
    if raw is None:
        st.info(
            "signals=(6, 65536), fs=25600Hz인 MAT 파일을 업로드해 주세요. 최대 20MB입니다."
        )
        st.button("분석 실행", disabled=True)
        render_history()
        return
    try:
        preview = preview_mat(raw)
    except Exception as error:
        st.error("입력 파일 확인: " + error_message(error))
        st.button("분석 실행", disabled=True)
        render_history()
        return
    st.subheader("CH3 파형 미리보기")
    cols = st.columns(3)
    cols[0].metric("샘플링 주파수", f"{preview['fs']:,.0f} Hz")
    cols[1].metric("CH3 샘플 수", f"{preview['samples']:,}")
    cols[2].metric("측정 길이", f"{preview['duration']:.2f} 초")
    # 파형 전체를 표시합니다. 추론에는 미리보기 표가 아니라 원본 bytes를 전달합니다.
    st.line_chart(
        pd.DataFrame({"시간 (초)": preview["time"], "CH3 진폭": preview["signal"]}),
        x="시간 (초)",
        y="CH3 진폭",
    )
    st.caption("원본 CH3 진폭입니다. 물리 단위는 센서·수집 설정 확인이 필요합니다.")
    if st.button("분석 실행", type="primary"):
        st.session_state.pop("analysis_result", None)
        try:
            with st.spinner("저장 모델로 분석 중입니다…"):
                engine = load_engine(rpm, MODEL_OPTIONS[choice])
                st.session_state["analysis_result"] = engine.predict_bytes(
                    raw, rpm=rpm, filename=upload_name(uploaded.name)
                )
                st.session_state["model_versions"] = {
                    name: "sha256:"
                    + engine.manifest["artifacts"][
                        engine.manifest["models"][name]["artifact"]
                    ]
                    for name in engine.models
                }
        except Exception as error:
            st.error("분석할 수 없습니다: " + error_message(error))
    result = st.session_state.get("analysis_result")
    if result is not None:
        st.subheader("분석 결과")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "모델": MODEL_LABELS[name],
                        "파일 점수": item["file_score"],
                        "저장 임계값": item["file_threshold"],
                        "판정": "이상" if item["predicted_anomaly"] else "정상",
                    }
                    for name, item in result["models"].items()
                ]
            ),
            hide_index=True,
            column_config={
                "파일 점수": st.column_config.NumberColumn(format="%.12g"),
                "저장 임계값": st.column_config.NumberColumn(format="%.12g"),
            },
        )
        st.caption(
            "파일 점수는 8개 구간 점수의 평균입니다. 점수가 저장 임계값을 초과할 때 이상으로 판정합니다. 모델 간 점수 크기는 직접 비교하지 마세요."
        )
        st.download_button(
            "결과 JSON 다운로드",
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            file_name="inference_result.json",
            mime="application/json",
        )
        save_result(result, st.session_state.get("model_versions", {}))
        render_details(result, preview)
    st.caption(
        "원본 업로드 파일은 저장하지 않습니다. 결과 저장 버튼을 누른 분석 정보만 SQLite에 저장합니다. 결함 후보 추정은 후속 단계입니다."
    )
    render_history()


if __name__ == "__main__":
    main()
