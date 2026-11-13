"""기존 화면 아래에 붙이는 저장·이력 UI."""

from pathlib import Path
import os

import pandas as pd
import streamlit as st

from vibration.history import HistoryStore, history_csv
from vibration.web_inference import MODEL_LABELS


def store():
    default = Path(__file__).resolve().parents[1] / "outputs/history/analyses.sqlite3"
    return HistoryStore(os.environ.get("ISU_CHEMICAL_HISTORY_DB", str(default)))


def save_result(result, versions):
    if st.button("결과 저장", key="save_analysis"):
        try:
            analysis_id, created = store().save(result, versions)
            if created:
                st.success(f"분석 #{analysis_id} 저장 완료")
            else:
                st.info(
                    f"이미 저장된 분석 #{analysis_id}입니다. 중복 저장하지 않았습니다."
                )
        except Exception:
            st.error(
                "결과 저장에 실패했습니다. DB 경로의 쓰기 권한·디스크 공간·파일 잠금 상태를 확인하고 다시 시도해 주세요. 저장 완료로 처리하지 않았습니다."
            )


def render_history():
    st.subheader("분석 이력")
    st.caption(
        "분석 날짜는 한국 시간(KST) 기준으로 검색합니다. 전체 모델 결과는 같은 분석 ID로 연결됩니다. 모델·판정 조건은 같은 모델 결과 행에 함께 적용됩니다."
    )
    cols = st.columns(2)
    start = cols[0].date_input("시작 날짜", value=None, key="history_start")
    end = cols[1].date_input("종료 날짜", value=None, key="history_end")
    cols = st.columns(3)
    rpm = cols[0].selectbox("이력 RPM", ("전체", 900, 1200, 1800), key="history_rpm")
    model = cols[1].selectbox(
        "이력 모델", ("전체", "AE", "OCSVM", "IF"), key="history_model"
    )
    decision = cols[2].selectbox(
        "이력 판정", ("전체", "정상", "이상"), key="history_decision"
    )
    if start and end and start > end:
        st.error("시작 날짜는 종료 날짜보다 늦을 수 없습니다.")
        return
    try:
        models = {label: name for name, label in MODEL_LABELS.items()}
        rows = store().search(
            start=start,
            end=end,
            rpm=None if rpm == "전체" else rpm,
            model=models.get(model),
            decision={"정상": 0, "이상": 1}.get(decision),
        )
        if not rows:
            st.info("검색 조건에 해당하는 저장 이력이 없습니다.")
            return
        st.caption(
            f"분석 {len({row['analysis_id'] for row in rows})}건 · 모델 결과 {len(rows)}개"
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.download_button(
            "검색 결과 CSV 다운로드",
            history_csv(rows),
            file_name="analysis_history.csv",
            mime="text/csv",
            key="history_csv",
        )
        selected = st.selectbox(
            "상세 조회할 분석 ID",
            list(dict.fromkeys(row["analysis_id"] for row in rows)),
            key="history_detail",
        )
        detail = store().detail(selected)
        with st.expander(f"분석 #{selected} 상세", expanded=True):
            st.json(detail["analysis"])
            st.dataframe(pd.DataFrame(detail["models"]), hide_index=True)
            st.caption(
                "상세에는 검색 조건으로 일부 모델만 표시된 경우에도 해당 분석 건의 모든 저장 모델 결과를 표시합니다. 모델 버전은 저장 모델 파일의 SHA256입니다."
            )
    except Exception:
        st.error(
            "분석 이력을 읽을 수 없습니다. DB 경로와 읽기·쓰기 권한을 확인해 주세요."
        )
