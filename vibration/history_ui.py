"""기존 화면 아래에 붙이는 저장·이력 UI."""

from pathlib import Path
import os

import pandas as pd
import streamlit as st

from vibration.history import HistoryStore, history_csv
from vibration.four_model_history import search_four_model_history
from vibration.dae_experimental_history import read_dae_experimental
from vibration.web_inference import MODEL_LABELS


def store():
    default = Path(__file__).resolve().parents[1] / "outputs/history/analyses.sqlite3"
    return HistoryStore(os.environ.get("ISU_CHEMICAL_HISTORY_DB", str(default)))


def save_result(result, versions, *, dae_result=None, raw=None, rpm=None):
    if st.button("결과 저장", key="save_analysis"):
        try:
            dae_payload = (
                (raw, rpm, dae_result)
                if dae_result is not None else None
            )
            analysis_id, created = store().save(
                result, versions,
                dae_experimental=dae_payload
            )
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
        "이력 모델", ("전체", "AE", "OCSVM", "IF", "DAE (실험용)"), key="history_model"
    )
    decision = cols[2].selectbox(
        "이력 판정", ("전체", "정상", "이상"), key="history_decision"
    )
    if start and end and start > end:
        st.error("시작 날짜는 종료 날짜보다 늦을 수 없습니다.")
        return
    try:
        models = {
            label: name
            for name, label in MODEL_LABELS.items()
        }
        models["DAE (실험용)"] = "dae_experimental"
        rows = search_four_model_history(
            store(),
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

        if any(
            row["model"] == "dae_experimental"
            for row in rows
        ):
            st.warning(
                "DAE 검색 결과는 날짜 기준 탐색적 실험입니다. "
                "물리적 Run 독립성과 운영 성능은 "
                "검증되지 않았습니다."
            )

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
        all_models = search_four_model_history(store())
        selected_models = [
            row for row in all_models
            if row["analysis_id"] == selected
        ]

        with st.expander(f"분석 #{selected} 상세", expanded=True):
            st.json(detail["analysis"])

            if selected_models:
                names = {
                    "autoencoder": "AE (기준)",
                    "ocsvm": "One-Class SVM",
                    "iforest": "Isolation Forest",
                    "dae_experimental": "DAE (실험용)",
                }
                display = [
                    {
                        "모델": names.get(r["model"], r["model"]),
                        "점수": r["score"],
                        "임계값": r["threshold"],
                        "판정": (
                            "이상" if r["predicted_anomaly"]
                            else "정상"
                        ),
                        "모델 버전": r["model_version"],
                    }
                    for r in selected_models
                ]
                st.markdown("**4개 모델 통합 저장 결과**")
                st.dataframe(
                    pd.DataFrame(display),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("저장된 모델 결과가 없습니다.")
            dae_saved = read_dae_experimental(store(), selected)
            if dae_saved is not None:
                st.markdown("**DAE 실험용 저장 결과**")
                st.dataframe(
                    pd.DataFrame([dae_saved]), hide_index=True
                )
                st.warning(
                    "DAE는 날짜 기준 탐색적 평가입니다. "
                    "운영 성능과 물리적 Run 독립성은 미검증입니다."
                )
            st.caption(
                "상세에는 검색 조건으로 일부 모델만 표시된 경우에도 해당 분석 건의 모든 저장 모델 결과를 표시합니다. 모델 버전은 저장 모델 파일의 SHA256입니다."
            )
    except Exception:
        st.error(
            "분석 이력을 읽을 수 없습니다. DB 경로와 읽기·쓰기 권한을 확인해 주세요."
        )
