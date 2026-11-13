"""User-facing rotating equipment vibration dashboard; original app is unchanged."""

from datetime import datetime
import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from vibration.dashboard_v2 import configuration, bundle_state, analyze, history_store, group_history
from vibration.detail_ui import spectrum_frame, segment_frame
from vibration.history import KST, history_csv
from vibration.web_inference import preview_mat, upload_name, error_message, MODEL_LABELS
from vibration.candidate_analysis import VERSION as CANDIDATE_VERSION, safe_candidate_analysis
from vibration.candidate_history import read_candidate
from vibration.candidate_ui import render_candidate

CSS = """
<style>
.stApp { background: #f4f7fb; }
.block-container { max-width: 1280px; padding-top: 4.5rem; }
[data-testid="stSidebar"] { background: #122338; }
[data-testid="stSidebar"] * { color: #e4edf7; }
[data-testid="stMetric"] { background: white; border: 1px solid #dce4ee;
 border-radius: 12px; padding: 16px 18px; }
[data-testid="stMetricLabel"] { color: #526477; }
[data-testid="stMetricValue"] { font-size: 1.35rem; }
.st-key-v2_input_card, .st-key-v2_result_card { background: white; border-radius: 12px; }
[data-testid="stButton"] button[kind="primary"] { background: #19699a; border-color: #19699a; color: white; }
.eyebrow { color: #357bba; letter-spacing: .16em; font-size: .75rem; font-weight: 700; }
.hero-title { color: #13283f; font-size: 2rem; font-weight: 750; margin: .35rem 0; }
.hero-copy { color: #5f7184; margin-bottom: 1.5rem; }
</style>
"""


def history_table(groups):
    return pd.DataFrame([{
        "분석 ID": r["analysis_id"], "분석 시각 (KST)": datetime.fromisoformat(r["analyzed_at"]).astimezone(KST).strftime("%Y-%m-%d %H:%M"),
        "파일": r["filename"], "RPM": r["rpm"], "상태": "이상 포함" if r["alarms"] else "정상",
        "저장 결과 수": r["results"],
    } for r in groups])


def render_history_page():
    st.subheader("분석 이력")
    st.caption("저장한 측정 결과를 날짜·운전 속도·판정으로 조회합니다. 날짜는 한국 시간 기준입니다.")
    left, middle, right = st.columns(3)
    with left:
        start = st.date_input("시작 날짜", value=None, key="v2_history_start")
        end = st.date_input("종료 날짜", value=None, key="v2_history_end")
    with middle:
        rpm = st.selectbox("운전 속도", ("전체", 900, 1200, 1800), key="v2_history_rpm")
    with right:
        decision = st.selectbox("판정", ("전체", "정상", "이상 포함"), key="v2_history_decision")
    if start and end and start > end:
        st.error("종료 날짜는 시작 날짜 이후로 선택해 주세요.")
        return
    try:
        rows = history_store().search(start=start, end=end, rpm=None if rpm == "전체" else rpm)
        groups = group_history(rows)
        if decision != "전체":
            groups = [g for g in groups if bool(g["alarms"]) == (decision == "이상 포함")]
        ids = {g["analysis_id"] for g in groups}
        if not groups:
            st.info("조건에 맞는 저장 기록이 없습니다.")
            return
        st.dataframe(history_table(groups), hide_index=True, width="stretch")
        st.caption("기존 다중 분석기 기록은 하나라도 이상 판정이 있으면 ‘이상 포함’으로 요약합니다.")
        st.download_button("조회 결과 CSV 다운로드", history_csv([r for r in rows if r["analysis_id"] in ids]),
                           file_name="inspection_history.csv", mime="text/csv", key="v2_history_csv")
        selected = st.selectbox("상세 기록", [g["analysis_id"] for g in groups], key="v2_history_detail")
        detail = history_store().detail(selected)
        with st.expander("기록 세부정보"):
            st.write(f"파일: {detail['analysis']['filename']} · {detail['analysis']['rpm']} RPM")
            st.dataframe(pd.DataFrame([{
                "분석기": MODEL_LABELS.get(r["model"], r["model"]), "이상 점수": r["score"],
                "임계값": r["threshold"], "판정": "이상" if r["predicted_anomaly"] else "정상",
                "버전": r["model_version"],
            } for r in detail["models"]]), hide_index=True)
            st.caption("기록 추적용 정보입니다. 서로 다른 분석기의 점수 크기는 직접 비교하지 마세요.")
            candidate = read_candidate(history_store(), selected)
            render_candidate(candidate)
            if candidate is not None:
                st.download_button('저장된 후보 근거 JSON 다운로드',
                                   json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False),
                                   file_name='candidate_evidence.json', mime='application/json', key='v2_history_candidate_json')
    except Exception:
        st.error("이력을 읽지 못했습니다. 저장 경로와 접근 권한을 확인해 주세요.")


def render_result(result, versions):
    name, item = next(iter(result["models"].items()))
    abnormal = bool(item["predicted_anomaly"])
    st.subheader("1. AE 이상 탐지 결과")
    notice = st.session_state.pop("v2_save_notice", None)
    if notice:
        (st.success if notice[0] else st.info)(notice[1])
    if abnormal:
        st.warning("이상 · 측정 조건과 설비 상태를 점검해 주세요.")
    else:
        st.success("정상 · 이번 측정의 이상 점수가 저장 기준을 넘지 않았습니다.")
    a, b, c = st.columns(3)
    a.metric("판정", "이상" if abnormal else "정상")
    b.metric("이상 점수", f"{item['file_score']:.6g}")
    c.metric("판정 기준", f"{item['file_threshold']:.6g}")
    st.caption("점수가 기준을 초과하면 이상입니다. 파일 점수는 8개 구간의 평균이며 고장 종류나 설비 안전을 확정하지 않습니다.")
    candidate = st.session_state.get('v2_candidate')
    st.divider()
    render_candidate(candidate)
    st.divider()
    if st.button("이 결과 저장", key="v2_save", type="primary"):
        try:
            record_id, created = history_store().save(result, versions, candidate_report=candidate)
            if created:
                st.session_state["v2_save_notice"] = (True, f"분석 #{record_id}을 저장했습니다.")
            else:
                st.session_state["v2_save_notice"] = (False, f"이미 저장된 분석 #{record_id}입니다.")
            st.rerun()
        except Exception:
            st.error("결과를 저장하지 못했습니다. 저장 경로·권한·디스크 공간을 확인해 주세요.")
    st.caption('저장 시 AE 판정과 후보 특징·근거를 함께 기록합니다. 원본 MAT와 파형은 DB에 저장하지 않습니다.')
    st.download_button("분석 결과 다운로드", json.dumps({**result, 'candidate_analysis': candidate}, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name="vibration_result.json", mime="application/json", key="v2_result_json")
    with st.expander("구간별 분석 근거"):
        frame = segment_frame(item)
        chart = alt.Chart(frame).mark_line(point=True, color="#247cb6").encode(
            x=alt.X("구간:O"), y=alt.Y("구간 점수:Q", scale=alt.Scale(zero=False)))
        rule = alt.Chart(pd.DataFrame({"기준": [item["segment_threshold"]]})).mark_rule(
            color="#cf654b", strokeDash=[5, 4]).encode(y="기준:Q")
        st.altair_chart(chart + rule, width="stretch")
        st.caption("구간 기준과 파일 판정 기준은 다릅니다. 일부 구간의 초과만으로 파일 판정을 바꾸지 않습니다.")
    with st.expander("분석기 정보"):
        st.write('이상 탐지 모델: 1D Convolutional AutoEncoder')
        st.caption("기존 AE 번들의 전처리·가중치·임계값을 그대로 사용합니다. 결함 후보를 분류하는 모델과는 별개입니다.")
        st.code(versions[name], language=None)


def main():
    st.set_page_config(page_title="회전체 진동 이상 탐지", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ◈ VIBRATION")
        st.caption("회전체 상태 점검")
        page = st.radio("작업 화면", ("대시보드", "분석 이력"), key="v2_page", label_visibility="collapsed")
        st.divider()
        st.caption("측정 파일 기반 점검\n\nCH3 · 25,600 Hz\n\n900 / 1200 / 1800 RPM")
    st.markdown('<div class="eyebrow">ROTATING EQUIPMENT · CONDITION MONITORING</div>'
                '<h1 class="hero-title">회전체 진동 이상 탐지</h1>'
                '<div class="hero-copy">측정 데이터를 확인하고, 저장 기준에 따라 진동 이상 징후를 점검합니다.</div>',
                unsafe_allow_html=True)
    st.info("연구 시제품 · AE 이상 탐지와 결함 후보 분석은 별개입니다. 정상 오탐이 보고되어 파형·측정 조건도 함께 확인하세요.")
    if page == "분석 이력":
        render_history_page()
        return
    try:
        groups = group_history(history_store().search())
        counts = len(groups), sum(bool(g["alarms"]) for g in groups)
    except Exception:
        counts = "—", "—"
        st.warning("저장 이력을 읽지 못했습니다. 현재 파일 분석과는 별도로 저장 경로를 확인해 주세요.")
    one, two, three = st.columns(3)
    one.metric("저장된 점검", counts[0])
    two.metric("이상 포함 기록", counts[1])
    three.metric("측정 방식", "MAT 파일")
    try:
        config = configuration()
        if config.model != 'autoencoder':
            raise ValueError('발표용 V2는 AE만 사용합니다. ISU_CHEMICAL_V2_MODEL=autoencoder로 설정하세요.')
    except ValueError as error:
        st.session_state.pop("v2_result", None)
        st.session_state.pop("v2_versions", None)
        st.session_state.pop("v2_save_notice", None)
        st.session_state.pop('v2_candidate', None)
        st.session_state.pop('v2_signal_details', None)
        st.error("분석기 설정을 확인해 주세요. 관리자 설정이 유효하지 않습니다.")
        with st.expander("설정 진단"):
            st.write(str(error))
        return
    inputs, results = st.columns([1, 1.8], gap="large")
    with inputs:
        with st.container(border=True, key="v2_input_card"):
            st.subheader("새 측정 분석")
            st.caption("1. 파일 업로드 → 2. 측정 RPM 선택 → 3. 분석 실행")
            upload = st.file_uploader("진동 측정 MAT", type=["mat"], key="v2_upload")
            rpm = st.selectbox("측정 RPM", (900, 1200, 1800), key="v2_rpm")
            st.caption("측정 당시 RPM을 선택하세요. 파일에서 운전 속도를 자동 측정하지 않습니다.")
            state = bundle_state(config, rpm)
            st.caption(state["message"])
            raw = upload.getvalue() if upload is not None else None
            identity = (hashlib.sha256(raw).hexdigest() if raw else None, rpm,
                        upload_name(upload.name) if upload else None, config, state["stamp"], CANDIDATE_VERSION)
            if st.session_state.get("v2_identity") != identity:
                st.session_state.pop("v2_result", None)
                st.session_state.pop("v2_versions", None)
                st.session_state.pop("v2_save_notice", None)
                st.session_state.pop('v2_candidate', None)
                st.session_state.pop('v2_signal_details', None)
            st.session_state["v2_identity"] = identity
            preview = None
            if upload is not None:
                try:
                    preview = preview_mat(raw)
                    st.success("입력 형식 확인 완료")
                    st.caption(f"{preview['samples']:,} 샘플 · {preview['duration']:.2f}초 · CH3")
                except Exception as error:
                    st.error("파일을 확인해 주세요: " + error_message(error))
            else:
                st.caption("지원 파일: 6채널 진동 MAT · 25,600 Hz · 최대 20MB")
            if not state["files_present"]:
                st.warning("분석기 파일이 준비되지 않았습니다. 관리자에게 문의해 주세요.")
            if st.button("분석 실행", key="v2_run", type="primary", width="stretch",
                         disabled=preview is None or not state["files_present"]):
                st.session_state.pop("v2_result", None)
                st.session_state.pop("v2_versions", None)
                st.session_state.pop('v2_candidate', None)
                st.session_state.pop('v2_signal_details', None)
                try:
                    with st.spinner("진동 이상 징후를 분석하고 있습니다…"):
                        result, versions = analyze(raw, rpm, upload_name(upload.name), config)
                        candidate, signal_details = safe_candidate_analysis(raw, rpm)
                    st.session_state["v2_result"], st.session_state["v2_versions"] = result, versions
                    st.session_state['v2_candidate'], st.session_state['v2_signal_details'] = candidate, signal_details
                except Exception as error:
                    st.error("분석을 완료하지 못했습니다: " + error_message(error))
            st.caption("원본 업로드는 저장하지 않습니다. 분석 결과는 저장 버튼을 눌렀을 때만 기록됩니다.")
    with results:
        with st.container(border=True, key="v2_result_card"):
            result = st.session_state.get("v2_result")
            if result is None:
                st.subheader("분석 대기")
                st.write("측정 파일을 업로드하고 분석을 실행하면 이곳에 판정과 점수가 표시됩니다.")
                st.caption("실시간 설비 상태가 아닌 업로드한 측정 파일의 분석 결과입니다.")
            else:
                render_result(result, st.session_state["v2_versions"])
    if preview is not None:
        st.subheader("진동 데이터")
        wave, spectrum, ae_spectrum = st.tabs(("CH3 시간 파형", "FFT 진폭 스펙트럼", "AE 입력 스펙트럼"))
        with wave:
            st.line_chart(pd.DataFrame({"시간 (초)": preview["time"], "CH3 진폭": preview["signal"]}),
                          x="시간 (초)", y="CH3 진폭", color="#247cb6")
            st.caption("원본 CH3 진폭입니다. 센서 물리 단위는 아직 확정되지 않았습니다.")
        with spectrum:
            details = st.session_state.get('v2_signal_details')
            if details is None:
                st.info('분석 실행 후 결함 후보 검토에 사용한 전체 기록 FFT가 표시됩니다.')
            else:
                frame = pd.DataFrame({'주파수 (Hz)': details['frequency'], 'FFT 진폭': details['amplitude']})
                display = st.radio('FFT 표시 범위', ('전체 0–10kHz', '저주파 0–200Hz'), horizontal=True, key='v2_fft_range')
                if display == '저주파 0–200Hz':
                    frame = frame[frame['주파수 (Hz)'] <= 200]
                # Embed values directly: whole-record FFT can exceed Altair's default 5000-row limit.
                chart = alt.Chart(alt.Data(values=frame.to_dict('records'))).mark_line(color='#247cb6').encode(
                    x=alt.X('주파수 (Hz):Q'), y=alt.Y('FFT 진폭:Q'))
                orders = pd.DataFrame({'주파수 (Hz)': [rpm / 60., rpm / 30.], '회전 성분': ['명목 1X', '명목 2X']})
                lines = alt.Chart(orders).mark_rule(color='#cc7850', strokeDash=[4, 4]).encode(
                    x='주파수 (Hz):Q', tooltip=['회전 성분', '주파수 (Hz)'])
                st.altair_chart(chart + lines, width='stretch')
                st.caption('전체 기록 평균 제거 · 대칭 Hann · 단측 진폭 보정 · 명목 1X/2X 표시. 선의 존재만으로 결함을 판정하지 않습니다.')
        with ae_spectrum:
            st.line_chart(spectrum_frame(preview), x="주파수 (Hz)", y="8구간 log1p 평균", color="#247cb6")
            st.caption("Hann FFT의 log1p 진폭 · 8구간 평균 · 학습 기준 Z-score 적용 전")


if __name__ == "__main__":
    main()
