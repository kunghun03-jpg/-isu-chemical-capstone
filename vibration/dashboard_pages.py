"""Working history, report and read-only settings pages over existing APIs."""
from datetime import datetime
import json

import pandas as pd
import streamlit as st

from vibration.candidate_engine import bundle_identity
from vibration.candidate_history import read_candidate
from vibration.candidate_ui import render_candidate
from vibration.dashboard_v2 import bundle_state, group_history, history_store
from vibration.history import KST, history_csv
from vibration.web_inference import MODEL_LABELS


def history_table(groups):
    return pd.DataFrame([{'분석 ID':r['analysis_id'],'분석 시각 (KST)':datetime.fromisoformat(r['analyzed_at']).astimezone(KST).strftime('%Y-%m-%d %H:%M'),
                         '파일':r['filename'],'RPM':r['rpm'],'상태':'이상 포함' if r['alarms'] else '정상','저장 결과 수':r['results']} for r in groups])


def render_history_page():
    with st.container(key='panel_history'):
        st.subheader('분석 이력')
        st.caption('저장한 분석을 조회합니다. 날짜는 분석 실행일(KST)이며 MAT 측정일과 다를 수 있습니다.')
        left, middle, right = st.columns(3)
        with left:
            start = st.date_input('시작 날짜',value=None,key='v2_history_start')
            end = st.date_input('종료 날짜',value=None,key='v2_history_end')
        with middle:
            rpm = st.selectbox('운전 속도',('전체',900,1200,1800),key='v2_history_rpm')
        with right:
            decision = st.selectbox('판정',('전체','정상','이상 포함'),key='v2_history_decision')
        if start and end and start > end:
            st.error('종료 날짜는 시작 날짜 이후로 선택해 주세요.')
            return
        try:
            rows = history_store().search(start=start,end=end,rpm=None if rpm == '전체' else rpm)
            groups = group_history(rows)
            if decision != '전체':
                groups = [g for g in groups if bool(g['alarms']) == (decision == '이상 포함')]
            if not groups:
                st.info('조건에 맞는 저장 기록이 없습니다.')
                return
            st.dataframe(history_table(groups),hide_index=True,width='stretch')
            ids = {g['analysis_id'] for g in groups}
            st.download_button('조회 결과 CSV 다운로드',history_csv([r for r in rows if r['analysis_id'] in ids]),
                               file_name='inspection_history.csv',mime='text/csv',key='v2_history_csv')
            st.caption('과거 다중 모델 기록은 하나라도 이상이면 ‘이상 포함’으로 집계합니다.')
            selected = st.selectbox('상세 기록',[g['analysis_id'] for g in groups],key='v2_history_detail')
            detail = history_store().detail(selected)
            with st.expander('기록 세부정보'):
                st.write(f"파일: {detail['analysis']['filename']} · {detail['analysis']['rpm']} RPM")
                st.dataframe(pd.DataFrame([{'분석기':MODEL_LABELS.get(r['model'],r['model']),'이상 점수':r['score'],
                    '임계값':r['threshold'],'판정':'이상' if r['predicted_anomaly'] else '정상','버전':r['model_version']} for r in detail['models']]),hide_index=True)
                try:
                    candidate = read_candidate(history_store(),selected)
                    render_candidate(candidate)
                    if candidate is not None:
                        st.download_button('저장된 후보 근거 JSON 다운로드',json.dumps(candidate,ensure_ascii=False,indent=2,allow_nan=False),
                                           file_name='candidate_evidence.json',mime='application/json',key='v2_history_candidate_json')
                except Exception:
                    st.warning('AE 이력은 조회했으나 후보 근거를 읽지 못했습니다. 후보 기록 무결성을 확인하세요.')
        except Exception:
            st.error('이력을 읽지 못했습니다. 저장 경로와 접근 권한을 확인해 주세요.')


def render_reports():
    with st.container(key='panel_reports'):
        st.subheader('통계 및 리포트')
        st.caption('실제 DB에 저장된 분석 건의 집계입니다. 모델 정확도·설비 가동률·실시간 감시 수치가 아닙니다.')
        try:
            rows = history_store().search()
            groups = group_history(rows)
        except Exception:
            st.error('저장 이력을 조회하지 못했습니다.')
            return
        if not groups:
            st.info('저장된 분석이 없습니다. 분석 결과를 저장하면 통계가 표시됩니다.')
            return
        a,b,c = st.columns(3)
        alerts = sum(bool(g['alarms']) for g in groups)
        a.metric('저장된 분석 건',len(groups)); b.metric('정상 기록',len(groups)-alerts); c.metric('이상 포함 기록',alerts)
        frame = history_table(groups)
        grouped = frame.groupby(['RPM','상태']).size().unstack(fill_value=0)
        st.bar_chart(grouped,color=['#20b982' if c=='정상' else '#f1a13c' for c in grouped.columns])
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('전체 리포트 CSV 다운로드',history_csv(rows),file_name='vibration_report.csv',mime='text/csv',key='v2_report_csv')
        st.caption('다중 모델 과거 기록은 분석 ID로 묶으며, 하나라도 이상이면 ‘이상 포함’입니다.')


def render_settings(config, error=None):
    with st.container(key='panel_settings'):
        st.subheader('시스템 설정')
        st.caption('현재 실행 설정의 읽기 전용 진단 화면입니다. 모델·임계값·환경변수를 이 화면에서 변경하지 않습니다.')
        if error:
            st.error(str(error))
        if config:
            st.write('이상 탐지 모델: 1D Convolutional AutoEncoder')
            st.code(str(config.bundle_root),language=None)
            states = [{'RPM':rpm,'파일 상태':'필요 파일 존재' if bundle_state(config,rpm)['files_present'] else '파일 미준비'} for rpm in (900,1200,1800)]
            st.dataframe(pd.DataFrame(states),hide_index=True)
            st.caption('파일 존재 확인이며 모델 버전·해시·실제 추론은 분석 실행 시 검사합니다.')
        folder, enabled, stamp = bundle_identity()
        st.write('후보 엔진 경로:',folder or '미설정')
        st.write('자동 후보명 표시 설정:','활성 요청 · 정책 통과 시만 표시' if enabled else '보류 모드')
        st.caption('후보 경로 등록은 성능 검증 완료를 뜻하지 않습니다. 상세 정책과 오류는 분석 결과에서 확인하세요.')
        st.write('이력 DB:')
        st.code(str(history_store().path),language=None)
        st.write('입력 계약: .mat / signals=(6,65536) / fs=25,600Hz / 분석 채널 CH3만 사용')
        st.button('설정 다시 확인',key='v2_refresh_settings')
