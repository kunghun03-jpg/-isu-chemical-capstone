"""Responsive V2 dashboard. Inference, candidate policy and storage APIs are unchanged."""
import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from vibration.dashboard_v2 import configuration, bundle_state, analyze, history_store
from vibration.detail_ui import spectrum_frame, segment_frame
from vibration.web_inference import preview_mat, upload_name, error_message
from vibration.candidate_analysis import STAGE_VERSION as CANDIDATE_VERSION, analyze_with_engine
from vibration.candidate_engine import bundle_identity as candidate_bundle_identity
from vibration.candidate_ui import render_candidate
from vibration.candidate_decision_ui import decision_view
from vibration.dashboard_pages import render_history_page, render_reports, render_settings
from vibration.dashboard_view import (CSS, WAVE_ICON, ae_card, analyzed_time, candidate_card,
    chart, features_card, header, navigate, number, summary_card, visualization_row)

PAGES = ('홈 대시보드', '데이터 업로드', '분석 이력', '통계 및 리포트', '시스템 설정')


def clear_result():
    for key in ('v2_result', 'v2_versions', 'v2_candidate', 'v2_signal_details', 'v2_save_notice'):
        st.session_state.pop(key, None)


def input_panel(config):
    with st.container(key='v2_input_card'):
        intro, uploader, speed, action = st.columns([1.5, 2., .8, .85], gap='medium', vertical_alignment='center')
        with intro:
            st.markdown('<div class="section-kicker">MAT FILE ANALYSIS</div>', unsafe_allow_html=True)
            st.subheader('진동 데이터 분석')
            st.markdown('<div class="input-intro">MAT 파일을 업로드해 AI 기반 이상 징후와 결함 후보의 근거를 확인하세요.</div>', unsafe_allow_html=True)
        with uploader:
            upload = st.file_uploader('MAT 파일 업로드', type=['mat'], key='v2_upload')
        with speed:
            rpm = st.selectbox('측정 RPM', (900, 1200, 1800), key='v2_rpm')
            st.caption('측정 당시 회전수')
        state = bundle_state(config, rpm)
        raw = upload.getvalue() if upload is not None else None
        identity = (hashlib.sha256(raw).hexdigest() if raw else None, rpm,
                    upload_name(upload.name) if upload else None, config, state['stamp'], CANDIDATE_VERSION,
                    candidate_bundle_identity())
        if st.session_state.get('v2_identity') != identity:
            clear_result()
        st.session_state['v2_identity'] = identity
        preview = None
        if upload is not None:
            try:
                preview = preview_mat(raw)
            except Exception as error:
                st.error('파일을 확인해 주세요: ' + error_message(error))
        if not state['files_present']:
            st.warning('분석기 파일이 준비되지 않았습니다. 시스템 설정의 경로를 확인하세요.')
        with action:
            if st.button('▶ 분석 시작',key='v2_run',type='primary',width='stretch',disabled=preview is None or not state['files_present']):
                clear_result()
                try:
                    with st.spinner('AE와 진동 특징을 분석하고 있습니다…'):
                        result, versions = analyze(raw, rpm, upload_name(upload.name), config)
                        candidate, details = analyze_with_engine(raw, rpm, result)
                    st.session_state['v2_result'], st.session_state['v2_versions'] = result, versions
                    st.session_state['v2_candidate'], st.session_state['v2_signal_details'] = candidate, details
                except Exception as error:
                    st.error('분석을 완료하지 못했습니다: ' + error_message(error))
        if preview:
            st.caption(f"입력 확인 · {upload_name(upload.name)} · {preview['samples']:,} 샘플 · {preview['duration']:.2f}초 · CH3 · 원본은 저장하지 않습니다.")
        else:
            st.caption('지원: .mat · signals=(6,65536) · fs=25,600Hz · CH3만 분석 · 최대 20MB')
    return preview, rpm, state


def render_actions(result, candidate):
    with st.container(key='panel_actions'):
        st.subheader('결과 저장 및 이력')
        notice = st.session_state.pop('v2_save_notice',None)
        if notice:
            (st.success if notice[0] else st.info)(notice[1])
        if st.button('결과 저장하기',key='v2_save',type='primary',width='stretch',disabled=result is None):
            try:
                record_id, created = history_store().save(result,st.session_state['v2_versions'],candidate_report=candidate)
                st.session_state['v2_save_notice'] = (created, f'분석 #{record_id}을 저장했습니다.' if created else f'이미 저장된 분석 #{record_id}입니다.')
                st.rerun()
            except Exception:
                st.error('결과를 저장하지 못했습니다. 저장 경로·권한·디스크 공간을 확인해 주세요.')
        st.button('분석 이력 조회 →',key='v2_go_history',on_click=navigate,args=('분석 이력',),width='stretch')
        if result:
            st.download_button('분석 결과 JSON',json.dumps({**result,'candidate_analysis':candidate},ensure_ascii=False,indent=2,allow_nan=False),
                               file_name='vibration_result.json',mime='application/json',key='v2_result_json',width='stretch')
        st.caption('AE와 후보 근거를 함께 저장합니다. 원본 MAT·파형은 DB에 저장하지 않습니다.')


def summary(result, candidate):
    item = result['models']['autoencoder'] if result else None
    with st.container(key='v2_summary_row'):
        content, actions = st.columns([4.,1.05],gap='small')
        with content, st.container(key='panel_summary'):
            st.subheader('분석 결과 요약')
            a,b,c,d = st.columns(4,gap='small')
            with a:
                summary_card('1. 데이터 정보',[('파일',result['input']['name'] if result else '업로드 후 분석'),
                    ('RPM',result['input']['rpm'] if result else '—'),('분석 시각 (KST)',analyzed_time(result))],'panel_summary_data')
            with b:
                summary_card('2. AE 이상 탐지',[('상태',('이상' if item['predicted_anomaly'] else '정상') if item else '분석 대기'),
                    ('점수',number(item['file_score']) if item else '—'),('임계값',number(item['file_threshold']) if item else '—')],'panel_summary_ae')
            with c:
                summary_card('3. RF 예측과 최종 후보',[('모델 참고 예측',decision_view(candidate)['raw_label'] if candidate else '분석 대기'),
                    ('최종 후보',candidate['candidate'] if candidate else '분석 대기'),
                    ('분석 상태',{'candidate':'후보 제시','abstained':'판정 보류','skipped_normal':'정상 · 추정 생략','unclear':'근거 부족','unavailable':'계산 불가'}.get(candidate['status'],candidate['status']) if candidate else '대기'),
                    ('보류·생략 사유',' / '.join(candidate.get('reasons',[])) if candidate else 'AE 분석 후 확인')],'panel_summary_candidate')
            with d:
                summary_card('4. 판단 근거',[('관측 특징','CH3 · RMS · 1X/2X · 주파수 대역' if candidate and candidate.get('features') else '실제 특징 계산 전'),
                    ('해석 한계','관측 특징은 결함 원인을 확정하지 않습니다.'),('상세','하단 상세 분석에서 전체 근거 확인')],'panel_summary_evidence')
        with actions:
            render_actions(result,candidate)


def details_panel(result,candidate,preview,details):
    with st.expander('상세 분석 · 원본 그래프와 전체 판단 근거'):
        show = st.toggle('상세 그래프와 근거 표시',value=False,key='v2_show_details')
        wave,fft,ae = st.tabs(('CH3 시간 파형','FFT 진폭 스펙트럼','AE 입력 스펙트럼'))
        if show and preview is not None:
            with wave:
                st.line_chart(pd.DataFrame({'시간 (초)':preview['time'],'CH3 진폭':preview['signal']}),x='시간 (초)',y='CH3 진폭',color='#2478ff')
                st.caption('전체 65,536개 실제 CH3 샘플 · 센서 raw 단위')
            with fft:
                if details is None:
                    st.info('전체 FFT는 분석 후 확인할 수 있습니다.')
                else:
                    frame = pd.DataFrame({'주파수 (Hz)':details['frequency'],'FFT 진폭':details['amplitude']})
                    st.altair_chart(chart(frame,'주파수 (Hz):Q','FFT 진폭:Q',height=260),width='stretch')
                    st.caption('전체 0–10kHz bin · 축약 없음 · 대칭 Hann 단측 진폭')
            with ae:
                st.line_chart(spectrum_frame(preview),x='주파수 (Hz)',y='8구간 log1p 평균',color='#2478ff')
                st.caption('기존 AE 입력 · 8구간 log1p 평균 · Train Z-score 전')
            if result:
                render_candidate(candidate)
                with st.expander('구간별 분석 근거'):
                    item = result['models']['autoencoder']
                    frame = segment_frame(item)
                    graph = alt.Chart(frame).mark_line(point=True).encode(x='구간:O',y='구간 점수:Q')
                    rule = alt.Chart(pd.DataFrame({'기준':[item['segment_threshold']]})).mark_rule(color='#de8f34',strokeDash=[4,4]).encode(y='기준:Q')
                    st.altair_chart(graph+rule,width='stretch')
                    st.caption('구간 임계값은 파일 판정 기준과 다릅니다.')
                with st.expander('분석기 정보'):
                    st.code(st.session_state['v2_versions']['autoencoder'],language=None)
        else:
            st.caption('표시를 켜면 원본 전체 파형·FFT·AE 스펙트럼과 후보의 전체 근거를 볼 수 있습니다.')


def main():
    st.set_page_config(page_title='회전기기 AI 진동 진단',page_icon='◈',layout='wide',initial_sidebar_state='auto')
    st.markdown(CSS,unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">'+WAVE_ICON+'<div>Rotating Machinery<br>Fault Diagnosis System</div></div>',unsafe_allow_html=True)
        page = st.radio('작업 화면',PAGES,key='v2_page',label_visibility='collapsed')
        st.markdown('<div class="sidebar-note"><strong>지원 포맷</strong><br>.mat 파일 · 6채널 입력<br>✓ CH3 분석 · 25,600Hz<br>900 / 1200 / 1800 RPM<br><small>파일 기반 분석 · 실시간 수집 아님</small></div>',unsafe_allow_html=True)
    config, error = None, None
    try:
        config = configuration()
        if config.model != 'autoencoder':
            raise ValueError('V2는 AE만 사용합니다. ISU_CHEMICAL_V2_MODEL=autoencoder로 설정하세요.')
    except ValueError as problem:
        error = problem
        config = None
        clear_result()
    top = st.container()
    if page in ('분석 이력','통계 및 리포트','시스템 설정'):
        with top:
            header(bundle_state(config,st.session_state.get('v2_rpm',900)) if config else None,st.session_state.get('v2_result'))
        {'분석 이력':render_history_page,'통계 및 리포트':render_reports,'시스템 설정':lambda:render_settings(config,error)}[page]()
        return
    if config is None:
        with top:
            header(None,None)
        st.error('분석기 설정을 확인해 주세요. 관리자 설정이 유효하지 않습니다.')
        st.write(str(error))
        return
    if page == '데이터 업로드':
        st.subheader('데이터 업로드')
        st.caption('지원 형식을 확인한 뒤 측정 당시 RPM으로 분석합니다. 파일명으로 결함 종류를 판단하지 않습니다.')
    preview,rpm,state = input_panel(config)
    result,candidate = st.session_state.get('v2_result'),st.session_state.get('v2_candidate')
    with top:
        header(state,result)
    with st.container(key='v2_results_row'):
        ae,fault,features = st.columns([1.,1.18,1.03],gap='small')
        with ae,st.container(key='panel_ae'):
            ae_card(result)
        with fault,st.container(key='panel_candidate'):
            candidate_card(candidate)
        with features,st.container(key='panel_features'):
            features_card(candidate)
    details = st.session_state.get('v2_signal_details')
    visualization_row(preview,details,candidate,rpm)
    summary(result,candidate)
    details_panel(result,candidate,preview,details)
    st.markdown('<div class="research-note">연구 시제품 · AE 정상 오탐 및 후보 분류의 일반화 한계가 있습니다. 분석 결과와 실제 설비 상태를 함께 점검하세요.</div>',unsafe_allow_html=True)


if __name__ == '__main__':
    main()
