"""Existing Streamlit dashboard; presentation changes only, same launch entry point."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit_app_v2 as v2
from vibration.dashboard_v2 import configuration,history_store,bundle_state
from vibration.dashboard_view import CSS,WAVE_ICON,candidate_card,features_card
from vibration.dashboard_pages import render_history_page,render_reports,render_settings
from vibration.dashboard_v3 import references,save_references,read_references,MODELS,analyze_selected
from vibration.dae_ui import bundle_status
from vibration.web_inference import preview_mat,error_message,upload_name
from vibration.candidate_analysis import analyze_with_engine
from vibration.candidate_engine import bundle_identity
from vibration.normal_comparison_ui import render_normal_comparison
from vibration.v3_presentation import (CSS as RESPONSIVE_CSS, dae_diagnostics, hero,
    overview, model_tiles, signal_panel, advice_panel, guide_page)

PAGES=('분석하기','분석 이력','모델 정보','사용 가이드')


def header(state,result,error=None):
    hero(state,result,error)


def input_panel(config):
    with st.container(key='v2_input_card'):
        st.subheader('❶ 분석 데이터 입력')
        upload=st.file_uploader('MAT 파일 업로드',type=['mat'],key='v3_upload')
        rpm=st.selectbox('측정 RPM',(900,1200,1800),key='v2_rpm')
        raw=upload.getvalue() if upload is not None else None
        folder=config.bundle_root/f'{rpm}rpm'
        stamps=tuple(sorted((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file())) if folder.is_dir() else ()
        identity=(hashlib.sha256(raw).hexdigest() if raw else None,upload_name(upload.name) if upload else None,rpm,config,stamps,bundle_identity(),bundle_status(rpm))
        if st.session_state.get('v3_input_identity')!=identity:
            v2.clear_result();st.session_state.pop('v3_rows',None)
        st.session_state['v3_input_identity']=identity
        preview=None
        if raw:
            try: preview=preview_mat(raw)
            except Exception as error: st.error('입력 오류: '+error_message(error))
        run=st.button('▶ 분석 시작',key='v3_run',type='primary',width='stretch',disabled=preview is None or 'v2_result' in st.session_state)
        if run and 'v2_result' not in st.session_state:
            v2.clear_result();st.session_state.pop('v3_rows',None)
            try:
                with st.spinner('AE 및 RF 분석 중…'):
                    result,versions=analyze_selected(raw,rpm,upload_name(upload.name),config)
                    candidate,details=analyze_with_engine(raw,rpm,result)
                st.session_state.update(v2_result=result,v2_versions=versions,v2_candidate=candidate,v2_signal_details=details)
                with st.spinner('추가 모델 참고 분석 중…'):
                    st.session_state['v3_rows']=references(raw,rpm,config,result)
                st.rerun()
            except Exception as error:
                st.session_state['v2_analysis_error']=error_message(error)
                st.error('분석 오류: '+error_message(error))
        if st.session_state.get('v2_analysis_error'): st.error('분석 오류: '+st.session_state['v2_analysis_error'])
        if 'v2_result' in st.session_state: st.caption('동일 입력의 중복 실행 방지 · 파일/RPM 변경 시 새 분석')
        st.caption('CH3 · 65,536샘플 · 25,600Hz · 원본은 저장하지 않습니다.')
    return preview,rpm,{'files_present':(folder/'manifest.json').is_file()}


def model_table(rows):
    names={'autoencoder':'AE (기준)','dae':'DAE','ocsvm':'OCSVM','iforest':'IF'}
    st.dataframe(pd.DataFrame([{'모델':names[r['model']],'점수':f"{r['score']:.8g}" if r['score'] is not None else '—',
        '판정':r['decision'] if r['status']=='완료' else r['status'],'임계값':f"{r['threshold']:.8g}" if r['threshold'] is not None else '—'} for r in rows]),hide_index=True,width='stretch')
    with st.expander('모델별 기준·버전·실행 상태'):
        for r in rows:
            st.write(r['label']+' · '+r['status']+' · '+r['rule'])
            st.caption('버전: '+str(r['version']))
    st.caption('AE만 운영 판정 기준입니다. 모델별 점수는 척도가 다르며 평균·신뢰도로 비교하지 않습니다.')
    for r in rows:
        if r.get('reason'): st.caption(r['label']+' · '+r['reason'])


def performance():
    st.subheader('모델 성능 비교 · 오프라인 자료')
    path=os.environ.get('ISU_CHEMICAL_V3_EVALUATION_CSV','')
    if not path:
        st.info('평가 자료 없음');return
    try:
        frame=pd.read_csv(Path(path).expanduser())
        if frame.empty:
            st.info('평가 자료 없음');return
        required={'model','rpm','date','n','split','metric','value'}
        if not required.issubset(frame.columns): raise ValueError('필수 열: '+', '.join(sorted(required)))
        if not set(frame['model']).issubset(set(MODELS)): raise ValueError('모델명 확인 필요')
        st.dataframe(frame,hide_index=True,width='stretch')
        st.caption('외부 오프라인 평가 기록입니다. 이 앱에서 한 파일을 분석한 결과나 검증된 운영 정확도를 뜻하지 않습니다. 분할·표본·모델 버전을 확인하세요.')
    except Exception as error: st.error('평가 자료를 읽지 못했습니다: '+str(error))


def history_page():
    render_history_page()
    selected=st.session_state.get('v2_history_detail')
    if selected:
        try:
            rows=read_references(history_store(),selected)
            if rows:
                st.subheader('저장된 추가 모델 참고 결과');model_table(rows)
            else: st.info('기존 V2 기록: 추가 모델 참고 결과 없음')
        except Exception as error: st.error(str(error))
    with st.expander('저장 이력 통계 및 리포트'):
        render_reports()


def actions(result,candidate,rows):
    with st.container(key='panel_actions'):
        st.subheader('결과 저장 및 이력')
        if st.button('결과 저장하기',key='v3_save',disabled=not result or not rows,type='primary',width='stretch'):
            try:
                ident,created=save_references(history_store(),result,st.session_state['v2_versions'],candidate,rows)
                st.success(f'분석 #{ident} 저장 완료' if created else f'기존 분석 #{ident} 확인 완료')
            except Exception as error: st.error('저장 실패 (AE 기록만 저장됐을 수 있습니다): '+str(error))
        st.button('분석 이력 조회 →',on_click=v2.navigate,args=('분석 이력',),width='stretch')
        if result:
            st.download_button('분석 결과 JSON',json.dumps({**result,'candidate_analysis':candidate,'reference_models':rows},ensure_ascii=False,indent=2,allow_nan=False),file_name='v3_vibration_result.json',mime='application/json',width='stretch')
        st.caption('기존 SQLite에 저장 · 원본 MAT와 파형은 저장하지 않습니다.')


def main():
    st.set_page_config(page_title='회전체 진동 이상 감지 시스템',page_icon='◈',layout='wide',initial_sidebar_state='auto')
    # Shared V2 presentation remains untouched; overrides are scoped to this existing entry point.
    st.html(CSS+RESPONSIVE_CSS)
    legacy_pages={'홈 대시보드':'분석하기','데이터 업로드':'분석하기','통계 및 리포트':'분석 이력','모델 성능 비교':'모델 정보','시스템 설정':'모델 정보'}
    if st.session_state.get('v2_page') in legacy_pages:
        st.session_state['v2_page']=legacy_pages[st.session_state['v2_page']]
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">'+WAVE_ICON+'<div>회전체 진동<br>이상 감지 시스템</div></div>',unsafe_allow_html=True)
        page=st.radio('작업 화면',PAGES,key='v2_page',label_visibility='collapsed')
        st.markdown('<div class="sidebar-note"><strong>ⓘ 시스템 안내</strong><br>진동 분석으로 설비 점검을 지원합니다. 결함을 확정 진단하는 장비가 아닙니다.<hr>MAT · CH3 · 25,600Hz<br>900 / 1200 / 1800 RPM<br>파일 기반 분석 · 실시간 수집 아님</div>',unsafe_allow_html=True)
    try:
        config=configuration()
        if config.model!='autoencoder': raise ValueError('운영 기준은 autoencoder여야 합니다.')
    except Exception as error:
        header(None,None,str(error));st.error(str(error));return
    if page!='분석하기':
        header(bundle_state(config,st.session_state.get('v2_rpm',900)),st.session_state.get('v2_result'),st.session_state.get('v2_analysis_error'))
        if page=='분석 이력': history_page()
        elif page=='모델 정보':
            with st.container(key='panel_model_info'):
                st.subheader('모델 정보 및 적용 제한')
                st.write('운영 판정은 AE 저장 임계값을 사용합니다. OCSVM·IF는 참고 비교이며 점수 척도가 다릅니다. DAE는 지원 범위와 성능 검증이 필요합니다.')
                render_settings(config,None)
                with st.expander('DAE 번들 진단 · 읽기 전용'): st.json(dae_diagnostics(st.session_state.get('v2_rpm',900)))
                performance()
                render_normal_comparison()
        else: guide_page()
        return
    top=st.container()
    with st.container(key='v3_workspace'):
        input_column,result_column=st.columns([1,3.15],gap='medium')
        with input_column:
            preview,rpm,state=input_panel(config)
        result=st.session_state.get('v2_result');candidate=st.session_state.get('v2_candidate');details=st.session_state.get('v2_signal_details');rows=st.session_state.get('v3_rows')
        with result_column,st.container(key='panel_results'):
            st.subheader('❷ 분석 결과 요약')
            overview(rpm,state,result,st.session_state.get('v2_analysis_error'))
            st.markdown('**모델별 판정 결과**')
            model_tiles(rows)
            with st.expander('모델별 점수·임계값·버전·실행 상태'):
                if rows: model_table(rows)
                else: st.info('분석 대기 · 실제 모델 결과 없음')
                st.caption('종합 판정은 기존 운영 기준인 AE 결과입니다. 모델 점수 평균·정규화·투표를 사용하지 않습니다.')
    with top: header(state,result,st.session_state.get('v2_analysis_error'))
    with st.container(key='v3_evidence_row'):
        signal_column,summary_column=st.columns([2.2,1],gap='medium')
        with signal_column:
            signal_panel(preview,details,candidate,rpm,result)
        with summary_column:
            advice_panel(result,candidate,rows)
            actions(result,candidate,rows)
    with st.expander('RF 참고 예측과 최종 후보 · 상세 근거',expanded=False):
        candidate_card(candidate)
        if candidate and candidate.get('engine'): st.caption('RF 모델 버전: '+candidate['engine']['model_version'])
        features_card(candidate)
    v2.details_panel(result,candidate,preview,details)

if __name__=='__main__': main()
