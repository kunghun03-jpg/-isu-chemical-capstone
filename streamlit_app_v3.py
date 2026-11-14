"""V3 reference-model dashboard. V2 and operational AE/RF remain unchanged."""
import hashlib
import json
import os
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
import streamlit_app_v2 as v2
from vibration.dashboard_v2 import configuration,history_store,bundle_state
from vibration.dashboard_view import CSS,WAVE_ICON,app_status,analyzed_time,ae_card,candidate_card,features_card,visualization_row,summary_card
from vibration.dashboard_pages import render_history_page,render_reports,render_settings
from vibration.dashboard_v3 import references,rotation_orders,save_references,read_references,MODELS,analyze_selected
from vibration.dae_ui import bundle_status
from vibration.web_inference import preview_mat,error_message,upload_name
from vibration.candidate_analysis import analyze_with_engine
from vibration.candidate_engine import bundle_identity

PAGES=('홈 대시보드','데이터 업로드','분석 이력','통계 및 리포트','모델 성능 비교','시스템 설정')


def header(state,result,error=None):
    with st.container(key='panel_topbar'):
        title,status,settings=st.columns([2.4,1.5,.6],vertical_alignment='center')
        with title:
            st.markdown('<div class="top-title">Rotating Machinery<br>Fault Diagnosis System · V3</div>',unsafe_allow_html=True)
        with status:
            text,ready=app_status(state,result,error)
            st.markdown('<div class="top-status">'+text+'<br>'+analyzed_time(result)+' · KST</div>',unsafe_allow_html=True)
        with settings:
            st.button('⚙ 설정',key='v3_settings',on_click=v2.navigate,args=('시스템 설정',))


def input_panel(config):
    with st.container(key='v2_input_card'):
        st.subheader('진동 데이터 분석 · V3')
        a,b,c=st.columns([3,1,1],vertical_alignment='bottom')
        with a: upload=st.file_uploader('MAT 파일 업로드',type=['mat'],key='v3_upload')
        with b: rpm=st.selectbox('측정 RPM',(900,1200,1800),key='v2_rpm')
        raw=upload.getvalue() if upload is not None else None
        folder=config.bundle_root/f'{rpm}rpm'
        stamps=tuple(sorted((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file())) if folder.is_dir() else ()
        identity=(hashlib.sha256(raw).hexdigest() if raw else None,rpm,config,stamps,bundle_identity(),bundle_status(rpm))
        if st.session_state.get('v3_input_identity')!=identity:
            v2.clear_result();st.session_state.pop('v3_rows',None)
        st.session_state['v3_input_identity']=identity
        preview=None
        if raw:
            try: preview=preview_mat(raw)
            except Exception as error: st.error('입력 오류: '+error_message(error))
        with c:
            run=st.button('▶ 분석 시작',key='v3_run',type='primary',disabled=preview is None or 'v2_result' in st.session_state)
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
        if 'v2_result' in st.session_state: st.caption('동일 입력의 중복 실행을 막았습니다. 파일 또는 RPM 변경 시 새 분석이 가능합니다.')
        st.caption((upload_name(upload.name)+' · ' if upload else '')+'CH3 · 65,536 samples · 25,600Hz · 원본은 DB에 저장하지 않습니다.')
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


def main():
    st.set_page_config(page_title='V3 다중 AI 진동 진단',page_icon='◈',layout='wide')
    st.markdown(CSS+'<style>[data-testid=stSidebar] [data-testid=stCaptionContainer] p{color:#c6d7f0!important}.top-title{font-size:17px}.st-key-panel_models{min-width:0}</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">'+WAVE_ICON+'<div>Rotating Machinery<br>Fault Diagnosis System · V3</div></div>',unsafe_allow_html=True)
        page=st.radio('작업 화면',PAGES,key='v2_page',label_visibility='collapsed')
        st.markdown('<div class="sidebar-note">MAT · 65,536샘플<br>25,600Hz · CH3<br>900 / 1200 / 1800RPM<br><br>로컬 운영 화면<br>관리자 로그인 기능 없음</div>',unsafe_allow_html=True)
    try:
        config=configuration()
        if config.model!='autoencoder': raise ValueError('운영 기준은 autoencoder여야 합니다.')
    except Exception as error:
        st.error(str(error));return
    if page not in ('홈 대시보드','데이터 업로드'):
        header(bundle_state(config,st.session_state.get('v2_rpm',900)),st.session_state.get('v2_result'),st.session_state.get('v2_analysis_error'))
        if page=='분석 이력':
            render_history_page()
            selected=st.session_state.get('v2_history_detail')
            if selected:
                try:
                    rows=read_references(history_store(),selected)
                    if rows:
                        st.subheader('저장된 추가 모델 참고 결과');model_table(rows)
                    else: st.info('기존 V2 기록: 추가 모델 참고 결과 없음')
                except Exception as error: st.error(str(error))
        elif page=='통계 및 리포트': render_reports()
        elif page=='모델 성능 비교': performance()
        else:
            render_settings(config,None)
            st.write('DAE 상태:',bundle_status(st.session_state.get('v2_rpm',900)))
        return
    top=st.container()
    preview,rpm,state=input_panel(config)
    result=st.session_state.get('v2_result');candidate=st.session_state.get('v2_candidate');details=st.session_state.get('v2_signal_details')
    rows=st.session_state.get('v3_rows')
    with top:
        header(state,result,st.session_state.get('v2_analysis_error'))
        st.caption('V3 · AE 운영 기준 / 추가 모델 참고 비교 · 준비 상태는 모델별 실제 실행 결과에서 확인합니다.')
    with st.container(key='v2_results_row'):
        a,b,c=st.columns([1,1.4,1.2],gap='small')
        with a,st.container(key='panel_ae'):
            st.subheader('종합 진단 · 운영 기준 AE');ae_card(result)
        with b,st.container(key='panel_models'):
            st.subheader('모델별 이상 탐지 결과')
            if rows:model_table(rows)
            else: st.info('분석 대기 · 실제 모델 결과 없음')
        with c,st.container(key='panel_candidate'):
            candidate_card(candidate)
            if candidate and candidate.get('engine'): st.caption('RF 모델 버전: '+candidate['engine']['model_version'])
    with st.container(key='panel_features'):features_card(candidate)
    visualization_row(preview,details,candidate,rpm)
    if preview:
        with st.expander('시간 파형 확대 · 드래그/휠 탐색'):
            frame=pd.DataFrame({'시간 (초)':preview['time'],'CH3 진폭 (raw)':preview['signal']})
            st.altair_chart(alt.Chart(frame).mark_line(color='#2478ff').encode(x='시간 (초):Q',y='CH3 진폭 (raw):Q').interactive(),width='stretch')
    if details:
        with st.container(key='panel_orders_v3'):
            st.subheader('실측 스펙트럼 회전 성분 · 1X–5X')
            orders=rotation_orders(details,rpm)
            st.altair_chart(alt.Chart(orders).mark_bar(color='#8257df').encode(x='회전 성분:N',y='진폭 (raw):Q',tooltip=list(orders.columns)),width='stretch')
            st.dataframe(orders,hide_index=True,width='stretch')
            st.caption('명목 RPM/60 × 차수 · ±1Hz 피크 탐색 (bin 해상도 고려) · 결함 진단 기준 아님')
    with st.container(key='panel_summary'):
        st.subheader('분석 결과 요약')
        a,b,c,d=st.columns(4)
        with a:summary_card('데이터 정보',[('파일',result['input']['name'] if result else '대기'),('RPM',rpm)],'panel_v3_data')
        with b:summary_card('모델별 판정',[(r['label'],r['decision']) for r in rows] if rows else [('상태','대기')],'panel_v3_models')
        with c:summary_card('RF 참고 예측',[('원시 예측',(candidate.get('engine') or {}).get('raw_prediction','실행 결과 없음') if candidate else '대기'),('최종 후보',candidate['candidate'] if candidate else '대기')],'panel_v3_rf')
        with d:summary_card('판정 근거',[('운영 기준','AE 저장 임계값'),('보류 사유',' / '.join(candidate.get('reasons',[])) if candidate else '대기')],'panel_v3_reason')
    with st.container(key='panel_actions'):
        st.subheader('결과 저장 및 이력')
        if st.button('결과 저장하기',key='v3_save',disabled=not result or not rows,type='primary'):
            try:
                ident,created=save_references(history_store(),result,st.session_state['v2_versions'],candidate,rows)
                st.success(f'분석 #{ident} 저장 완료' if created else f'기존 분석 #{ident} 확인 완료')
            except Exception as error: st.error('저장 실패 (AE 기록만 저장됐을 수 있습니다): '+str(error))
        st.button('분석 이력 조회 →',on_click=v2.navigate,args=('분석 이력',))
        if result:
            st.download_button('분석 결과 JSON',json.dumps({**result,'candidate_analysis':candidate,'reference_models':rows},ensure_ascii=False,indent=2,allow_nan=False),file_name='v3_vibration_result.json',mime='application/json')
    v2.details_panel(result,candidate,preview,details)

if __name__=='__main__': main()
