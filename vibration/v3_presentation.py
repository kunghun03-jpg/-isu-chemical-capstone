"""V3-only layout and artifact diagnostics; no model/policy changes."""
from pathlib import Path
from vibration.dae_readiness import probe as bundle_status



def dae_diagnostics(rpm):
    state=bundle_status(rpm)
    folder=Path(state['folder']) if state.get('folder') else None
    files={name:bool(folder and (folder/name).is_file()) for name in ('manifest.json','autoencoder_model.pth','normalization.npz','threshold.npz')}
    return {**state,'configured_folder':str(folder) if folder else '','files_present':files,
        'explanation':'모델 미준비는 bundle manifest 또는 필수 파일을 찾지 못한 상태입니다. legacy NPZ/PTH만 있는 폴더는 추론 번들이 아닙니다.' if state['label']=='모델 미준비' else state.get('reason',state['label'])}

# Loaded only by streamlit_app_v3.py. Shared V2 styles and behavior stay intact.
CSS = '''<style>
:root{--ink:#14263f;--muted:#4a607b;--line:#d8e3f1;--surface:#f3f6fb}
.stApp{background:#f3f6fb;color:#14263f}
.block-container{max-width:1920px;padding:0 20px 24px!important}
[data-testid="stMarkdownContainer"]:has(>div){margin-bottom:0!important}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]{gap:16px}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#1c2b40,#101f34 85%)!important;border-right:1px solid #283e59}
[data-testid="stSidebarContent"]{padding:0!important}
[data-testid="stSidebarUserContent"]{padding:28px 18px!important}
.sidebar-brand{width:auto!important;margin:0!important;padding:2px 0 34px;font-size:16px;line-height:1.45;white-space:normal;gap:12px}
.sidebar-brand svg{width:38px;height:38px}
.st-key-v2_page{width:100%!important;margin:0!important}
.st-key-v2_page [data-testid="stRadioOption"]{min-height:54px;padding:13px 14px!important;gap:8px;margin:3px 0;font-size:16px}
.st-key-v2_page [data-testid="stRadioOption"] p{font-size:16px!important;font-weight:600}
.st-key-v2_page [data-testid="stRadioOption"]:has(input:checked){background:linear-gradient(100deg,#1255a9,#205eaa)!important;box-shadow:inset 0 0 0 1px #3376bf}
.sidebar-note{width:100%!important;margin:clamp(42px,20vh,230px) 0 0!important;line-height:1.8;font-size:13px;border:1px solid #3b506d;background:#22354c55}
.sidebar-note hr{border-color:#3a506a;margin:12px 0}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#c9d8e9!important}
[class*="st-key-panel_"]{border-radius:12px;min-width:0;box-shadow:0 2px 8px #233e6005;padding:16px;box-sizing:border-box;overflow-wrap:anywhere}
.st-key-panel_topbar{position:relative;background:linear-gradient(105deg,#f7fbff,#e8f0fa 70%,#d8e5f2)!important;border:0;border-radius:0!important;margin:0 -20px;padding:22px 28px!important;overflow:hidden;min-height:114px}
.hero-title{font-size:clamp(23px,2vw,34px);line-height:1.25;letter-spacing:-.055em;font-weight:850;color:#0b172c;margin:0 0 8px;position:relative;z-index:1}
.hero-subtitle{font-size:15px;color:#3c506a;line-height:1.5;max-width:80%;position:relative;z-index:1}
.hero-status{font-size:12px;color:#4b6380;margin-top:8px;position:relative;z-index:1}
.hero-art{position:absolute;right:14px;top:0;bottom:0;width:240px;opacity:.18;pointer-events:none}
.hero-art svg{width:100%;height:100%}
h3{font-size:18px!important;font-weight:750!important;color:#132e56;line-height:1.4;padding:0 0 12px!important}
[data-testid="stCaptionContainer"] p{font-size:13px!important;line-height:1.6;color:#4a607b!important}
.st-key-v2_input_card{padding:17px 16px;box-shadow:none;min-width:0;border-radius:12px;background:linear-gradient(135deg,white,#f9fbff)}
.st-key-v2_input_card [data-testid="stVerticalBlock"]{gap:12px}
.st-key-v2_input_card [data-testid="stFileUploader"]{min-width:0}
.st-key-v2_input_card [data-testid="stFileUploaderDropzone"]{flex-wrap:wrap;padding:12px;background:#f5f8ff;border:1px dashed #c5d5ec;border-radius:9px;gap:8px}
.st-key-v2_input_card [data-testid="stFileUploaderDropzoneInstructions"]{min-width:0;width:100%;margin:0;flex:1 1 100%}
.st-key-v2_input_card [data-testid="stFileUploaderDropzoneInstructions"]>div{min-width:0}
.st-key-v2_input_card [data-testid="stFileUploaderFile"]{min-width:0}
.st-key-v2_input_card button[kind="primary"]{min-height:46px}
.st-key-v2_input_card button[kind="primary"]:disabled{background:#e2e9f4;color:#506680;border:1px solid #d1ddec}
.st-key-v2_input_card [data-testid="stFileUploaderFile"]{width:100%;max-width:100%;overflow:hidden}
.st-key-v2_input_card [data-testid="stFileUploaderFile"]>div{min-width:0;max-width:100%}
.st-key-v2_input_card [data-testid="stCaptionContainer"] p{font-size:12px!important}
.st-key-panel_results{padding:16px;min-height:100%;background:linear-gradient(145deg,#fff,#fafcff)}
.overview-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 14px}
.kpi-card{padding:16px 14px 12px;border:1px solid #d9e7fb;border-radius:10px;background:#f0f6ff;min-width:0;min-height:116px}
.kpi-card.state{background:#eefaf7;border-color:#cceee4}.kpi-card.decision{background:#f4f7fb;border-color:#dfe6ee}.kpi-card.alert{background:#fff3f1;border-color:#ffddd9}.kpi-card.normal{background:#eefbf3;border-color:#ccebdd}.kpi-card.score{background:#f8faff}
.kpi-label{font-size:13px;color:#2d4463;font-weight:650;line-height:1.4;min-height:20px}.kpi-icon{color:#1769ff;margin-right:6px}
.kpi-value{font-size:clamp(23px,1.7vw,30px);font-weight:800;letter-spacing:-.03em;color:#10233d;line-height:1.4;margin:6px 0 3px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.alert .kpi-value{color:#df493d}.normal .kpi-value{color:#088353}.kpi-value small{font-size:15px;font-weight:600;margin-left:4px}
.kpi-note{font-size:12px;color:#536781;line-height:1.5}.model-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.model-tile{border:1px solid #dbe5f2;border-radius:9px;padding:12px;background:#fff;min-width:0;min-height:95px}
.model-tile.alert{background:linear-gradient(110deg,#fff,#fff3f3)}.model-tile.normal{background:linear-gradient(110deg,#fff,#f1fbf6)}.model-tile.disabled{background:#f0f3f8;color:#61718a}
.model-heading{font-size:15px;font-weight:750;display:flex;justify-content:space-between;align-items:start;gap:4px;flex-wrap:wrap;line-height:1.4}.model-state{font-size:13px;color:#4a607b}.model-tile.alert .model-state{color:#d64a40}.model-tile.normal .model-state{color:#008455}.model-score{font-size:17px;font-weight:700;line-height:1.6;margin-top:5px}.model-rule{font-size:11px;color:#526781;line-height:1.5}
.st-key-panel_signal [data-testid="stTabs"]{min-width:0}.st-key-panel_signal [role="tablist"]{gap:5px;overflow-x:auto;border-bottom:1px solid #dbe5f2;scrollbar-width:thin}.st-key-panel_signal [role="tab"]{background:#f8faff;border:1px solid #dce5f3;border-radius:8px 8px 0 0;padding:10px 12px;white-space:nowrap;height:auto}.st-key-panel_signal [role="tab"][aria-selected="true"]{background:#edf5ff;color:#1268d6;font-weight:750}.st-key-panel_signal [data-testid="stExpander"]{margin-top:10px}
.st-key-panel_signal [data-testid="stVegaLiteChart"]{min-width:0;width:100%;max-width:100%}
.advice-box{padding:14px;border-radius:10px;background:#edf5ff;line-height:1.7;font-size:14px;margin-bottom:12px;color:#213956}.advice-box.action{background:#f2efff}.advice-box.caution{background:#fff9e6}.advice-box strong{font-size:15px;color:#122a4b}.advice-box p{margin:5px 0 0}.advice-box .small{font-size:12px;color:#52637c}
.rf-split{border-top:1px solid #d9e5f6;margin-top:10px;padding-top:9px}.rf-split b{color:#1c3b66}.final-candidate{margin-top:7px}.st-key-panel_actions{background:#fff;border-color:#dbe5f2;padding:16px;margin-top:0}.st-key-panel_actions button[kind="primary"]{background:linear-gradient(110deg,#0a6be9,#0754c8)}
.st-key-panel_model_info,.st-key-panel_guide{background:white}.st-key-v3_workspace [data-testid="stColumn"],.st-key-v3_evidence_row [data-testid="stColumn"]{min-width:0!important}
[data-testid="stMetricValue"]{white-space:normal!important;overflow-wrap:anywhere}
button:focus-visible,a:focus-visible,[role="tab"]:focus-visible{outline:3px solid #448aff!important;outline-offset:2px}
@media(min-width:1051px){[data-testid="stSidebar"]{width:242px!important;min-width:242px!important;max-width:242px!important}}
@media(max-width:1200px){.overview-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.model-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.hero-subtitle{max-width:90%}.st-key-v3_evidence_row>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]{flex-wrap:wrap}.st-key-v3_evidence_row>[data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}}
@media(max-width:850px){[data-testid="stToolbar"]{display:flex!important}[data-testid="stAppDeployButton"],[data-testid="stMainMenu"]{display:none!important}[data-testid="stSidebarHeader"]{height:34px!important;min-height:34px!important}.block-container{padding:0 12px 18px!important}.st-key-panel_topbar{margin:0 -12px;padding:20px 18px!important}.hero-subtitle{max-width:100%;font-size:13px}.hero-art{opacity:.08}.hero-title{font-size:23px}.st-key-v3_workspace [data-testid="stHorizontalBlock"],.st-key-v3_evidence_row [data-testid="stHorizontalBlock"]{flex-wrap:wrap}.st-key-v3_workspace [data-testid="stColumn"],.st-key-v3_evidence_row [data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}.st-key-panel_signal [data-testid="stHorizontalBlock"]{flex-wrap:wrap}.st-key-panel_signal [data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}.overview-grid,.model-grid{gap:8px}.kpi-card{padding:12px;min-height:105px}.model-tile{padding:10px}.sidebar-note{margin-top:30px!important}[data-testid="stHeader"]{height:34px!important;z-index:1000!important}[data-testid="stSidebarCollapsedControl"]{background:#e8eff8;border-radius:0 0 8px 0}}
</style>'''

from html import escape
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from vibration.dashboard_view import app_status, analyzed_time, number, chart, envelope_points
from vibration.candidate_decision_ui import decision_view
from scripts.analyze_fault_signatures import analyze_signal
from scipy.signal import butter, sosfiltfilt, hilbert

MOTOR_ART = '<svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg"><g stroke="#173c66" stroke-width="3"><path fill="#6f98c5" d="M52 45h152v73H52z"/><ellipse cx="204" cy="81" rx="32" ry="38" fill="#315f91"/><ellipse cx="204" cy="81" rx="21" ry="27"/><path d="M52 55H32v50h20M236 70h43v23h-43M74 118v13h35v-13M159 118v13h35v-13"/><path d="M83 45V22h75v23M95 25v18M113 25v18M131 25v18M149 25v18M66 51v59M81 51v59M96 51v59M111 51v59M126 51v59M141 51v59M156 51v59M171 51v59"/></g></svg>'


def hero(state, result, error=None):
    status, _ = app_status(state, result, error)
    with st.container(key='panel_topbar'):
        st.markdown('<div class="hero-title">회전체 진동 이상 감지 시스템</div>'
            '<div class="hero-subtitle">AI 기반 진동 신호 분석으로 이상 징후를 확인하고, 설비 점검을 위한 분석 근거를 제공합니다.</div>'
            '<div class="hero-status">● '+escape(status)+' · '+escape(analyzed_time(result))+' KST</div>'
            '<div class="hero-art" aria-hidden="true">'+MOTOR_ART+'</div>', unsafe_allow_html=True)


def overview_values(rpm, result, error=None):
    item = result['models']['autoencoder'] if result else None
    return dict(rpm=rpm, status='분석 오류' if error else '완료' if item else '분석 대기',
        decision='판정 불가' if error else ('이상 징후' if item['predicted_anomaly'] else '정상') if item else '분석 대기',
        score=item['file_score'] if item else None, threshold=item['file_threshold'] if item else None,
        abnormal=bool(item['predicted_anomaly']) if item else None)


def overview(rpm, state, result, error=None):
    v=overview_values(rpm,result,error)
    cards=[('◴','운전 속도 (RPM)',str(rpm)+'<small>RPM</small>','선택한 측정 조건',''),
           ('✓','분석 상태',v['status'],'입력 검증·AE 계산 완료' if result else 'MAT 파일 업로드 후 실행','state'),
           ('△','종합 판정',v['decision'],'운영 기준: AE 저장 임계값','decision '+('alert' if v['abnormal'] else 'normal' if v['abnormal'] is False else '')),
           ('◉','AE 이상 점수',number(v['score']),'임계값 '+number(v['threshold'])+' · 원시 MSE','score')]
    html='<div class="overview-grid">'
    for icon,label,value,note,style in cards:
        html+=f'<div class="kpi-card {style}"><div class="kpi-label"><span class="kpi-icon">{icon}</span>{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{escape(note)}</div></div>'
    st.markdown(html+'</div>',unsafe_allow_html=True)


def model_tile_values(rows):
    by_model={r['model']:r for r in rows or []}
    tiles=[]
    for model,name in (('autoencoder','AE'),('ocsvm','OCSVM'),('iforest','IF'),('dae','DAE')):
        r=by_model.get(model)
        complete=bool(r and r['status']=='완료')
        blocked=model=='dae' and not complete
        status='사용 중지 / 검증 필요' if blocked else r['decision'] if complete else r['status'] if r else '분석 대기'
        tiles.append(dict(model=model,name=name,status=status,score=r['score'] if complete else None,
            threshold=r['threshold'] if complete else None,
            style='disabled' if blocked else 'alert' if complete and r['decision']=='이상' else 'normal' if complete and r['decision']=='정상' else '',
            note='운영 기준 · 복원 MSE' if model=='autoencoder' else 'RPM·성능 검증 필요' if blocked else '참고 · 점수 척도 다름',
            reason=r.get('reason','') if r else ''))
    return tiles


def model_tiles(rows):
    html='<div class="model-grid">'
    for t in model_tile_values(rows):
        html+=f'<div class="model-tile {t["style"]}" data-model="{t["model"]}"><div class="model-heading">{t["name"]}<span class="model-state">{escape(t["status"])}</span></div><div class="model-score" title="{escape(str(t["score"]))}">{number(t["score"])}</div><div class="model-rule">{escape(t["note"])}</div></div>'
    st.markdown(html+'</div>',unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=3)
def display_signals(signal, fs, rpm):
    """Display-only arrays, following the existing EDA envelope contract; no model input changes."""
    observed=analyze_signal(signal,fs,rpm)
    x=np.asarray(signal,dtype=np.float64)
    centered=x-x.mean()
    sos=butter(4,(2000.,8000.),btype='bandpass',fs=fs,output='sos')
    envelope=np.abs(hilbert(sosfiltfilt(sos,centered)))
    trim=int(round(.1*fs))
    envelope=envelope[trim:-trim]
    if not np.isfinite(envelope).all(): raise ValueError('포락 표시 값이 유한하지 않습니다.')
    return {**observed, 'envelope_time':np.arange(trim,len(signal)-trim)/fs,'envelope_signal':envelope}


def fft_chart(details,rpm,upper=200,height=285):
    f,a=np.asarray(details['frequency']),np.asarray(details['amplitude'])
    mask=f<=upper;f,a=f[mask],a[mask]
    edges=np.linspace(0,len(f),min(1200,len(f))+1,dtype=int)
    indices=[i+int(a[i:j].argmax()) for i,j in zip(edges[:-1],edges[1:])]
    frame=pd.DataFrame({'주파수 (Hz)':f[indices],'FFT 진폭 (raw)':a[indices]})
    orders=pd.DataFrame({'주파수 (Hz)':[rpm*k/60 for k in (1,2,3)],'차수':[f'{k}X' for k in (1,2,3)]})
    line=chart(frame,'주파수 (Hz):Q','FFT 진폭 (raw):Q',height=height)
    rules=alt.Chart(orders).mark_rule(color='#ee8b36',strokeDash=[4,4],opacity=.6).encode(x='주파수 (Hz):Q',tooltip=['차수:N'])
    labels=alt.Chart(orders).mark_text(align='left',dx=4,dy=8,color='#9b5b24',fontSize=11).encode(x='주파수 (Hz):Q',y=alt.value(5),text='차수:N')
    return line+rules+labels


def signal_panel(preview,details,candidate,rpm,result):
    with st.container(key='panel_signal'):
        st.subheader('❸ 신호 분석 결과 · 분석 근거')
        if preview is None:
            st.info('MAT 파일을 업로드하면 실제 CH3 파형을 확인할 수 있습니다.')
            st.caption('시간 파형 · FFT · 포락 신호/스펙트럼 · 명목 1X·2X·3X')
            return
        view=None
        if result is not None:
            try: view=display_signals(preview['signal'],preview['fs'],rpm)
            except Exception as e: st.warning('신호 상세 표시 불가: '+str(e)+' · AE 결과는 유지됩니다.')
        time_tab,fft_tab,envelope_tab,order_tab=st.tabs(['시간 영역 파형','주파수 스펙트럼','포락 분석','회전 성분'])
        with time_tab:
            wave,spectrum=st.columns([1.2,1],gap='medium')
            with wave:
                st.markdown('**CH3 시간 영역 파형**')
                st.altair_chart(chart(envelope_points(preview['time'],preview['signal']),'시간 (초):Q','CH3 진폭:Q',height=140),width='stretch')
                if view is not None:
                    st.markdown('**포락 신호 · 2–8kHz 반송 대역**')
                    frame=envelope_points(view['envelope_time'],view['envelope_signal']).rename(columns={'CH3 진폭':'포락 진폭 (raw)'})
                    st.altair_chart(chart(frame,'시간 (초):Q','포락 진폭 (raw):Q',color='#8d5bd7',height=120),width='stretch')
                else: st.caption('분석 후 포락 신호와 스펙트럼을 표시합니다.')
            with spectrum:
                st.markdown('**주파수 스펙트럼 · 저주파 확대**')
                if details is not None:
                    st.altair_chart(fft_chart(details,rpm,200,285),width='stretch')
                    st.caption('점선: 명목 1X·2X·3X · RPM/60 × 차수')
                else: st.info('분석 실행 후 실제 FFT를 표시합니다.')
            st.caption('센서 raw 진폭 · 포락 가장자리 0.1초씩 제외 · 속도 적분은 제공하지 않습니다.')
        with fft_tab:
            upper=st.radio('FFT 표시 범위',('전체 0–10kHz','저주파 0–200Hz'),horizontal=True,key='v2_fft_range')
            if details is not None: st.altair_chart(fft_chart(details,rpm,200 if upper.startswith('저주파') else 10000,280),width='stretch')
            else: st.info('분석 후 FFT를 표시합니다.')
            st.caption('Hann 단측 진폭 스펙트럼입니다. ASD 또는 물리 단위로 환산한 값이 아닙니다.')
            with st.expander('시간 파형 확대 · 드래그/휠 탐색'):
                st.altair_chart(chart(envelope_points(preview['time'],preview['signal']),'시간 (초):Q','CH3 진폭:Q',height=240).interactive(),width='stretch')
        with envelope_tab:
            if view is not None:
                frame=pd.DataFrame({'주파수 (Hz)':view['envelope_frequency'],'포락 FFT 진폭 (raw)':view['envelope_amplitude']})
                st.altair_chart(chart(frame,'주파수 (Hz):Q','포락 FFT 진폭 (raw):Q',color='#8d5bd7',height=260),width='stretch')
                st.caption('기존 EDA: 4차 Butterworth 2–8kHz → 양방향 필터 → Hilbert → 양끝 0.1초 제외. 반송 대역의 현장 타당성과 베어링 결함 주파수는 미검증입니다.')
            else: st.info('분석 후 실제 포락 스펙트럼을 표시합니다.')
        with order_tab:
            if details is not None:
                from vibration.dashboard_v3 import rotation_orders
                orders=rotation_orders(details,rpm)
                st.altair_chart(alt.Chart(orders).mark_bar(color='#3884e9').encode(x=alt.X('회전 성분:N',axis=alt.Axis(labelAngle=0)),y='진폭 (raw):Q',tooltip=list(orders.columns)).properties(height=230),width='stretch')
                st.dataframe(orders,hide_index=True,width='stretch')
                st.caption('1X–5X · 명목 RPM 기준, 실제 회전수 계측 아님 · ±1Hz 탐색(bin 해상도 고려) · 확정 결함 기준 아님')
            else: st.info('분석 후 실제 회전 성분을 표시합니다.')


def advice_panel(result,candidate,rows):
    with st.container(key='panel_advice'):
        st.subheader('❹ 분석 요약 및 권장 조치')
        item=result['models']['autoencoder'] if result else None
        verdict='AE 저장 기준을 초과해 이상 징후가 관측됐습니다.' if item and item['predicted_anomaly'] else 'AE 저장 기준을 초과하지 않았습니다. 결함 부재를 보장하지 않습니다.' if item else '분석 실행 후 실제 판정과 주요 특징을 표시합니다.'
        view=decision_view(candidate) if candidate else None
        summary='<div class="advice-box"><strong>▤ 분석 요약</strong><p>'+escape(verdict)+'</p>'
        if candidate and candidate.get('features'):
            f=candidate['features'];summary+='<p>실제 1X '+number(f.get('one_x_amplitude'))+' · 2X '+number(f.get('two_x_amplitude'))+'<br>고주파 평균제곱 '+number(f.get('energy_high'))+' (raw²)</p>'
        if rows:
            summary+='<p class="small">'+escape(' · '.join(r['label']+': '+(r['decision'] if r['status']=='완료' else r['status']) for r in rows if r['model']!='dae'))+'</p>'
        if view:
            summary+='<div class="rf-split">모델 참고 예측: <b>'+escape(view['raw_label'])+'</b><br><span class="small">RF 원시 예측 · 최종 진단 아님</span><div class="final-candidate">최종 후보: <b>'+escape(view['final_candidate'])+'</b></div></div>'
        st.markdown(summary+'</div>',unsafe_allow_html=True)
        if view and view['reasons']:
            st.caption('실제 보류·생략 사유')
            st.info(view['reasons'][0])
            if len(view['reasons'])>1:
                with st.expander('추가 보류 사유'):
                    for reason in view['reasons'][1:]: st.write(reason)
        action='이상 징후가 반복되면 설비 점검을 권장합니다. 동일 RPM·측정 조건으로 재측정해 추세를 확인하세요.' if item and item['predicted_anomaly'] else '필요 시 동일 RPM·측정 조건으로 재측정하고 과거 기록과 비교하세요.' if item else '측정 당시 RPM을 선택하고 MAT 입력을 확인해 분석하세요.'
        st.markdown('<div class="advice-box action"><strong>⚒ 권장 조치</strong><p>'+escape(action)+'</p></div>'
            '<div class="advice-box caution"><strong>△ 분석 시 유의사항</strong><p>DAE: 사용 중지 / 검증 필요<br>AI 판정은 점검을 지원하는 참고 정보이며 결함 원인을 확정하지 않습니다.</p><p class="small">RF 점수는 미보정 모델 점수입니다. 정상 비교 기준·센서 단위·운전 조건에 따른 한계를 확인하세요.</p></div>',unsafe_allow_html=True)


def guide_page():
    with st.container(key='panel_guide'):
        st.subheader('사용 가이드')
        st.markdown('1. **분석하기**에서 MAT를 업로드하고 측정 당시 RPM을 선택합니다.\n2. **분석 시작**을 누르고 AE 운영 판정과 모델별 결과를 확인합니다.\n3. 파형·FFT·포락 분석과 RF 참고 예측, 최종 보류 사유를 구분해 확인합니다.\n4. **결과 저장하기**로 저장하고 **분석 이력**에서 조회·다운로드합니다.')
        st.info('입력: signals=(6, 65536), fs=25,600Hz, CH3만 분석, .mat 최대 20MB. 파일 또는 RPM이 바뀌면 이전 결과를 초기화합니다.')
        st.write('종합 판정은 AE 저장 기준만 사용합니다. 모델 점수를 평균하거나 공통 신뢰도로 환산하지 않습니다. DAE는 지원 범위·성능 검증 전까지 사용 중지 상태입니다.')
        st.write('RF는 AE 이상일 때만 실행하며 기존 후보 보류 정책을 유지합니다. 파일명은 화면 표시용이며 결함 예측 입력으로 사용하지 않습니다.')
        st.write('저장 이력에는 원본 MAT·파형을 보관하지 않습니다. 이전 기록의 원신호를 다시 보려면 원본 파일을 재업로드하세요. 모델 경로·DB 경로 및 오프라인 성능 자료는 모델 정보에서 확인합니다.')

# Decorative menu icons; native radio inputs and accessible labels remain in place.
from urllib.parse import quote
_MENU_ICONS=(
    '<path d="M2 12h5l3-8 4 16 3-8h5"/>',
    '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 3"/>',
    '<path d="m3 7 9-5 9 5-9 5-9-5Zm0 5 9 5 9-5M3 17l9 5 9-5"/>',
    '<path d="M12 5v16M3 4c4-1 7-1 9 1 2-2 5-2 9-1v16c-4-1-7-1-9 1-2-2-5-2-9-1V4Z"/>')
for _i,_path in enumerate(_MENU_ICONS):
    _svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#d7e7fa" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round">'+_path+'</svg>'
    CSS+='<style>.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="'+str(_i)+'"])::before{background-image:url("data:image/svg+xml,'+quote(_svg)+'")}</style>'
