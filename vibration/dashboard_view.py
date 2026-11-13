"""Dashboard presentation only. No model fitting, thresholds or diagnosis decisions."""
from datetime import datetime
from html import escape
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from vibration.candidate_analysis import FEATURE_LABELS
from vibration.history import KST

WAVE_ICON = '<svg width="40" height="30" viewBox="0 0 44 32" fill="none"><path d="M2 17h8l4-12 6 24 6-22 4 10h12" stroke="#2585ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
CSS = '''<style>
:root{--ink:#14243d;--muted:#6e7e94;--blue:#1769ff;--navy:#101f3c;--line:#dce4f2;--surface:#f3f6fc}
.stApp{background:var(--surface);color:var(--ink);font-family:Inter,"Noto Sans KR",sans-serif}
.block-container{max-width:1880px;padding:.25rem 1rem 1.1rem}
[data-testid="stHeader"]{background:transparent;height:0;min-height:0;pointer-events:none!important}
[data-testid="stHeader"] button{pointer-events:auto}
[data-testid="stToolbar"]{display:none}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#142849 0%,#0b1930 100%);border-right:1px solid #254267}
[data-testid="stSidebarUserContent"]{padding:18px 16px!important}
[data-testid="stSidebarUserContent"]>div{padding:0!important}
[data-testid="stSidebarHeader"]{height:0;min-height:0;padding:0}
[data-testid="stSidebarCollapseButton"]{position:absolute;right:3px;top:2px;z-index:2}
[data-testid="stSidebarCollapseButton"] span{color:#bfcee2}
.st-key-v2_page{width:196px!important;margin-left:-20px}
.st-key-v2_page [data-testid="stRadio"],.st-key-v2_page [data-testid="stRadioGroup"]{width:100%!important}
.st-key-v2_page [data-testid="stRadioGroup"]>div{width:100%}
.st-key-v2_page [data-testid="stRadioOption"]{display:flex!important;align-items:center}
.st-key-v2_page [data-testid="stRadioOption"]>div>div:first-child{display:none}
.st-key-v2_page [data-testid="stRadioOption"]::before{content:'▦';font-size:20px;color:#eaf3ff;margin-right:10px;line-height:1}
.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="0"])::before{content:'⌂'}
.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="1"])::before{content:'↥'}
.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="2"])::before{content:'◷'}
.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="3"])::before{content:'↗'}
.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="4"])::before{content:'⚙'}
.st-key-v2_fft_range [data-testid="stRadioOption"][data-selected="true"]>div>div:first-child{background:#1769ff;border-color:#1769ff}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:#edf4ff}
[data-testid="stSidebar"] [role="radiogroup"]{gap:8px}
[data-testid="stSidebar"] label{padding:10px 12px;border-radius:9px;font-weight:550;width:100%;box-sizing:border-box}
[data-testid="stSidebar"] label:has(input:checked){background:#1671f6;box-shadow:0 4px 12px #04162d44}
.sidebar-brand{display:flex;align-items:center;gap:8px;color:white;font-size:12px;font-weight:750;line-height:1.5;padding:0 0 26px;width:196px;margin-left:-20px;white-space:nowrap}
.sidebar-brand svg{width:32px;flex-shrink:0}
.sidebar-note{margin-top:clamp(30px,16vh,160px);padding:17px 14px;border:1px solid #405576;border-radius:14px;color:#c6d7f0;font-size:12px;line-height:2;width:196px;margin-left:-20px;box-sizing:border-box}
.sidebar-note strong{color:white;font-size:14px}
[class*="st-key-panel_"],.st-key-v2_input_card{background:white;border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 3px 16px #142b5010;gap:.45rem}
[class*="st-key-panel_"] [data-testid="stVerticalBlock"],.st-key-v2_input_card [data-testid="stVerticalBlock"]{gap:.4rem}
.st-key-panel_topbar{background:#101f3c;color:white;border-color:#101f3c;padding:11px 18px;border-radius:8px;margin-bottom:3px}
.top-title{font-size:19px;font-weight:750;color:white;line-height:1.3}.top-sub{font-size:10px;color:#aabbd3;letter-spacing:.1em;margin-top:4px}
.top-status{color:#d9e8ff;font-size:12px;line-height:1.8}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#37d393;margin-right:6px}.dot.wait{background:#f4b345}
.st-key-panel_topbar button{background:#213b5c;color:white;border-color:#3a5679;font-size:12px}
.st-key-panel_ae,.st-key-panel_candidate,.st-key-panel_features{height:100%;min-height:300px}
.st-key-panel_wave,.st-key-panel_fft,.st-key-panel_orders{height:100%;min-height:235px}
.st-key-panel_candidate_status{background:linear-gradient(110deg,#fff6df,#fffaf0);border-color:#ffda95;box-shadow:none;padding:12px}
.st-key-panel_candidate_skipped{background:#edf9f5;border-color:#c8eade;box-shadow:none;padding:12px}
.st-key-panel_candidate_status [data-testid="stMetricValue"],.st-key-panel_candidate_skipped [data-testid="stMetricValue"]{white-space:normal;overflow-wrap:anywhere;font-size:1.22rem}
h1{font-size:1.4rem!important;letter-spacing:-.03em}h2{font-size:1.2rem!important}h3{font-size:1.04rem!important;letter-spacing:-.02em;padding-top:0!important;padding-bottom:16px!important}
[data-testid="stCaptionContainer"]{font-size:11px;line-height:1.6;color:var(--muted)}
[data-testid="stMetricValue"]{font-size:1.3rem;font-weight:750;color:var(--ink)}
[data-testid="stMetricLabel"]{font-size:12px;color:var(--muted)}
[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(135deg,#287fff,#075cf3);border:0;color:white;border-radius:8px;font-weight:650;min-height:42px}
[data-testid="stButton"] button{border-radius:8px}
.input-intro{font-size:13px;line-height:1.8;color:#526784}.section-kicker{color:#246ce3;font-size:10px;font-weight:750;letter-spacing:.13em;margin-bottom:6px}
.status-center{display:flex;justify-content:center;align-items:center;gap:20px;padding:6px 0 10px}
.status-ring{height:126px;width:126px;flex-shrink:0;border-radius:50%;border:12px solid #29cf91;box-shadow:inset 0 0 0 1px #c5f1df;display:flex;align-items:center;justify-content:center;flex-direction:column;color:#00885a;font-weight:800;font-size:25px;background:#f5fffb}
.status-ring.alert{border-color:#ffad38;color:#b76000;background:#fffaf0;box-shadow:inset 0 0 0 1px #ffe9c5}
.status-ring.pending{border-color:#dce5f1;color:#7e8fa7;background:#f8fafc;box-shadow:none}
.status-ring small{font-size:11px;font-weight:500;margin-top:4px}.status-copy{font-size:12px;color:#63758e;line-height:1.8}
.candidate-banner{background:linear-gradient(110deg,#fff8e7,#fffdf6);border:1px solid #ffdfa4;border-radius:9px;padding:15px 14px;margin-bottom:12px;overflow-wrap:anywhere}
.candidate-banner.neutral{background:#f2f6fd;border-color:#dce6f5}.candidate-banner .overline{font-size:11px;color:#92713c;margin-bottom:6px}.candidate-banner strong{font-size:20px;color:#172a43}
.observation{padding:9px 0;border-bottom:1px solid #edf1f8;font-size:12px;line-height:1.7}.observation:last-child{border:0}.observation b{display:inline-block;width:75px;color:#1e3960}.observation span{color:#677b95}
.feature-table{width:100%;border-collapse:collapse;font-size:12px}.feature-table th{background:#f1f4fa;color:#364a68;text-align:left;padding:7px;border-bottom:1px solid #dae3f1}.feature-table td{padding:6px 7px;border-bottom:1px solid #e9eef7;color:#52657d}.feature-table td:last-child{text-align:right;color:#172d4b;font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
.summary-label{font-size:10px;color:#6d809a;margin:8px 0 3px}.summary-value{font-size:12px;color:#213b5b;overflow-wrap:anywhere;line-height:1.6}.summary-title{font-size:13px;font-weight:750;color:#14396c;margin-bottom:9px}
.st-key-panel_summary_data,.st-key-panel_summary_ae,.st-key-panel_summary_candidate,.st-key-panel_summary_evidence{padding:9px;box-shadow:none;background:#f8fbff}.summary-label{margin-top:5px}.summary-title{margin-bottom:5px}
.st-key-panel_actions{background:#fafffc;border-color:#d3e9de}.st-key-panel_actions button[kind="primary"]{background:linear-gradient(135deg,#08b884,#009567)}
.research-note{font-size:11px;color:#7e8ba0;margin:2px 0 8px}
@media(min-width:1051px){[data-testid="stSidebar"]{width:244px!important;min-width:244px!important;max-width:244px!important}.st-key-v2_mobile_nav{display:none}}
@media(max-width:1050px){
 .block-container{padding:.25rem 12px 1rem}
 [data-testid="stHeader"] button span{color:#dce9fa}
 .st-key-v2_results_row [data-testid="stHorizontalBlock"],.st-key-v2_charts_row [data-testid="stHorizontalBlock"],.st-key-v2_summary_row [data-testid="stHorizontalBlock"],.st-key-v2_input_card [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
 .st-key-v2_results_row [data-testid="stColumn"],.st-key-v2_charts_row [data-testid="stColumn"],.st-key-v2_summary_row [data-testid="stColumn"],.st-key-v2_input_card [data-testid="stColumn"]{flex:1 1 100%;width:100%;min-width:0}
 .status-ring{width:112px;height:112px}.top-title{font-size:16px}
 .st-key-panel_topbar [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
 .st-key-panel_topbar [data-testid="stColumn"]{flex:1 1 100%;width:100%;min-width:0}
}
</style>'''

# Decorative SVGs avoid OS-dependent missing glyphs; native radio inputs remain accessible.
_NAV_PATHS = (
    '<path d="M3 10l9-7 9 7M5 9v12h14V9M9 21v-7h6v7"/>',
    '<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
    '<path d="M3 12a9 9 0 1 0 3-7M3 3v6h6M12 7v6l4 2"/>',
    '<path d="M4 3v18h17M7 15l4-5 4 3 6-8"/>',
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2"/>',
)
for _i, _path in enumerate(_NAV_PATHS):
    _svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#eaf3ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + _path + '</svg>'
    CSS += '<style>.st-key-v2_page [data-testid="stRadioOption"]:has(input[value="'+str(_i)+'"])::before{content:"";height:20px;width:20px;flex:0 0 20px;background-image:url("data:image/svg+xml,'+quote(_svg)+'");background-size:contain;background-repeat:no-repeat}</style>'


def number(value):
    return '—' if value is None else f'{float(value):.6g}'


def analyzed_time(result):
    return datetime.fromisoformat(result['analyzed_at_utc']).astimezone(KST).strftime('%Y.%m.%d %H:%M:%S') if result else '아직 분석하지 않음'


def navigate(page):
    st.session_state['v2_page'] = page


def header(state, result):
    with st.container(key='panel_topbar'):
        title, status, settings = st.columns([2.4, 1.5, .55], vertical_alignment='center')
        with title:
            st.markdown('<div class="top-title">회전기기 AI 진동 진단 시스템</div><div class="top-sub">VIBRATION ANALYSIS · AE + FAULT CANDIDATE</div>', unsafe_allow_html=True)
            with st.container(key='v2_mobile_nav'):
                with st.popover('☰ 메뉴'):
                    for i,page in enumerate(('홈 대시보드','데이터 업로드','분석 이력','통계 및 리포트','시스템 설정')):
                        st.button(page,key=f'v2_nav_{i}',on_click=navigate,args=(page,),width='stretch')
        with status:
            present = state and state['files_present']
            text = '선택 RPM의 AE 파일 확인' if present else 'AE 파일·설정 확인 필요'
            st.markdown(f'<div class="top-status"><span class="dot {"" if present else "wait"}"></span>{text}<br>{escape(analyzed_time(result))} · KST</div>', unsafe_allow_html=True)
        with settings:
            st.button('⚙ 설정', key='v2_open_settings', on_click=navigate, args=('시스템 설정',), width='stretch')


def ae_card(result):
    st.subheader('1. AE 이상 탐지 결과')
    item = result['models']['autoencoder'] if result else None
    status = '이상' if item and item['predicted_anomaly'] else '정상' if item else '대기'
    kind = 'alert' if status == '이상' else '' if item else 'pending'
    subtitle = '기준 초과' if status == '이상' else '기준 이내' if item else 'MAT 분석 전'
    st.markdown(f'<div class="status-center"><div class="status-ring {kind}">{status}<small>{subtitle}</small></div><div class="status-copy">1D Conv AutoEncoder<br>CH3 · 8구간 평균<br>파일 단위 이상 탐지</div></div>', unsafe_allow_html=True)
    if item:
        a, b = st.columns(2)
        a.metric('이상 점수', f"{item['file_score']:.6g}")
        b.metric('판정 기준', f"{item['file_threshold']:.6g}")
        (st.warning if item['predicted_anomaly'] else st.success)('이상 징후가 탐지됐습니다.' if item['predicted_anomaly'] else '저장된 AE 기준을 초과하지 않았습니다.')
    else:
        st.subheader('분석 대기')
        st.caption('MAT 파일을 업로드하고 분석을 시작하세요.')
    st.caption('점수 > 임계값일 때 이상 · 고장 종류와 설비 안전을 확정하지 않습니다.')


def candidate_card(report):
    st.subheader('2. 결함 후보 분석')
    if report is None:
        st.markdown('<div class="candidate-banner neutral"><div class="overline">AE 분석 후 확인</div><strong>결과 대기</strong></div>', unsafe_allow_html=True)
        st.caption('AE 이상일 때만 후보 엔진을 실행합니다.')
        return
    # Keep native metric semantics for accessibility and existing tests.
    with st.container(key='panel_candidate_skipped' if report['status']=='skipped_normal' else 'panel_candidate_status'):
        st.metric('결함 후보', report['candidate'])
    st.caption('확정 진단이 아닌 탐색적 후보 · 분류 점수는 신뢰도가 아닙니다.')
    reasons = report.get('reasons', [])
    if reasons:
        st.info(reasons[0])
        if len(reasons) > 1:
            with st.expander(f'추가 보류 사유 {len(reasons)-1}개'):
                for reason in reasons[1:]:
                    st.info(reason)
    if report.get('failure'):
        st.warning(report['failure'])
    engine = report.get('engine')
    if engine:
        st.caption('실제 엔진 점수 · 미보정 (확률·신뢰도 아님)')
        names={'Bearing':'베어링','Unbalance':'불평형','Looseness':'느슨함','Normal':'정상(원시)'}
        values = [{'클래스': names.get(key,key), '분류 점수': value} for key, value in engine['scores_uncalibrated'].items()]
        st.altair_chart(alt.Chart(pd.DataFrame(values)).mark_bar(color='#2578ff',cornerRadiusEnd=3).encode(
            x=alt.X('분류 점수:Q',title=None,axis=alt.Axis(tickCount=4),scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('클래스:N',title=None,sort='-x')).properties(height=130), width='stretch')
    elif report.get('features'):
        f = report['features']
        st.markdown('<div class="observation"><b>베어링 검토</b><span>고주파 평균제곱 '+number(f.get('energy_high'))+'</span></div>'
                    '<div class="observation"><b>불평형 검토</b><span>1X 진폭 '+number(f.get('one_x_amplitude'))+'</span></div>'
                    '<div class="observation"><b>느슨함 검토</b><span>2X 진폭 '+number(f.get('two_x_amplitude'))+'</span></div>',unsafe_allow_html=True)
    st.caption('정상 대비 증가·결함 원인은 별도 검증이 필요합니다. 자세한 근거는 하단 상세 보기에 있습니다.')


def feature_rows(report):
    features = report.get('features', {}) if report else {}
    keys = ('rms_ac','peak_ac','crest_factor_ac','one_x_amplitude','two_x_amplitude','energy_high','envelope_rms_ac')
    return [(FEATURE_LABELS[k][0], features.get(k), FEATURE_LABELS[k][1]) for k in keys]


def features_card(report):
    st.subheader('3. 주요 진동 특징량')
    rows = ''.join(f'<tr><td>{escape(label)}</td><td title="{escape(str(value))}">{number(value)}</td></tr>' for label,value,_ in feature_rows(report))
    st.markdown('<table class="feature-table"><thead><tr><th>특징량</th><th style="text-align:right">측정값</th></tr></thead><tbody>'+rows+'</tbody></table>', unsafe_allow_html=True)
    st.caption('센서 raw 단위 · 에너지는 raw² · Crest Factor는 무차원. 검증된 정상 범위는 제공하지 않습니다.')


def envelope_points(time, signal, bins=1600):
    """Plot-only min/max sampling: preserve spikes and actual sample times/values."""
    time, signal = np.asarray(time), np.asarray(signal)
    if len(signal) <= bins * 2:
        indices = np.arange(len(signal))
    else:
        indices = []
        for start, end in zip(np.linspace(0,len(signal),bins+1,dtype=int)[:-1], np.linspace(0,len(signal),bins+1,dtype=int)[1:]):
            block = signal[start:end]
            indices.extend((start + int(block.argmin()), start + int(block.argmax())))
        indices = np.unique([0, *indices, len(signal)-1])
    return pd.DataFrame({'시간 (초)': time[indices], 'CH3 진폭': signal[indices]})


def rotation_frame(report, rpm):
    f = report.get('features', {}) if report else {}
    return pd.DataFrame([{'회전 성분': f'{order}X · {rpm*order/60:g}Hz', '진폭': f.get(key)}
                         for order,key in ((1,'one_x_amplitude'),(2,'two_x_amplitude'))])


def chart(data, x, y, color='#2274ff', height=145):
    return alt.Chart(alt.Data(values=data.to_dict('records'))).mark_line(color=color,strokeWidth=1.3).encode(
        x=alt.X(x,axis=alt.Axis(grid=True,tickCount=5)),y=alt.Y(y,axis=alt.Axis(grid=True,tickCount=4))).properties(height=height)


def visualization_row(preview, details, report, rpm):
    with st.container(key='v2_charts_row'):
        wave, fft, orders = st.columns(3, gap='small')
        with wave, st.container(key='panel_wave'):
            st.subheader('시간 영역 파형 (CH3)')
            if preview is None:
                st.caption('업로드 후 실제 CH3 파형을 표시합니다.')
            else:
                st.altair_chart(chart(envelope_points(preview['time'], preview['signal']), '시간 (초):Q','CH3 진폭:Q'), width='stretch')
                st.caption('원본 raw 진폭 · 전체 2.56초 · 화면용 min/max 축약')
        with fft, st.container(key='panel_fft'):
            st.subheader('주파수 스펙트럼 (FFT)')
            if details is None:
                st.caption('분석 후 실제 FFT와 회전 성분을 표시합니다.')
            else:
                display = st.radio('FFT 표시 범위', ('전체 0–10kHz', '저주파 0–200Hz'), horizontal=True, key='v2_fft_range',label_visibility='collapsed')
                f, a = details['frequency'], details['amplitude']
                mask = f <= (200 if display == '저주파 0–200Hz' else 10000)
                f, a = f[mask], a[mask]
                # Max per display bin preserves actual peak frequencies; analysis is never downsampled.
                indices = [i + int(a[i:j].argmax()) for i,j in zip(np.linspace(0,len(f),min(1400,len(f))+1,dtype=int)[:-1],np.linspace(0,len(f),min(1400,len(f))+1,dtype=int)[1:])]
                frame = pd.DataFrame({'주파수 (Hz)': f[indices], 'FFT 진폭': a[indices]})
                rule = alt.Chart(pd.DataFrame({'주파수 (Hz)': [rpm/60,rpm/30]})).mark_rule(color='#ee9d37',strokeDash=[4,4]).encode(x='주파수 (Hz):Q')
                st.altair_chart(chart(frame,'주파수 (Hz):Q','FFT 진폭:Q','#7955df') + rule,width='stretch')
                st.caption('Hann 단측 진폭 · 피크 보존 축약 · 점선: 명목 1X/2X')
        with orders, st.container(key='panel_orders'):
            st.subheader('주요 회전 성분')
            if not report or not report.get('features'):
                st.caption('실제 계산된 1X·2X 진폭을 표시합니다.')
            else:
                frame = rotation_frame(report,rpm)
                st.altair_chart(alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4,size=36).encode(
                    x=alt.X('회전 성분:N',axis=alt.Axis(labelAngle=0)),y=alt.Y('진폭:Q'),color=alt.Color('회전 성분:N',legend=None,scale=alt.Scale(range=['#2478ff','#8960ef']))).properties(height=145),width='stretch')
                st.caption('명목 RPM 기준 · raw 진폭 · 결함 판정 기준선이 아닙니다.')


def summary_card(title, pairs, key):
    with st.container(key=key):
        body = '<div class="summary-title">'+escape(title)+'</div>'
        for label, value in pairs:
            body += f'<div class="summary-label">{escape(label)}</div><div class="summary-value">{escape(str(value))}</div>'
        st.markdown(body,unsafe_allow_html=True)
