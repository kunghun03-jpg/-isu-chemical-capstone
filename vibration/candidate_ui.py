"""Presentation of measured evidence without converting it to unvalidated diagnoses."""
import pandas as pd
import streamlit as st

from vibration.candidate_analysis import FEATURE_LABELS
from vibration.candidate_decision_ui import render_decision


def render_candidate(report):
    st.subheader('2. 결함 후보 분석')
    if report is None:
        st.info('이 기록에는 결함 후보 근거가 없습니다. 새 분석에서 확인할 수 있습니다.')
        return
    render_decision(report, prefix='detail')
    st.caption('이상 탐지와 별도 결과입니다. 확정 진단이 아닌 탐색적 결함 후보이며 실제 점검이 필요합니다.')
    if report['status'] == 'unavailable':
        st.warning(report['failure'])
        return
    features = report['features']
    if report['status'] == 'candidate':
        st.write('학습된 후보 엔진의 출력입니다. 아래 관측 특징이 결함의 물리적 원인을 확정하지는 않습니다.')
    elif report['status'] != 'skipped_normal':
        st.write('결함 종류 제시를 보류했습니다. 실제 관측값과 보류 사유를 함께 확인하세요.')
    engine = report.get('engine')
    if engine is not None:
        with st.expander('후보 엔진 출력 · 보류 판단 근거'):
            st.write('원시 분류 결과: ' + engine['raw_prediction'])
            st.caption('원시 결과는 보류될 수 있습니다. 분류 점수와 점수 차이는 신뢰도나 보정된 확률이 아닙니다.')
            st.dataframe(pd.DataFrame([{'클래스': k, '분류 점수 (미보정)': v} for k, v in engine['scores_uncalibrated'].items()]), hide_index=True)
            st.write(f"1·2위 점수 차이: {engine['score_margin']:.6g}")
            st.caption('엔진: ' + engine['kind'] + ' · ' + engine['model_version'])
            if engine['policy_rule'] is not None:
                st.write('Validation에서 선택한 제시 기준:', engine['policy_rule'])
    st.caption(f"실제 관측값 · 1X {features['one_x_amplitude']:.5g} · 2X {features['two_x_amplitude']:.5g} · "
               f"고주파 평균제곱 {features['energy_high']:.5g} (센서 raw 단위)")
    with st.expander('실제 진동 특징량 9개와 측정 조건'):
        frame = pd.DataFrame([{'특징량': label, '측정값': features[key], '단위': unit}
                              for key, (label, unit) in FEATURE_LABELS.items()])
        st.dataframe(frame, hide_index=True, width='stretch')
        order = report['orders']
        st.caption(f"명목 1X {order['one_x_hz']:.2f}Hz / 측정 피크 {order['one_x_peak_hz']:.3f}Hz · "
                   f"명목 2X {order['two_x_hz']:.2f}Hz / 측정 피크 {order['two_x_peak_hz']:.3f}Hz · 탐색 ±1Hz")
        st.caption('전체 2.56초 FFT · 해상도 0.390625Hz · 엔벨로프 carrier 2–8kHz(탐색용). 결측값은 판정값으로 대체하지 않습니다.')
    with st.expander('후보별 관측 근거와 점검 한계'):
        for review in report['reviews']:
            st.markdown(f"**{review['family']} 검토**")
            st.write(review['interpretation'])
            st.caption(review['limitation'])
        st.caption('위 세 항목은 순위나 진단 결과가 아니라 검토 대상입니다.')
    with st.expander('결과 해석 시 주의사항'):
        for limitation in report['limitations']:
            st.write('• ' + limitation)
