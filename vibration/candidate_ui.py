"""Presentation of measured evidence without converting it to unvalidated diagnoses."""
import pandas as pd
import streamlit as st

from vibration.candidate_analysis import FEATURE_LABELS


def render_candidate(report):
    st.subheader('2. 결함 후보 분석')
    if report is None:
        st.info('이 기록에는 결함 후보 근거가 없습니다. 새 분석에서 확인할 수 있습니다.')
        return
    st.metric('결함 후보', report['candidate'])
    st.caption('이상 탐지와 별도 결과입니다. 결함 종류를 뒷받침할 검증된 기준이 없어 확정하지 않습니다.')
    if report['status'] == 'unavailable':
        st.warning(report['failure'])
        return
    features = report['features']
    st.write('베어링 · 불평형 · 느슨함을 검토했으나, 정상 비교 기준과 검증된 분류 근거가 부족합니다.')
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
