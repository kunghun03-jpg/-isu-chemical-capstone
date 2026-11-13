"""Read-only separation of recorded RF reference output and the final policy decision."""
import streamlit as st

RAW_LABELS = {'Bearing': '베어링 결함', 'Unbalance': '불평형', 'Looseness': '느슨함',
              'Normal': '정상 클래스'}

CSS = '''<style>
[class*="st-key-rf_reference_"]{background:#f1f6fd;border:1px solid #c9d9ee;border-radius:9px;padding:12px;box-shadow:none;gap:.35rem}
[class*="st-key-rf_final_"]{background:#fff9ee;border:1px solid #f1d6a7;border-radius:9px;padding:12px;box-shadow:none;gap:.35rem}
[class*="st-key-rf_reference_"] [data-testid="stMetricValue"], [class*="st-key-rf_final_"] [data-testid="stMetricValue"]{font-size:1.08rem!important;white-space:normal!important;overflow-wrap:anywhere}
</style>'''


def decision_view(report):
    """Never derive a class from a filename, score argmax or final candidate label."""
    report = report or {}
    skipped = report.get('status') == 'skipped_normal'
    engine = report.get('engine') or {}
    raw = engine.get('raw_prediction') if not skipped else None
    return {'raw_class': raw if raw in RAW_LABELS else None,
            'raw_label': RAW_LABELS[raw] if raw in RAW_LABELS else '실행 생략' if skipped else '원시 예측 없음',
            'final_candidate': report.get('candidate', '기록 없음'),
            'reasons': tuple(report.get('reasons', [])), 'skipped': skipped}


def render_decision(report, prefix='detail', compact=False):
    """Show existing payload fields; never mutate policy output or persistence data."""
    view = decision_view(report)
    st.markdown(CSS, unsafe_allow_html=True)

    def raw_panel():
        with st.container(key='rf_reference_' + prefix):
            st.caption('모델 참고 예측 · 보류 정책 적용 전')
            st.metric('RF 원시 예측', view['raw_label'])
            if view['raw_class']:
                st.caption('RF 클래스: ' + view['raw_class'] + ' · 최종 진단이 아닙니다.')
            elif view['skipped']:
                st.caption('AE 정상으로 RF 엔진을 실행하지 않았습니다.')
            else:
                st.caption('저장된 결과에 RF 원시 예측이 없습니다.')

    def final_panel():
        with st.container(key='rf_final_' + prefix):
            st.caption('최종 결함 후보 · 기존 보류 정책 적용 후')
            # Preserve the original metric label for existing dashboards/history consumers.
            st.metric('결함 후보', view['final_candidate'])
            st.caption('참고 예측과 별개이며 확정 진단이 아닙니다.')

    if compact:
        left, right = st.columns(2, gap='small')
        with left:
            raw_panel()
        with right:
            final_panel()
    else:
        raw_panel()
        final_panel()
    if view['reasons']:
        st.caption('실제 판정 보류·생략 사유')
        st.info(view['reasons'][0])
        if len(view['reasons']) > 1:
            with st.expander(f'추가 사유 {len(view["reasons"])-1}개'):
                for reason in view['reasons'][1:]:
                    st.info(reason)
