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
from vibration.ae_reconstruction_ui import render_reconstruction
from vibration.reconstruction_cache import (
    get_reconstruction_data, clear_reconstruction_cache,
)
from vibration.bandwise_reconstruction import analyze_reconstruction_data
from vibration.bandwise_analysis_ui import render_bandwise_analysis

from vibration.class_comparison_ui import render_class_comparison
from vibration.v3_presentation import (CSS as RESPONSIVE_CSS, dae_diagnostics, hero,
    overview, model_tiles, signal_panel, advice_panel, guide_page)

PAGES=('분석하기','Class별 신호 비교','분석 이력','모델 정보','사용 가이드')


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
            clear_reconstruction_cache(st.session_state)
        st.session_state['v3_input_identity']=identity
        preview=None
        if raw:
            try: preview=preview_mat(raw)
            except Exception as error: st.error('입력 오류: '+error_message(error))
        run=st.button('▶ 분석 시작',key='v3_run',type='primary',width='stretch',disabled=preview is None or 'v2_result' in st.session_state)
        if run and 'v2_result' not in st.session_state:
            v2.clear_result();st.session_state.pop('v3_rows',None)
            clear_reconstruction_cache(st.session_state)
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
        if st.session_state.get('v2_analysis_error'): st.error('분석 오류: '+st.session_state['v2_analysis_error'])
        if 'v2_result' in st.session_state: st.caption('동일 입력의 중복 실행 방지 · 파일/RPM 변경 시 새 분석')
        st.caption('CH3 · 65,536샘플 · 25,600Hz · 원본은 저장하지 않습니다.')
    return preview,rpm,{'files_present':(folder/'manifest.json').is_file()}


def model_table(rows):
    names={'autoencoder':'AE (기준)','dae':'DAE (실험용)','ocsvm':'OCSVM','iforest':'IF'}
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
    from vibration.four_model_performance_ui import (
        render_four_model_performance,
    )

    render_four_model_performance()

    with st.expander(
        '이전 모델 성능 평가 · 참고 자료',
        expanded=False,
    ):
        legacy_performance()


def legacy_performance():
    st.subheader('모델 성능 비교 · 오프라인 평가')

    # 1. 기존 AE / OCSVM / IF 평가 자료
    default = (
        Path(__file__).resolve().parent
        / 'outputs/date_prelim_20261106_reports/comparison_file.csv'
    )
    path = Path(
        os.environ.get('ISU_CHEMICAL_V3_EVALUATION_CSV') or default
    ).expanduser()

    if not path.is_file():
        st.info(f'평가 자료를 찾지 못했습니다: {path}')
        return

    try:
        frame = pd.read_csv(path)

        required = {
            'rpm', 'model', 'normal_count', 'anomaly_count',
            'false_positive_rate', 'detection_rate', 'auroc'
        }

        if not required.issubset(frame.columns):
            st.warning('평가 CSV 형식이 예상과 다릅니다.')
            st.dataframe(frame, hide_index=True, width='stretch')
            return

        names = {
            'autoencoder': 'AE',
            'ocsvm': 'OCSVM',
            'iforest': 'IF'
        }

        frame = frame[frame['model'].isin(names)].copy()

        for column in required - {'model'}:
            frame[column] = pd.to_numeric(
                frame[column], errors='coerce'
            )

        frame = frame.dropna(
            subset=['rpm', 'auroc', 'detection_rate',
                    'false_positive_rate']
        )

        if frame.empty:
            st.info('표시할 평가 결과가 없습니다.')
            return

        frame['모델'] = frame['model'].map(names)

        rpms = sorted(frame['rpm'].astype(int).unique())

        rpm = st.selectbox(
            'RPM 선택',
            rpms,
            key='v3_perf_rpm',
            format_func=lambda x: f'{x} RPM'
        )


        # 2. 기존 표에 DAE 추가
        st.markdown('### AE · OCSVM · IF · DAE 성능 비교')

        chosen = frame[
            frame['rpm'] == rpm
        ].sort_values('auroc', ascending=False).copy()

        # DAE 별도 평가 자료
        dae_path = Path(
            Path(__file__).resolve().parent / 'outputs/web_artifacts/dae_evaluation/dae_rpm_summary.csv'
        )

        if dae_path.is_file():
            try:
                dae_frame = pd.read_csv(dae_path)
                dae_frame['rpm'] = pd.to_numeric(
                    dae_frame['rpm'], errors='coerce'
                )

                dae_selected = dae_frame[
                    dae_frame['rpm'] == rpm
                ]

                if not dae_selected.empty:
                    dae_row = dae_selected.iloc[0]

                    dae_entry = pd.DataFrame([{
                        'rpm': rpm,
                        'model': 'dae',
                        '모델': 'DAE',
                        'auroc': float(
                            dae_row['observed_auroc']
                        ),
                        'detection_rate': float(
                            dae_row['fault_alarm_fraction']
                        ),
                        'false_positive_rate': float(
                            dae_row['normal_alarm_fraction']
                        )
                    }])

                    chosen = pd.concat(
                        [chosen, dae_entry],
                        ignore_index=True
                    )

            except Exception as error:
                st.warning(
                    'DAE 평가 자료를 읽지 못했습니다: '
                    + str(error)
                )

        # AUROC 기준 정렬 (참고용)
        chosen = chosen.sort_values(
            'auroc', ascending=False
        ).copy()

        chosen['AUROC'] = chosen['auroc'].round(4)

        chosen['이상 탐지율 (%)'] = (
            chosen['detection_rate'] * 100
        ).round(1)

        chosen['정상 오탐률 (%)'] = (
            chosen['false_positive_rate'] * 100
        ).round(1)

        chosen['평가 구분'] = chosen['모델'].apply(
            lambda x: (
                '별도 예비 평가'
                if x == 'DAE'
                else '기존 비교 평가'
            )
        )

        # DAE는 별도 평가이므로 공식 순위 제외
        chosen['순위'] = '—'

        original_mask = chosen['모델'] != 'DAE'

        chosen.loc[original_mask, '순위'] = (
            chosen.loc[original_mask, 'auroc']
            .rank(method='min', ascending=False)
            .astype(int)
            .astype(str)
        )

        st.dataframe(
            chosen[
                [
                    '순위', '모델', '평가 구분',
                    'AUROC', '이상 탐지율 (%)',
                    '정상 오탐률 (%)'
                ]
            ],
            hide_index=True,
            width='stretch'
        )

        st.caption(
            'AE·OCSVM·IF의 순위는 동일 비교 자료의 '
            'AUROC 기준입니다. DAE는 별도 실제 MAT '
            '예비 평가 결과로, 순위에서 제외했습니다.'
        )

        st.markdown('**AUROC 비교**')
        st.bar_chart(
            chosen.set_index('모델')[['AUROC']],
            y_label='AUROC'
        )

        st.markdown('**이상 탐지율 · 정상 오탐률**')
        st.bar_chart(
            chosen.set_index('모델')[
                ['이상 탐지율 (%)', '정상 오탐률 (%)']
            ],
            y_label='비율 (%)'
        )

        if (chosen['false_positive_rate'] >= 0.5).any():
            st.warning(
                '정상 오탐률이 50% 이상인 모델이 있습니다. '
                '현재 임계값의 운영 성능은 추가 개선이 필요합니다.'
            )


        # 3. RPM별 AUROC 단순 평균
        means = (
            frame.groupby('모델', as_index=False)['auroc']
            .mean()
            .sort_values('auroc', ascending=False)
        )

        means['평균 AUROC'] = means['auroc'].round(4)

        st.markdown('**RPM별 AUROC 단순 평균 · 참고**')
        st.dataframe(
            means[['모델', '평균 AUROC']],
            hide_index=True,
            width='stretch'
        )

        # 4. DAE 실제 MAT 예비 평가 결과
        st.divider()
        st.markdown('### DAE 실제 MAT 예비 평가 · 별도 결과')

        dae_default = Path(
            Path(__file__).resolve().parent / 'outputs/web_artifacts/dae_evaluation/dae_rpm_summary.csv'
        )

        dae_path = Path(
            os.environ.get('ISU_CHEMICAL_DAE_EVALUATION_CSV')
            or dae_default
        ).expanduser()

        if dae_path.is_file():
            try:
                dae = pd.read_csv(dae_path)

                dae_required = {
                    'rpm',
                    'normal_n',
                    'fault_n',
                    'observed_auroc',
                    'normal_alarm_fraction',
                    'fault_alarm_fraction',
                    'success_unique_files'
                }

                if not dae_required.issubset(dae.columns):
                    st.warning(
                        'DAE 평가 CSV 형식이 예상과 다릅니다.'
                    )
                else:
                    dae = dae[
                        [
                            'rpm',
                            'normal_n',
                            'fault_n',
                            'success_unique_files',
                            'observed_auroc',
                            'normal_alarm_fraction',
                            'fault_alarm_fraction'
                        ]
                    ].copy()

                    for column in dae.columns:
                        dae[column] = pd.to_numeric(
                            dae[column], errors='coerce'
                        )

                    dae = dae.rename(columns={
                        'rpm': 'RPM',
                        'normal_n': '정상 파일 수',
                        'fault_n': '결함 파일 수',
                        'success_unique_files': '평가 성공 파일 수',
                        'observed_auroc': 'AUROC',
                        'normal_alarm_fraction': '정상 오탐률 (%)',
                        'fault_alarm_fraction': '결함 탐지율 (%)'
                    })

                    dae['AUROC'] = dae['AUROC'].round(4)

                    dae['정상 오탐률 (%)'] = (
                        dae['정상 오탐률 (%)'] * 100
                    ).round(1)

                    dae['결함 탐지율 (%)'] = (
                        dae['결함 탐지율 (%)'] * 100
                    ).round(1)

                    dae = dae.sort_values('RPM')

                    st.dataframe(
                        dae,
                        hide_index=True,
                        width='stretch'
                    )

                    st.markdown('**DAE RPM별 AUROC**')
                    st.bar_chart(
                        dae.set_index('RPM')[['AUROC']],
                        y_label='AUROC'
                    )

                    st.markdown(
                        '**DAE 결함 탐지율 · 정상 오탐률**'
                    )
                    st.bar_chart(
                        dae.set_index('RPM')[
                            ['결함 탐지율 (%)', '정상 오탐률 (%)']
                        ],
                        y_label='비율 (%)'
                    )

                    st.warning(
                        'DAE는 실제 MAT 파일을 이용한 '
                        '독립성 미확인 예비 평가입니다. '
                        'AE·OCSVM·IF와 평가 표본 및 조건이 다르므로 '
                        '동일 조건의 성능 순위로 해석할 수 없습니다.'
                    )

                    st.info(
                        'DAE의 웹 운영 판정은 아직 지원 범위 '
                        '검증이 완료되지 않아 차단된 상태입니다. '
                        '이 표는 저장된 오프라인 평가 결과입니다.'
                    )

            except Exception as dae_error:
                st.error(
                    'DAE 평가 자료를 읽지 못했습니다: '
                    + str(dae_error)
                )
        else:
            st.info(
                f'DAE 예비 평가 결과 파일을 찾지 못했습니다: '
                f'{dae_path}'
            )

        # 5. 평가 자료 해석 안내

        # 4개 모델 통합 성능 비교
        st.divider()
        st.markdown('### 4개 모델 통합 성능 비교 · 참고용')

        comparison = frame[
            frame['rpm'] == rpm
        ][
            ['모델', 'auroc', 'detection_rate',
             'false_positive_rate']
        ].copy()

        comparison = comparison.rename(columns={
            'auroc': 'AUROC',
            'detection_rate': '결함 탐지율 (%)',
            'false_positive_rate': '정상 오탐률 (%)'
        })

        comparison['결함 탐지율 (%)'] *= 100
        comparison['정상 오탐률 (%)'] *= 100

        if dae_path.is_file():
            try:
                dae_compare = pd.read_csv(dae_path)

                dae_compare['rpm'] = pd.to_numeric(
                    dae_compare['rpm'], errors='coerce'
                )

                dae_selected = dae_compare[
                    dae_compare['rpm'] == rpm
                ]

                if not dae_selected.empty:
                    dae_row = dae_selected.iloc[0]

                    dae_entry = pd.DataFrame([{
                        '모델': 'DAE',
                        'AUROC': float(
                            dae_row['observed_auroc']
                        ),
                        '결함 탐지율 (%)': float(
                            dae_row['fault_alarm_fraction']
                        ) * 100,
                        '정상 오탐률 (%)': float(
                            dae_row['normal_alarm_fraction']
                        ) * 100
                    }])

                    comparison = pd.concat(
                        [comparison, dae_entry],
                        ignore_index=True
                    )

            except Exception as compare_error:
                st.warning(
                    'DAE 통합 비교 자료를 읽지 못했습니다: '
                    + str(compare_error)
                )

        comparison['AUROC'] = comparison['AUROC'].round(4)
        comparison['결함 탐지율 (%)'] = (
            comparison['결함 탐지율 (%)'].round(1)
        )
        comparison['정상 오탐률 (%)'] = (
            comparison['정상 오탐률 (%)'].round(1)
        )

        comparison['평가 구분'] = comparison['모델'].apply(
            lambda model:
            '별도 예비 평가' if model == 'DAE'
            else '기존 비교 평가'
        )

        st.dataframe(
            comparison[
                ['모델', '평가 구분', 'AUROC',
                 '결함 탐지율 (%)', '정상 오탐률 (%)']
            ],
            hide_index=True,
            width='stretch'
        )

        st.markdown('**4개 모델 AUROC 비교**')
        st.bar_chart(
            comparison.set_index('모델')[['AUROC']],
            y_label='AUROC'
        )

        st.markdown('**4개 모델 결함 탐지율 · 정상 오탐률**')
        st.bar_chart(
            comparison.set_index('모델')[
                ['결함 탐지율 (%)', '정상 오탐률 (%)']
            ],
            y_label='비율 (%)'
        )

        st.warning(
            'DAE는 다른 날짜·표본 선정 방식으로 수행된 '
            '독립성 미확인 예비 평가입니다. '
            '네 모델의 수치는 참고용으로만 나란히 표시하며, '
            '동일 조건의 성능 순위로 해석하면 안 됩니다.'
        )

        st.divider()

        st.info(
            'RF는 결함 후보 분류 모델이므로 '
            '이상 탐지 모델 AUROC 순위에서 제외했습니다.'
        )

        st.caption(
            '저장된 오프라인 평가 결과입니다. '
            '현재 웹 모델과 평가 모델의 버전·분할·임계값 '
            '일치 여부는 별도 확인이 필요합니다.'
        )

        with st.expander('AE · OCSVM · IF 원본 평가 데이터'):
            st.dataframe(
                frame,
                hide_index=True,
                width='stretch'
            )

    except Exception as error:
        st.error(
            '평가 자료를 읽지 못했습니다: ' + str(error)
        )


def history_page():
    from vibration.history_ui import render_history

    legacy_tab, integrated_tab = st.tabs(
        ['기존 분석 이력 · RF 근거',
         '4개 모델 통합 이력 · DAE 실험용']
    )

    with legacy_tab:
        render_history_page()

    with integrated_tab:
        render_history()
def actions(result,candidate,rows):
    with st.container(key='panel_actions'):
        st.subheader('결과 저장 및 이력')
        if st.button('결과 저장하기',key='v3_save',disabled=not result or not rows,type='primary',width='stretch'):
            try:
                ident,created=save_references(history_store(),result,st.session_state['v2_versions'],candidate,rows,raw=st.session_state['v3_upload'].getvalue())
                st.success(f'분석 #{ident} 저장 완료' if created else f'기존 분석 #{ident} 확인 완료')
            except Exception as error:
                st.error(
                    '분석 결과 저장을 완료하지 못했습니다. '
                    '기본 분석 기록은 저장되었을 수 있습니다. '
                    '분석 이력에서 저장 상태를 확인하세요. '
                    '오류: ' + str(error)
                )
        st.button('분석 이력 조회 →',on_click=v2.navigate,args=('분석 이력',),width='stretch')
        if result:
            st.download_button('분석 결과 JSON',json.dumps({**result,'candidate_analysis':candidate,'reference_models':rows},ensure_ascii=False,indent=2,allow_nan=False),file_name='v3_vibration_result.json',mime='application/json',width='stretch')
        st.caption('기존 SQLite에 저장 · 원본 MAT와 파형은 저장하지 않습니다.')


def main():
    st.set_page_config(page_title='회전체 진동 이상 감지 시스템',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
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
        if page=='Class별 신호 비교': render_class_comparison()
        elif page=='분석 이력': history_page()
        elif page=='모델 정보':
            with st.container(key='panel_model_info'):
                st.subheader('모델 정보 및 적용 제한')
                st.write('운영 판정은 AE 저장 임계값을 사용합니다. OCSVM·IF는 참고 비교이며 점수 척도가 다릅니다. DAE는 지원 범위와 성능 검증이 필요합니다.')
                render_settings(config,None)
                with st.expander('DAE 번들 진단 · 읽기 전용'): st.json(dae_diagnostics(st.session_state.get('v2_rpm',900)))
                performance()
                render_normal_comparison()

                # 측정 상태 변화 탐색 결과 - 읽기 전용
                from pathlib import Path as _shift_pathlib_Path
                import pandas as _shift_pandas
                with st.expander(
                    '측정 상태 변화 분석 · 실험용',
                    expanded=False
                ):
                    shift_path = _shift_pathlib_Path(
                        'outputs/measurement_shift/'
                        'normal2_validation_20260916.json'
                    )

                    st.caption(
                        '2026-09-16 Normal2 Validation 데이터에 대한 '
                        '오프라인 탐색 결과입니다. '
                        '현재 업로드한 MAT 파일의 분석 결과가 아닙니다.'
                    )

                    if not shift_path.is_file():
                        st.info(
                            '측정 상태 변화 분석 결과 파일이 없습니다.'
                        )
                    else:
                        try:
                            shift_data = json.loads(
                                shift_path.read_text(
                                    encoding='utf-8'
                                )
                            )

                            if (
                                shift_data.get('schema_version') != 1
                                or shift_data.get('feature')
                                != 'measurement_score_shift'
                                or shift_data.get('status')
                                != 'experimental_only'
                                or shift_data.get('modifies_model_decisions')
                                is not False
                                or shift_data.get('modifies_thresholds')
                                is not False
                                or shift_data.get('run_independence_verified')
                                is not False
                            ):
                                raise ValueError(
                                    '실험용 진단 데이터 정책 불일치'
                                )

                            results = shift_data.get('results', [])
                            expected = {
                                (rpm, model)
                                for rpm in (900, 1200, 1800)
                                for model in (
                                    'autoencoder',
                                    'ocsvm',
                                    'iforest'
                                )
                            }
                            actual = {
                                (r['rpm'], r['model'])
                                for r in results
                            }

                            if len(results) != 9 or actual != expected:
                                raise ValueError(
                                    '진단 결과 9개 구성 불일치'
                                )

                            model_names = {
                                'autoencoder': 'AE',
                                'ocsvm': 'OCSVM',
                                'iforest': 'IF'
                            }

                            table_rows = []

                            for item in sorted(
                                results,
                                key=lambda r: (
                                    r['rpm'],
                                    r['model']
                                )
                            ):
                                if (
                                    item['status']
                                    != 'exploratory_diagnostic'
                                    or item['window_size'] != 20
                                ):
                                    raise ValueError(
                                        '분석 조건 불일치'
                                    )

                                table_rows.append({
                                    'RPM': item['rpm'],
                                    '모델': model_names[item['model']],
                                    '초기 중앙값': round(
                                        item['early_median'], 6
                                    ),
                                    '마지막 중앙값': round(
                                        item['late_median'], 6
                                    ),
                                    '점수 변화': round(
                                        item['median_delta'], 6
                                    ),
                                    '변화 방향': {
                                        'increased': '증가',
                                        'decreased': '감소',
                                        'unchanged': '변화 없음'
                                    }[item['score_direction']],
                                    '초기 경보': item['early_alarm_count'],
                                    '마지막 경보': item['late_alarm_count']
                                })

                            st.dataframe(
                                _shift_pandas.DataFrame(table_rows),
                                hide_index=True,
                                width='stretch'
                            )

                            st.info(
                                '900·1800 RPM에서는 초기보다 마지막 '
                                '구간의 점수가 낮아졌고, 1200 RPM에서는 '
                                '반대 경향이 관찰됐습니다. '
                                '측정 상태 변화의 원인은 확정되지 않았습니다.'
                            )

                            st.caption(
                                '기존 Validation 기반 탐색 결과이며 '
                                '물리적 Run 독립성은 확인되지 않았습니다. '
                                '모델의 Normal/Abnormal 판정과 '
                                '운영 임계값은 변경하지 않습니다.'
                            )

                        except (
                            ValueError, KeyError, TypeError, OSError
                        ) as error:
                            st.warning(
                                '측정 상태 변화 자료 표시 불가: '
                                + str(error)
                            )
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
            # AE 운영 판정과 관측 특징을 표시하며, 결함명은 추정하지 않습니다.
            if result is not None:
                st.markdown('**이상 신호 분석**')
                with st.container(key='v3_candidate_summary'):
                    ae = result.get('models', {}).get('autoencoder')
                    if ae:
                        abnormal = bool(ae['predicted_anomaly'])
                        left, right = st.columns(2)
                        with left:
                            if abnormal:
                                st.error('AE 이상 감지')
                            else:
                                st.success('AE 기준 이상 미검출')
                            st.caption(
                                f"파일 MSE {ae['file_score']:.6f} · "
                                f"Q95 {ae['file_threshold']:.6f}"
                            )
                        with right:
                            if not abnormal:
                                st.info('결함 후보: 분석 생략')
                                st.caption('AE 기준 이상 미검출로 후보 분석을 생략했습니다.')
                            elif isinstance(candidate, dict) and candidate.get('status') == 'candidate':
                                st.warning(f"RF 결함 후보: {candidate.get('candidate', '결함 후보 보류')}")
                                st.caption('RF 검증 정책을 통과한 참고 후보이며, 확정 진단은 아닙니다.')
                            else:
                                rf = (
                                    candidate.get('engine')
                                    if isinstance(candidate, dict)
                                    else None
                                )

                                if isinstance(rf, dict):
                                    raw_class = rf.get(
                                        'raw_prediction', 'Unknown'
                                    )

                                    class_names = {
                                        'Bearing': '베어링 결함 의심',
                                        'Looseness': '풀림 결함 의심',
                                        'Unbalance': '불균형 의심',
                                        'Normal': 'RF 정상 예측',
                                    }

                                    st.warning(
                                        f'RF 실험용 예측: {raw_class}'
                                    )

                                    st.write(
                                        class_names.get(
                                            raw_class, raw_class
                                        )
                                    )

                                    st.caption(
                                        '자동 후보 판정: 보류 · '
                                        '확정 진단 아님'
                                    )
                                else:
                                    st.info('RF 결함 후보: 보류')

                                if isinstance(candidate, dict):
                                    reasons = candidate.get(
                                        'reasons', []
                                    )

                                    for reason in reasons:
                                        st.caption(f'보류 사유: {reason}')
                                else:
                                    st.caption(
                                        'RF 후보 분석 결과 없음'
                                    )
                    else:
                        st.info('AE 판정 결과를 확인할 수 없습니다.')

                    # RF_MAIN_EXPERIMENTAL_RESULT_V1
                    rf_result = (
                        candidate.get('engine')
                        if isinstance(candidate, dict)
                        else None
                    )

                    if isinstance(rf_result, dict):
                        st.markdown('**RF 결함별 원시 분류 점수**')

                        scores = rf_result.get(
                            'scores_uncalibrated', {}
                        )

                        for label, score in sorted(
                            scores.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        ):
                            value = float(score)

                            st.write(
                                f'{label}: {value:.2%}'
                            )

                            st.progress(
                                min(1.0, max(0.0, value))
                            )

                        margin = rf_result.get('score_margin')
                        rule = rf_result.get('policy_rule')

                        cols = st.columns(2)

                        if margin is not None:
                            cols[0].metric(
                                '실제 분류 점수 차이',
                                f'{float(margin):.6f}'
                            )

                        if isinstance(rule, dict):
                            minimum = float(rule['margin_min'])

                            cols[1].metric(
                                '검증 기준',
                                f'{minimum:.6f}'
                            )

                            if margin is not None:
                                if float(margin) >= minimum:
                                    st.caption(
                                        '분류 점수 차이 기준 충족'
                                    )
                                else:
                                    st.caption(
                                        '분류 점수 차이 기준 미충족'
                                    )
                        else:
                            cols[1].metric(
                                '검증 기준', '미확립'
                            )

                        st.caption(
                            'RF 원시 분류 점수는 미보정 출력이며 '
                            '실제 결함 발생 확률이나 확정 진단이 아닙니다.'
                        )

                    features = candidate.get('features') if isinstance(candidate, dict) else None
                    if isinstance(features, dict) and features:
                        st.markdown('**관측된 진동 특징**')
                        cols = st.columns(3)
                        for col, label, key in zip(
                            cols,
                            ('1X 진폭', '2X 진폭', '고주파 에너지'),
                            ('one_x_amplitude', 'two_x_amplitude', 'energy_high'),
                        ):
                            value = features.get(key)
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                col.metric(label, f'{value:.5g}')
                            else:
                                col.metric(label, '—')
                        st.caption(
                            '특징량은 입력 신호의 관측값이며 정상 대비 증가나 결함 원인을 확정하지 않습니다.'
                        )
                    else:
                        st.caption('추가 진동 특징량이 제공되지 않았습니다.')

                    if isinstance(candidate, dict) and candidate.get('engine'):
                        st.caption('RF 참고 결과가 있습니다. 하단 상세 정보에서 확인할 수 있습니다.')
                    st.caption('운영 이상 판정은 AE만 사용하며, 결함 종류는 자동 확정하지 않습니다.')

            with st.expander('비교 모델 참고 결과 · 실험용', expanded=False):
                if rows:
                    model_table(rows)
                else:
                    st.info('추가 모델 참고 결과 없음')
                st.caption('종합 판정은 기존 AE 결과만 사용합니다.')
    with top: header(state,result,st.session_state.get('v2_analysis_error'))
    with st.container(key='v3_evidence_row'):
        signal_column,summary_column=st.columns([2.2,1],gap='medium')
        with signal_column:
            signal_panel(preview,details,candidate,rpm,result)
        with summary_column:
            advice_panel(result,candidate,rows)
            actions(result,candidate,rows)
    if isinstance(candidate, dict) and candidate.get('engine'):
        with st.expander('RF 참고 예측 · 상세 근거', expanded=False):
            candidate_card(candidate, prefix='rf_detail')
            engine = candidate['engine']
            if isinstance(engine, dict) and engine.get('model_version'):
                st.caption('RF 모델 버전: ' + str(engine['model_version']))
            features_card(candidate)

    # EXPERIMENTAL_RF_INDEPENDENT_V1
    with st.expander('실험용 결함 유형 분석 · AE 판정과 독립', expanded=False):
        st.caption(
            '운영 AE 판정과 별개로 RF 원시 분류 결과를 확인합니다. '
            '결과는 확정 진단이나 검증된 결함 확률이 아닙니다.'
        )

        import hashlib
        import os
        from pathlib import Path
        import pandas as pd
        from vibration.candidate_analysis import candidate_analysis, FEATURE_LABELS
        from vibration.candidate_engine import CandidateEngine

        current_raw = None
        current_sha = None

        if result is not None and preview is not None:
            upload = st.session_state.get('v3_upload')
            if upload is not None:
                current_raw = upload.getvalue()
                current_sha = hashlib.sha256(current_raw).hexdigest()

        identity_ok = (
            result is not None
            and current_raw is not None
            and current_sha == result.get('input', {}).get('sha256')
            and rpm == result.get('input', {}).get('rpm')
        )

        saved = st.session_state.get('v3_rf_experiment')
        if saved is not None and (
            not identity_ok
            or saved.get('sha256') != current_sha
            or saved.get('rpm') != rpm
        ):
            st.session_state.pop('v3_rf_experiment', None)
            saved = None

        run_rf = st.button(
            '실험용 RF 결함 분석 실행',
            key='v3_rf_experimental_run',
            disabled=not identity_ok,
        )

        if run_rf:
            st.session_state.pop('v3_rf_experiment', None)
            try:
                report, _ = candidate_analysis(current_raw, rpm)

                bundle = os.environ.get(
                    'ISU_CHEMICAL_CANDIDATE_BUNDLE',
                    Path(__file__).resolve().parent / 'outputs/web_artifacts/rf_candidate'
                )

                engine = CandidateEngine(Path(bundle))

                # 자동 후보 허용 정책은 비활성 상태로 유지
                prediction = engine.predict(
                    report['features'],
                    rpm,
                    labels_enabled=False,
                )

                saved = {
                    'sha256': current_sha,
                    'rpm': rpm,
                    'report': report,
                    'prediction': prediction,
                }
                st.session_state['v3_rf_experiment'] = saved

            except Exception as error:
                st.error(
                    '실험용 RF 분석 실패: '
                    + type(error).__name__ + ': ' + str(error)
                )

        if saved is not None:
            report = saved['report']
            prediction = saved['prediction']

            raw_class = prediction['raw_prediction']
            labels = {
                'Bearing': '베어링 결함',
                'Unbalance': '불평형',
                'Looseness': '느슨함',
                'Normal': '정상 클래스',
            }

            st.markdown('#### RF 원시 결함 분류')
            st.metric(
                '실험용 RF 예측',
                labels.get(raw_class, raw_class)
            )

            st.warning(
                '운영 판정과 별개인 실험용 결과입니다. '
                '자동 결함 진단이나 확정 판정이 아닙니다.'
            )

            score_rows = [
                {
                    '클래스': labels.get(name, name),
                    'RF 원시 점수': float(value),
                }
                for name, value
                in prediction['scores_uncalibrated'].items()
            ]

            st.dataframe(
                pd.DataFrame(score_rows),
                hide_index=True,
                width='stretch',
            )

            st.caption(
                'RF 원시 점수는 트리 투표에 따른 미보정 출력으로, '
                '실제 결함 발생 확률이나 진단 정확도를 뜻하지 않습니다.'
            )

            # Experimental RF feature sensitivity
            from vibration.rf_explanation import feature_sensitivity

            st.markdown('#### RF 예측에 영향을 주는 특징 분석')

            try:
                original_score, influence = feature_sensitivity(
                    CandidateEngine(
                        Path(
                            os.environ.get(
                                'ISU_CHEMICAL_CANDIDATE_BUNDLE',
                                Path(__file__).resolve().parent / 'outputs/web_artifacts/rf_candidate'
                            )
                        )
                    ),
                    report['features'],
                    raw_class,
                )

                sensitivity_rows = [
                    {
                        '특징량': FEATURE_LABELS[item['특징량']][0],
                        '실제 측정값': item['실제 측정값'],
                        '학습 중앙값': item['학습 중앙값'],
                        '점수 변화': item['원래 점수와 차이'],
                    }
                    for item in influence
                ]

                st.caption(
                    f'{raw_class} 원래 RF 점수: {original_score:.4f}'
                )

                st.dataframe(
                    pd.DataFrame(sensitivity_rows),
                    hide_index=True,
                    width='stretch',
                )

                st.caption(
                    '점수 변화 = 원래 클래스 점수 − '
                    '해당 특징만 학습 중앙값으로 대체한 점수. '
                    '양수는 원래 측정값이 해당 클래스 점수를 '
                    '높이는 방향임을 뜻합니다.'
                )

                if influence:
                    top = influence[0]
                    name = FEATURE_LABELS[top['특징량']][0]
                    st.info(
                        f'이번 입력에서 가장 민감한 특징: {name} '
                        f'(점수 변화 {top["원래 점수와 차이"]:+.4f})'
                    )

                st.warning(
                    '이 분석은 한 특징씩 바꿔보는 민감도 실험입니다. '
                    '특징 간 상관관계를 반영하지 않으며, '
                    '점수 변화의 합은 전체 예측을 설명하지 않습니다. '
                    '결함의 실제 원인이나 진단 정확도를 증명하지 않습니다.'
                )

            except Exception as error:
                st.warning(
                    'RF 특징 민감도 분석 불가: '
                    + type(error).__name__
                )

            st.markdown('#### 실제 측정된 진동 특징')

            feature_rows = [
                {
                    '특징량': label,
                    '측정값': report['features'].get(key),
                    '단위': unit,
                }
                for key, (label, unit) in FEATURE_LABELS.items()
            ]

            st.dataframe(
                pd.DataFrame(feature_rows),
                hide_index=True,
                width='stretch',
            )



            # RF_RELATIVE_EXPERIMENT_V1
            st.markdown('#### 상대 특징 RF · 실험용 비교')

            try:
                import json as _relative_json
                import hashlib as _relative_hashlib
                import joblib as _relative_joblib
                import numpy as _relative_np

                _relative_dir = Path(
                    Path(__file__).resolve().parent / 'outputs/web_artifacts/rf_relative_candidate'
                )

                from vibration.rf_relative_inference import (
                    predict_relative_rf,
                )

                _relative_result = predict_relative_rf(
                    report['features'],
                    folder=_relative_dir,
                )

                _relative_prediction = (
                    _relative_result['raw_prediction']
                )

                _relative_scores = (
                    _relative_result['scores_uncalibrated']
                )

                _relative_provenance = (
                    _relative_result['provenance_mode']
                )

                st.dataframe(
                    pd.DataFrame([
                        {
                            '모델': '기존 4분류 RF',
                            '원시 예측': str(raw_class),
                        },
                        {
                            '모델': '상대 특징 3분류 RF',
                            '원시 예측': _relative_prediction,
                        },
                    ]),
                    hide_index=True,
                    width='stretch',
                )

                st.markdown('**상대 특징 RF 클래스별 원시 점수**')

                st.dataframe(
                    pd.DataFrame([
                        {
                            '결함': name,
                            'RF 원시 점수': round(score, 4),
                        }
                        for name, score in sorted(
                            _relative_scores.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]),
                    hide_index=True,
                    width='stretch',
                )

                st.caption(
                    '상대 특징 RF는 기존 Train의 결함 데이터만 '
                    '학습한 실험 모델입니다. Normal을 구분하지 '
                    '못하며, 원시 점수는 보정된 결함 확률이 '
                    '아닙니다. 기존 AE 판정과 RF 운영 모델에 '
                    '영향을 주지 않습니다.'
                )

            except Exception as _relative_error:
                st.warning(
                    '상대 특징 RF 비교 불가: '
                    + type(_relative_error).__name__
                    + ': ' + str(_relative_error)
                )

            # RF_NORMAL_TRAIN_REFERENCE_V1
            st.markdown('#### 정상 Train 기준 대비 진동 특징')

            try:
                import json as _json
                import hashlib as _hashlib

                ref_path = Path(
                    Path(__file__).resolve().parent / 'outputs/web_artifacts/normal_reference/isu_normal_feature_reference_v1.json'
                )
                ref_data = _json.loads(
                    ref_path.read_text(encoding='utf-8')
                )

                split_path = Path(ref_data['source'])
                verified = (
                    ref_data.get('source_split') == 'train'
                    and ref_data.get('source_label') == 'Normal'
                    and ref_data.get('validated_as_fault_threshold') is False
                    and _hashlib.sha256(
                        split_path.read_bytes()
                    ).hexdigest() == ref_data['source_sha256']
                )

                if not verified:
                    raise ValueError('정상 기준 출처·무결성 불일치')

                rpm_reference = ref_data['reference'][str(rpm)]
                normal_features = rpm_reference['features']
                comparison = []

                for key, (label, unit) in FEATURE_LABELS.items():
                    value = report['features'].get(key)
                    stats = normal_features[key]

                    if value is None:
                        status = '측정값 없음'
                    elif value < stats['p05']:
                        status = 'P5 미만'
                    elif value > stats['p95']:
                        status = 'P95 초과'
                    else:
                        status = '참고 범위 내'

                    comparison.append({
                        '특징량': label,
                        '입력값': value,
                        '정상 중앙값': stats['median'],
                        '정상 P5': stats['p05'],
                        '정상 P95': stats['p95'],
                        '비교': status,
                        '단위': unit,
                    })

                st.dataframe(
                    pd.DataFrame(comparison),
                    hide_index=True,
                    width='stretch',
                )

                outside = [
                    row['특징량'] for row in comparison
                    if row['비교'] in ('P5 미만', 'P95 초과')
                ]

                st.caption(
                    f"{rpm} RPM · 정상 Train "
                    f"{rpm_reference['normal_train_files']}개 기준 · "
                    '5~95백분위 참고 범위'
                )

                if outside:
                    st.info(
                        '정상 참고 범위를 벗어난 특징: '
                        + ', '.join(outside)
                    )
                else:
                    st.info(
                        '이번 입력에서 정상 참고 범위를 벗어난 '
                        '측정 특징은 없습니다.'
                    )

                st.caption(
                    '정상 기준은 RF 학습 분할의 Normal Train만으로 '
                    '계산했습니다. 결함명이나 업로드 날짜로 기준을 '
                    '선택하지 않습니다. P5~P95는 탐색적 참고 범위이며 '
                    '검증된 이상 판정 임계값이 아닙니다. '
                    '이 범위 밖이라는 이유만으로 결함을 확정하지 않습니다.'
                )

            except Exception as error:
                st.warning(
                    '정상 Train 특징 비교 불가: '
                    + type(error).__name__ + ': ' + str(error)
                )

            family = {
                'Bearing': '베어링',
                'Unbalance': '불평형',
                'Looseness': '느슨함',
            }.get(raw_class)

            if family:
                review = next(
                    (
                        item for item in report['reviews']
                        if item['family'] == family
                    ),
                    None,
                )

                if review is not None:
                    st.markdown('#### 결함 후보와 관련된 진동 특성')
                    st.write(review['interpretation'])

                    related = [
                        {
                            '관련 특징': FEATURE_LABELS[key][0],
                            '측정값': report['features'].get(key),
                        }
                        for key in review['feature_keys']
                    ]

                    st.dataframe(
                        pd.DataFrame(related),
                        hide_index=True,
                        width='stretch',
                    )

                    st.caption('판단 한계: ' + review['limitation'])

            st.info(
                '위 특징은 해당 결함과 관련하여 검토할 관측값입니다. '
                'RF가 이 특징 때문에 해당 클래스를 선택했다는 '
                '설명이나 정상 대비 증가를 입증하는 결과는 아닙니다.'
            )

            with st.expander('RF 판단 보류 사유'):
                for reason in prediction['reasons']:
                    st.write('- ' + reason)

            st.caption(
                '운영 AE 결과 및 분석 이력은 변경하지 않습니다. '
                '실험용 분석 결과는 현재 웹 세션에서만 유지합니다.'
            )


    with st.expander('AE 입력·복원 Spectrum 상세 분석', expanded=False):
        if result is not None and preview is not None:
            try:
                raw = st.session_state['v3_upload'].getvalue()
                data = get_reconstruction_data(
                    st.session_state, raw, rpm, config, result
                )
                render_reconstruction(data, result)

                # Supplementary analysis reuses the existing AE reconstruction.
                st.divider()
                try:
                    bandwise_analysis = analyze_reconstruction_data(
                        data, result, rpm
                    )
                    render_bandwise_analysis(bandwise_analysis)
                except Exception as bandwise_error:
                    st.warning(
                        '대역별 보조 분석 불가: ' + str(bandwise_error)
                    )
            except Exception as error:
                st.warning('AE 복원 시각화 불가: ' + str(error))
        else:
            st.info('MAT 파일 분석 후 복원 Spectrum을 확인할 수 있습니다.')
    v2.details_panel(result,candidate,preview,details)

if __name__=='__main__': main()