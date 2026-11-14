"""Read-only, offline Normal1/Normal2 evidence; no live inference or RPM policy."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_FOLDER = Path(__file__).resolve().parents[1] / 'outputs/normal_condition_comparison'
SUMMARY_NAME = 'normal1_vs_normal2_summary.csv'
CAUTION = '날짜 기준 예비 평가입니다. 실제 Run 독립성은 확인되지 않았으며 독립 시험 성능이나 모델 적용 범위 인증으로 해석할 수 없습니다.'
FEATURES = {
    'rms_median': ('RMS 중앙값', ('rms_median', 'rms_ac_median', 'median_rms')),
    'one_x_median': ('1X 진폭 중앙값', ('one_x_amplitude_median', 'one_x_median', '1x_median')),
    'two_x_median': ('2X 진폭 중앙값', ('two_x_amplitude_median', 'two_x_median', '2x_median')),
    'three_x_median': ('3X 진폭 중앙값', ('three_x_amplitude_median', 'three_x_median', '3x_median')),
}


def read_frame(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    names = [str(c).strip().lower() for c in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError('중복된 CSV 열 이름을 확인하세요.')
    frame.columns = names
    return frame


def find_column(frame, aliases, required=True):
    matches = [name for name in aliases if name in frame.columns]
    if len(matches) > 1:
        raise ValueError('같은 의미의 열이 여러 개 있습니다: ' + ', '.join(matches))
    if not matches and required:
        raise ValueError('필요한 CSV 열: ' + ' 또는 '.join(aliases))
    return matches[0] if matches else None


def load_comparison(path):
    """Normalize explicit median columns only. Never fit, impute or pool summaries."""
    source = read_frame(path)
    rpm = find_column(source, ('rpm',))
    condition = find_column(source, ('condition', 'normal_condition', 'normal_type'), required=False)
    groups = {k: find_column(source, aliases, required=False) for k, aliases in
              (('date', ('date', 'measurement_date')), ('run', ('run', 'run_id')))}
    optional_n = find_column(source, ('n', 'n_files', 'file_count'), required=False)
    parts = []
    if condition:
        part = source[[rpm, condition]].rename(columns={rpm: 'rpm', condition: 'condition'}).copy()
        for key, (_, aliases) in FEATURES.items():
            part[key] = source[find_column(source, aliases)]
        if optional_n:
            part['n'] = source[optional_n]
        for key, column in groups.items():
            if column:
                part[key] = source[column]
        parts.append(part)
    else:
        # Paired columns, e.g. normal1_rms_median / normal2_rms_median.
        for name in ('Normal1', 'Normal2'):
            part = source[[rpm]].rename(columns={rpm: 'rpm'}).copy()
            part['condition'] = name
            for key, (_, aliases) in FEATURES.items():
                part[key] = source[find_column(source, tuple(name.lower() + '_' + a for a in aliases))]
            for key, column in groups.items():
                if column:
                    part[key] = source[column]
            count_col = find_column(source, tuple(name.lower() + '_' + c for c in ('n', 'n_files', 'file_count')), required=False)
            if count_col:
                part['n'] = source[count_col]
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame['condition'] = frame['condition'].str.strip().str.lower().map({'normal1': 'Normal1', 'normal2': 'Normal2'})
    numeric = ['rpm', *FEATURES]
    if 'n' in frame:
        numeric.append('n')
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if frame.empty:
        return frame
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError('중앙값/RPM/표본 수에 빈 값·숫자 형식 오류·NaN/Inf가 있습니다. 값을 임의로 보완하지 않습니다.')
    if frame['condition'].isna().any() or not frame['rpm'].isin((900, 1200, 1800)).all():
        raise ValueError('조건은 Normal1/Normal2, RPM은 900/1200/1800이어야 합니다.')
    if (frame[list(FEATURES)] < 0).any().any():
        raise ValueError('RMS·진폭 중앙값은 음수가 될 수 없습니다.')
    if 'n' in frame and ((frame['n'] < 1).any() or (frame['n'] % 1 != 0).any()):
        raise ValueError('표본 수는 1 이상의 정수여야 합니다.')
    keys = ['rpm', 'condition']
    for group, column in groups.items():
        if column:
            frame[group] = frame[group].str.strip()
            if frame[group].eq('').any():
                raise ValueError('날짜/Run 열의 빈 값을 확인하세요. 다른 날짜와 합치지 않습니다.')
            keys.append(group)
    if frame.duplicated(keys).any():
        raise ValueError('동일 RPM·조건·날짜/Run 집계가 중복됩니다. 중앙값을 다시 평균내지 않습니다.')
    frame['rpm'] = frame['rpm'].astype(int)
    return frame.sort_values(keys).reset_index(drop=True)


def comparison_long(frame):
    return frame.melt(id_vars=[c for c in frame.columns if c not in FEATURES],
                      value_vars=list(FEATURES), var_name='feature', value_name='median')


def load_exclusions(path):
    frame = read_frame(path)
    # A file inventory may legitimately carry both a basename and a full path.
    def first(aliases):
        return next((name for name in aliases if name in frame), None)
    path_col = first(('path', 'file_path'))
    name_col = first(('filename', 'file_name', 'file'))
    reason_col = first(('reason', 'exclusion_reason', 'excluded_reason', 'error'))
    if not (path_col or name_col) or not reason_col:
        raise ValueError('제외 파일 CSV에는 filename/path 및 reason/error 열이 필요합니다.')
    columns = {
        '파일': path_col or name_col,
        'RPM': find_column(frame, ('rpm',)),
        '조건': find_column(frame, ('condition', 'normal_condition', 'normal_type')),
        '제외 사유': reason_col,
    }
    result = pd.DataFrame({label: frame[col].str.strip() for label, col in columns.items()})
    if result['파일'].eq('').any() or result['제외 사유'].eq('').any():
        raise ValueError('제외 파일명과 제외 사유의 빈 값을 확인하세요.')
    result['파일명'] = (frame[name_col].str.strip() if name_col else
                        result['파일'].str.replace('\\', '/', regex=False).str.rsplit('/').str[-1])
    basename = result['파일'].str.replace('\\', '/', regex=False).str.rsplit('/').str[-1]
    result['파일명'] = result['파일명'].where(result['파일명'].ne(''), basename)
    result['RPM'] = result['RPM'].replace('', '미기록')
    result['조건'] = result['조건'].replace('', '미기록')
    if 'error' in frame and reason_col != 'error':
        result['오류 상세'] = frame['error']
    return result[['파일명', 'RPM', '조건', '제외 사유', '파일', *(['오류 상세'] if '오류 상세' in result else [])]]


def comparison_charts(frame):
    long = comparison_long(frame)
    for pair in (('rms_median', 'one_x_median'), ('two_x_median', 'three_x_median')):
        for column, key in zip(st.columns(2), pair):
            with column:
                label = FEATURES[key][0]
                st.markdown('**' + label + '**')
                selected = long[long['feature'] == key]
                chart = alt.Chart(selected).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X('rpm:O', title='RPM', sort=[900, 1200, 1800], axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset('condition:N', sort=['Normal1', 'Normal2']),
                    y=alt.Y('median:Q', title='중앙값 (센서 raw 단위)', scale=alt.Scale(zero=True)),
                    color=alt.Color('condition:N', title='정상 조건', sort=['Normal1', 'Normal2'],
                                    scale=alt.Scale(domain=['Normal1', 'Normal2'], range=['#246ee9', '#21a790'])),
                    tooltip=[alt.Tooltip('rpm:O', title='RPM'), alt.Tooltip('condition:N', title='조건'),
                             alt.Tooltip('median:Q', title=label, format='.8g')]).properties(height=220)
                st.altair_chart(chart, width='stretch')


def render_normal_comparison(folder=None):
    folder = Path(folder) if folder is not None else DEFAULT_FOLDER
    st.subheader('정상 조건 검증 · Normal1 vs Normal2')
    st.warning(CAUTION)
    st.caption('기존 오프라인 집계 CSV를 읽기만 합니다. 정상 기준·모델 정확도·운영 임계값으로 사용하지 않습니다.')
    path = folder / SUMMARY_NAME
    if not path.is_file():
        st.info('정상 조건 비교 자료 없음 · outputs/normal_condition_comparison/' + SUMMARY_NAME + ' 파일을 배치해 주세요.')
    else:
        try:
            frame = load_comparison(path)
            if frame.empty:
                st.info('정상 조건 비교 CSV에 표시할 집계 행이 없습니다.')
            else:
                selected = frame
                for key, label in (('date', '측정 날짜'), ('run', '원본 Run 표기')):
                    if key in frame:
                        options = sorted(selected[key].unique())
                        choice = st.selectbox(label, options, key='normal_compare_' + key) if len(options) > 1 else options[0]
                        selected = selected[selected[key] == choice]
                        st.caption(label + ': ' + str(choice))
                if 'date' not in frame:
                    st.caption('CSV에 날짜 열이 없어 날짜별 구분을 확인할 수 없습니다.')
                missing = [f'{rpm}RPM {name}' for rpm in (900, 1200, 1800) for name in ('Normal1', 'Normal2')
                           if selected[(selected['rpm'] == rpm) & (selected['condition'] == name)].empty]
                if missing:
                    st.info('현재 선택 범위의 누락 집계: ' + ', '.join(missing) + ' · 누락값은 0으로 표시하지 않습니다.')
                comparison_charts(selected)
                labels = {'rpm': 'RPM', 'condition': '조건', 'date': '측정 날짜', 'run': '원본 Run 표기', 'n': '원본 표본 수',
                          **{key: value[0] for key, value in FEATURES.items()}}
                st.dataframe(selected.rename(columns=labels), hide_index=True, width='stretch')
                st.caption('CSV의 중앙값을 그대로 표시합니다. 특징 간 척도·센서 단위를 임의로 통일하지 않습니다. Run 열이 있어도 독립성이 검증됐다는 뜻은 아닙니다.')
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.warning('정상 조건 비교 자료를 표시하지 못했습니다: ' + str(error))
    # Keep quality evidence available even if the comparison file is missing or invalid.
    st.subheader('데이터 품질 · 제외 파일')
    excluded = folder / 'excluded_files.csv'
    if not excluded.is_file():
        st.info('제외 파일 자료 없음 · excluded_files.csv가 없으며 제외 파일이 없다는 의미는 아닙니다.')
    else:
        try:
            quality = load_exclusions(excluded)
            if quality.empty:
                st.info('excluded_files.csv에 제외 파일 기록이 없습니다.')
            else:
                st.caption(f'CSV에 기록된 제외 항목 {len(quality)}건 · 현재 그래프 선택 범위와 별도로 전체 기록을 표시합니다.')
                st.dataframe(quality, hide_index=True, width='stretch')
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.warning('제외 파일 자료를 표시하지 못했습니다: ' + str(error))
