"""Observational CH3 evidence, deliberately separate from AE anomaly inference."""
import hashlib

import numpy as np

# Reuse the existing EDA math, not classifiers or harmonic improvement experiments.
from scripts.analyze_fault_signatures import analyze_signal
from vibration.web_inference import preview_mat
from vibration.candidate_engine import CandidateEngine, LABELS, bundle_identity

VERSION = 'ch3_candidate_evidence_v1'
STAGE_VERSION = 'ch3_candidate_engine_v2'
SKIPPED = 'AE 정상 · 후보 추정 생략'
UNCLEAR = '결함 후보 불명확'
FEATURE_LABELS = {
    'rms_ac': ('AC RMS', 'raw units'), 'peak_ac': ('AC Peak', 'raw units'),
    'crest_factor_ac': ('Crest Factor', '무차원'),
    'one_x_amplitude': ('1X 진폭', 'raw units'), 'two_x_amplitude': ('2X 진폭', 'raw units'),
    'energy_low': ('저주파 평균제곱 · 0–200Hz', 'raw units²'),
    'energy_mid': ('중주파 평균제곱 · 200–2000Hz', 'raw units²'),
    'energy_high': ('고주파 평균제곱 · 2000–10000Hz', 'raw units²'),
    'envelope_rms_ac': ('엔벨로프 AC RMS', 'raw units'),
}
LIMITATIONS = [
    '검증된 결함 분류 모델·판정 기준이 없어 종류별 진단과 후보 확률을 제공하지 않습니다.',
    '동일 RPM·측정 조건의 정상 기준이 없어 현재 특징량의 증가 여부를 확정할 수 없습니다.',
    '기존 날짜 분리 평가에서 느슨함 분류 성능이 매우 낮아 단순 고조파로 느슨함을 판정하지 않습니다.',
    '명목 RPM의 1X·2X이며 실제 회전수 계측값이 아닙니다. 센서 물리 단위와 부하 차이도 확인해야 합니다.',
    '이상 탐지와 결함 종류는 별개입니다. AE 정상은 결함 부재, AE 이상은 특정 결함을 보장하지 않습니다.',
]


def base_report(raw, rpm):
    if rpm not in (900, 1200, 1800):
        raise ValueError('지원 RPM을 확인하세요.')
    return {'schema_version': 1, 'analyzer_version': VERSION,
            'input_sha256': hashlib.sha256(raw).hexdigest(), 'rpm': rpm, 'channel': 3,
            'candidate': UNCLEAR, 'status': 'unclear', 'confidence': None,
            'features': {}, 'reviews': [], 'limitations': list(LIMITATIONS)}


def candidate_analysis(raw, rpm):
    """No filename, condition label, learned artifact or diagnostic threshold is used."""
    report = base_report(raw, rpm)
    loaded = preview_mat(raw)
    details = analyze_signal(loaded['signal'], loaded['fs'], rpm)
    features = {key: None if details['features'][key] is None else float(details['features'][key])
                for key in FEATURE_LABELS}
    if any(v is not None and not np.isfinite(v) for v in features.values()):
        raise ValueError('후보 분석 특징량이 유한하지 않습니다.')
    report.update(features=features,
                  measurement={'fs': loaded['fs'], 'samples': loaded['samples'], 'duration_seconds': loaded['duration'],
                               'fft_resolution_hz': loaded['fs'] / loaded['samples'],
                               'order_tolerance_hz': 1., 'envelope_carrier_hz': [2000., 8000.]},
                  orders={'one_x_hz': rpm / 60., 'two_x_hz': rpm / 30.,
                          'one_x_peak_hz': float(details['features']['one_x_peak_hz']),
                          'two_x_peak_hz': float(details['features']['two_x_peak_hz'])},
                  reviews=[
                      {'family': '베어링', 'feature_keys': ['energy_high', 'envelope_rms_ac', 'crest_factor_ac'],
                       'interpretation': '고주파·엔벨로프 특징을 관측했습니다. 다른 원인도 가능하며 정상 대비 증가가 검증되지 않았습니다.',
                       'limitation': '베어링 형상·결함 주파수·센서 대역폭 근거가 없어 IR/OR/Roller로 구분하지 않습니다.'},
                      {'family': '불평형', 'feature_keys': ['one_x_amplitude', 'two_x_amplitude', 'rms_ac'],
                       'interpretation': '명목 1X·2X 주변의 실제 FFT 진폭을 측정했습니다. 1X가 존재한다는 사실만으로 불평형을 판정하지 않습니다.',
                       'limitation': '실제 회전수, 정상 대비 기준, 부하와 센서 방향이 추가로 필요합니다.'},
                      {'family': '느슨함', 'feature_keys': ['two_x_amplitude', 'crest_factor_ac', 'peak_ac'],
                       'interpretation': '2X·파형 첨두 특성을 관측했습니다. 이 특징만으로 느슨함을 구분하는 성능은 검증되지 않았습니다.',
                       'limitation': '기존 Looseness Recall이 매우 낮아 후보를 확정하지 않습니다. 별도 점검과 독립 데이터 검증이 필요합니다.'},
                  ])
    return report, {'frequency': details['frequency'], 'amplitude': details['amplitude']}


def safe_candidate_analysis(raw, rpm):
    """A feature failure must never change or discard a successful AE result."""
    try:
        return candidate_analysis(raw, rpm)
    except Exception:
        report = base_report(raw, rpm)
        report.update(status='unavailable', failure='진동 특징 계산을 완료하지 못했습니다. AE 결과와 별도로 입력·수치 상태를 확인하세요.')
        return report, None


def analyze_with_engine(raw, rpm, ae_result):
    """Extract observations for display; execute RF only when this same input is AE-abnormal."""
    report, spectrum = safe_candidate_analysis(raw, rpm)
    report.update(schema_version=2, analyzer_version=STAGE_VERSION, abstained=True, reasons=[], engine=None)
    report['limitations'][0] = '결함 후보 엔진의 적용 여부와 보류 사유를 확인하세요. 후보는 확정 진단이 아니며 진단 성능은 별도 검증이 필요합니다.'
    if (report['input_sha256'] != ae_result['input']['sha256'] or rpm != ae_result['input']['rpm'] or
            set(ae_result['models']) != {'autoencoder'}):
        raise ValueError('AE와 후보 분석 입력이 일치하지 않습니다.')
    item = ae_result['models']['autoencoder']
    abnormal = bool(item['predicted_anomaly'])
    if abnormal != (item['file_score'] > item['file_threshold']):
        raise ValueError('AE 판정 규칙이 일치하지 않습니다.')
    if not abnormal:
        report['reasons'] = ['AE 정상으로 결함 후보 엔진 실행을 생략했습니다. 결함 부재를 보장하지 않습니다.']
        if report['status'] != 'unavailable':
            report.update(candidate=SKIPPED, status='skipped_normal', abstained=False)
        return report, spectrum
    if report['status'] == 'unavailable':
        report['reasons'] = [report['failure']]
        return report, spectrum
    folder, enabled, _ = bundle_identity()
    if folder is None:
        report.update(status='abstained', reasons=['후보 모델 미설정: ISU_CHEMICAL_CANDIDATE_BUNDLE 경로를 확인하세요.'])
        return report, spectrum
    try:
        engine = CandidateEngine(folder).predict(report['features'], rpm, labels_enabled=enabled)
        report.update(engine=engine, reasons=engine['reasons'], status='candidate' if engine['accepted'] else 'abstained',
                      abstained=not engine['accepted'], candidate=LABELS[engine['raw_prediction']] if engine['accepted'] else UNCLEAR)
        report['limitations'][0] = 'Random Forest 분류 점수는 보정된 확률·신뢰도가 아닙니다. 후보는 확정 진단이 아닙니다.'
    except Exception as error:
        report.update(status='abstained', reasons=[f'후보 모델 로드·입력·추론 오류: {type(error).__name__}: {error}'])
    return report, spectrum
