"""Versioned local RF inference and validation-derived abstention; no MAT labels used."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from scripts.evaluate_fault_classifier import CLASSES, FEATURES, RF_CONFIG

SCHEMA = 'ch3_eda9_v1'
CONFIG = {'channel': 3, 'fs': 25600, 'samples': 65536, 'order_tolerance_hz': 1.,
          'envelope_carrier_hz': [2000., 8000.], 'envelope_trim_seconds': .1,
          'bands_hz': [[0, 200], [200, 2000], [2000, 10000]]}
LABELS = {'Bearing': '베어링 결함 의심', 'Unbalance': '불평형 의심', 'Looseness': '느슨함 의심'}
POLICY_CONFIG = {'min_validation_accepts': 5, 'target_validation_precision': .8,
                 'require_training_range': True, 'require_no_missing': True}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_identity():
    folder = os.environ.get('ISU_CHEMICAL_CANDIDATE_BUNDLE', '').strip()
    enabled = os.environ.get('ISU_CHEMICAL_CANDIDATE_LABELS', '0') == '1'
    if not folder:
        return None, enabled, None
    folder = Path(folder).expanduser().resolve()
    try:
        stamp = tuple((name, (folder / name).stat().st_mtime_ns, (folder / name).stat().st_size)
                      for name in ('manifest.json', 'model.joblib'))
    except OSError:
        stamp = 'missing'
    return str(folder), enabled, stamp


def feature_matrix(mappings):
    rows = []
    for mapping in mappings:
        if set(mapping) != set(FEATURES):
            raise ValueError('특징량 이름·스키마가 모델과 다릅니다.')
        row = [np.nan if mapping[f] is None else float(mapping[f]) for f in FEATURES]
        if any(np.isinf(v) or (np.isfinite(v) and not 0 <= v <= np.finfo(np.float32).max) for v in row):
            raise ValueError('모델 입력에 비유한값 또는 범위 오류가 있습니다.')
        if np.isnan(row).all():
            raise ValueError('모델 입력 특징량이 모두 결측입니다.')
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(-1, len(FEATURES))


def train_pipeline(x, labels):
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('forest', RandomForestClassifier(**RF_CONFIG))]).fit(x, labels)


def training_bounds(x):
    return {'min': [float(np.nanmin(v)) if np.isfinite(v).any() else None for v in x.T],
            'max': [float(np.nanmax(v)) if np.isfinite(v).any() else None for v in x.T]}


def raw_predictions(pipeline, x):
    votes = pipeline.predict_proba(x)
    if not np.isfinite(votes).all():
        raise ValueError('RF 분류 점수가 유한하지 않습니다.')
    index = votes.argmax(axis=1)
    sorted_votes = np.sort(votes, axis=1)
    margins = sorted_votes[:, -1] - sorted_votes[:, -2]
    return pipeline.classes_[index], votes, margins


def in_training_range(values, bounds):
    violations = []
    for j, value in enumerate(values):
        if np.isfinite(value) and (bounds['min'][j] is None or not bounds['min'][j] <= value <= bounds['max'][j]):
            violations.append(FEATURES[j])
    return violations


def derive_policy(pipeline, x, labels, rpms, bounds, supported_rpms=None):
    """Use validation only; test data is neither an argument nor a policy switch."""
    labels, rpms = np.asarray(labels), np.asarray(rpms)
    predicted, _, margins = raw_predictions(pipeline, x)
    eligible = np.asarray([not np.isnan(v).any() and not in_training_range(v, bounds) for v in x])
    rules, diagnostics = {}, []
    for rpm in sorted(set(rpms.tolist())):
        for label in LABELS:
            indices = np.flatnonzero((rpms == rpm) & (predicted == label) & eligible & (margins > 0))
            best = None
            if (sum((rpms == rpm) & (labels == label)) < POLICY_CONFIG['min_validation_accepts']
                    or (supported_rpms is not None and rpm not in supported_rpms)):
                indices = np.asarray([], dtype=int)
            for threshold in sorted(set(float(margins[i]) for i in indices)):
                selected = [i for i in indices if margins[i] >= threshold]
                accuracy = float(np.mean(labels[selected] == label))
                if len(selected) >= POLICY_CONFIG['min_validation_accepts'] and accuracy >= POLICY_CONFIG['target_validation_precision']:
                    candidate = {'margin_min': threshold, 'validation_accepts': len(selected),
                                 'validation_correct': int(sum(labels[selected] == label)), 'validation_precision': accuracy}
                    if best is None or len(selected) > best['validation_accepts']:
                        best = candidate
            if best is not None:
                rules[f'{rpm}:{label}'] = best
            diagnostics.append({'rpm': rpm, 'class': label, 'validation_truth_n': int(sum((rpms == rpm) & (labels == label))),
                                'eligible_predictions': len(indices), 'policy_available': best is not None})
    return {'config': dict(POLICY_CONFIG), 'rules': rules, 'diagnostics': diagnostics,
            'limitations': 'Thresholds selected on validation; validation precision is optimistic, not a calibrated guarantee. Test is evaluated once without refitting.'}


def decide(predicted, margin, values, rpm, bounds, policy, supported_rpms, labels_enabled):
    reasons = []
    if predicted == 'Normal':
        reasons.append('AE 이상과 RF Normal의 판정 불일치')
    if rpm not in supported_rpms:
        reasons.append('학습 RPM 적용 범위 밖')
    if np.isnan(values).any():
        reasons.append('특징 결측: 훈련 중앙값으로 계산했으나 자동 후보는 보류')
    outside = in_training_range(values, bounds)
    if outside:
        reasons.append('훈련 특징 범위 이탈: ' + ', '.join(outside))
    rule = policy['rules'].get(f'{rpm}:{predicted}')
    if rule is None:
        reasons.append('해당 RPM·클래스의 보류 기준 검증 표본 또는 성능 부족')
    elif margin <= 0 or margin < rule['margin_min']:
        reasons.append('분류 점수 차이가 검증 기준 미만')
    if not labels_enabled:
        reasons.append('자동 후보명 표시 비활성: 실제 평가 검토 후 설정 필요')
    return not reasons, reasons, rule


class CandidateEngine:
    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        self.manifest = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        training = json.loads((self.folder / 'training_report.json').read_text(encoding='utf-8'))
        if (training.get('completed') is not True or training.get('inputs_unchanged') is not True or
                training.get('status') != 'trained_evaluated_no_refit' or
                training.get('manifest_sha256') != sha256(self.folder / 'manifest.json')):
            raise ValueError('학습 보고서가 미완료이거나 manifest 무결성이 일치하지 않습니다.')
        if (m.get('schema_version') != 1 or m.get('feature_schema') != SCHEMA or m.get('features') != list(FEATURES)
                or m.get('feature_config') != CONFIG or m.get('classes') != list(CLASSES) or m.get('rf_config') != RF_CONFIG
                or m.get('policy', {}).get('config') != POLICY_CONFIG or m.get('auto_labels_default') is not False):
            raise ValueError('후보 모델 특징량·전처리·보류 정책 스키마가 호환되지 않습니다.')
        if m.get('versions') != {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__}:
            raise ValueError('후보 모델 패키지 버전이 현재 환경과 다릅니다.')
        if sha256(self.folder / 'model.joblib') != m['model_sha256']:
            raise ValueError('후보 모델 SHA256 불일치: 손상 또는 교체 여부를 확인하세요.')
        if sha256(self.folder / 'split_files.csv') != m['split_files_sha256']:
            raise ValueError('후보 모델 학습·평가 파일 목록 해시가 다릅니다.')
        bounds = m['training_bounds']
        if any(len(bounds[k]) != len(FEATURES) for k in ('min', 'max')):
            raise ValueError('훈련 특징 범위가 호환되지 않습니다.')
        for lo, hi in zip(bounds['min'], bounds['max']):
            if (lo is None) != (hi is None) or (lo is not None and (not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi)):
                raise ValueError('훈련 특징 범위가 유효하지 않습니다.')
        dates = [set(m['split_dates'][role]) for role in ('train', 'validation', 'test')]
        if any(not d for d in dates) or any(dates[i] & dates[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('후보 모델 날짜 분리가 유효하지 않습니다.')
        for key, rule in m['policy']['rules'].items():
            rpm, label = key.split(':')
            if (int(rpm) not in m['training_rpms'] or label not in LABELS or
                    not np.isfinite(rule['margin_min']) or not 0 < rule['margin_min'] <= 1 or
                    rule['validation_accepts'] < POLICY_CONFIG['min_validation_accepts'] or
                    not POLICY_CONFIG['target_validation_precision'] <= rule['validation_precision'] <= 1):
                raise ValueError('후보 보류 정책이 유효하지 않습니다.')
        # Only load trusted locally trained artifacts; MAT upload never accepts a pickle.
        self.pipeline = joblib.load(self.folder / 'model.joblib')
        if (not isinstance(self.pipeline, Pipeline) or list(self.pipeline.named_steps) != ['imputer', 'forest']
                or not isinstance(self.pipeline.named_steps['imputer'], SimpleImputer)
                or not isinstance(self.pipeline.named_steps['forest'], RandomForestClassifier)
                or self.pipeline.n_features_in_ != len(FEATURES) or set(self.pipeline.classes_) != set(CLASSES)):
            raise ValueError('후보 모델 파이프라인이 호환되지 않습니다.')
        imputer, forest = self.pipeline.named_steps['imputer'], self.pipeline.named_steps['forest']
        if imputer.strategy != 'median' or not imputer.keep_empty_features or any(forest.get_params()[k] != v for k, v in RF_CONFIG.items()):
            raise ValueError('저장 모델과 학습 설정이 다릅니다.')
        if imputer.statistics_.shape != (len(FEATURES),) or not np.isfinite(imputer.statistics_).all():
            raise ValueError('저장 결측 처리 통계가 유효하지 않습니다.')

    def predict(self, features, rpm, labels_enabled=False):
        x = feature_matrix([features])
        predicted, votes, margins = raw_predictions(self.pipeline, x)
        accepted, reasons, rule = decide(str(predicted[0]), float(margins[0]), x[0], rpm,
                                         self.manifest['training_bounds'], self.manifest['policy'],
                                         self.manifest['training_rpms'], labels_enabled)
        return {'kind': 'RandomForest', 'model_version': self.manifest['model_version'],
                'model_sha256': self.manifest['model_sha256'], 'feature_schema': SCHEMA,
                'raw_prediction': str(predicted[0]), 'scores_uncalibrated': dict(zip(self.pipeline.classes_.tolist(), votes[0].tolist())),
                'score_margin': float(margins[0]), 'accepted': accepted, 'reasons': reasons, 'policy_rule': rule,
                'labels_enabled': labels_enabled}
