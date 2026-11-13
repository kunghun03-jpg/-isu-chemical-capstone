"""Optional additive evidence table; legacy analyses/model_results remain unchanged."""
from contextlib import closing
import hashlib
import json
import math
import sqlite3

from vibration.candidate_analysis import FEATURE_LABELS, UNCLEAR, VERSION, STAGE_VERSION, SKIPPED
from vibration.candidate_engine import LABELS


def encode_candidate(report, result):
    legacy = report.get('schema_version') == 1 and report.get('analyzer_version') == VERSION
    stage = report.get('schema_version') == 2 and report.get('analyzer_version') == STAGE_VERSION
    status = report.get('status')
    valid_decision = ((legacy and status in ('unclear', 'unavailable') or stage and status in ('unavailable', 'abstained'))
                      and report.get('candidate') == UNCLEAR)
    if stage and status == 'skipped_normal':
        valid_decision = report.get('candidate') == SKIPPED and not result['models']['autoencoder']['predicted_anomaly']
    if stage and status == 'candidate':
        engine = report.get('engine') or {}
        valid_decision = (engine.get('raw_prediction') in LABELS and engine.get('accepted') is True and report.get('abstained') is False
                          and engine.get('labels_enabled') is True and engine.get('policy_rule') is not None
                          and report.get('candidate') == LABELS.get(engine.get('raw_prediction'))
                          and bool(result['models']['autoencoder']['predicted_anomaly']))
    if (not (legacy or stage) or not valid_decision or
            report.get('confidence') is not None or report.get('channel') != 3 or
            report.get('input_sha256') != result['input']['sha256'] or report.get('rpm') != result['input']['rpm'] or
            set(result['models']) != {'autoencoder'}):
        raise ValueError('후보 근거와 AE 입력의 대응을 확인하세요.')
    features = report.get('features', {})
    if report['status'] != 'unavailable' and set(features) != set(FEATURE_LABELS):
        raise ValueError('후보 특징량이 누락되었습니다.')
    if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0)
           for v in features.values()):
        raise ValueError('후보 특징량을 확인하세요.')
    return json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)


def insert_candidate(connection, analysis_id, payload):
    """Run inside HistoryStore.save's existing transaction; no executescript/commit."""
    connection.execute('''CREATE TABLE IF NOT EXISTS candidate_reports (
        analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
        analyzer_version TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
        PRIMARY KEY (analysis_id, analyzer_version)
    )''')
    checksum = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    version = json.loads(payload)['analyzer_version']
    existing = connection.execute('SELECT payload_sha256 FROM candidate_reports WHERE analysis_id=? AND analyzer_version=?',
                                  (analysis_id, version)).fetchone()
    if existing is not None:
        if existing[0] != checksum:
            raise ValueError('같은 분석의 기존 후보 기록은 덮어쓰지 않습니다.')
        return
    connection.execute('INSERT INTO candidate_reports VALUES (?, ?, ?, ?)', (analysis_id, version, payload, checksum))


def read_candidate(store, analysis_id):
    """Read-only; older DBs are accepted without creating/migrating a table."""
    if not store.path.is_file():
        return None
    with closing(sqlite3.connect(store.path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='candidate_reports'").fetchone() is None:
            return None
        record = connection.execute('''SELECT payload_json, payload_sha256 FROM candidate_reports
                                    WHERE analysis_id=? AND analyzer_version IN (?, ?)
                                    ORDER BY CASE WHEN analyzer_version=? THEN 0 ELSE 1 END LIMIT 1''',
                                    (int(analysis_id), STAGE_VERSION, VERSION, STAGE_VERSION)).fetchone()
        if record is None:
            return None
        if hashlib.sha256(record[0].encode('utf-8')).hexdigest() != record[1]:
            raise ValueError('저장된 후보 근거 무결성을 확인하세요.')
        return json.loads(record[0])
