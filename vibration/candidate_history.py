"""Optional additive evidence table; legacy analyses/model_results remain unchanged."""
from contextlib import closing
import hashlib
import json
import math
import sqlite3

from vibration.candidate_analysis import FEATURE_LABELS, UNCLEAR, VERSION


def encode_candidate(report, result):
    if (report.get('schema_version') != 1 or report.get('analyzer_version') != VERSION or
            report.get('candidate') != UNCLEAR or report.get('status') not in ('unclear', 'unavailable') or
            report.get('confidence') is not None or report.get('channel') != 3 or
            report.get('input_sha256') != result['input']['sha256'] or report.get('rpm') != result['input']['rpm'] or
            set(result['models']) != {'autoencoder'}):
        raise ValueError('후보 근거와 AE 입력의 대응을 확인하세요.')
    features = report.get('features', {})
    if report['status'] == 'unclear' and set(features) != set(FEATURE_LABELS):
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
    existing = connection.execute('SELECT payload_sha256 FROM candidate_reports WHERE analysis_id=? AND analyzer_version=?',
                                  (analysis_id, VERSION)).fetchone()
    if existing is not None:
        if existing[0] != checksum:
            raise ValueError('같은 분석의 기존 후보 기록은 덮어쓰지 않습니다.')
        return
    connection.execute('INSERT INTO candidate_reports VALUES (?, ?, ?, ?)', (analysis_id, VERSION, payload, checksum))


def read_candidate(store, analysis_id):
    """Read-only; older DBs are accepted without creating/migrating a table."""
    if not store.path.is_file():
        return None
    with closing(sqlite3.connect(store.path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='candidate_reports'").fetchone() is None:
            return None
        record = connection.execute('SELECT payload_json, payload_sha256 FROM candidate_reports WHERE analysis_id=? AND analyzer_version=?',
                                    (int(analysis_id), VERSION)).fetchone()
        if record is None:
            return None
        if hashlib.sha256(record[0].encode('utf-8')).hexdigest() != record[1]:
            raise ValueError('저장된 후보 근거 무결성을 확인하세요.')
        return json.loads(record[0])
