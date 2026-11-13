"""SQLite 분석 건/모델 결과 저장. 원본 MAT나 파형은 저장하지 않습니다."""

from contextlib import closing
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import sqlite3

KST = timezone(timedelta(hours=9))


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            self._initialize(connection)
        except Exception:
            connection.close()
            raise
        return connection

    @staticmethod
    def _initialize(connection):
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY, request_key TEXT NOT NULL UNIQUE,
                analyzed_at TEXT NOT NULL, analyzed_date TEXT NOT NULL,
                saved_at TEXT NOT NULL, filename TEXT NOT NULL, file_sha256 TEXT NOT NULL,
                rpm INTEGER NOT NULL, channel INTEGER NOT NULL,
                bundle_id TEXT NOT NULL, source_experiment TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_results (
                analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                model TEXT NOT NULL, model_version TEXT NOT NULL,
                score REAL NOT NULL, threshold REAL NOT NULL,
                predicted_anomaly INTEGER NOT NULL CHECK(predicted_anomaly IN (0,1)),
                PRIMARY KEY(analysis_id, model)
            );
            CREATE INDEX IF NOT EXISTS history_date_rpm ON analyses(analyzed_date, rpm);
        """)

    def save(self, result, model_versions):
        timestamp = datetime.fromisoformat(result["analyzed_at_utc"])
        if timestamp.tzinfo is None:
            raise ValueError("분석 시각에 시간대가 필요합니다.")
        models = []
        for name, item in sorted(result["models"].items()):
            score, threshold = float(item["file_score"]), float(item["file_threshold"])
            predicted = int(item["predicted_anomaly"])
            if (
                name not in {"autoencoder", "ocsvm", "iforest"}
                or not math.isfinite(score)
                or not math.isfinite(threshold)
                or predicted != int(score > threshold)
            ):
                raise ValueError("저장 결과의 점수·임계값·판정을 확인해 주세요.")
            version = model_versions[name]
            if not version:
                raise ValueError("모델 버전이 필요합니다.")
            models.append((name, version, score, threshold, predicted))
        if not models:
            raise ValueError("저장할 모델 결과가 없습니다.")
        item = result["input"]
        parent = (
            timestamp.astimezone(timezone.utc).isoformat(),
            timestamp.astimezone(KST).date().isoformat(),
            item["name"],
            item["sha256"],
            int(item["rpm"]),
            int(item["channel"]),
            result["bundle_id"],
            result["source_experiment"],
        )
        # 저장 클릭 시각이 아닌 분석 시각과 결과로 키를 생성합니다. DB UNIQUE가 동시 저장도 차단합니다.
        request_key = hashlib.sha256(
            json.dumps([parent, models], ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO analyses
                (request_key, analyzed_at, analyzed_date, filename, file_sha256, rpm, channel, bundle_id, source_experiment, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_key) DO NOTHING""",
                (request_key, *parent, datetime.now(timezone.utc).isoformat()),
            )
            created = cursor.rowcount == 1
            analysis_id = connection.execute(
                "SELECT id FROM analyses WHERE request_key=?", (request_key,)
            ).fetchone()["id"]
            if created:
                connection.executemany(
                    "INSERT INTO model_results VALUES (?, ?, ?, ?, ?, ?)",
                    [(analysis_id, *row) for row in models],
                )
        return analysis_id, created

    def search(self, start=None, end=None, rpm=None, model=None, decision=None):
        clauses, params = [], []
        for value, op in ((start, ">="), (end, "<=")):
            if value:
                normalized = date.fromisoformat(str(value)).isoformat()
                clauses.append(f"a.analyzed_date {op} ?")
                params.append(normalized)
        if rpm is not None:
            clauses.append("a.rpm=?")
            params.append(int(rpm))
        if model is not None:
            clauses.append("m.model=?")
            params.append(model)
        if decision is not None:
            clauses.append("m.predicted_anomaly=?")
            params.append(int(decision))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """SELECT a.id AS analysis_id, a.analyzed_at, a.analyzed_date,
                a.saved_at, a.filename, a.file_sha256, a.rpm, a.channel, a.bundle_id, a.source_experiment,
                m.model, m.model_version, m.score, m.threshold, m.predicted_anomaly
                FROM analyses a JOIN model_results m ON a.id=m.analysis_id"""
                    + where
                    + " ORDER BY a.analyzed_at DESC, a.id DESC, m.model",
                    params,
                )
            ]

    def detail(self, analysis_id):
        with closing(self._connect()) as connection:
            parent = connection.execute(
                "SELECT * FROM analyses WHERE id=?", (int(analysis_id),)
            ).fetchone()
            if parent is None:
                raise ValueError("저장된 분석 건을 찾을 수 없습니다.")
            models = connection.execute(
                "SELECT * FROM model_results WHERE analysis_id=? ORDER BY model",
                (int(analysis_id),),
            ).fetchall()
            return {"analysis": dict(parent), "models": [dict(row) for row in models]}


def history_csv(rows):
    """검색된 모델 결과를 한 행씩 내보냅니다. 분석 ID로 전체 모델을 묶을 수 있습니다."""
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # 사용자 파일명이 스프레드시트 수식으로 실행되지 않게 문자열만 보호합니다.
        safe = {
            key: (
                "'" + value
                if isinstance(value, str)
                and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
            )
            for key, value in row.items()
        }
        writer.writerow(safe)
    return stream.getvalue().encode("utf-8-sig")
