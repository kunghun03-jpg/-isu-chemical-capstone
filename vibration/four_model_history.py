"""4개 모델의 분석 이력 통합 조회. DB 읽기 전용."""

from contextlib import closing
from datetime import date
import sqlite3


def _search_core_four_model_history(
    store, start=None, end=None,
    rpm=None, model=None, decision=None
):
    if not store.path.is_file():
        return []

    if start is not None:
        start = date.fromisoformat(str(start)).isoformat()
    if end is not None:
        end = date.fromisoformat(str(end)).isoformat()
    if start and end and start > end:
        raise ValueError("날짜 범위 오류")

    if rpm is not None and rpm not in (900, 1200, 1800):
        raise ValueError("RPM 오류")

    if model is not None and model not in (
        "autoencoder", "ocsvm", "iforest",
        "dae_experimental"
    ):
        raise ValueError("모델 필터 오류")

    if decision is not None and decision not in (0, 1):
        raise ValueError("판정 필터 오류")

    uri = store.path.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row

        has_dae = conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table'
               AND name='dae_experimental_results'"""
        ).fetchone() is not None

        # 기존 세 모델의 조회 형식을 유지한다.
        legacy = """
            SELECT
                a.id AS analysis_id,
                a.analyzed_at,
                a.analyzed_date,
                a.saved_at,
                a.filename,
                a.file_sha256,
                a.rpm,
                a.channel,
                a.bundle_id,
                a.source_experiment,
                m.model,
                m.model_version,
                m.score,
                m.threshold,
                m.predicted_anomaly,
                'date_based_exploratory'
                    AS evaluation_scope,
                NULL AS validated_for_operation
            FROM analyses a
            JOIN model_results m
              ON m.analysis_id = a.id
        """

        dae = """
            SELECT
                a.id AS analysis_id,
                a.analyzed_at,
                a.analyzed_date,
                a.saved_at,
                a.filename,
                a.file_sha256,
                a.rpm,
                a.channel,
                a.bundle_id,
                a.source_experiment,
                'dae_experimental' AS model,
                d.model_version,
                d.score,
                d.threshold,
                d.predicted_anomaly,
                d.evaluation_scope,
                d.validated_for_operation
            FROM analyses a
            JOIN dae_experimental_results d
              ON d.analysis_id = a.id
        """

        combined = (
            legacy + " UNION ALL " + dae
            if has_dae else legacy
        )

        clauses = []
        params = []

        if start is not None:
            clauses.append("analyzed_date >= ?")
            params.append(start)

        if end is not None:
            clauses.append("analyzed_date <= ?")
            params.append(end)

        if rpm is not None:
            clauses.append("rpm = ?")
            params.append(rpm)

        if model is not None:
            clauses.append("model = ?")
            params.append(model)

        if decision is not None:
            clauses.append("predicted_anomaly = ?")
            params.append(decision)

        sql = "SELECT * FROM (" + combined + ")"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += (
            " ORDER BY analyzed_at DESC,"
            " analysis_id DESC, model"
        )

        return [
            dict(row)
            for row in conn.execute(sql, params)
        ]


def search_four_model_history(
    store, start=None, end=None,
    rpm=None, model=None, decision=None
):
    """기존 이력 + V3 참고 모델을 중복 없이 통합 조회.

    데이터베이스에는 기록하지 않는 읽기 전용 함수.
    """
    import math
    from vibration.dashboard_v3 import read_references

    allowed = {
        "autoencoder", "ocsvm",
        "iforest", "dae_experimental"
    }

    if model is not None and model not in allowed:
        raise ValueError("모델 필터 오류")

    if decision is not None and decision not in (0, 1):
        raise ValueError("판정 필터 오류")

    # 모델·판정 조건을 적용하기 전에 분석 ID를 모두 확보
    rows = _search_core_four_model_history(
        store,
        start=start,
        end=end,
        rpm=rpm,
        model=None,
        decision=None,
    )

    if not rows:
        return []

    # 분석 ID별 기존 DB 기록 확보
    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["analysis_id"], []
        ).append(row)

    for analysis_id, existing in grouped.items():
        # 저장된 V3 JSON의 SHA256 무결성 검증 포함
        references = read_references(
            store, analysis_id
        )

        if not references:
            continue

        # 기존 model_results에 저장된 모델 중복 방지
        present = {
            row["model"] for row in existing
        }

        parent = existing[0]

        for ref in references:
            name = ref.get("model")

            # AE와 DAE는 각각 기존 전용 테이블 사용
            if name not in ("ocsvm", "iforest"):
                continue

            if name in present:
                continue

            if ref.get("status") != "완료":
                continue

            score = float(ref["score"])
            threshold = float(ref["threshold"])
            version = ref.get("version")
            predicted = int(score > threshold)

            if (
                not math.isfinite(score)
                or not math.isfinite(threshold)
                or not isinstance(version, str)
                or not version
                or ref.get("decision") != (
                    "이상" if predicted else "정상"
                )
            ):
                raise ValueError(
                    f"분석 #{analysis_id} {name} 결과 오류"
                )

            new_row = dict(parent)
            new_row.update(
                model=name,
                model_version=version,
                score=score,
                threshold=threshold,
                predicted_anomaly=predicted,
                evaluation_scope="date_based_exploratory",
                validated_for_operation=None,
            )

            rows.append(new_row)
            present.add(name)

    # 통합된 결과에 모델·판정 필터 적용
    if model is not None:
        rows = [
            row for row in rows
            if row["model"] == model
        ]

    if decision is not None:
        rows = [
            row for row in rows
            if row["predicted_anomaly"] == decision
        ]

    rows.sort(
        key=lambda row: (
            row["analyzed_at"],
            row["analysis_id"],
        ),
        reverse=True,
    )

    return rows
