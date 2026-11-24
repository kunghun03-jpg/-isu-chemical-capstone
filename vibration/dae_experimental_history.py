"""DAE 탐색적 결과 추가 저장.

기존 analyses/model_results의 구조와 저장 정책을 변경하지 않는다.
"""
from contextlib import closing
from pathlib import Path
import hashlib
import json
import math
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    ROOT / "outputs/dae_group_split_experiment_20261119"
)

ARTIFACTS = (
    "dae_group_split_best.pth",
    "dae_normalization.npz",
    "dae_thresholds.npz",
)


def model_version():
    """가중치·정규화·임계값 세 파일의 결합 지문."""
    hashes = {}

    for name in ARTIFACTS:
        path = ARTIFACT_DIR / name
        digest = hashlib.sha256()

        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

        hashes[name] = digest.hexdigest()

    encoded = json.dumps(hashes, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def save_dae_experimental(store, analysis_id, raw, rpm, result):
    """기존 분석 ID의 입력과 일치하는 DAE 결과만 연결."""
    if type(raw) is not bytes or not raw:
        raise ValueError("원본 MAT bytes가 필요합니다.")

    if type(analysis_id) is not int or analysis_id < 1:
        raise ValueError("분석 ID 오류")

    if type(rpm) is not int or rpm not in (900, 1200, 1800):
        raise ValueError("RPM 오류")

    if (
        result.get("model") != "DAE"
        or result.get("rpm") != rpm
        or result.get("evaluation_scope") != "date_based_exploratory"
        or result.get("validated_for_operation") is not False
    ):
        raise ValueError("DAE 실험용 결과 형식 오류")

    score = float(result["score"])
    threshold = float(result["threshold"])
    decision = result["predicted_anomaly"]

    if (
        not math.isfinite(score)
        or not math.isfinite(threshold)
        or score < 0
        or threshold < 0
        or type(decision) is not int
        or decision not in (0, 1)
        or decision != int(score > threshold)
    ):
        raise ValueError("DAE 점수·임계값·판정 오류")

    sha = hashlib.sha256(raw).hexdigest()
    version = model_version()

    with closing(store._connect()) as connection, connection:
        parent = connection.execute(
            """SELECT file_sha256, rpm, channel
               FROM analyses WHERE id=?""",
            (analysis_id,)
        ).fetchone()

        if parent is None:
            raise ValueError("기존 분석 ID를 찾을 수 없습니다.")

        if (
            parent["file_sha256"] != sha
            or parent["rpm"] != rpm
            or parent["channel"] != 3
        ):
            raise ValueError(
                "기존 분석과 DAE 입력 파일·RPM·채널이 다릅니다."
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS dae_experimental_results (
                analysis_id INTEGER PRIMARY KEY
                    REFERENCES analyses(id) ON DELETE CASCADE,
                model_version TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                rpm INTEGER NOT NULL,
                score REAL NOT NULL,
                threshold REAL NOT NULL,
                predicted_anomaly INTEGER NOT NULL
                    CHECK(predicted_anomaly IN (0,1)),
                evaluation_scope TEXT NOT NULL,
                validated_for_operation INTEGER NOT NULL
                    CHECK(validated_for_operation=0)
            )
        """)

        payload = (
            analysis_id,
            version,
            sha,
            rpm,
            score,
            threshold,
            decision,
            "date_based_exploratory",
            0,
        )

        previous = connection.execute(
            """SELECT analysis_id, model_version,
                      input_sha256, rpm, score, threshold,
                      predicted_anomaly, evaluation_scope,
                      validated_for_operation
               FROM dae_experimental_results
               WHERE analysis_id=?""",
            (analysis_id,)
        ).fetchone()

        if previous is not None:
            if tuple(previous) != payload:
                raise ValueError(
                    "기존 DAE 기록과 다릅니다. 덮어쓰지 않습니다."
                )
            return False

        connection.execute(
            """INSERT INTO dae_experimental_results
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            payload,
        )

    return True


def read_dae_experimental(store, analysis_id):
    """DAE 테이블이 없는 기존 DB도 안전하게 조회."""
    if not store.path.is_file():
        return None

    uri = store.path.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        exists = connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table'
               AND name='dae_experimental_results'"""
        ).fetchone()

        if exists is None:
            return None

        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """SELECT * FROM dae_experimental_results
               WHERE analysis_id=?""",
            (int(analysis_id),)
        ).fetchone()

        return dict(row) if row is not None else None


def insert_dae_experimental(connection, analysis_id, raw, rpm, result):
    """HistoryStore.save의 동일한 트랜잭션에서 DAE 결과 저장."""
    if type(raw) is not bytes or not raw:
        raise ValueError("DAE 입력 bytes 오류")

    if type(rpm) is not int or rpm not in (900, 1200, 1800):
        raise ValueError("DAE RPM 오류")

    if (
        result.get("model") != "DAE"
        or result.get("rpm") != rpm
        or result.get("evaluation_scope") != "date_based_exploratory"
        or result.get("validated_for_operation") is not False
    ):
        raise ValueError("DAE 실험용 결과 형식 오류")

    score = float(result["score"])
    threshold = float(result["threshold"])
    decision = result["predicted_anomaly"]

    if (
        not math.isfinite(score)
        or not math.isfinite(threshold)
        or score < 0
        or threshold < 0
        or type(decision) is not int
        or decision not in (0, 1)
        or decision != int(score > threshold)
    ):
        raise ValueError("DAE 점수·임계값·판정 오류")

    sha = hashlib.sha256(raw).hexdigest()

    parent = connection.execute(
        "SELECT file_sha256, rpm, channel FROM analyses WHERE id=?",
        (analysis_id,)
    ).fetchone()

    if parent is None:
        raise ValueError("분석 ID 없음")

    if (
        parent["file_sha256"] != sha
        or parent["rpm"] != rpm
        or parent["channel"] != 3
    ):
        raise ValueError("DAE 입력과 기존 분석 정보 불일치")

    version = model_version()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS dae_experimental_results (
            analysis_id INTEGER PRIMARY KEY
                REFERENCES analyses(id) ON DELETE CASCADE,
            model_version TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            rpm INTEGER NOT NULL,
            score REAL NOT NULL,
            threshold REAL NOT NULL,
            predicted_anomaly INTEGER NOT NULL
                CHECK(predicted_anomaly IN (0,1)),
            evaluation_scope TEXT NOT NULL,
            validated_for_operation INTEGER NOT NULL
                CHECK(validated_for_operation=0)
        )
    """)

    payload = (
        analysis_id, version, sha, rpm, score, threshold,
        decision, "date_based_exploratory", 0
    )

    previous = connection.execute(
        "SELECT * FROM dae_experimental_results WHERE analysis_id=?",
        (analysis_id,)
    ).fetchone()

    if previous is not None:
        if tuple(previous) != payload:
            raise ValueError("기존 DAE 결과를 덮어쓰지 않습니다.")
        return False

    connection.execute(
        "INSERT INTO dae_experimental_results VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        payload
    )
    return True


def save_dae_standalone(
    store, raw, rpm, filename, result
):
    """DAE 단독 분석을 기존 이력 구조에 원자적으로 저장.

    같은 입력·RPM·DAE 모델 버전·결과는 중복 저장하지 않는다.
    """
    from datetime import datetime, timezone

    if type(raw) is not bytes or not raw:
        raise ValueError("MAT 원본 bytes가 필요합니다.")

    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("MAT 파일은 20MB 이하여야 합니다.")

    if type(rpm) is not int or rpm not in (
        900, 1200, 1800
    ):
        raise ValueError("지원하지 않는 RPM입니다.")

    if not isinstance(filename, str) or not filename:
        raise ValueError("파일 이름이 필요합니다.")

    if (
        result.get("model") != "DAE"
        or result.get("rpm") != rpm
        or result.get("evaluation_scope")
            != "date_based_exploratory"
        or result.get("validated_for_operation") is not False
    ):
        raise ValueError("DAE 실험용 결과 형식 오류")

    score = float(result["score"])
    threshold = float(result["threshold"])
    decision = result["predicted_anomaly"]

    if (
        not math.isfinite(score)
        or not math.isfinite(threshold)
        or score < 0
        or threshold < 0
        or type(decision) is not int
        or decision not in (0, 1)
        or decision != int(score > threshold)
    ):
        raise ValueError("DAE 점수·임계값·판정 오류")

    sha = hashlib.sha256(raw).hexdigest()
    version = model_version()

    # 동일 입력과 동일 모델 결과는 중복 저장하지 않는다.
    identity = [
        "dae_standalone_v1",
        sha,
        rpm,
        version,
        score,
        threshold,
        decision,
    ]

    request_key = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False
        ).encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    kst_date = now.astimezone(
        timezone(__import__("datetime").timedelta(hours=9))
    ).date().isoformat()

    # 사용자 파일명은 표시용으로만 사용한다.
    safe_filename = Path(
        filename.replace("\\", "/")
    ).name[:200] or "uploaded.mat"

    with closing(store._connect()) as connection, connection:
        cursor = connection.execute(
            """INSERT INTO analyses
               (request_key, analyzed_at, analyzed_date,
                saved_at, filename, file_sha256,
                rpm, channel, bundle_id, source_experiment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(request_key) DO NOTHING""",
            (
                request_key,
                timestamp,
                kst_date,
                timestamp,
                safe_filename,
                sha,
                rpm,
                3,
                version,
                "dae_standalone_date_exploratory",
            )
        )

        created = cursor.rowcount == 1

        analysis_id = connection.execute(
            "SELECT id FROM analyses WHERE request_key=?",
            (request_key,)
        ).fetchone()["id"]

        # 동일 트랜잭션 안에서 기존 DAE 검증 저장 함수를 사용
        inserted = insert_dae_experimental(
            connection,
            analysis_id,
            raw,
            rpm,
            result
        )

        if created != inserted:
            raise ValueError(
                "DAE 단독 분석 기록의 저장 상태가 일치하지 않습니다."
            )

    return analysis_id, created
