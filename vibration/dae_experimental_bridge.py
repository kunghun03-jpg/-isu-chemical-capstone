
"""Persistent subprocess bridge for experimental DAE inference.

Operational DAE readiness and AE decision policies remain unchanged.
"""

import atexit
import hashlib
import json
import math
import os
import selectors
import struct
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DAE_PYTHON = ROOT / ".venv/bin/python"

DAE_WORKER = ROOT / "scripts/dae_experimental_worker.py"

ARTIFACT_ROOT = (
    ROOT / "outputs/dae_group_split_experiment_20261119"
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_RESPONSE_BYTES = 65536

STARTUP_TIMEOUT = 120
REQUEST_TIMEOUT = 120
SHUTDOWN_TIMEOUT = 5

SUPPORTED_RPMS = (900, 1200, 1800)

ARTIFACT_NAMES = (
    "dae_group_split_best.pth",
    "dae_normalization.npz",
    "dae_thresholds.npz",
)


def artifact_identity():
    """Identify the current runtime and model artifacts."""

    paths = [
        DAE_PYTHON,
        DAE_WORKER,
        ROOT / "scripts/dae_experimental_inference.py",
        ROOT / "vibration/dae_preprocessing.py",
        ROOT / "dae_legacy/model.py",
    ]

    paths.extend(
        ARTIFACT_ROOT / name
        for name in ARTIFACT_NAMES
    )

    identity = []

    for path in paths:

        if not path.is_file():
            raise FileNotFoundError(
                f"DAE 실행 파일이 없습니다: {path.name}"
            )

        stat = path.stat()

        identity.append((
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ))

    return tuple(identity)


def validate_input(raw, rpm):

    if type(raw) is not bytes:
        raise ValueError(
            "MAT 파일은 bytes 형식이어야 합니다."
        )

    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "20MB 이하 MAT 파일만 지원합니다."
        )

    if type(rpm) is not int or rpm not in SUPPORTED_RPMS:
        raise ValueError(
            "RPM은 900, 1200, 1800 중 선택하세요."
        )


def validate_result(result, rpm):

    if not isinstance(result, dict):
        raise ValueError("DAE 결과 형식 오류")

    score = float(result["score"])
    threshold = float(result["threshold"])
    decision = result["predicted_anomaly"]

    if not (
        math.isfinite(score)
        and math.isfinite(threshold)
        and score >= 0
        and threshold > 0
    ):
        raise ValueError("DAE 점수 또는 임계값 오류")

    if type(decision) is not int or decision not in (0, 1):
        raise ValueError("DAE 판정 형식 오류")

    if result["rpm"] != rpm:
        raise ValueError("DAE 결과 RPM 불일치")

    if decision != int(score > threshold):
        raise ValueError("DAE 점수와 판정 불일치")

    if (
        result.get("evaluation_scope")
        != "date_based_exploratory"
    ):
        raise ValueError("DAE 평가 범위 불일치")

    if result.get("validated_for_operation") is not False:
        raise ValueError("DAE 운영 판정 제한 오류")

    return {
        "model": "DAE",
        "rpm": rpm,
        "score": score,
        "threshold": threshold,
        "predicted_anomaly": decision,
        "prediction": "이상" if decision else "정상",
        "evaluation_scope": "date_based_exploratory",
        "validated_for_operation": False,
    }


class ExperimentalDAEBridge:

    def __init__(self):

        self._lock = threading.RLock()
        self._process = None
        self._identity = None

        self._request_id = 0
        self._buffer = bytearray()

        self._starts = 0
        self._completed = 0

    def _remaining(self, deadline):

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                "DAE 프로세스 응답 시간이 초과되었습니다."
            )

        return remaining

    def _read_response(self, deadline):

        process = self._process

        if process is None:
            raise RuntimeError("DAE 프로세스 미실행")

        fd = process.stdout.fileno()

        with selectors.DefaultSelector() as selector:

            selector.register(
                fd,
                selectors.EVENT_READ,
            )

            while True:

                position = self._buffer.find(b"\n")

                if position >= 0:

                    if position > MAX_RESPONSE_BYTES:
                        raise ValueError(
                            "DAE 응답 크기 초과"
                        )

                    line = bytes(
                        self._buffer[:position]
                    )

                    del self._buffer[:position + 1]

                    result = json.loads(
                        line.decode("utf-8")
                    )

                    if not isinstance(result, dict):
                        raise ValueError(
                            "DAE 응답 형식 오류"
                        )

                    return result

                if len(self._buffer) > MAX_RESPONSE_BYTES:
                    raise ValueError(
                        "DAE 응답 크기 초과"
                    )

                events = selector.select(
                    self._remaining(deadline)
                )

                if not events:
                    raise TimeoutError(
                        "DAE 응답 시간 초과"
                    )

                chunk = os.read(fd, 4096)

                if not chunk:
                    raise RuntimeError(
                        "DAE 프로세스가 종료되었습니다."
                    )

                self._buffer.extend(chunk)

    def _write_request(self, header, raw, deadline):

        process = self._process

        if process is None:
            raise RuntimeError("DAE 프로세스 미실행")

        encoded = json.dumps(
            header,
            allow_nan=False,
        ).encode("utf-8")

        if not 0 < len(encoded) <= MAX_HEADER_BYTES:
            raise ValueError("DAE 요청 헤더 크기 오류")

        parts = (
            struct.pack("!I", len(encoded)),
            encoded,
            raw,
        )

        fd = process.stdin.fileno()

        with selectors.DefaultSelector() as selector:

            selector.register(
                fd,
                selectors.EVENT_WRITE,
            )

            for part in parts:

                view = memoryview(part)

                while view:

                    events = selector.select(
                        self._remaining(deadline)
                    )

                    if not events:
                        raise TimeoutError(
                            "DAE 요청 전송 시간 초과"
                        )

                    try:
                        written = os.write(
                            fd, view[:65536]
                        )

                    except BlockingIOError:
                        continue

                    if written <= 0:
                        raise RuntimeError(
                            "DAE 요청 전송 실패"
                        )

                    view = view[written:]

    def _stop(self, graceful=False):

        process = self._process

        if process is None:
            return

        try:

            if graceful and process.poll() is None:

                try:

                    self._request_id += 1

                    deadline = (
                        time.monotonic()
                        + SHUTDOWN_TIMEOUT
                    )

                    self._write_request(
                        {
                            "command": "shutdown",
                            "request_id": self._request_id,
                        },
                        b"",
                        deadline,
                    )

                    response = self._read_response(
                        deadline
                    )

                    if (
                        response.get("status") != "shutdown"
                        or response.get("request_id")
                        != self._request_id
                    ):
                        raise RuntimeError(
                            "DAE 종료 응답 오류"
                        )

                    process.wait(
                        timeout=self._remaining(deadline)
                    )

                except Exception:
                    pass

            if process.poll() is None:

                process.terminate()

                try:
                    process.wait(
                        timeout=SHUTDOWN_TIMEOUT
                    )

                except subprocess.TimeoutExpired:

                    process.kill()
                    process.wait(
                        timeout=SHUTDOWN_TIMEOUT
                    )

        finally:

            for stream in (
                process.stdin,
                process.stdout,
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

            self._process = None
            self._identity = None
            self._buffer.clear()

    def _start(self, identity):

        self._stop()

        environment = os.environ.copy()

        environment.pop("PYTHONHOME", None)

        environment["PYTHONPATH"] = str(ROOT)
        environment["PYTHONNOUSERSITE"] = "1"

        self._process = subprocess.Popen(
            [
                str(DAE_PYTHON),
                "-u",
                str(DAE_WORKER),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
            env=environment,
            bufsize=0,
        )

        self._buffer.clear()

        try:

            os.set_blocking(
                self._process.stdin.fileno(),
                False,
            )

            deadline = (
                time.monotonic()
                + STARTUP_TIMEOUT
            )

            response = self._read_response(deadline)

            if (
                response.get("status") != "ready"
                or response.get("protocol_version") != 1
                or response.get("validated_for_operation")
                is not False
            ):
                raise RuntimeError(
                    "DAE Worker 초기화 응답 오류"
                )

            if artifact_identity() != identity:
                raise RuntimeError(
                    "DAE 초기화 중 모델 파일이 변경되었습니다."
                )

            self._identity = identity
            self._starts += 1

        except Exception:

            self._stop()
            raise

    def _ensure_process(self):

        identity = artifact_identity()

        if (
            self._process is None
            or self._process.poll() is not None
            or self._identity != identity
        ):
            self._start(identity)

    def predict(self, raw, rpm):

        validate_input(raw, rpm)

        with self._lock:

            try:

                self._ensure_process()

                self._request_id += 1
                request_id = self._request_id

                deadline = (
                    time.monotonic()
                    + REQUEST_TIMEOUT
                )

                self._write_request(
                    {
                        "command": "predict",
                        "request_id": request_id,
                        "rpm": rpm,
                        "size": len(raw),
                    },
                    raw,
                    deadline,
                )

                response = self._read_response(
                    deadline
                )

                if (
                    response.get("request_id")
                    != request_id
                ):
                    raise RuntimeError(
                        "DAE 요청·응답 ID 불일치"
                    )

                if response.get("status") != "ok":
                    raise RuntimeError(
                        "DAE 실험용 추론에 실패했습니다."
                    )

                result = validate_result(
                    response["result"],
                    rpm,
                )

                if artifact_identity() != self._identity:
                    raise RuntimeError(
                        "추론 중 DAE 모델 파일이 변경되었습니다."
                    )

                self._completed += 1

                return result

            except Exception:

                self._stop()
                raise

    def status(self):

        with self._lock:

            running = (
                self._process is not None
                and self._process.poll() is None
            )

            return {
                "running": running,
                "pid": (
                    self._process.pid
                    if running else None
                ),
                "process_starts": self._starts,
                "completed_requests": self._completed,
                "validated_for_operation": False,
            }

    def close(self):

        with self._lock:
            self._stop(graceful=True)


_BRIDGE = ExperimentalDAEBridge()

atexit.register(_BRIDGE.close)


def predict_experimental_dae(raw, rpm):
    """Preserve the existing dashboard interface."""

    return _BRIDGE.predict(raw, rpm)


def experimental_dae_status():
    """Return process diagnostics without inference."""

    return _BRIDGE.status()


def close_experimental_dae():
    """Explicitly release the DAE worker."""

    _BRIDGE.close()
