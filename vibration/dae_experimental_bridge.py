"""DAE 실험용 별도 프로세스 추론 연결.

운영용 DAE readiness/번들을 변경하지 않는다.
"""
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAE_PYTHON = ROOT / ".venv/bin/python"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DAE_SCRIPT = """
import json
import sys
from scripts.dae_experimental_inference import ExperimentalDAE

rpm = int(sys.argv[1])
raw = sys.stdin.buffer.read()

engine = ExperimentalDAE()
result = engine.predict_bytes(raw, rpm)

print(json.dumps(result, allow_nan=False))
"""


def predict_experimental_dae(raw, rpm):
    """MAT bytes를 DAE 전용 환경으로 전달한다."""

    if type(raw) is not bytes:
        raise ValueError("MAT 파일은 bytes 형식이어야 합니다.")

    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("20MB 이하 MAT 파일만 지원합니다.")

    if type(rpm) is not int or rpm not in (900, 1200, 1800):
        raise ValueError("RPM은 900, 1200, 1800 중 선택하세요.")

    if not DAE_PYTHON.is_file():
        raise FileNotFoundError(
            "DAE 전용 Python 환경을 찾을 수 없습니다."
        )

    try:
        process = subprocess.run(
            [
                str(DAE_PYTHON),
                "-c",
                DAE_SCRIPT,
                str(rpm),
            ],
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            timeout=120,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "DAE 실험용 추론 제한 시간을 초과했습니다."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "DAE 실험용 추론에 실패했습니다. "
            "서버의 DAE 환경과 입력 파일을 확인하세요."
        ) from error

    try:
        result = json.loads(process.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "DAE 추론 결과 형식이 올바르지 않습니다."
        ) from error

    score = float(result["score"])
    threshold = float(result["threshold"])
    decision = result["predicted_anomaly"]

    if not (math.isfinite(score) and math.isfinite(threshold)):
        raise ValueError("DAE 추론 결과에 비유한 값이 있습니다.")

    if type(decision) is not int or decision not in (0, 1):
        raise ValueError("DAE 판정 형식 오류")

    if result["rpm"] != rpm:
        raise ValueError("DAE 결과 RPM 불일치")

    if decision != int(score > threshold):
        raise ValueError("DAE 점수와 판정 불일치")

    if result.get("evaluation_scope") != "date_based_exploratory":
        raise ValueError("DAE 실험용 결과 범위 불일치")

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
