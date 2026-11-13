"""웹 입력 미리보기와 오류 안내. 추론·판정은 기존 저장 모델 API에 위임합니다."""

from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from vibration.preprocessing import ch3_log_spectra, frequency_bins

MODEL_OPTIONS = {
    "전체": ("autoencoder", "ocsvm", "iforest"),
    "AE": ("autoencoder",),
    "OCSVM": ("ocsvm",),
    "IF": ("iforest",),
}
MODEL_LABELS = {"autoencoder": "AE", "ocsvm": "OCSVM", "iforest": "IF"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def preview_mat(raw):
    """업로드를 디스크에 쓰지 않고 기존 추론과 동일한 입력 조건을 확인합니다."""
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("비어 있지 않은 20MB 이하 MAT 파일을 업로드해 주세요.")
    try:
        variables = {name: shape for name, shape, _ in whosmat(BytesIO(raw))}
    except Exception as error:
        raise ValueError(
            "MAT 파일을 읽을 수 없습니다. 지원 MAT 형식을 확인해 주세요."
        ) from error
    if (
        variables.get("signals") != (6, 65536)
        or np.prod(variables.get("fs", (0,))) != 1
    ):
        raise ValueError(
            "signals=(6, 65536) 배열과 단일 값 fs가 필요합니다. MAT 변수 형태를 확인해 주세요."
        )
    try:
        data = loadmat(BytesIO(raw), variable_names=("signals", "fs"))
    except Exception as error:
        raise ValueError(
            "MAT 파일을 읽을 수 없습니다. signals와 fs가 있는 지원 MAT 형식을 확인해 주세요."
        ) from error
    if "signals" not in data or "fs" not in data:
        raise ValueError("MAT 파일에 signals 배열과 fs(샘플링 주파수)가 필요합니다.")
    # 전처리와 동일한 형식·전체 6채널 품질 검사를 적용합니다. 통계는 구하지 않습니다.
    log_spectra = ch3_log_spectra(data["signals"], data["fs"])
    signal = data["signals"][2].copy()
    return {
        "signal": signal,
        "time": np.arange(signal.size) / 25600.0,
        "fs": 25600.0,
        "samples": signal.size,
        "duration": signal.size / 25600.0,
        "log_spectra": log_spectra,
        "frequency": frequency_bins(),
    }


def upload_name(name):
    """표시용 이름만 남깁니다. 이 이름으로 경로를 생성하거나 상태를 추정하지 않습니다."""
    return Path(str(name).replace("\\", "/")).name[:200] or "uploaded.mat"


def error_message(error):
    if isinstance(error, FileNotFoundError):
        return "선택한 RPM의 저장 모델 또는 정규화 파일이 없습니다. outputs/inference_bundles/date_prelim_20261106 아래의 추론 패키지를 확인해 주세요. 이 화면에서는 모델을 새로 학습하지 않습니다."
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "추론에 필요한 패키지가 없습니다. 프로젝트 requirements.txt를 설치한 Python 환경으로 실행해 주세요."
    if isinstance(error, ValueError):
        return str(error)
    return "저장 모델을 읽거나 분석하는 중 오류가 발생했습니다. 추론 패키지와 설치 환경을 확인해 주세요."
