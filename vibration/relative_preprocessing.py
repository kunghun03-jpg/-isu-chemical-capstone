"""상대 스펙트럼 기반 실험용 AE 전처리."""

import numpy as np

from vibration.preprocessing import (
    PREPROCESSING_CONFIG,
    ch3_log_spectra,
)

RELATIVE_PREPROCESSING_CONFIG = {
    **PREPROCESSING_CONFIG,
    "version": "ch3_fft_relative_l2_v1",
    "transform": "segment_l2_normalized_fft_amplitude",
    "experimental_only": True,
}


def ch3_relative_spectra(signals, fs):
    """
    CH3 상대 스펙트럼을 계산합니다.

    입력: (6, 65536)
    출력: (8, 1, 3201)
    """

    # 기존 입력 품질 검사와 FFT 처리 재사용
    log_spectra = ch3_log_spectra(signals, fs)

    # log1p 변환 이전의 FFT 진폭 복원
    amplitude = np.expm1(
        log_spectra.astype(np.float64)
    )

    if not np.isfinite(amplitude).all():
        raise ValueError("FFT 진폭 복원 오류")

    # 구간별 스펙트럼 L2 크기
    norm = np.linalg.norm(
        amplitude,
        axis=2,
        keepdims=True,
    )

    if not np.isfinite(norm).all():
        raise ValueError("스펙트럼 크기 오류")

    if (norm <= 1e-12).any():
        raise ValueError("유효한 진동 스펙트럼이 없습니다.")

    # 절대 진폭 대신 상대적인 주파수 패턴 사용
    relative = (
        amplitude / norm
    ).astype(np.float32)

    if relative.shape != (8, 1, 3201):
        raise ValueError("상대 스펙트럼 크기 오류")

    if not np.isfinite(relative).all():
        raise ValueError("비정상 상대 스펙트럼")

    return relative
