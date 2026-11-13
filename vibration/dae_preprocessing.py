"""Legacy file-level Welch features, separate from the existing FFT pipeline."""

from io import BytesIO

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch


def dae_log_spectrum(signals, fs):
    signals = np.asarray(signals)
    fs = np.asarray(fs)
    if signals.shape != (6, 65536):
        raise ValueError("신호 크기는 (6, 65536)이어야 합니다.")
    if signals.dtype.kind not in "fiu":
        raise ValueError("신호는 실수 배열이어야 합니다.")
    if fs.size != 1 or fs.dtype.kind not in "fiu" or float(fs.item()) != 25600:
        raise ValueError("샘플링 주파수는 25600 Hz 단일 값이어야 합니다.")
    # Preserve legacy quality scope: only CH3 participates in these features.
    ch3 = np.asarray(signals[2, :], dtype=np.float64)
    if not np.isfinite(ch3).all():
        raise ValueError("원본 CH3 신호에 이상값이 있습니다.")
    freq, psd = welch(
        ch3, fs=25600.0, window="hann", nperseg=8192,
        noverlap=4096, nfft=8192,
    )
    spectrum = np.sqrt(psd)[(freq >= 0) & (freq < 10000)]
    if spectrum.shape != (3200,):
        raise ValueError("DAE 스펙트럼 크기 오류")
    values = np.log10(spectrum.reshape(800, 4).mean(axis=1) + 1e-12)
    if not np.isfinite(values).all():
        raise ValueError("전처리 결과에 이상값이 있습니다.")
    return values.astype(np.float32)


def preprocess_mat_bytes(raw):
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("비어 있지 않은 MAT bytes가 필요합니다.")
    try:
        data = loadmat(BytesIO(raw), variable_names=("signals", "fs"))
    except Exception as error:
        raise ValueError("MAT 파일을 읽을 수 없습니다.") from error
    if "signals" not in data or "fs" not in data:
        raise ValueError("MAT에 signals와 fs가 필요합니다.")
    return dae_log_spectrum(data["signals"], data["fs"])


def preprocess_uploaded_mat(uploaded_file):
    """Compatibility entry point for legacy BytesIO/Streamlit uploads."""
    return preprocess_mat_bytes(uploaded_file.getvalue())


def validate_statistics(mean, std):
    for name, value in (("mean", mean), ("std", std)):
        if (
            value.size not in (1, 800)
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"DAE {name}의 크기 또는 값이 올바르지 않습니다.")
    try:
        shape = np.broadcast_shapes((800,), mean.shape, std.shape)
    except ValueError as error:
        raise ValueError("DAE 정규화 통계의 형태가 호환되지 않습니다.") from error
    if np.prod(shape) != 800:
        raise ValueError("DAE 정규화 결과는 800개 값이어야 합니다.")
    if np.any(std <= 0):
        raise ValueError("정규화 표준편차는 0보다 커야 합니다.")


def normalize_spectrum(log_spectrum, mean, std):
    mean, std = np.asarray(mean), np.asarray(std)
    validate_statistics(mean, std)
    if log_spectrum.shape != (800,) or not np.isfinite(log_spectrum).all():
        raise ValueError("DAE 입력 스펙트럼은 유한한 800개 값이어야 합니다.")
    # Do not cast statistics or clamp std: preserve legacy NumPy arithmetic.
    normalized = (log_spectrum - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("정규화 결과에 이상값이 있습니다.")
    return normalized
