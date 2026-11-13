
import io

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch


def preprocess_uploaded_mat(uploaded_file):
    # Streamlit 업로드 파일 읽기
    mat_data = loadmat(
        io.BytesIO(uploaded_file.getvalue())
    )

    signals = np.asarray(mat_data["signals"])
    fs = float(np.asarray(mat_data["fs"]).squeeze())

    if signals.shape != (6, 65536):
        raise ValueError("신호 크기 오류")

    if fs != 25600:
        raise ValueError("샘플링 주파수 오류")

    # CH3 추출
    ch3 = np.asarray(
        signals[2, :],
        dtype=np.float64
    )

    if not np.isfinite(ch3).all():
        raise ValueError("원본 신호에 이상값 있음")

    # Welch PSD
    freq, psd = welch(
        ch3,
        fs=fs,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        nfft=8192
    )

    spectrum = np.sqrt(psd)

    # 0~10 kHz
    mask = (freq >= 0) & (freq < 10000)
    spectrum_selected = spectrum[mask]

    if len(spectrum_selected) != 3200:
        raise ValueError("스펙트럼 크기 오류")

    # 3200 -> 800
    spectrum_800 = spectrum_selected.reshape(
        800, 4
    ).mean(axis=1)

    # 로그 변환
    log_spectrum = np.log10(
        spectrum_800 + 1e-12
    )

    if not np.isfinite(log_spectrum).all():
        raise ValueError("전처리 결과에 이상값 있음")

    return log_spectrum.astype(np.float32)
