"""학습용 배치와 저장 모델 추론이 공유하는 고정 CH3 전처리."""

import numpy as np
from vibration.channel import select_ch3

PREPROCESSING_CONFIG = {
    "version": "ch3_fft_log1p_v1",
    "raw_shape": [6, 65536],
    "fs": 25600.0,
    "channel_number": 3,
    "channel_index": 2,
    "segments": 8,
    "window_size": 8192,
    "remove_mean": "per_segment",
    "window": "numpy.hanning_symmetric",
    "fft": "numpy.rfft",
    "amplitude_scale": "abs_fft_divide_window_sum_double_interior_bins",
    "frequency_range_hz": [0, 10000],
    "upper_frequency_inclusive": True,
    "frequency_bins": 3201,
    "transform": "numpy.log1p",
    "pre_z_dtype": "float32",
    "quality_scope": "all_6_raw_channels",
    "extreme_value_limit": 1e100,
}


def frequency_bins():
    frequency = np.fft.rfftfreq(8192, d=1 / 25600.0)
    return frequency[frequency <= 10000]


def ch3_log_spectra(signals, fs):
    """원본을 바꾸지 않고 기존 배치와 같은 (8, 1, 3201)을 반환합니다."""
    if not isinstance(signals, np.ndarray) or signals.shape != (6, 65536):
        raise ValueError("signals는 (6, 65536) 배열이어야 합니다.")
    if (
        signals.dtype.kind not in "fiu"
        or not np.isfinite(signals).all()
        or (np.abs(signals) > 1e100).any()
    ):
        raise ValueError("원본 전체 6채널의 실수·유한값·극단값 품질 조건을 확인하세요.")
    fs_value = np.asarray(fs)
    if (
        fs_value.size != 1
        or fs_value.dtype.kind not in "fiu"
        or float(fs_value.item()) != 25600.0
    ):
        raise ValueError("샘플링 주파수는 25600 Hz 단일 값이어야 합니다.")
    selected = select_ch3(signals)
    segments = selected.reshape(1, 8, 8192).transpose(1, 0, 2)
    segments = segments - segments.mean(axis=2, keepdims=True)
    window = np.hanning(8192)
    amplitude = np.abs(np.fft.rfft(segments * window, axis=2)) / window.sum()
    amplitude[:, :, 1:-1] *= 2
    frequency = np.fft.rfftfreq(8192, d=1 / 25600.0)
    values = np.log1p(amplitude[:, :, frequency <= 10000]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("전처리 결과가 유한하지 않습니다.")
    return values


def apply_normalization(values, mean, std):
    if mean.shape != (1, 1, 3201) or std.shape != mean.shape:
        raise ValueError("선정된 주파수별 Train 정규화 기준 (1, 1, 3201)이 필요합니다.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError("저장된 정규화 통계가 잘못되었습니다.")
    result = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("정규화 결과가 유한하지 않습니다.")
    return result
