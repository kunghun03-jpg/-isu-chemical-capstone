# 회전 주파수 주변의 채널별 최대 진폭을 추출합니다.

"""파일 전체 2.56초의 회전 주파수 ±2Hz 최대 진폭. 모든 분할·상태에 적용."""

from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
from vibration.channel import select_ch3


def run(project_dir, output_dir, rpm):
    with (output_dir / f"split_{rpm}rpm.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    # 전체 2.56초 신호에서 회전 주파수(RPM / 60)의 ±2Hz 범위를 봅니다.
    n, fs = 65536, 25600.0
    window = np.hanning(n)
    frequency = np.fft.rfftfreq(n, 1 / fs)
    band = (frequency >= rpm / 60 - 2) & (frequency <= rpm / 60 + 2)
    records = []
    for row in rows:
        data = loadmat(project_dir / row["file"])
        signals = data["signals"].copy()
        if signals.shape != (6, n) or float(data["fs"].item()) != fs:
            raise ValueError(f"데이터 형식 확인 필요: {row['file']}")
        if not np.isfinite(signals).all() or (np.abs(signals) > 1e100).any():
            raise ValueError(f"손상 의심 파일: {row['file']}")

        # 평균을 제거한 뒤 Hann 창과 단측 FFT로 채널별 진폭을 구합니다.
        signals = select_ch3(signals)
        signals -= signals.mean(axis=1, keepdims=True)
        amplitude = np.abs(np.fft.rfft(signals * window, axis=1)) / window.sum()
        amplitude[:, 1:-1] *= 2

        # 대역 안의 최대 진폭을 채널별 파일 특징으로 저장합니다.
        peaks = amplitude[:, band].max(axis=1)
        record = dict(row)
        record["CH3_peak"] = float(peaks[0])
        records.append(record)
    if not records:
        raise ValueError("분할 목록이 비어 있습니다.")
    with (output_dir / "rotation_features.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"회전 주파수 {rpm/60:g}Hz 주변 특징: {len(records)}개 파일 저장")
