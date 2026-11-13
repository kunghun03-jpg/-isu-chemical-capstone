# 보관된 초기 실험 코드: check_data. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from scipy.io import loadmat

# check_data.py가 있는 폴더를 기준으로 데이터 찾기
project_dir = Path(__file__).resolve().parent
normal_dir = project_dir / "data" / "20260916_900RPM" / "Normal1"

files = sorted(normal_dir.rglob("*.mat"))

print("Normal1의 MAT 파일 개수:", len(files))

if not files:
    raise FileNotFoundError("Normal1 안에서 MAT 파일을 찾지 못했습니다.")

# 첫 번째 파일 하나만 읽기
file_path = files[0]
data = loadmat(file_path)

signals = data["signals"]
fs = data["fs"].item()

print("읽은 파일:", file_path.name)
print("데이터 모양:", signals.shape)
print("샘플링 주파수:", fs, "Hz")
print("측정 시간:", signals.shape[1] / fs, "초")
print("첫 번째 채널의 앞쪽 값 10개:")
print(signals[0, :10])

import numpy as np
import matplotlib.pyplot as plt

# 각 측정값에 해당하는 시간
time = np.arange(signals.shape[1]) / fs

# 6개 채널을 각각 그래프로 표시
fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for ch in range(6):
    axes[ch].plot(time, signals[ch], linewidth=0.5)
    axes[ch].set_ylabel(f"CH {ch + 1}")
    axes[ch].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Normal vibration - {file_path.name}")
plt.tight_layout()
plt.show()

# 채널별 평균값 제거
centered = signals - signals.mean(axis=1, keepdims=True)

# 스펙트럼 계산 전 Hann 창 적용
n = signals.shape[1]
window = np.hanning(n)

# FFT로 주파수별 진폭 계산
freq = np.fft.rfftfreq(n, d=1 / fs)
amplitude = np.abs(np.fft.rfft(centered * window, axis=1)) / window.sum()

# 단측 스펙트럼 보정 (0Hz와 나이퀴스트 주파수 제외)
amplitude[:, 1:-1] *= 2

fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for ch in range(6):
    axes[ch].plot(freq, amplitude[ch], linewidth=0.7)
    axes[ch].set_ylabel(f"CH {ch + 1}")
    axes[ch].grid(True, alpha=0.3)

axes[-1].set_xlabel("Frequency (Hz)")
axes[-1].set_xlim(0, 1000)  # 우선 0~1000Hz 구간 확대
fig.suptitle("Normal vibration - Frequency spectrum")
plt.tight_layout()
plt.show()
