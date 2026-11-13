from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch, detrend
from scipy.integrate import cumulative_trapezoid

# 조원이 공유한 데이터 폴더
DATA_ROOT = Path(
    "/mnt/c/Users/kungh/OneDrive/바탕 화면/isu_chemical/data/20260919/20260919_900RPM"
)

OUTPUT = Path("outputs/feedback1_signal_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal1",
    "Normal2",
    "Normal3",
    "IR",
    "Looeness0.1",
    "Unbalance_-1",
]

results = {}

for class_name in CLASSES:
    folder = DATA_ROOT / class_name

    if not folder.exists():
        print(f"[건너뜀] 폴더 없음: {folder}")
        continue

    mat_files = sorted(folder.rglob("*.mat"))

    if not mat_files:
        print(f"[건너뜀] MAT 파일 없음: {folder}")
        continue

    mat = loadmat(mat_files[0])

    acceleration = np.asarray(
        mat["signals"][2, :], dtype=np.float64
    )
    fs = float(np.asarray(mat["fs"]).squeeze())

    # DC 및 저주파 드리프트 제거
    acceleration = detrend(acceleration)
    sos = butter(
        4, 5.0, btype="highpass", fs=fs, output="sos"
    )
    acceleration_filtered = sosfiltfilt(sos, acceleration)

    # 가속도 적분 → 속도 (m/s, 원본 가속도가 m/s²인 경우)
    velocity = cumulative_trapezoid(
        acceleration_filtered,
        dx=1 / fs,
        initial=0
    )

    # 적분 후 잔류 드리프트 제거
    velocity = detrend(velocity)
    velocity = sosfiltfilt(sos, velocity)

    # 속도 스펙트럼
    freq, psd = welch(
        velocity,
        fs=fs,
        window="hann",
        nperseg=8192,
        noverlap=4096
    )

    results[class_name] = {
        "time": np.arange(len(velocity)) / fs,
        "velocity": velocity,
        "freq": freq,
        "spectrum": np.sqrt(psd),
        "rms": np.sqrt(np.mean(velocity ** 2)),
    }

    print(
        f"{class_name}: "
        f"Velocity RMS = {results[class_name]['rms']:.6g}"
    )

if not results:
    raise RuntimeError(
        "분석할 MAT 파일을 찾지 못했습니다. DATA_ROOT를 확인하세요."
    )

# 1. 속도 시간파형
fig, axes = plt.subplots(
    len(results), 1,
    figsize=(13, 2.1 * len(results)),
    sharex=True
)

axes = np.atleast_1d(axes)

for ax, (name, data) in zip(axes, results.items()):
    ax.plot(
        data["time"],
        data["velocity"],
        linewidth=0.65
    )
    ax.set_ylabel(name)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("900 RPM - Velocity Time Waveforms (CH3)")
fig.tight_layout()
fig.savefig(
    OUTPUT / "900rpm_velocity_time.png",
    dpi=180
)
plt.close(fig)

# 2. 속도 주파수 스펙트럼
fig, ax = plt.subplots(figsize=(13, 6))

for name, data in results.items():
    mask = data["freq"] <= 500
    ax.plot(
        data["freq"][mask],
        data["spectrum"][mask],
        label=name,
        linewidth=1
    )

ax.set_title("900 RPM - Velocity Spectrum (0-500 Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Velocity ASD")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT / "900rpm_velocity_spectrum.png",
    dpi=180
)
plt.close(fig)

print("\n그래프 저장 완료:")
print(OUTPUT / "900rpm_velocity_time.png")
print(OUTPUT / "900rpm_velocity_spectrum.png")
