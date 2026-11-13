from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch, detrend
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919/20260919_900RPM"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal1", "Normal2", "Normal3",
    "IR", "Looeness0.1", "Unbalance_-1"
]

results = {}

for name in CLASSES:
    files = sorted((DATA / name).rglob("*.mat"))

    if not files:
        print(f"[건너뜀] {name}: MAT 파일 없음")
        continue

    mat = loadmat(files[0])
    acceleration = np.asarray(mat["signals"][2, :], dtype=float)
    fs = float(np.asarray(mat["fs"]).squeeze())

    # 저주파 드리프트 제거 후 적분
    sos = butter(4, 5, btype="highpass", fs=fs, output="sos")
    acceleration = sosfiltfilt(sos, detrend(acceleration))

    velocity = cumulative_trapezoid(
        acceleration, dx=1/fs, initial=0
    )
    velocity = sosfiltfilt(sos, detrend(velocity))

    freq, psd = welch(
        velocity, fs=fs, nperseg=8192, noverlap=4096
    )

    results[name] = {
        "time": np.arange(len(velocity)) / fs,
        "velocity": velocity,
        "freq": freq,
        "asd": np.sqrt(psd),
    }

    print(f"[완료] {name}: {files[0].name}")

if not results:
    raise RuntimeError("MAT 파일을 찾지 못했습니다.")

# 속도 시간파형
fig, axes = plt.subplots(
    len(results), 1,
    figsize=(13, 2.2 * len(results)),
    sharex=True
)
axes = np.atleast_1d(axes)

for ax, (name, data) in zip(axes, results.items()):
    ax.plot(data["time"], data["velocity"], linewidth=0.6)
    ax.set_ylabel(name)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("900 RPM - Velocity Time Waveforms (CH3)")
fig.tight_layout()
fig.savefig(OUT / "900rpm_velocity_time.png", dpi=180)
plt.close(fig)

# 속도 주파수 스펙트럼
fig, ax = plt.subplots(figsize=(13, 6))

for name, data in results.items():
    mask = data["freq"] <= 500
    ax.plot(
        data["freq"][mask],
        data["asd"][mask],
        label=name
    )

ax.set_title("900 RPM - Velocity Spectrum (CH3, 0-500 Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Velocity ASD (unit/sqrt(Hz))")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "900rpm_velocity_spectrum.png", dpi=180)
plt.close(fig)

print("\n그래프 저장 완료:")
print(OUT / "900rpm_velocity_time.png")
print(OUT / "900rpm_velocity_spectrum.png")
