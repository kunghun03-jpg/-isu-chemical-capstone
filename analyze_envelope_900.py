from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, detrend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919/20260919_900RPM"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal1",
    "Normal2",
    "Normal3",
    "IR",
    "Looeness0.1",
    "Unbalance_-1",
]

results = {}

for name in CLASSES:
    files = sorted((DATA / name).rglob("*.mat"))

    if not files:
        print(f"[건너뜀] {name}: MAT 파일 없음")
        continue

    mat = loadmat(files[0])
    signals = np.asarray(mat["signals"])
    fs = float(np.asarray(mat["fs"]).squeeze())

    if signals.shape != (6, 65536) or fs != 25600:
        raise ValueError(f"{name}: 신호 크기 또는 샘플링 주파수 오류")

    acceleration = detrend(
        np.asarray(signals[2, :], dtype=np.float64)
    )

    if not np.isfinite(acceleration).all():
        raise ValueError(f"{name}: 비정상적인 신호값 발견")

    freq, psd = welch(
        acceleration,
        fs=fs,
        window="hann",
        nperseg=8192,
        noverlap=4096
    )

    results[name] = (freq, np.sqrt(psd))
    print(f"[완료] {name}: {files[0].name}")

if not results:
    raise RuntimeError("분석할 MAT 파일이 없습니다.")

# 고주파 스펙트럼 비교
fig, axes = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True
)

for name, (freq, asd) in results.items():
    mask = (freq >= 500) & (freq <= 10000)

    axes[0].plot(
        freq[mask], asd[mask], label=name, linewidth=0.8
    )

    axes[1].plot(
        freq[mask],
        20 * np.log10(asd[mask] + 1e-15),
        label=name,
        linewidth=0.8
    )

axes[0].set_ylabel("Acceleration ASD")
axes[0].set_title("900 RPM - High Frequency Acceleration Spectrum")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Acceleration ASD (dB, relative)")
axes[1].grid(alpha=0.3)

fig.tight_layout()

output_file = OUT / "900rpm_acceleration_highfreq.png"
fig.savefig(output_file, dpi=180)
plt.close(fig)

print("\n그래프 저장 완료:")
print(output_file)
