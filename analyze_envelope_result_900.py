from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, welch, detrend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919/20260919_900RPM"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal1", "Normal2", "Normal3",
    "IR", "Looeness0.1", "Unbalance_-1"
]

LOWCUT = 1800
HIGHCUT = 2800

results = {}

for name in CLASSES:
    files = sorted((DATA / name).rglob("*.mat"))

    if not files:
        print(f"[건너뜀] {name}: MAT 파일 없음")
        continue

    mat = loadmat(files[0])
    fs = float(np.asarray(mat["fs"]).squeeze())
    signals = np.asarray(mat["signals"])

    if signals.shape != (6, 65536) or fs != 25600:
        raise ValueError(f"{name}: 신호 형식 오류")

    acceleration = detrend(
        np.asarray(signals[2, :], dtype=np.float64)
    )

    # 고주파 대역통과 필터
    sos = butter(
        4,
        [LOWCUT, HIGHCUT],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(sos, acceleration)

    # Hilbert 변환을 이용한 포락선
    envelope = np.abs(hilbert(filtered))

    # 포락선의 평균값 제거 후 주파수 분석
    envelope_ac = envelope - np.mean(envelope)

    freq, psd = welch(
        envelope_ac,
        fs=fs,
        window="hann",
        nperseg=16384,
        noverlap=8192
    )

    results[name] = {
        "time": np.arange(len(envelope)) / fs,
        "envelope": envelope,
        "freq": freq,
        "asd": np.sqrt(psd),
    }

    print(f"[완료] {name}: {files[0].name}")

if not results:
    raise RuntimeError("MAT 파일을 찾지 못했습니다.")

# 1. 포락 시간파형
fig, axes = plt.subplots(
    len(results), 1,
    figsize=(13, 2.2 * len(results)),
    sharex=True
)

axes = np.atleast_1d(axes)

for ax, (name, data) in zip(axes, results.items()):
    ax.plot(
        data["time"],
        data["envelope"],
        linewidth=0.6
    )
    ax.set_ylabel(name)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(
    "900 RPM - Envelope Time Waveforms "
    "(CH3, 1800-2800 Hz)"
)
fig.tight_layout()
fig.savefig(
    OUT / "900rpm_envelope_time.png",
    dpi=180
)
plt.close(fig)

# 2. 포락 주파수 스펙트럼
fig, ax = plt.subplots(figsize=(13, 6))

for name, data in results.items():
    mask = (data["freq"] >= 0) & (data["freq"] <= 500)

    ax.plot(
        data["freq"][mask],
        data["asd"][mask],
        label=name,
        linewidth=1
    )

ax.set_title(
    "900 RPM - Envelope Spectrum "
    "(CH3, Bandpass 1800-2800 Hz)"
)
ax.set_xlabel("Modulation Frequency (Hz)")
ax.set_ylabel("Envelope ASD")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "900rpm_envelope_spectrum.png",
    dpi=180
)
plt.close(fig)

print("\n포락 분석 그래프 저장 완료:")
print(OUT / "900rpm_envelope_time.png")
print(OUT / "900rpm_envelope_spectrum.png")
