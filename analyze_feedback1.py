from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch

BASE = Path("data/20260919/20260919_900RPM")
OUT = Path("outputs/feedback1_signal_analysis")
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "Normal1",
    "Normal2",
    "Normal3",
    "IR",
    "Looeness0.1",
    "Unbalance_-1",
]

fig, axes = plt.subplots(
    len(CONDITIONS), 2,
    figsize=(15, 15),
    constrained_layout=True
)

for i, condition in enumerate(CONDITIONS):
    files = sorted((BASE / condition / "Vib").glob("*.mat"))

    if not files:
        print(f"[SKIP] {condition}: MAT file not found")
        continue

    path = files[0]
    data = loadmat(path)

    fs = float(data["fs"].item())
    signal = np.asarray(data["signals"][2, :], dtype=float)

    # DC offset removal
    signal = signal - np.mean(signal)

    # Time waveform: first 0.2 seconds
    duration = 0.2
    n = int(fs * duration)

    time = np.arange(n) / fs

    axes[i, 0].plot(time, signal[:n], linewidth=0.7)
    axes[i, 0].set_title(f"{condition} - Acceleration waveform")
    axes[i, 0].set_xlabel("Time (s)")
    axes[i, 0].set_ylabel("CH3 amplitude (raw unit)")
    axes[i, 0].grid(alpha=0.3)

    # Welch power spectral density
    freq, psd = welch(
        signal,
        fs=fs,
        nperseg=8192,
        noverlap=4096,
    )

    axes[i, 1].semilogy(freq, np.maximum(psd, 1e-20))
    axes[i, 1].set_xlim(0, 200)
    axes[i, 1].set_title(f"{condition} - Acceleration PSD")
    axes[i, 1].set_xlabel("Frequency (Hz)")
    axes[i, 1].set_ylabel("PSD (raw unit²/Hz)")
    axes[i, 1].grid(alpha=0.3)

    print(f"[OK] {condition}: {path.name}")

output = OUT / "900rpm_acceleration_lowfreq_comparison.png"

fig.savefig(output, dpi=200)
plt.close(fig)

print(f"\nSaved: {output}")