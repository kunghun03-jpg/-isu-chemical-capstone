from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, detrend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FS = 25600
RPMS = [900, 1200, 1800]

# 발표용 대표 상태
CLASSES = [
    "Normal1",
    "IR",
    "OR",
    "Roller",
    "Looeness0.1",
    "Unbalance_-2"
]

hp = butter(
    4, 5,
    btype="highpass",
    fs=FS,
    output="sos"
)

bp = butter(
    4, [1800, 2800],
    btype="bandpass",
    fs=FS,
    output="sos"
)

def process(file):
    mat = loadmat(file)
    x = np.asarray(mat["signals"][2, :], dtype=np.float64)
    fs = float(np.asarray(mat["fs"]).squeeze())

    if len(x) != 65536 or fs != FS:
        raise ValueError("신호 길이 또는 fs 오류")

    if not np.isfinite(x).all():
        raise ValueError("원본 NaN/Inf")

    acc = detrend(x)

    acc_hp = sosfiltfilt(hp, acc)

    vel = np.zeros_like(acc_hp)
    vel[1:] = np.cumsum(
        (acc_hp[:-1] + acc_hp[1:]) / (2 * FS)
    )
    vel = sosfiltfilt(hp, detrend(vel))

    filtered = sosfiltfilt(bp, acc)
    env = np.abs(hilbert(filtered))
    env_ac = env - env.mean()

    if not np.isfinite(vel).all() or not np.isfinite(env_ac).all():
        raise ValueError("처리 결과 NaN/Inf")

    return acc, vel, env_ac

for rpm in RPMS:
    fig, axes = plt.subplots(
        len(CLASSES), 3,
        figsize=(17, 15),
        sharex=True
    )

    for i, name in enumerate(CLASSES):
        folder = DATA / f"20260919_{rpm}RPM" / name
        files = sorted(folder.rglob("*.mat"))

        result = None

        print(f"[확인] {rpm}RPM {name}: MAT 파일 {len(files)}개")

        for file in files:
            try:
                result = process(file)
                print(f"[선택] {rpm}RPM {name}: {file.name}")
                break
            except Exception as e:
                print(f"[제외] {file.name}: {e}")

        if result is None:
            print(f"[실패] {rpm}RPM {name}")
            continue

        acc, vel, env = result

        # 첫 0.2초 표시
        n = int(FS * 0.2)
        t = np.arange(n) / FS

        for j, signal in enumerate([acc, vel, env]):
            axes[i, j].plot(
                t, signal[:n],
                linewidth=0.7
            )
            axes[i, j].grid(alpha=0.3)

        axes[i, 0].set_ylabel(name)

    for j, title in enumerate([
        "Acceleration",
        "Velocity (integrated)",
        "Envelope AC (1800-2800 Hz)"
    ]):
        axes[0, j].set_title(title)
        axes[-1, j].set_xlabel("Time (s)")

    fig.suptitle(
        f"{rpm} RPM - Representative Time Waveforms",
        fontsize=16
    )
    fig.tight_layout()

    path = OUT / f"{rpm}rpm_representative_timewave.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"[저장] {path}")

print("\n대표 시간파형 분석 완료!")
