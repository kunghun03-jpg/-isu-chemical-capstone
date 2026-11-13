from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import welch

# 1. 기본 설정
DATA_DIR = Path(
    "/mnt/c/Users/kungh/OneDrive/바탕 화면/캡스톤 디자인"
)

DATES = ["20260912", "20260916", "20260919"]
RPMS = [900, 1200, 1800]

FAULT_FOLDERS = [
    "IR",
    "OR",
    "Roller",
    "Looseness0.05",
    "Looseness0.1",
    "LoosenessFull",
    "Unbalance_-1",
    "Unbalance_-2"
]

OUTPUT_DIR = Path(__file__).resolve().parent
# 2. 기존 정상 데이터와 동일한 전처리
def preprocess_mat(file_path):
    data = sio.loadmat(file_path)

    signals = data["signals"]
    fs = float(np.asarray(data["fs"]).squeeze())

    if signals.shape != (6, 65536):
        raise ValueError("신호 크기 오류")

    if fs != 25600:
        raise ValueError("샘플링 주파수 오류")

    ch3 = np.asarray(signals[2, :], dtype=np.float64)

    if not np.isfinite(ch3).all():
        raise ValueError("원본 신호에 이상값 있음")

    freq, psd = welch(
        ch3,
        fs=fs,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        nfft=8192
    )

    spectrum = np.sqrt(psd)

    mask = (freq >= 0) & (freq < 10000)
    spectrum_selected = spectrum[mask]

    if len(spectrum_selected) != 3200:
        raise ValueError("스펙트럼 크기 오류")

    spectrum_800 = spectrum_selected.reshape(
        800, 4
    ).mean(axis=1)

    log_spectrum = np.log10(
        spectrum_800 + 1e-12
    )

    if not np.isfinite(log_spectrum).all():
        raise ValueError("전처리 결과에 이상값 있음")

    return log_spectrum

# 3. 고장 데이터 수집
X = []
labels = []
groups = []
file_names = []
excluded_files = []
total_found = 0
for date in DATES:
    print(f"\n===== {date} =====")

    for rpm in RPMS:
        rpm_dir = DATA_DIR / date / f"{date}_{rpm}RPM"

        for fault_name in FAULT_FOLDERS:
            fault_dir = rpm_dir / fault_name

            if not fault_dir.is_dir():
                print("폴더 없음:", fault_dir)
                continue
            files = sorted(fault_dir.rglob("*.mat"))
            total_found += len(files)
            print(
                f"{rpm}RPM / {fault_name}: "
                f"{len(files)}개"
            )
            for file_path in files:
                try:
                    spectrum = preprocess_mat(file_path)
                    X.append(spectrum)
                    labels.append(fault_name)
                    groups.append(f"{date}_{rpm}RPM")
                    file_names.append(str(file_path))

                except Exception as e:
                    print("제외:", file_path.name, "|", e)
                    excluded_files.append(str(file_path))

# 4. 결과 저장
X = np.asarray(X, dtype=np.float32)
labels = np.asarray(labels)
groups = np.asarray(groups)
file_names = np.asarray(file_names)

print("\n===== 고장 데이터 수집 결과 =====")
print("발견한 파일:", total_found)
print("정상 처리:", len(X))
print("제외한 파일:", len(excluded_files))
print("데이터 크기:", X.shape)

if len(X) == 0:
    raise RuntimeError("처리된 고장 데이터가 없습니다.")
np.savez(
    OUTPUT_DIR / "fault_dataset.npz",
    X=X,
    labels=labels,
    groups=groups,
    file_names=file_names
)
(OUTPUT_DIR / "excluded_fault_files.txt").write_text(
    "\n".join(excluded_files),
    encoding="utf-8"
)
print("\n고장 데이터 전처리 완료!")
print("저장 파일: fault_dataset.npz")
