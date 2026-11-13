# 보관된 초기 실험 코드: prepare_spectra. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "900rpm" / "first"
output_dir.mkdir(exist_ok=True)

# 첫 실험 설정
WINDOW_SIZE = 8192  # 구간당 0.32초
EXPECTED_FS = 25600.0
MAX_FREQ = 10000
CHECK_LIMIT = 1e100  # 숫자 손상 후보 확인용 임시 기준

window = np.hanning(WINDOW_SIZE)
freq = np.fft.rfftfreq(WINDOW_SIZE, d=1 / EXPECTED_FS)
freq_mask = freq <= MAX_FREQ

# 학습·검증·평가 파일 목록 읽기
with (output_dir / "split_900rpm.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

excluded_files = []

for split in ("train", "validation", "test_normal", "test_anomaly"):
    selected = [row for row in rows if row["split"] == split]

    spectra_list = []
    source_files = []
    conditions = []
    segment_ids = []
    excluded_count = 0

    for row in selected:
        data = loadmat(project_dir / row["file"])
        signals = data["signals"].copy()
        fs = float(data["fs"].item())

        if signals.shape != (6, 65536) or fs != EXPECTED_FS:
            raise ValueError(f"데이터 형식 확인 필요: {row['file']}")

        # NaN·무한대 또는 극단값이 있는 파일은 전체 제외
        nonfinite = ~np.isfinite(signals)
        extreme = np.isfinite(signals) & (np.abs(signals) > CHECK_LIMIT)

        if nonfinite.any() or extreme.any():
            excluded_files.append(
                {
                    "file": row["file"],
                    "condition": row["condition"],
                    "split": split,
                    "nonfinite_count": int(nonfinite.sum()),
                    "extreme_count": int(extreme.sum()),
                }
            )
            excluded_count += 1
            print("손상 의심 파일 제외:", row["file"])
            continue

        # 파일 하나를 겹치지 않는 8개 구간으로 분할
        # 모양: (8개 구간, 6채널, 8192개 측정값)
        segments = signals.reshape(6, 8, WINDOW_SIZE).transpose(1, 0, 2)

        # 구간별·채널별 평균 제거
        segments = segments - segments.mean(axis=2, keepdims=True)

        # Hann 창 적용 후 FFT 및 단측 진폭 계산
        amplitude = np.abs(np.fft.rfft(segments * window, axis=2)) / window.sum()

        amplitude[:, :, 1:-1] *= 2

        # 0~10000Hz 선택 후 로그 변환
        log_spectra = np.log1p(amplitude[:, :, freq_mask])

        if not np.isfinite(log_spectra).all():
            raise ValueError(f"변환 결과 확인 필요: {row['file']}")

        spectra_list.append(log_spectra.astype(np.float32))

        # 각 구간의 원본 파일과 상태 기록
        for segment_id in range(8):
            source_files.append(row["file"])
            conditions.append(row["condition"])
            segment_ids.append(segment_id)

    if not spectra_list:
        raise ValueError(f"{split}: 사용할 데이터가 없습니다.")

    X = np.concatenate(spectra_list, axis=0)

    # 정규화 전 스펙트럼 저장
    np.savez_compressed(
        output_dir / f"{split}_spectra.npz",
        X=X,
        frequency=freq[freq_mask],
        source_file=np.array(source_files),
        condition=np.array(conditions),
        segment_id=np.array(segment_ids),
    )

    print(
        f"\n{split}: {X.shape}, "
        f"사용 파일 {len(source_files) // 8}개, "
        f"제외 파일 {excluded_count}개\n"
    )

    del X, spectra_list

# 제외 파일 목록 저장
report_path = output_dir / "excluded_files_900rpm.csv"

with report_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["file", "condition", "split", "nonfinite_count", "extreme_count"]
    )
    writer.writeheader()
    writer.writerows(excluded_files)

print("스펙트럼 저장 완료!")
print("전체 제외 파일:", len(excluded_files), "개")
print("제외 목록 저장:", report_path)
