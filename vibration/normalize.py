# 정상 학습 데이터의 평균·표준편차로 모든 분할을 정규화합니다.

from pathlib import Path
import json
import numpy as np
from vibration.channel import validate_spectra


def run(project_dir, output_dir, rpm, *, mode="per_frequency"):
    if mode not in {"per_frequency", "global_train"}:
        raise ValueError(f"지원하지 않는 정규화 방식: {mode}")
    with np.load(output_dir / "train_spectra.npz") as data:
        train = data["X"].astype(np.float64)
    validate_spectra(train)

    # 학습 데이터만으로 기준을 정해 검증·평가 정보의 유입을 방지합니다.
    if mode == "global_train":
        # 해당 RPM의 Train 로그 스펙트럼 전체에서 단일 평균·표준편차를 구합니다.
        mean = train.mean(keepdims=True)
        std = train.std(keepdims=True)
    else:
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)

    # 표준편차 하한을 두어 0 또는 매우 작은 값으로 나누지 않도록 합니다.
    std = np.maximum(std, 1e-06)
    np.savez_compressed(output_dir / f"normalization_{rpm}rpm.npz", mean=mean, std=std)
    config = {"mode": mode, "reference_split": "train", "ddof": 0,
              "statistic_shape": list(mean.shape), "std_floor": 1e-6,
              "train_segments": len(train), "train_values": int(train.size),
              "mean_min": float(mean.min()), "mean_max": float(mean.max()),
              "std_min": float(std.min()), "std_max": float(std.max())}
    (output_dir / f"normalization_config_{rpm}rpm.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    del train

    # 모든 분할에 같은 기준을 적용하고 파일·구간 정보를 보존합니다.
    for split in ("train", "validation", "test_normal", "test_anomaly"):
        source_path = output_dir / f"{split}_spectra.npz"
        with np.load(source_path) as data:
            X = data["X"].astype(np.float64)
            validate_spectra(X)
            X = ((X - mean) / std).astype(np.float32)
            if not np.isfinite(X).all():
                raise ValueError(f"{split}: 정규화 결과 확인 필요")
            np.savez_compressed(
                output_dir / f"{split}_normalized.npz",
                X=X,
                frequency=data["frequency"],
                source_file=data["source_file"],
                condition=data["condition"],
                segment_id=data["segment_id"],
                **{
                    key: data[key]
                    for key in ("date", "rpm", "run_id", "date_group")
                    if key in data.files
                },
            )
        print(f"{split}: {X.shape}, 유한값 확인 완료")
        del X
    print("정규화 완료!")
