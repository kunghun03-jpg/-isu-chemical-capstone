
"""Experimental RF single-feature sensitivity; not causal attribution."""
import numpy as np

from vibration.candidate_engine import FEATURES, feature_matrix


def feature_sensitivity(engine, features, predicted):
    pipeline = engine.pipeline
    classes = list(pipeline.classes_)

    if predicted not in classes:
        raise ValueError("RF 예측 클래스가 일치하지 않습니다.")

    x = feature_matrix([features])
    target = classes.index(predicted)

    original = float(pipeline.predict_proba(x)[0, target])
    medians = pipeline.named_steps["imputer"].statistics_

    rows = []

    for index, name in enumerate(FEATURES):
        if not np.isfinite(x[0, index]):
            continue

        changed = x.copy()
        changed[0, index] = medians[index]

        alternative = float(
            pipeline.predict_proba(changed)[0, target]
        )

        rows.append({
            "특징량": name,
            "실제 측정값": float(x[0, index]),
            "학습 중앙값": float(medians[index]),
            "대체 후 RF 점수": alternative,
            "원래 점수와 차이": original - alternative,
        })

    rows.sort(
        key=lambda item: abs(item["원래 점수와 차이"]),
        reverse=True
    )

    return original, rows
