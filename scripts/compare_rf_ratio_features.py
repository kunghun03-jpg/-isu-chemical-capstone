"""Offline RF feature-ablation experiment; production models unchanged."""

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline

from scripts.evaluate_fault_classifier import RF_CONFIG, FEATURES
from scripts.experiment_rf_features import EXTRA_FEATURES

PATH = Path.home() / "isu_rf_extended_features_run01.csv"
CLASSES = ("Normal", "Bearing", "Looseness", "Unbalance")

FEATURE_SETS = {
    "A_baseline_9": FEATURES,
    "B_extended_15": FEATURES + EXTRA_FEATURES,
    "C_relative_5": (
        "crest_factor_ac",
        "ratio_2x_1x",
        "ratio_3x_1x",
        "harmonic_energy_fraction",
        "energy_high",
    ),
    "D_harmonic_6": (
        "one_x_amplitude",
        "two_x_amplitude",
        "three_x_amplitude",
        "four_x_amplitude",
        "five_x_amplitude",
        "harmonic_energy_fraction",
    ),
}

with PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

train = [r for r in rows if r["date"] == "20260912"]
validation = [r for r in rows if r["date"] == "20260916"]
test = [r for r in rows if r["date"] == "20260919"]

def xy(subset, features):
    x = np.array([
        [float(r[name]) if r[name].strip().lower() not in ("", "nan")
         else np.nan for name in features]
        for r in subset
    ])
    y = np.array([r["family"] for r in subset])
    return x, y

for name, features in FEATURE_SETS.items():
    x_train, y_train = xy(train, features)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_CONFIG)),
    ])
    model.fit(x_train, y_train)

    print(f"\n=== {name} ===")

    for split_name, subset in (("Validation", validation), ("Test", test)):
        x, y = xy(subset, features)
        pred = model.predict(x)

        recalls = recall_score(
            y, pred, labels=CLASSES,
            average=None, zero_division=0
        )

        print(
            split_name,
            "accuracy=", round(accuracy_score(y, pred), 3),
            "recall=", {
                c: round(float(v), 3)
                for c, v in zip(CLASSES, recalls)
            },
        )

        if split_name == "Validation":
            cm = confusion_matrix(y, pred, labels=CLASSES)
            print(
                "실제 Looseness 예측 분포:",
                dict(zip(CLASSES, cm[2].tolist()))
            )
