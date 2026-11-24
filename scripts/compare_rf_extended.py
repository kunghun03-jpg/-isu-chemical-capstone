"""Offline comparison of baseline and extended RF features."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from scripts.evaluate_fault_classifier import FEATURES as BASE_FEATURES
from scripts.evaluate_fault_classifier import RF_CONFIG
from scripts.experiment_rf_features import EXTRA_FEATURES

CLASSES = ("Normal", "Bearing", "Looseness", "Unbalance")
TRAIN_DATE = "20260912"
VALIDATION_DATE = "20260916"
TEST_DATE = "20260919"


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_xy(rows, features):
    x = np.array([
        [
            float(row[name]) if row[name].strip().lower() not in ("", "nan")
            else np.nan
            for name in features
        ]
        for row in rows
    ], dtype=float)

    y = np.array([row["family"] for row in rows])
    return x, y


def evaluate(name, features, rows):
    train = [r for r in rows if r["date"] == TRAIN_DATE]
    validation = [r for r in rows if r["date"] == VALIDATION_DATE]
    test = [r for r in rows if r["date"] == TEST_DATE]

    x_train, y_train = make_xy(train, features)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_CONFIG)),
    ])

    model.fit(x_train, y_train)

    print(f"\n========== {name} ==========")
    print("특징 수:", len(features))

    for split_name, subset in (
        ("Validation", validation),
        ("Test", test),
    ):
        x, y = make_xy(subset, features)
        prediction = model.predict(x)

        print(f"\n--- {split_name} ({len(subset)}개) ---")
        print(classification_report(
            y, prediction,
            labels=CLASSES,
            zero_division=0,
            digits=3,
        ))
        print("혼동행렬")
        print("행=실제, 열=예측:", CLASSES)
        print(confusion_matrix(y, prediction, labels=CLASSES))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input.expanduser())

    if len(rows) != 2378:
        raise ValueError(f"예상 데이터 2378개와 다름: {len(rows)}")

    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("중복 MAT 경로가 있습니다.")

    evaluate("기존 RF (9개 특징)", BASE_FEATURES, rows)
    evaluate("개선 RF (15개 특징)", BASE_FEATURES + EXTRA_FEATURES, rows)


if __name__ == "__main__":
    main()
