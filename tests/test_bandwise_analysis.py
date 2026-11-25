"""Regression tests for supplementary bandwise analysis."""

from copy import deepcopy
from unittest.mock import patch

import json
import numpy as np
import pytest

from vibration.bandwise_ae import build_frequency_bands
from vibration.bandwise_reconstruction import (
    analyze_reconstruction_data,
)
from vibration.bandwise_analysis_ui import (
    render_bandwise_analysis,
)


def make_sample(rpm):

    frequency, masks = build_frequency_bands(rpm)

    inputs = np.zeros(
        (8, 3201), dtype=np.float64
    )

    reconstructed = np.zeros_like(inputs)

    reconstructed[:, masks["0_100Hz"]] = 3.0
    reconstructed[:, masks["100_1000Hz"]] = 2.0
    reconstructed[:, masks["1000_10000Hz"]] = 1.0

    error = (reconstructed - inputs) ** 2
    scores = error.mean(axis=1)

    data = {
        "frequency": frequency.copy(),
        "input": inputs,
        "reconstruction": reconstructed,
        "scores": scores,
    }

    result = {
        "input": {"rpm": rpm},
        "bundle_id": f"synthetic-{rpm}",
        "models": {
            "autoencoder": {
                "file_score": float(scores.mean()),
                "file_threshold": 0.5,
                "predicted_anomaly": True,
                "segments": [
                    {
                        "segment_id": i,
                        "score": float(score),
                    }
                    for i, score in enumerate(scores)
                ],
            }
        },
    }

    return data, result


@pytest.mark.parametrize(
    "rpm", [900, 1200, 1800]
)
def test_bandwise_calculation(rpm):

    data, result = make_sample(rpm)

    analysis = analyze_reconstruction_data(
        data, result, rpm
    )

    bands = {
        band["band"]: band
        for band in analysis["bands"]
    }

    assert len(bands) == 5

    assert bands["0_100Hz"]["mse"] == pytest.approx(9)
    assert bands["100_1000Hz"]["mse"] == pytest.approx(4)
    assert bands["1000_10000Hz"]["mse"] == pytest.approx(1)

    assert bands["1X"]["mse"] == pytest.approx(9)
    assert bands["2X"]["mse"] == pytest.approx(9)

    assert analysis["highest_mse_band"] == "0_100Hz"

    assert (
        analysis["highest_contribution_band"]
        == "1000_10000Hz"
    )

    broad = [
        band for band in analysis["bands"]
        if band["band_type"] == "broad"
    ]

    assert sum(
        band["error_percentage"]
        for band in broad
    ) == pytest.approx(100)

    assert sum(
        band["global_contribution"]
        for band in broad
    ) == pytest.approx(
        analysis["overall_mse"]
    )


def test_operational_result_unchanged():

    data, result = make_sample(900)

    before = deepcopy(result)

    analysis = analyze_reconstruction_data(
        data, result, 900
    )

    assert result == before

    assert (
        analysis["analysis_status"]
        == "supplementary_only"
    )

    for key in (
        "anomaly_decision_provided",
        "fault_classification_provided",
        "approved_for_production",
        "operational_model_changed",
    ):
        assert analysis[key] is False

    assert "predicted_anomaly" not in analysis
    assert "threshold" not in analysis

    assert json.loads(
        json.dumps(analysis, allow_nan=False)
    ) == analysis


def test_zero_reconstruction_error():

    data, result = make_sample(900)

    data["reconstruction"][:] = 0
    data["scores"][:] = 0

    item = result["models"]["autoencoder"]

    item["file_score"] = 0.0
    item["predicted_anomaly"] = False

    for segment in item["segments"]:
        segment["score"] = 0.0

    analysis = analyze_reconstruction_data(
        data, result, 900
    )

    assert analysis["overall_mse"] == 0

    assert analysis["highest_mse_band"] is None
    assert analysis["highest_contribution_band"] is None

    for band in analysis["bands"]:

        assert band["mse"] == 0
        assert band["error_percentage"] == 0
        assert band["global_contribution"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "wrong_rpm",
        "wrong_frequency",
        "wrong_shape",
        "nan",
        "inf",
        "segment_mismatch",
        "file_mismatch",
        "reference_segment_mismatch",
    ],
)
def test_invalid_reconstruction_rejected(case):

    data, result = make_sample(900)
    rpm = 900

    if case == "wrong_rpm":
        rpm = 1200

    elif case == "wrong_frequency":
        data["frequency"][0] += 1

    elif case == "wrong_shape":
        data["input"] = data["input"][:, :-1]

    elif case == "nan":
        data["reconstruction"][0, 0] = np.nan

    elif case == "inf":
        data["input"][0, 0] = np.inf

    elif case == "segment_mismatch":
        data["scores"][0] += 1

    elif case == "file_mismatch":
        result["models"]["autoencoder"][
            "file_score"
        ] += 1

    elif case == "reference_segment_mismatch":
        result["models"]["autoencoder"][
            "segments"
        ][0]["score"] += 1

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        analyze_reconstruction_data(
            data, result, rpm
        )


def test_ui_renders_charts_and_table():

    data, result = make_sample(900)

    analysis = analyze_reconstruction_data(
        data, result, 900
    )

    with patch(
        "vibration.bandwise_analysis_ui.st"
    ) as ui:

        from unittest.mock import MagicMock

        ui.columns.return_value = (
            MagicMock(),
            MagicMock(),
        )

        render_bandwise_analysis(analysis)

        assert ui.altair_chart.call_count == 2
        assert ui.dataframe.call_count == 1

        for call in ui.altair_chart.call_args_list:

            chart = call.args[0]

            # Validate Altair chart schema.
            chart.to_dict(validate=True)

        frame = ui.dataframe.call_args.args[0]

        assert len(frame) == 5

        assert set(frame["대역"]) == {
            "1X",
            "2X",
            "저주파",
            "중주파",
            "고주파",
        }


def test_ui_rejects_unsafe_policy():

    data, result = make_sample(900)

    analysis = analyze_reconstruction_data(
        data, result, 900
    )

    analysis["anomaly_decision_provided"] = True

    with patch(
        "vibration.bandwise_analysis_ui.st"
    ):

        with pytest.raises(ValueError):
            render_bandwise_analysis(analysis)


def test_ui_without_analysis():

    with patch(
        "vibration.bandwise_analysis_ui.st"
    ) as ui:

        render_bandwise_analysis(None)

        ui.info.assert_called_once()
        ui.altair_chart.assert_not_called()
        ui.dataframe.assert_not_called()
