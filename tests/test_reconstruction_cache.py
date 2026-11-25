"""Tests for session-local AE reconstruction caching."""

import hashlib
from copy import deepcopy
from unittest.mock import patch

import pytest

from vibration.reconstruction_cache import (
    CACHE_KEY,
    get_reconstruction_data,
    clear_reconstruction_cache,
)


def make_context(raw=b"sample-mat", rpm=900):

    config = "test-config"

    result = {
        "input": {
            "rpm": rpm,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "bundle_id": "test-bundle",
    }

    state = {
        "v3_input_identity": (
            hashlib.sha256(raw).hexdigest(),
            rpm,
            "bundle-stamp-1",
        ),
        "v2_result": result,
    }

    return state, raw, rpm, config, result


def test_same_input_reuses_reconstruction():

    state, raw, rpm, config, result = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ) as reconstruct:

        first = get_reconstruction_data(
            state, raw, rpm, config, result
        )

        for _ in range(5):

            current = get_reconstruction_data(
                state, raw, rpm, config, result
            )

            assert current is first

        assert reconstruct.call_count == 1

    assert CACHE_KEY in state


@pytest.mark.parametrize(
    "change",
    [
        "file",
        "rpm",
        "bundle_stamp",
        "bundle_id",
        "config",
        "new_result",
    ],
)
def test_changed_input_reconstructs(change):

    state, raw, rpm, config, result = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ) as reconstruct:

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        result = deepcopy(result)

        if change == "file":

            raw = b"new-mat"

            result["input"]["sha256"] = (
                hashlib.sha256(raw).hexdigest()
            )

        elif change == "rpm":

            rpm = 1200
            result["input"]["rpm"] = rpm

        elif change == "bundle_stamp":

            state["v3_input_identity"] = (
                "changed-bundle-stamp",
            )

        elif change == "bundle_id":

            result["bundle_id"] = "new-bundle"

        elif change == "config":

            config = "new-config"

        if change in (
            "file",
            "rpm",
            "new_result",
        ):
            state["v2_result"] = result
        else:
            # Keep the same result instance to test
            # cache-key invalidation independently.
            current = state["v2_result"]
            current.update(result)
            result = current

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        assert reconstruct.call_count == 2


def test_explicit_cache_reset():

    state, raw, rpm, config, result = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ) as reconstruct:

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        clear_reconstruction_cache(state)

        assert CACHE_KEY not in state

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        assert reconstruct.call_count == 2


@pytest.mark.parametrize(
    "case",
    [
        "missing_identity",
        "old_result",
        "wrong_rpm",
        "wrong_hash",
    ],
)
def test_invalid_context_clears_cache(case):

    state, raw, rpm, config, result = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ) as reconstruct:

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        if case == "missing_identity":

            state.pop("v3_input_identity")

        elif case == "old_result":

            state["v2_result"] = deepcopy(result)

        elif case == "wrong_rpm":

            rpm = 1800

        elif case == "wrong_hash":

            raw = b"wrong-file"

        with pytest.raises(ValueError):

            get_reconstruction_data(
                state, raw, rpm, config, result
            )

        assert CACHE_KEY not in state

        assert reconstruct.call_count == 1


def test_failed_reconstruction_is_not_cached():

    state, raw, rpm, config, result = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data",
        side_effect=RuntimeError("AE failure"),
    ):

        with pytest.raises(RuntimeError):

            get_reconstruction_data(
                state, raw, rpm, config, result
            )

    assert CACHE_KEY not in state


def test_sessions_do_not_share_cache():

    first = make_context()
    second = make_context()

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ) as reconstruct:

        get_reconstruction_data(*first)
        get_reconstruction_data(*second)

        assert reconstruct.call_count == 2

    clear_reconstruction_cache(first[0])

    assert CACHE_KEY not in first[0]
    assert CACHE_KEY in second[0]


def test_operational_result_unchanged():

    state, raw, rpm, config, result = make_context()

    before = deepcopy(result)

    with patch(
        "vibration.reconstruction_cache.reconstruction_data"
    ):

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

        get_reconstruction_data(
            state, raw, rpm, config, result
        )

    assert result == before
