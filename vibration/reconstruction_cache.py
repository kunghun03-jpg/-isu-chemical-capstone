"""Session-local cache for verified AE reconstruction data."""

import hashlib

from vibration.ae_reconstruction_ui import reconstruction_data


CACHE_KEY = "v3_ae_reconstruction_cache"


def clear_reconstruction_cache(state):

    state.pop(CACHE_KEY, None)


def get_reconstruction_data(
    state, raw, rpm, config, result
):

    identity = state.get("v3_input_identity")

    if identity is None:
        clear_reconstruction_cache(state)
        raise ValueError("Missing input identity")

    if state.get("v2_result") is not result:
        clear_reconstruction_cache(state)
        raise ValueError("Analysis result is not current")

    if result["input"]["rpm"] != rpm:
        clear_reconstruction_cache(state)
        raise ValueError("RPM mismatch")

    raw_hash = hashlib.sha256(raw).hexdigest()

    if raw_hash != result["input"]["sha256"]:
        clear_reconstruction_cache(state)
        raise ValueError("Input file hash mismatch")

    key = (
        identity,
        raw_hash,
        rpm,
        config,
        result["bundle_id"],
    )

    cached = state.get(CACHE_KEY)

    if (
        cached is not None
        and cached["key"] == key
        and cached["result"] is result
    ):
        return cached["data"]

    clear_reconstruction_cache(state)

    data = reconstruction_data(
        raw, rpm, config, result
    )

    state[CACHE_KEY] = {
        "key": key,
        "result": result,
        "data": data,
    }

    return data
