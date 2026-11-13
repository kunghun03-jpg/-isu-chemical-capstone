"""CH3 selection and spectrum shape checks shared by the current pipeline."""

import numpy as np

CHANNEL_NUMBER = 3
CHANNEL_INDEX = 2
INPUT_CHANNELS = 1
FREQUENCY_BINS = 3201
INPUT_SHAPE = (INPUT_CHANNELS, FREQUENCY_BINS)


def select_ch3(signals):
    """Select CH3 while retaining the channel axis; never modify the source array."""
    if signals.shape != (6, 65536):
        raise ValueError(f"Expected raw MAT shape (6, 65536), got {signals.shape}")
    return signals[CHANNEL_INDEX:CHANNEL_INDEX + 1].copy()


def validate_spectra(values):
    """Reject old six-channel artifacts before NumPy can broadcast them silently."""
    if values.ndim != 3 or values.shape[1:] != INPUT_SHAPE:
        raise ValueError(
            f"Expected CH3 spectra (N, 1, 3201), got {values.shape}. "
            "Regenerate CH3 spectra and normalization; six-channel artifacts are incompatible."
        )
    if not np.isfinite(values).all():
        raise ValueError("CH3 spectra contain NaN or infinity")
