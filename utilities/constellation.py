"""
Magnitude spectrogram → constellation points utility.

Same local-max + grid-density algorithm as
first_iteration/generate_constellation.py, but returns a sparse (N, 2)
array of accepted [freq_idx, time_idx] pairs instead of a full
spec-shaped binary mask.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from .config import ConstellationConfig


_DEFAULT_CONFIG = ConstellationConfig()


def _find_local_maxima(
    spec: np.ndarray,
    config: ConstellationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_max = maximum_filter(spec, size=config.filter_size)
    is_peak = (spec == local_max) & (spec > config.noise_floor)
    freqs, times = np.nonzero(is_peak)
    amplitudes = spec[freqs, times]
    return freqs, times, amplitudes


def _apply_density_constraint(
    freqs: np.ndarray,
    times: np.ndarray,
    amplitudes: np.ndarray,
    spec_shape: tuple[int, int],
    config: ConstellationConfig,
) -> np.ndarray:
    order = np.argsort(amplitudes)[::-1]
    freqs = freqs[order]
    times = times[order]

    n_freq_blocks = int(np.ceil(spec_shape[0] / config.grid_size[0]))
    n_time_blocks = int(np.ceil(spec_shape[1] / config.grid_size[1]))
    block_counts = np.zeros((n_freq_blocks, n_time_blocks), dtype=int)

    accepted_f: list[int] = []
    accepted_t: list[int] = []

    for f, t in zip(freqs, times):
        bf = f // config.grid_size[0]
        bt = t // config.grid_size[1]

        if block_counts[bf, bt] < config.k:
            accepted_f.append(int(f))
            accepted_t.append(int(t))
            block_counts[bf, bt] += 1

    if not accepted_f:
        return np.empty((0, 2), dtype=np.intp)
    return np.column_stack([accepted_f, accepted_t]).astype(np.intp)


def generate_constellation(
    spec: np.ndarray,
    config: ConstellationConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Return an (N, 2) array of [freq_idx, time_idx] for each accepted
    peak. No fixed order is promised — the hash step re-sorts anyway.
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {spec.shape}")
    freqs, times, amplitudes = _find_local_maxima(spec, config)
    return _apply_density_constraint(freqs, times, amplitudes, spec.shape, config)
