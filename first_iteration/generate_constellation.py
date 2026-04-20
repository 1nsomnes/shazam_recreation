"""
generate_constellation.py

Reads a 2D magnitude spectrogram from a .npy file, finds local maxima,
applies a grid-based density constraint, and outputs a binary
constellation map as a .npy file (same shape, 1 at each accepted peak,
0 everywhere else).
"""

import sys
import os

import numpy as np
from scipy.ndimage import maximum_filter


# --- Configurable parameters ---
FILTER_SIZE = (15, 15)  # 2D neighborhood for local maximum detection (freq, time)
GRID_SIZE = (32, 32)    # Coarse grid block dimensions for density constraint
K = 5                   # Max peaks allowed per grid block
NOISE_FLOOR = 1e-3      # Minimum magnitude to consider (ignore near-silence)


def load_spectrogram(npy_path: str) -> np.ndarray:
    """Load a 2D magnitude spectrogram from a .npy file."""
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Spectrogram file not found: {npy_path}")

    spec = np.load(npy_path)
    if spec.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {spec.shape}")

    print(f"Loaded: {npy_path}")
    print(f"  Shape : {spec.shape}  (freq_bins x time_frames)")
    return spec


def find_local_maxima(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a 2D maximum filter and return coordinates of points whose
    original value equals the local maximum (i.e. they ARE the local max).

    Returns (freq_indices, time_indices, amplitudes) for all candidates
    above the noise floor.
    """
    local_max = maximum_filter(spec, size=FILTER_SIZE)

    # A point is a candidate peak if it equals the local max and exceeds the noise floor
    is_peak = (spec == local_max) & (spec > NOISE_FLOOR)

    freqs, times = np.nonzero(is_peak)
    amplitudes = spec[freqs, times]

    print(f"Local maxima found: {len(freqs)}")
    return freqs, times, amplitudes


def apply_density_constraint(
    freqs: np.ndarray,
    times: np.ndarray,
    amplitudes: np.ndarray,
    spec_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grid-and-Sort: keep at most K peaks per coarse grid block,
    preferring the loudest candidates.

    Returns (freq_indices, time_indices) of accepted peaks.
    """
    # Sort candidates by amplitude, loudest first
    order = np.argsort(amplitudes)[::-1]
    freqs = freqs[order]
    times = times[order]

    # Determine grid block counts
    n_freq_blocks = int(np.ceil(spec_shape[0] / GRID_SIZE[0]))
    n_time_blocks = int(np.ceil(spec_shape[1] / GRID_SIZE[1]))
    block_counts = np.zeros((n_freq_blocks, n_time_blocks), dtype=int)

    accepted_f = []
    accepted_t = []

    for f, t in zip(freqs, times):
        bf = f // GRID_SIZE[0]
        bt = t // GRID_SIZE[1]

        if block_counts[bf, bt] < K:
            accepted_f.append(f)
            accepted_t.append(t)
            block_counts[bf, bt] += 1

    accepted_f = np.array(accepted_f, dtype=np.intp)
    accepted_t = np.array(accepted_t, dtype=np.intp)

    total_blocks = n_freq_blocks * n_time_blocks
    active_blocks = int(np.count_nonzero(block_counts))
    print(f"Density constraint applied:")
    print(f"  Grid           : {n_freq_blocks} x {n_time_blocks} blocks ({total_blocks} total)")
    print(f"  Active blocks  : {active_blocks}")
    print(f"  Accepted peaks : {len(accepted_f)}")
    return accepted_f, accepted_t


def build_constellation_map(
    accepted_f: np.ndarray,
    accepted_t: np.ndarray,
    spec_shape: tuple[int, int],
) -> np.ndarray:
    """Create a binary array: 1 at each accepted peak, 0 elsewhere."""
    constellation = np.zeros(spec_shape, dtype=np.uint8)
    constellation[accepted_f, accepted_t] = 1
    return constellation


DEFAULT_OUTPUT_DIR = "constellations"


def save_constellation(constellation: np.ndarray, output_path: str) -> None:
    """Save the binary constellation map to a .npy file."""
    np.save(output_path, constellation)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved constellation map to: {output_path}")
    print(f"  Shape  : {constellation.shape}  (freq_bins x time_frames)")
    print(f"  Dtype  : {constellation.dtype}")
    print(f"  Ones   : {int(constellation.sum())}  ({constellation.sum() / constellation.size * 100:.2f}% of pixels)")
    print(f"  Size   : {size_kb:.0f} KB")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <spectrogram.npy> [-o output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]

    output_dir = DEFAULT_OUTPUT_DIR
    if "-o" in sys.argv:
        output_dir = sys.argv[sys.argv.index("-o") + 1]

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(input_path).replace("_spectrogram.npy", "")
    output_path = os.path.join(output_dir, f"{basename}_constellation.npy")

    spec = load_spectrogram(input_path)
    freqs, times, amplitudes = find_local_maxima(spec)
    accepted_f, accepted_t = apply_density_constraint(freqs, times, amplitudes, spec.shape)
    constellation = build_constellation_map(accepted_f, accepted_t, spec.shape)

    save_constellation(constellation, output_path)


if __name__ == "__main__":
    main()
