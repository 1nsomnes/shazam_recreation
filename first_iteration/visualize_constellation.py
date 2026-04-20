"""
visualize_constellation.py

Loads a binary constellation map from a .npy file and renders it as a
PNG scatter plot with frequency (Hz) and time (s) axes.
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Must match the parameters used in compute_spectrogram.py
SAMPLE_RATE = 8000
HOP_LENGTH = 512


def load_constellation(npy_path: str) -> np.ndarray:
    """Load a 2D binary constellation map from a .npy file."""
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Constellation file not found: {npy_path}")

    constellation = np.load(npy_path)
    if constellation.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {constellation.shape}")

    peak_count = int(constellation.sum())
    print(f"Loaded: {npy_path}")
    print(f"  Shape : {constellation.shape}  (freq_bins x time_frames)")
    print(f"  Peaks : {peak_count}")
    return constellation


def render_constellation(constellation: np.ndarray, output_path: str) -> None:
    """Render constellation points as a scatter plot on a black background."""
    n_freq_bins, n_time_frames = constellation.shape

    freq_idx, time_idx = np.nonzero(constellation)

    # Convert indices to physical units
    time_s = time_idx * (HOP_LENGTH / SAMPLE_RATE)
    freq_hz = freq_idx * (SAMPLE_RATE / 2) / (n_freq_bins - 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_facecolor("black")
    ax.scatter(time_s, freq_hz, s=0.4, c="cyan", marker=".", alpha=0.7)
    ax.set_xlim(0, n_time_frames * (HOP_LENGTH / SAMPLE_RATE))
    ax.set_ylim(0, SAMPLE_RATE / 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(os.path.basename(output_path))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved image to: {output_path}  ({size_kb:.0f} KB)")


DEFAULT_OUTPUT_DIR = "outputs"


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <constellation.npy> [-o output_dir]")
        sys.exit(1)

    npy_path = sys.argv[1]

    output_dir = DEFAULT_OUTPUT_DIR
    if "-o" in sys.argv:
        output_dir = sys.argv[sys.argv.index("-o") + 1]

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.png")

    constellation = load_constellation(npy_path)
    render_constellation(constellation, output_path)


if __name__ == "__main__":
    main()
