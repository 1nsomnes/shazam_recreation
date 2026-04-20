"""
visualize_spectrogram.py

Loads a magnitude spectrogram from a .npy file and renders it as a
PNG image with frequency (Hz) and time (s) axes.
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt


# Must match the parameters used in compute_spectrogram.py
SAMPLE_RATE = 8000
HOP_LENGTH = 512


def load_spectrogram(npy_path: str) -> np.ndarray:
    """Load a 2D magnitude matrix from a .npy file."""
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Spectrogram file not found: {npy_path}")

    magnitude = np.load(npy_path)
    if magnitude.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {magnitude.shape}")

    print(f"Loaded: {npy_path}")
    print(f"  Shape : {magnitude.shape}  (freq_bins x time_frames)")
    print(f"  Dtype : {magnitude.dtype}")
    return magnitude


def render_spectrogram(magnitude: np.ndarray, output_path: str) -> None:
    """
    Convert magnitude to dB scale and render as a PNG with labelled axes.

    The dB conversion (20*log10) compresses the dynamic range so quiet
    detail is visible alongside loud peaks.
    """
    n_freq_bins, n_time_frames = magnitude.shape

    # Convert to decibels; clamp floor to avoid log(0)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))

    # Build physical axis values
    time_axis = np.arange(n_time_frames) * (HOP_LENGTH / SAMPLE_RATE)  # seconds
    freq_axis = np.linspace(0, SAMPLE_RATE / 2, n_freq_bins)           # Hz

    fig, ax = plt.subplots(figsize=(14, 5))
    img = ax.pcolormesh(
        time_axis,
        freq_axis,
        magnitude_db,
        shading="auto",
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(os.path.basename(output_path))
    fig.colorbar(img, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved image to: {output_path}  ({size_kb:.0f} KB)")


DEFAULT_OUTPUT_DIR = "outputs"


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <spectrogram.npy> [-o output_dir]")
        sys.exit(1)

    npy_path = sys.argv[1]

    output_dir = DEFAULT_OUTPUT_DIR
    if "-o" in sys.argv:
        output_dir = sys.argv[sys.argv.index("-o") + 1]

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.png")

    magnitude = load_spectrogram(npy_path)
    render_spectrogram(magnitude, output_path)


if __name__ == "__main__":
    main()
