"""
compute_spectrogram.py

Loads an MP3 file, computes its Short-Time Fourier Transform (STFT),
extracts the magnitude spectrogram, and saves it as a .npy binary file
for downstream constellation-map generation.
"""

import sys
import os

# Workaround: pyenv Python may be built without lzma C extension.
# Provide a stub so librosa's transitive dependency (pooch) can import.
try:
    import lzma  # noqa: F401
except ImportError:
    import types as _types
    _lzma = _types.ModuleType("_lzma")
    _lzma._encode_filter_properties = lambda f: (_ for _ in ()).throw(NotImplementedError)
    _lzma._decode_filter_properties = lambda fid, raw: (_ for _ in ()).throw(NotImplementedError)
    for _name in ("LZMACompressor", "LZMADecompressor"):
        setattr(_lzma, _name, type(_name, (), {}))

    class _LZMAError(Exception):
        pass
    _lzma.LZMAError = _LZMAError
    for _attr in (
        "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
        "CHECK_ID_MAX", "CHECK_UNKNOWN", "FILTER_LZMA1", "FILTER_LZMA2",
        "FILTER_DELTA", "FILTER_X86", "FILTER_IA64", "FILTER_ARM",
        "FILTER_ARMTHUMB", "FILTER_SPARC", "FILTER_POWERPC",
        "FORMAT_AUTO", "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW",
        "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
        "MODE_FAST", "MODE_NORMAL", "PRESET_DEFAULT", "PRESET_EXTREME",
    ):
        setattr(_lzma, _attr, 0)
    sys.modules["_lzma"] = _lzma

import numpy as np
import librosa


# --- STFT parameters ---
SAMPLE_RATE = 8000      # Target sample rate in Hz
N_FFT = 1024            # Window size (number of FFT points)
HOP_LENGTH = 512        # Hop between successive frames
# Frequency bins = n_fft // 2 + 1 = 513


def load_audio(filepath: str) -> np.ndarray:
    """Load an MP3 file, convert to mono, and resample to SAMPLE_RATE."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # sr=SAMPLE_RATE forces resampling; mono=True mixes stereo to single channel
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    print(f"Loaded: {filepath}")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {len(signal) / sr:.2f} s")
    print(f"  Samples     : {len(signal)}")
    return signal


def compute_magnitude_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    Compute the STFT and return only the magnitude (phase discarded).

    Returns
    -------
    magnitude : np.ndarray, shape (n_freq_bins, n_time_frames)
        n_freq_bins  = N_FFT // 2 + 1 = 513
        n_time_frames = 1 + (len(signal) - N_FFT) // HOP_LENGTH  (approx)
    """
    # librosa.stft returns a complex matrix of shape (1 + n_fft/2, n_frames)
    stft_matrix = librosa.stft(
        signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
    )

    # Discard phase, keep magnitude only
    magnitude = np.abs(stft_matrix)

    print(f"STFT complete:")
    print(f"  Frequency bins : {magnitude.shape[0]}")
    print(f"  Time frames    : {magnitude.shape[1]}")
    return magnitude


def save_spectrogram(magnitude: np.ndarray, output_path: str) -> None:
    """Save the 2D magnitude matrix to a .npy binary file."""
    np.save(output_path, magnitude)
    print(f"Saved spectrogram to: {output_path}")
    print(f"  Shape  : {magnitude.shape}  (freq_bins x time_frames)")
    print(f"  Dtype  : {magnitude.dtype}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size   : {size_mb:.2f} MB")


DEFAULT_OUTPUT_DIR = "spectrograms"


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.mp3> [-o output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]

    output_dir = DEFAULT_OUTPUT_DIR
    if "-o" in sys.argv:
        output_dir = sys.argv[sys.argv.index("-o") + 1]

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_spectrogram.npy")

    signal = load_audio(input_path)
    magnitude = compute_magnitude_spectrogram(signal)

    # Sanity check: n_fft=1024 must produce exactly 513 frequency bins
    assert magnitude.shape[0] == 513, (
        f"Expected 513 frequency bins, got {magnitude.shape[0]}"
    )

    save_spectrogram(magnitude, output_path)


if __name__ == "__main__":
    main()
