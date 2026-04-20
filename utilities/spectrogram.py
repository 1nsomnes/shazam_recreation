"""
MP3 → magnitude spectrogram utility.

Lifted from first_iteration/compute_spectrogram.py, stripped of the CLI
and print statements, and parameterized by SpectrogramConfig.
"""

import os
import sys

# Workaround: pyenv Python may be built without the _lzma C extension.
# librosa's transitive dep (pooch) imports lzma at load time, so we stub
# it out before importing librosa. Kept self-contained in this module.
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

from .config import SpectrogramConfig


_DEFAULT_CONFIG = SpectrogramConfig()


def load_audio(filepath: str, config: SpectrogramConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """Load an MP3, convert to mono, and resample to config.sample_rate."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    signal, _ = librosa.load(filepath, sr=config.sample_rate, mono=True)
    return signal


def compute_magnitude_spectrogram(
    signal: np.ndarray,
    config: SpectrogramConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Run STFT on a 1D signal and return the magnitude matrix of shape
    (n_freq_bins, n_time_frames). Phase is discarded.
    """
    stft_matrix = librosa.stft(
        signal,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
    )
    return np.abs(stft_matrix)


def mp3_to_spectrogram(
    filepath: str,
    config: SpectrogramConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    """Convenience wrapper: file path → magnitude spectrogram."""
    signal = load_audio(filepath, config)
    return compute_magnitude_spectrogram(signal, config)
