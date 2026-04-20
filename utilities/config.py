"""
Centralized config for the fingerprinting pipeline.

Each stage has its own frozen dataclass so callers can override only the
knobs they care about. Defaults match the values validated in
first_iteration/ and are what the pipeline ran with end-to-end.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 8000
    n_fft: int = 1024
    hop_length: int = 512
    window: str = "hann"

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass(frozen=True)
class ConstellationConfig:
    filter_size: tuple[int, int] = (15, 15)
    grid_size: tuple[int, int] = (32, 32)
    k: int = 5
    noise_floor: float = 1e-3


@dataclass(frozen=True)
class HashConfig:
    delta_t_min: int = 1
    delta_t_max: int = 127
    delta_f_max: int = 64
    fan_out: int = 5
    freq_quant_bits: int = 2
    # Width of each packed field. Defaults assume n_fft=1024 (513 bins → 10 raw
    # bits of freq → 10 - freq_quant_bits = 8 after quantization). If you change
    # n_fft, bump f_field_bits accordingly.
    f_field_bits: int = 8
    dt_field_bits: int = 7
