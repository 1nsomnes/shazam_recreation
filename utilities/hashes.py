"""
Constellation points → fingerprint hashes utility.

Same anchor/target pairing and bit-packing as
first_iteration/generate_hashes.py, but returns a plain
list[(hash, t_anchor)] with no song_id baked in and no JSON I/O. The
caller decides where those rows land (e.g. INSERT into Postgres with
the song_id from the songs table).
"""

import numpy as np

from .config import HashConfig


_DEFAULT_CONFIG = HashConfig()


def generate_hashes(
    peaks: np.ndarray,
    config: HashConfig = _DEFAULT_CONFIG,
) -> list[tuple[int, int]]:
    """
    Pair each anchor peak with up to fan_out targets in its forward-
    looking target zone and bit-pack each pair into a single integer.

    Hash layout (with defaults: 8 + 8 + 7 = 23 bits):
        [f_anchor_q : f_field_bits] [f_target_q : f_field_bits] [delta_t : dt_field_bits]

    Parameters
    ----------
    peaks : np.ndarray, shape (N, 2)
        Rows of [freq_idx, time_idx]. Order does not matter; this
        function re-sorts by time then freq internally.

    Returns
    -------
    list of (hash, t_anchor) tuples, in the order they were emitted.
    """
    if len(peaks) == 0:
        return []
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError(f"Expected peaks of shape (N, 2), got {peaks.shape}")

    # Time-sorted, freq-tiebroken — lets us break on delta_t overflow.
    order = np.lexsort((peaks[:, 0], peaks[:, 1]))
    peaks = peaks[order]

    f_shift = config.f_field_bits + config.dt_field_bits
    t_shift = config.dt_field_bits

    out: list[tuple[int, int]] = []
    n_peaks = len(peaks)

    for i in range(n_peaks):
        f_anchor_raw = int(peaks[i, 0])
        f_anchor_q = f_anchor_raw >> config.freq_quant_bits
        t_anchor = int(peaks[i, 1])
        fan_count = 0

        for j in range(i + 1, n_peaks):
            f_target_raw = int(peaks[j, 0])
            t_target = int(peaks[j, 1])
            delta_t = t_target - t_anchor

            if delta_t < config.delta_t_min:
                continue
            if delta_t > config.delta_t_max:
                break
            if abs(f_target_raw - f_anchor_raw) > config.delta_f_max:
                continue

            f_target_q = f_target_raw >> config.freq_quant_bits
            hash_val = (f_anchor_q << f_shift) | (f_target_q << t_shift) | delta_t
            out.append((hash_val, t_anchor))
            fan_count += 1

            if fan_count >= config.fan_out:
                break

    return out
