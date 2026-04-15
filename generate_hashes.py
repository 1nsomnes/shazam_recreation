"""
generate_hashes.py

Reads a binary constellation map (.npy), generates bit-packed fingerprint
hashes via anchor/target pairing with a forward-looking target zone, and
appends them to a persistent inverted index (hashes.json).
"""

import sys
import os
import json
from collections import defaultdict

import numpy as np


# --- Configurable parameters ---
DELTA_T_MIN = 1       # Minimum time-frame offset for target zone
DELTA_T_MAX = 127     # Maximum time-frame offset (7 bits)
FAN_OUT = 5           # Max targets per anchor

DEFAULT_INDEX_PATH = "hashes.json"


def load_constellation(npy_path: str) -> np.ndarray:
    """Load a 2D binary constellation map from a .npy file."""
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Constellation file not found: {npy_path}")

    constellation = np.load(npy_path)
    if constellation.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {constellation.shape}")

    return constellation


def extract_peaks(constellation: np.ndarray) -> np.ndarray:
    """
    Extract (freq_idx, time_idx) coordinates of all 1s in the map.

    Returns an array of shape (N, 2) sorted by time_idx then freq_idx,
    which is the natural order for the forward-looking target zone scan.
    """
    coords = np.argwhere(constellation == 1)  # shape (N, 2): [freq, time]

    # Sort by time (column 1), then frequency (column 0) for tie-breaking
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    coords = coords[order]

    print(f"  Peaks extracted : {len(coords)}")
    return coords


def generate_fingerprints(
    peaks: np.ndarray,
    song_id: str,
) -> dict[str, list[list]]:
    """
    Pair each anchor with up to FAN_OUT targets in its forward-looking
    target zone, bit-pack each pair into a 32-bit hash, and collect
    (song_id, t_anchor) payloads keyed by hash.

    Hash structure (27 bits of data in a uint32):
        bits [26:17]  f_anchor   (10 bits, max 1023)
        bits [16:7]   f_target   (10 bits, max 1023)
        bits [6:0]    delta_t    (7 bits,  max 127)

    hash_val = (f_anchor << 17) | (f_target << 7) | delta_t
    """
    n_peaks = len(peaks)
    hashes: dict[str, list[list]] = defaultdict(list)
    total_pairs = 0

    for i in range(n_peaks):
        f_anchor = int(peaks[i, 0])
        t_anchor = int(peaks[i, 1])
        fan_count = 0

        # Scan forward for targets — peaks are sorted by time so we walk
        # sequentially until we leave the target zone or exhaust fan-out.
        for j in range(i + 1, n_peaks):
            f_target = int(peaks[j, 0])
            t_target = int(peaks[j, 1])
            delta_t = t_target - t_anchor

            if delta_t < DELTA_T_MIN:
                continue
            if delta_t > DELTA_T_MAX:
                break

            hash_val = (f_anchor << 17) | (f_target << 7) | delta_t
            hashes[str(hash_val)].append([song_id, t_anchor])
            fan_count += 1
            total_pairs += 1

            if fan_count >= FAN_OUT:
                break

    print(f"  Fingerprints   : {total_pairs}")
    print(f"  Unique hashes  : {len(hashes)}")
    return hashes


def load_index(index_path: str) -> dict[str, list[list]]:
    """Load the existing inverted index from disk, or start fresh."""
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            data = json.load(f)
        print(f"  Loaded index   : {index_path} ({len(data)} keys)")
        return defaultdict(list, data)

    print(f"  New index      : {index_path}")
    return defaultdict(list)


def merge_and_save(
    index: dict[str, list[list]],
    new_hashes: dict[str, list[list]],
    index_path: str,
) -> None:
    """Merge new fingerprints into the index and write to disk."""
    for key, payloads in new_hashes.items():
        index[key].extend(payloads)

    with open(index_path, "w") as f:
        json.dump(index, f, separators=(",", ":"))

    size_kb = os.path.getsize(index_path) / 1024
    print(f"  Saved index    : {index_path} ({len(index)} keys, {size_kb:.0f} KB)")


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <constellation.npy> <song_id> [-i index.json]")
        sys.exit(1)

    npy_path = sys.argv[1]
    song_id = sys.argv[2]

    index_path = DEFAULT_INDEX_PATH
    if "-i" in sys.argv:
        index_path = sys.argv[sys.argv.index("-i") + 1]

    print(f"Indexing song {song_id} from {npy_path}")
    constellation = load_constellation(npy_path)
    peaks = extract_peaks(constellation)
    new_hashes = generate_fingerprints(peaks, song_id)

    index = load_index(index_path)
    merge_and_save(index, new_hashes, index_path)


if __name__ == "__main__":
    main()
