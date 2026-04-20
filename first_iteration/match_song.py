"""
match_song.py

Reads a binary constellation map (.npy), generates fingerprint hashes,
and looks them up against the persistent inverted index (hashes.json)
to identify which song best matches the input sample.

Matching is based on time-offset histogram voting:
  offset = db_time - sample_time
Matches are grouped by (song_id, offset_bucket) and ranked by group size.
Offsets are quantized into buckets of OFFSET_TOLERANCE frames so that
near-misses (due to resampling or encoding differences) still count
toward the same group.
"""

import sys
import os
import json
from collections import Counter

import numpy as np

# Reuse core functions from generate_hashes
from generate_hashes import (
    load_constellation,
    extract_peaks,
    DELTA_T_MIN,
    DELTA_T_MAX,
    DELTA_F_MAX,
    FAN_OUT,
    FREQ_QUANT_BITS,
    F_FIELD_BITS,
    DT_FIELD_BITS,
    DEFAULT_INDEX_PATH,
)

# --- Configurable parameters ---
OFFSET_TOLERANCE = 9  # Bucket width in frames (~0.6 s at hop=512/sr=8000)
DEFAULT_DEBUG_DIR = "debug"


def generate_hash_pairs(peaks: np.ndarray) -> list[tuple[str, int]]:
    """
    Same anchor/target pairing as generate_fingerprints, but returns
    (hash_key, t_anchor) pairs instead of building an inverted index.
    """
    n_peaks = len(peaks)
    pairs: list[tuple[str, int]] = []
    f_shift = F_FIELD_BITS + DT_FIELD_BITS
    t_shift = DT_FIELD_BITS

    for i in range(n_peaks):
        f_anchor_raw = int(peaks[i, 0])
        f_anchor_q = f_anchor_raw >> FREQ_QUANT_BITS
        t_anchor = int(peaks[i, 1])
        fan_count = 0

        for j in range(i + 1, n_peaks):
            f_target_raw = int(peaks[j, 0])
            t_target = int(peaks[j, 1])
            delta_t = t_target - t_anchor

            if delta_t < DELTA_T_MIN:
                continue
            if delta_t > DELTA_T_MAX:
                break
            if abs(f_target_raw - f_anchor_raw) > DELTA_F_MAX:
                continue

            f_target_q = f_target_raw >> FREQ_QUANT_BITS
            hash_val = (f_anchor_q << f_shift) | (f_target_q << t_shift) | delta_t
            pairs.append((str(hash_val), t_anchor))
            fan_count += 1

            if fan_count >= FAN_OUT:
                break

    print(f"  Fingerprints   : {len(pairs)}")
    return pairs


def match_against_index(
    pairs: list[tuple[str, int]],
    index: dict[str, list[list]],
    top_n: int = 10,
    debug_matches: list[dict] | None = None,
) -> list[tuple[str, int, int]]:
    """
    Look up each sample hash in the index and vote on (song_id, offset_bucket).

    offset = db_time - sample_time
    offset_bucket = offset // OFFSET_TOLERANCE

    Quantizing into buckets lets near-miss offsets reinforce each other
    instead of fragmenting into separate single-vote groups.

    Returns the top_n groups as (song_id, offset_bucket, count) sorted by
    count descending. If `debug_matches` is provided, every individual
    (hash, song, sample_time, db_time, offset, bucket) hit is appended.
    """
    votes: Counter[tuple[str, int]] = Counter()
    hit_count = 0

    for hash_key, sample_time in pairs:
        if hash_key not in index:
            continue
        for song_id, db_time in index[hash_key]:
            offset = db_time - sample_time
            bucket = offset // OFFSET_TOLERANCE
            votes[(song_id, bucket)] += 1
            hit_count += 1
            if debug_matches is not None:
                debug_matches.append({
                    "hash": hash_key,
                    "song_id": song_id,
                    "sample_time": sample_time,
                    "db_time": db_time,
                    "offset": offset,
                    "bucket": int(bucket),
                })

    print(f"  Hash hits      : {hit_count}")
    print(f"  Unique groups  : {len(votes)}")

    top = votes.most_common(top_n)
    return [(song_id, bucket * OFFSET_TOLERANCE, count)
            for (song_id, bucket), count in top]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <constellation.npy> [-i index.json] [--debug [path]]")
        sys.exit(1)

    npy_path = sys.argv[1]

    index_path = DEFAULT_INDEX_PATH
    if "-i" in sys.argv:
        index_path = sys.argv[sys.argv.index("-i") + 1]

    debug_path: str | None = None
    if "--debug" in sys.argv:
        i = sys.argv.index("--debug")
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
            debug_path = sys.argv[i + 1]
        else:
            os.makedirs(DEFAULT_DEBUG_DIR, exist_ok=True)
            basename = os.path.basename(npy_path).replace("_constellation.npy", "")
            debug_path = os.path.join(DEFAULT_DEBUG_DIR, f"{basename}_matches.json")

    if not os.path.isfile(index_path):
        print(f"Error: index file not found: {index_path}")
        sys.exit(1)

    print(f"Matching {npy_path} against {index_path}")

    constellation = load_constellation(npy_path)
    peaks = extract_peaks(constellation)
    pairs = generate_hash_pairs(peaks)

    print(f"  Loading index...")
    with open(index_path, "r") as f:
        index = json.load(f)
    print(f"  Index keys     : {len(index)}")

    debug_matches: list[dict] | None = [] if debug_path else None
    results = match_against_index(pairs, index, debug_matches=debug_matches)

    print(f"\n{'Rank':<6}{'Song':<25}{'Offset':<10}{'Matches'}")
    print("-" * 50)
    for rank, (song_id, offset, count) in enumerate(results, 1):
        print(f"{rank:<6}{song_id:<25}{offset:<10}{count}")

    if debug_path is not None:
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        payload = {
            "sample": os.path.basename(npy_path),
            "index": os.path.basename(index_path),
            "offset_tolerance": OFFSET_TOLERANCE,
            "sample_peaks": int(peaks.shape[0]),
            "sample_fingerprints": len(pairs),
            "total_hits": len(debug_matches),
            "top_results": [
                {"song_id": s, "offset": o, "count": c} for s, o, c in results
            ],
            "matches": debug_matches,
        }
        with open(debug_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nDebug matches written to: {debug_path}")


if __name__ == "__main__":
    main()
