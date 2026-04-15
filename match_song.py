"""
match_song.py

Reads a binary constellation map (.npy), generates fingerprint hashes,
and looks them up against the persistent inverted index (hashes.json)
to identify which song best matches the input sample.

Matching is based on time-offset histogram voting:
  offset = db_time - sample_time
Matches are grouped by (song_id, offset) and ranked by group size.
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
    FAN_OUT,
    DEFAULT_INDEX_PATH,
)


def generate_hash_pairs(peaks: np.ndarray) -> list[tuple[str, int]]:
    """
    Same anchor/target pairing as generate_fingerprints, but returns
    (hash_key, t_anchor) pairs instead of building an inverted index.
    """
    n_peaks = len(peaks)
    pairs: list[tuple[str, int]] = []

    for i in range(n_peaks):
        f_anchor = int(peaks[i, 0])
        t_anchor = int(peaks[i, 1])
        fan_count = 0

        for j in range(i + 1, n_peaks):
            f_target = int(peaks[j, 0])
            t_target = int(peaks[j, 1])
            delta_t = t_target - t_anchor

            if delta_t < DELTA_T_MIN:
                continue
            if delta_t > DELTA_T_MAX:
                break

            hash_val = (f_anchor << 17) | (f_target << 7) | delta_t
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
) -> list[tuple[str, int, int]]:
    """
    Look up each sample hash in the index and vote on (song_id, offset).

    offset = db_time - sample_time

    Returns the top_n groups as (song_id, offset, count) sorted by
    count descending.
    """
    votes: Counter[tuple[str, int]] = Counter()
    hit_count = 0

    for hash_key, sample_time in pairs:
        if hash_key not in index:
            continue
        for song_id, db_time in index[hash_key]:
            offset = db_time - sample_time
            votes[(song_id, offset)] += 1
            hit_count += 1

    print(f"  Hash hits      : {hit_count}")
    print(f"  Unique groups  : {len(votes)}")

    top = votes.most_common(top_n)
    return [(song_id, offset, count) for (song_id, offset), count in top]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <constellation.npy> [-i index.json]")
        sys.exit(1)

    npy_path = sys.argv[1]

    index_path = DEFAULT_INDEX_PATH
    if "-i" in sys.argv:
        index_path = sys.argv[sys.argv.index("-i") + 1]

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

    results = match_against_index(pairs, index)

    print(f"\n{'Rank':<6}{'Song':<25}{'Offset':<10}{'Matches'}")
    print("-" * 50)
    for rank, (song_id, offset, count) in enumerate(results, 1):
        print(f"{rank:<6}{song_id:<25}{offset:<10}{count}")


if __name__ == "__main__":
    main()
