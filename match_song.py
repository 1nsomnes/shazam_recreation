"""
match_song.py

Fingerprint an MP3 sample with the utilities pipeline and match it
against the Postgres hash index (schema from step 1). Ranking is the
same time-offset histogram voting as first_iteration/match_song.py:

    offset = db_time - sample_time
    bucket = offset // OFFSET_TOLERANCE

Votes are counted per (song_id, offset_bucket) so near-miss offsets
(from resampling / encoding drift) still reinforce the same group.

Usage:
    ./venv/bin/python match_song.py <sample.mp3>
    ./venv/bin/python match_song.py <sample.mp3> --db-url postgresql:///shazam_recreation
    ./venv/bin/python match_song.py <sample.mp3> --debug              # → debug/<sample>_matches.json
    ./venv/bin/python match_song.py <sample.mp3> --debug path.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import psycopg

from utilities import (
    ConstellationConfig,
    HashConfig,
    SpectrogramConfig,
    generate_constellation,
    generate_hashes,
    mp3_to_spectrogram,
)


# Bucket width in frames (~0.6 s at hop=512 / sr=8000). Purely a
# matching-time knob; not part of the fingerprint, so it lives here
# rather than in utilities/config.py.
OFFSET_TOLERANCE = 9
DEFAULT_DEBUG_DIR = "debug"


def fingerprint_sample(
    mp3_path: str,
    spec_cfg: SpectrogramConfig,
    const_cfg: ConstellationConfig,
    hash_cfg: HashConfig,
):
    spec = mp3_to_spectrogram(mp3_path, spec_cfg)
    peaks = generate_constellation(spec, const_cfg)
    pairs = generate_hashes(peaks, hash_cfg)
    return peaks, pairs


def index_by_hash(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """hash → [sample_time, ...]. A single hash can appear at multiple anchors."""
    by_hash: dict[int, list[int]] = defaultdict(list)
    for h, t in pairs:
        by_hash[h].append(t)
    return by_hash


def fetch_db_hits(
    conn: psycopg.Connection,
    hashes: list[int],
) -> list[tuple[int, int, int]]:
    """Return (hash, song_id, t_anchor) rows from the DB for any matching hash."""
    if not hashes:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT hash, song_id, t_anchor FROM hashes WHERE hash = ANY(%s);",
            (hashes,),
        )
        return cur.fetchall()


def tally_votes(
    sample_by_hash: dict[int, list[int]],
    db_rows: list[tuple[int, int, int]],
    debug_matches: list[dict] | None = None,
) -> tuple[Counter, int]:
    """Vote (song_id, bucket) across every (sample_time, db_time) combination."""
    votes: Counter = Counter()
    hit_count = 0

    for db_hash, song_id, db_time in db_rows:
        for sample_time in sample_by_hash.get(db_hash, ()):
            offset = db_time - sample_time
            bucket = offset // OFFSET_TOLERANCE
            votes[(song_id, bucket)] += 1
            hit_count += 1
            if debug_matches is not None:
                debug_matches.append({
                    "hash": int(db_hash),
                    "song_id": int(song_id),
                    "sample_time": int(sample_time),
                    "db_time": int(db_time),
                    "offset": int(offset),
                    "bucket": int(bucket),
                })

    return votes, hit_count


def resolve_songs(
    conn: psycopg.Connection,
    song_ids: set[int],
) -> dict[int, tuple[str, str]]:
    """song_id → (artist, song_name) for the top ranked songs."""
    if not song_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT song_id, artist, song_name FROM songs WHERE song_id = ANY(%s);",
            (list(song_ids),),
        )
        return {sid: (artist, name) for sid, artist, name in cur.fetchall()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match an MP3 sample against the Postgres fingerprint DB."
    )
    parser.add_argument("sample", help="Path to the MP3 sample to identify.")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", "postgresql:///shazam_recreation"),
        help="Postgres connection URL. Falls back to $DATABASE_URL, then "
             "postgresql:///shazam_recreation.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--debug",
        nargs="?",
        const="",
        default=None,
        help="Write per-hit JSON dump. Bare --debug uses debug/<sample>_matches.json.",
    )
    return parser.parse_args()


def resolve_debug_path(sample_path: str, debug_arg: str | None) -> str | None:
    if debug_arg is None:
        return None
    if debug_arg == "":
        os.makedirs(DEFAULT_DEBUG_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(sample_path))[0]
        return os.path.join(DEFAULT_DEBUG_DIR, f"{base}_matches.json")
    return debug_arg


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.sample):
        sys.exit(f"Sample not found: {args.sample}")

    debug_path = resolve_debug_path(args.sample, args.debug)

    spec_cfg = SpectrogramConfig()
    const_cfg = ConstellationConfig()
    hash_cfg = HashConfig()

    print(f"Fingerprinting {args.sample}")
    peaks, pairs = fingerprint_sample(args.sample, spec_cfg, const_cfg, hash_cfg)
    print(f"  Peaks        : {len(peaks)}")
    print(f"  Fingerprints : {len(pairs)}")

    sample_by_hash = index_by_hash(pairs)

    print(f"Connecting to {args.db_url}")
    with psycopg.connect(args.db_url) as conn:
        db_rows = fetch_db_hits(conn, list(sample_by_hash.keys()))
        print(f"  DB rows       : {len(db_rows)}")

        debug_matches: list[dict] | None = [] if debug_path else None
        votes, hit_count = tally_votes(sample_by_hash, db_rows, debug_matches)
        print(f"  Weighted hits : {hit_count}")
        print(f"  Unique groups : {len(votes)}")

        top = votes.most_common(args.top_n)
        top_song_ids = {sid for (sid, _), _ in top}
        song_names = resolve_songs(conn, top_song_ids)

    results = [
        (sid, song_names.get(sid, ("?", "?")), bucket * OFFSET_TOLERANCE, count)
        for (sid, bucket), count in top
    ]

    print(f"\n{'Rank':<6}{'Song':<40}{'Offset':<10}{'Matches'}")
    print("-" * 70)
    for rank, (sid, (artist, name), offset, count) in enumerate(results, 1):
        label = f"{artist} — {name}"
        if len(label) > 38:
            label = label[:37] + "…"
        print(f"{rank:<6}{label:<40}{offset:<10}{count}")

    if debug_path is not None:
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        payload = {
            "sample": os.path.basename(args.sample),
            "db_url": args.db_url,
            "offset_tolerance": OFFSET_TOLERANCE,
            "sample_peaks": int(len(peaks)),
            "sample_fingerprints": len(pairs),
            "total_hits": len(debug_matches) if debug_matches else 0,
            "top_results": [
                {
                    "song_id": int(sid),
                    "artist": song_names.get(sid, ("?", "?"))[0],
                    "song_name": song_names.get(sid, ("?", "?"))[1],
                    "offset": int(offset),
                    "count": int(count),
                }
                for sid, _, offset, count in results
            ],
            "matches": debug_matches or [],
        }
        with open(debug_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nDebug matches written to: {debug_path}")


if __name__ == "__main__":
    main()
