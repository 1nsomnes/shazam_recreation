"""
index_songs.py

Walk a directory of MP3s, run each through the fingerprint pipeline
(spectrogram → constellation → hashes), and upsert the results into the
local Postgres schema from step 1 (songs + hashes tables).

Usage:
    ./venv/bin/python index_songs.py --artist "Smashing Pumpkins"
    ./venv/bin/python index_songs.py --artist X --songs-dir songs --db-url postgresql:///shazam

Re-running is idempotent: each song's existing hashes are deleted before
the new set is inserted, so config changes get picked up cleanly.
"""

import argparse
import os
import re
import sys

import psycopg

from utilities import (
    ConstellationConfig,
    HashConfig,
    SpectrogramConfig,
    generate_constellation,
    generate_hashes,
    mp3_to_spectrogram,
)


def title_from_filename(path: str) -> str:
    """cherub_rock.mp3 → 'Cherub Rock'."""
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_+", " ", base).strip().title()


def upsert_song(conn: psycopg.Connection, artist: str, song_name: str) -> int:
    """Insert-or-fetch: returns song_id for (artist, song_name)."""
    with conn.cursor() as cur:
        # DO UPDATE (no-op) instead of DO NOTHING so RETURNING always fires.
        cur.execute(
            """
            INSERT INTO songs (artist, song_name)
            VALUES (%s, %s)
            ON CONFLICT (artist, song_name) DO UPDATE
                SET song_name = EXCLUDED.song_name
            RETURNING song_id;
            """,
            (artist, song_name),
        )
        row = cur.fetchone()
        assert row is not None
        return row[0]


def replace_hashes(
    conn: psycopg.Connection,
    song_id: int,
    rows: list[tuple[int, int]],
) -> None:
    """Delete existing hashes for this song and bulk-insert the new ones."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM hashes WHERE song_id = %s;", (song_id,))
        if not rows:
            return
        with cur.copy("COPY hashes (hash, song_id, t_anchor) FROM STDIN") as cp:
            for hash_val, t_anchor in rows:
                cp.write_row((hash_val, song_id, t_anchor))


def process_song(
    conn: psycopg.Connection,
    mp3_path: str,
    artist: str,
    spec_cfg: SpectrogramConfig,
    const_cfg: ConstellationConfig,
    hash_cfg: HashConfig,
) -> None:
    song_name = title_from_filename(mp3_path)
    print(f"[{song_name}] fingerprinting {mp3_path}")

    spec = mp3_to_spectrogram(mp3_path, spec_cfg)
    peaks = generate_constellation(spec, const_cfg)
    fingerprints = generate_hashes(peaks, hash_cfg)
    print(
        f"[{song_name}]   spec={spec.shape} peaks={len(peaks)} "
        f"hashes={len(fingerprints)}"
    )

    song_id = upsert_song(conn, artist, song_name)
    replace_hashes(conn, song_id, fingerprints)
    conn.commit()
    print(f"[{song_name}]   inserted under song_id={song_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument(
        "--artist",
        required=True,
        help="Artist name applied to every MP3 in --songs-dir.",
    )
    parser.add_argument(
        "--songs-dir",
        default="songs",
        help="Directory of .mp3 files to index (default: songs).",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", "postgresql:///shazam_recreation"),
        help="Postgres connection URL. Falls back to $DATABASE_URL, then "
             "postgresql:///shazam_recreation.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.songs_dir):
        sys.exit(f"Not a directory: {args.songs_dir}")

    mp3s = sorted(
        os.path.join(args.songs_dir, f)
        for f in os.listdir(args.songs_dir)
        if f.lower().endswith(".mp3")
    )
    if not mp3s:
        sys.exit(f"No MP3s found in {args.songs_dir}")

    spec_cfg = SpectrogramConfig()
    const_cfg = ConstellationConfig()
    hash_cfg = HashConfig()

    print(f"Connecting to {args.db_url}")
    with psycopg.connect(args.db_url) as conn:
        for mp3 in mp3s:
            process_song(conn, mp3, args.artist, spec_cfg, const_cfg, hash_cfg)

    print(f"Done. Indexed {len(mp3s)} song(s) for artist '{args.artist}'.")


if __name__ == "__main__":
    main()
