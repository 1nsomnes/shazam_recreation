"""
index_songs.py

Walk songs/<Artist>/<Album>/<track>.mp3, run each track through the
fingerprint pipeline (spectrogram → constellation → hashes), and upsert
the results into the local Postgres schema from step 1.

Artist is taken from the top-level directory under --songs-dir. Album
directories are traversed but the album itself is not stored (the songs
table has no album column in the current schema).

Usage:
    ./venv/bin/python index_songs.py
    ./venv/bin/python index_songs.py --songs-dir songs --db-url postgresql:///shazam_recreation

Re-running is idempotent: each song's existing hashes are deleted before
the new set is inserted, so config changes get picked up cleanly. A
failure on any single song is logged and the run continues.
"""

import argparse
import os
import re
import sys
from typing import Iterator

import psycopg

from utilities import (
    ConstellationConfig,
    HashConfig,
    SpectrogramConfig,
    generate_constellation,
    generate_hashes,
    mp3_to_spectrogram,
)


# Patterns stripped from filename stems, applied in order. Kept narrow
# to avoid mangling titles that legitimately contain parentheticals
# (e.g. "(2006 Remaster)" is informative and stays).
_BITRATE_SUFFIX = re.compile(r"\s*\(\d{2,4}k\)\s*$", re.IGNORECASE)
_TRAILING_NUM = re.compile(r"\s+\d+\s*$")
_OFFICIAL_TAIL = re.compile(r"\s*-\s*.+?\s+Official\s*$", re.IGNORECASE)


def clean_song_name(stem: str, artist: str) -> str:
    """
    Strip common youtube-dl suffixes from a filename stem.

    Examples
    --------
    >>> clean_song_name("Altogether - Slowdive Official (128k)", "Slowdive")
    'Altogether'
    >>> clean_song_name("Play for Today (2006 Remaster) - The Cure (128k)", "The Cure")
    'Play for Today (2006 Remaster)'
    >>> clean_song_name("In Your House (Remastered Version) 0", "The Cure")
    'In Your House (Remastered Version)'
    """
    s = stem
    # Two passes because a bitrate tag can sit behind an artist trailer.
    for _ in range(2):
        s = _BITRATE_SUFFIX.sub("", s)
        s = _OFFICIAL_TAIL.sub("", s)
        artist_tail = re.compile(
            rf"\s*-\s*{re.escape(artist)}\s*$", re.IGNORECASE
        )
        s = artist_tail.sub("", s)
        s = _TRAILING_NUM.sub("", s)
    return s.strip() or stem


def iter_tracks(songs_dir: str) -> Iterator[tuple[str, str]]:
    """
    Yield (artist, mp3_path) for every .mp3 under songs_dir, treating
    the immediate child directory of songs_dir as the artist name.
    """
    for artist in sorted(os.listdir(songs_dir)):
        artist_dir = os.path.join(songs_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
        for root, _, files in os.walk(artist_dir):
            for name in sorted(files):
                if name.lower().endswith(".mp3"):
                    yield artist, os.path.join(root, name)


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
    stem = os.path.splitext(os.path.basename(mp3_path))[0]
    song_name = clean_song_name(stem, artist)
    label = f"{artist} — {song_name}"
    print(f"[{label}] fingerprinting {mp3_path}")

    spec = mp3_to_spectrogram(mp3_path, spec_cfg)
    peaks = generate_constellation(spec, const_cfg)
    fingerprints = generate_hashes(peaks, hash_cfg)
    print(
        f"[{label}]   spec={spec.shape} peaks={len(peaks)} "
        f"hashes={len(fingerprints)}"
    )

    song_id = upsert_song(conn, artist, song_name)
    replace_hashes(conn, song_id, fingerprints)
    conn.commit()
    print(f"[{label}]   inserted under song_id={song_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument(
        "--songs-dir",
        default="songs",
        help="Root of the songs/<artist>/<album>/*.mp3 tree (default: songs).",
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

    tracks = list(iter_tracks(args.songs_dir))
    if not tracks:
        sys.exit(f"No MP3s found under {args.songs_dir}")

    print(f"Found {len(tracks)} track(s) across "
          f"{len({a for a, _ in tracks})} artist(s)")

    spec_cfg = SpectrogramConfig()
    const_cfg = ConstellationConfig()
    hash_cfg = HashConfig()

    print(f"Connecting to {args.db_url}")
    indexed = 0
    failed: list[tuple[str, str]] = []
    with psycopg.connect(args.db_url) as conn:
        for artist, mp3 in tracks:
            try:
                process_song(conn, mp3, artist, spec_cfg, const_cfg, hash_cfg)
                indexed += 1
            except Exception as e:
                # Rollback the failed song's transaction so the next one
                # starts clean; keep going through the rest of the tree.
                conn.rollback()
                print(f"[ERROR] {mp3}: {e}", file=sys.stderr)
                failed.append((mp3, str(e)))

    print(f"\nDone. Indexed {indexed}/{len(tracks)} track(s).")
    if failed:
        print(f"{len(failed)} failure(s):", file=sys.stderr)
        for path, err in failed:
            print(f"  {path}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
