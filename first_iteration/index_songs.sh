#!/usr/bin/env bash
#
# index_songs.sh
#
# Iterates through all MP3 files in songs/, runs the full fingerprinting
# pipeline (spectrogram → constellation → hashes), and appends results
# to hashes.json. Intermediate .npy files are written to a temp directory
# and cleaned up automatically on exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"
SONGS_DIR="$SCRIPT_DIR/songs"
INDEX_PATH="$SCRIPT_DIR/hashes.json"

if [ ! -d "$SONGS_DIR" ]; then
    echo "Error: songs/ directory not found"
    exit 1
fi

# Create a temp directory for intermediates; clean it up on exit no matter what
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

count=0
for mp3 in "$SONGS_DIR"/*.mp3; do
    [ -f "$mp3" ] || continue

    filename="$(basename "$mp3" .mp3)"
    echo "=== [$filename] ==="

    spec_file="$TMPDIR/${filename}_spectrogram.npy"
    const_file="$TMPDIR/${filename}_constellation.npy"

    "$PYTHON" "$SCRIPT_DIR/compute_spectrogram.py" "$mp3" -o "$TMPDIR"
    "$PYTHON" "$SCRIPT_DIR/generate_constellation.py" "$spec_file" -o "$TMPDIR"
    "$PYTHON" "$SCRIPT_DIR/generate_hashes.py" "$const_file" "$filename" -i "$INDEX_PATH"

    count=$((count + 1))
    echo ""
done

echo "Done. Indexed $count songs into $INDEX_PATH"
