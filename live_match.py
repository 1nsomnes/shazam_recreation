"""
live_match.py

PySide6 GUI that captures microphone audio and matches it against the
Postgres hash index at 3s, 6s, 9s, 12s, 15s, 18s, and 21s — each window
is the full buffer from t=0, not just the most recent 3s. Reuses the
fingerprint pipeline (utilities/) and voting logic (match_song.py) so
there's exactly one implementation of each.

Note: tkinter would be the obvious choice, but the pyenv Python in this
project was built without _tkinter, so we use PySide6 (pip-installable
with self-contained wheels). Worker/recorder code is toolkit-agnostic.

Usage:
    ./venv/bin/python live_match.py
    DATABASE_URL=postgresql:///other ./venv/bin/python live_match.py
"""

import os
import sys
import threading
import time

import librosa
import numpy as np
import psycopg
import sounddevice as sd
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from match_song import (
    OFFSET_TOLERANCE,
    fetch_db_hits,
    index_by_hash,
    resolve_songs,
    tally_votes,
)
from utilities import (
    ConstellationConfig,
    HashConfig,
    SpectrogramConfig,
    compute_magnitude_spectrogram,
    generate_constellation,
    generate_hashes,
)


TARGET_SR = 8000
INTERVAL_S = 3
MAX_DURATION_S = 21
CHECKPOINTS = tuple(range(INTERVAL_S, MAX_DURATION_S + 1, INTERVAL_S))
TOP_N = 10


# ---------------------------------------------------------------------------
# Audio capture (sounddevice callback thread → shared buffer)
# ---------------------------------------------------------------------------

class AudioRecorder:
    """Append-only mono capture at the device's native sample rate."""

    def __init__(self) -> None:
        info = sd.query_devices(kind="input")
        self.device_sr = int(info["default_samplerate"])
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self.start_time: float | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Overflow / underflow. Not fatal — log and keep going.
            print(f"[audio] {status}", file=sys.stderr)
        flat = indata[:, 0] if indata.ndim > 1 else indata
        with self._lock:
            self._chunks.append(flat.copy())

    def start(self) -> None:
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.device_sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.start_time = time.monotonic()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def elapsed(self) -> float:
        return 0.0 if self.start_time is None else time.monotonic() - self.start_time

    def snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# Fingerprint + DB match (reuses match_song.py helpers)
# ---------------------------------------------------------------------------

def match_buffer(
    audio: np.ndarray,
    orig_sr: int,
    conn: psycopg.Connection,
    spec_cfg: SpectrogramConfig,
    const_cfg: ConstellationConfig,
    hash_cfg: HashConfig,
) -> list[tuple[str, str, int, int]]:
    """audio@orig_sr → list[(artist, song_name, offset, count)] top-N."""
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    if len(audio) < spec_cfg.n_fft:
        return []

    spec = compute_magnitude_spectrogram(audio, spec_cfg)
    peaks = generate_constellation(spec, const_cfg)
    pairs = generate_hashes(peaks, hash_cfg)
    if not pairs:
        return []

    sample_by_hash = index_by_hash(pairs)
    db_rows = fetch_db_hits(conn, list(sample_by_hash.keys()))
    votes, _ = tally_votes(sample_by_hash, db_rows)
    top = votes.most_common(TOP_N)
    if not top:
        return []

    song_ids = {sid for (sid, _), _ in top}
    names = resolve_songs(conn, song_ids)
    return [
        (
            names.get(sid, ("?", "?"))[0],
            names.get(sid, ("?", "?"))[1],
            bucket * OFFSET_TOLERANCE,
            count,
        )
        for (sid, bucket), count in top
    ]


# ---------------------------------------------------------------------------
# Session worker — Qt signals to marshal results back to the GUI thread
# ---------------------------------------------------------------------------

class SessionWorker(QObject):
    """Runs on a plain Python thread. Emits Qt signals; Qt routes to GUI."""

    result_ready = Signal(int, list, float, bool)  # checkpoint, results, ms, done
    error = Signal(str)

    def __init__(self, recorder: AudioRecorder, db_url: str) -> None:
        super().__init__()
        self.recorder = recorder
        self.db_url = db_url

    def run(self) -> None:
        spec_cfg = SpectrogramConfig()
        const_cfg = ConstellationConfig()
        hash_cfg = HashConfig()

        try:
            with psycopg.connect(self.db_url) as conn:
                self.recorder.start()
                try:
                    for checkpoint in CHECKPOINTS:
                        while self.recorder.elapsed() < checkpoint:
                            time.sleep(0.05)

                        audio = self.recorder.snapshot()
                        target_len = checkpoint * self.recorder.device_sr
                        audio = audio[:target_len]

                        t0 = time.monotonic()
                        results = match_buffer(
                            audio, self.recorder.device_sr, conn,
                            spec_cfg, const_cfg, hash_cfg,
                        )
                        match_ms = (time.monotonic() - t0) * 1000

                        self.result_ready.emit(
                            checkpoint, results, match_ms,
                            checkpoint == CHECKPOINTS[-1],
                        )
                finally:
                    self.recorder.stop()
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, db_url: str) -> None:
        super().__init__()
        self.db_url = db_url
        self.recorder: AudioRecorder | None = None
        self.worker: SessionWorker | None = None
        self.worker_thread: threading.Thread | None = None

        self.setWindowTitle("Shazam Prototype — Live Match")
        self.resize(820, 480)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.status_label = QLabel("Press Start to begin listening.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(self.status_label, stretch=1)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet("font-size: 13px;")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.elapsed_label)
        layout.addLayout(header)

        checkpoints_label = QLabel(
            "Checkpoints: " + "  ".join(f"{c}s" for c in CHECKPOINTS)
        )
        checkpoints_label.setStyleSheet("color: #666;")
        layout.addWidget(checkpoints_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Artist", "Song", "Offset", "Matches"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Listening")
        self.start_btn.clicked.connect(self._start)
        button_row.addWidget(self.start_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        # Elapsed-time ticker — only active during a session.
        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(100)
        self.tick_timer.timeout.connect(self._tick_elapsed)

    def _start(self) -> None:
        self.start_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.status_label.setText("Listening…")
        self.elapsed_label.setText(f"0.0s / {MAX_DURATION_S}s")

        try:
            self.recorder = AudioRecorder()
        except Exception as e:
            QMessageBox.critical(
                self, "Audio device error", f"{type(e).__name__}: {e}"
            )
            self.start_btn.setEnabled(True)
            return

        self.worker = SessionWorker(self.recorder, self.db_url)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)

        self.worker_thread = threading.Thread(
            target=self.worker.run, daemon=True
        )
        self.worker_thread.start()
        self.tick_timer.start()

    def _tick_elapsed(self) -> None:
        if self.recorder and self.recorder.start_time is not None:
            el = min(self.recorder.elapsed(), MAX_DURATION_S)
            self.elapsed_label.setText(f"{el:.1f}s / {MAX_DURATION_S}s")

    def _on_result(
        self,
        checkpoint: int,
        results: list,
        match_ms: float,
        done: bool,
    ) -> None:
        self.table.setRowCount(len(results))
        for row, (artist, song, offset, count) in enumerate(results):
            values = [str(row + 1), artist, song, str(offset), str(count)]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col in (0, 3, 4):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, col, item)

        if done:
            self.tick_timer.stop()
            self.elapsed_label.setText(f"{MAX_DURATION_S}.0s / {MAX_DURATION_S}s")
            top = self._top_label(results)
            self.status_label.setText(
                f"Done at {checkpoint}s — top match: {top}  ({match_ms:.0f} ms)"
            )
            self.start_btn.setEnabled(True)
        else:
            n = len(results)
            plural = "" if n == 1 else "s"
            self.status_label.setText(
                f"{checkpoint}s window → {n} candidate{plural} "
                f"({match_ms:.0f} ms) — listening…"
            )

    def _on_error(self, message: str) -> None:
        self.tick_timer.stop()
        self.status_label.setText(f"Error: {message}")
        self.start_btn.setEnabled(True)

    @staticmethod
    def _top_label(results: list[tuple[str, str, int, int]]) -> str:
        if not results:
            return "— no match"
        artist, song, _, _ = results[0]
        return f"{artist} — {song}"


def main() -> None:
    db_url = os.environ.get("DATABASE_URL", "postgresql:///shazam_recreation")
    app = QApplication(sys.argv)
    window = MainWindow(db_url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
