"""
Simple local audio playback for generated track URLs.

Uses ffplay (FFmpeg) to stream and play audio URLs.
Also provides a reference audio buffer for acoustic echo cancellation (AEC).
"""

from __future__ import annotations

import collections
import shutil
import subprocess
import threading
from typing import Optional

import numpy as np


AEC_SAMPLE_RATE = 16000
AEC_CHANNELS = 1
AEC_READ_CHUNK = 160  # 10ms worth of samples at 16kHz
AEC_BUFFER_SECONDS = 30  # keep last 30s of reference audio


class MusicPlaybackError(RuntimeError):
    """Raised when audio download or playback fails."""


class MusicPlayer:
    def __init__(
        self,
        ffplay_path: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.ffplay_path = ffplay_path or shutil.which("ffplay")
        if not self.ffplay_path:
            raise MusicPlaybackError(
                "ffplay was not found on PATH. Install FFmpeg and ensure `ffplay` is available."
            )
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._playback_stop = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_generation = 0

        # Reference audio buffer for AEC
        max_chunks = (AEC_SAMPLE_RATE * AEC_BUFFER_SECONDS) // AEC_READ_CHUNK
        self._reference_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=max_chunks
        )
        self._reference_lock = threading.Lock()
        self._decoder_proc: Optional[subprocess.Popen] = None
        self._decoder_thread: Optional[threading.Thread] = None
        self._decoder_stop = threading.Event()

    def stop(self) -> None:
        self._playback_stop.set()
        thread = self._playback_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._playback_thread = None

        self._stop_reference_decoder()
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _spawn_ffplay(self, url: str) -> subprocess.Popen:
        return subprocess.Popen(
            [
                self.ffplay_path,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _playback_loop(self, url: str, generation: int) -> None:
        while (
            not self._playback_stop.is_set()
            and generation == self._playback_generation
        ):
            with self._proc_lock:
                proc = self._proc
            if proc is None:
                break

            # Wait for current playback to end (or stop request).
            while proc.poll() is None:
                if self._playback_stop.wait(timeout=0.2):
                    break

            if self._playback_stop.is_set() or generation != self._playback_generation:
                break

            # Track ended naturally. Restart from the beginning.
            self._stop_reference_decoder()
            if self.ffmpeg_path:
                self._start_reference_decoder(url)
            try:
                next_proc = self._spawn_ffplay(url)
            except Exception:
                with self._proc_lock:
                    self._proc = None
                break
            with self._proc_lock:
                self._proc = next_proc

    def play_url(self, url: str) -> str:
        self.stop()
        self._playback_stop.clear()
        self._playback_generation += 1
        generation = self._playback_generation
        try:
            proc = self._spawn_ffplay(url)
            with self._proc_lock:
                self._proc = proc
        except Exception as err:
            raise MusicPlaybackError(f"Failed to launch ffplay for url {url}: {err}") from err

        # Start reference decoder for AEC if ffmpeg is available
        if self.ffmpeg_path:
            self._start_reference_decoder(url)
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(url, generation),
            daemon=True,
        )
        self._playback_thread.start()

        return url

    def _start_reference_decoder(self, url: str) -> None:
        """Spawn ffmpeg to decode audio URL to raw 16kHz mono S16LE PCM for AEC."""
        self._stop_reference_decoder()
        with self._reference_lock:
            self._reference_buffer.clear()
        self._decoder_stop.clear()

        try:
            self._decoder_proc = subprocess.Popen(
                [
                    self.ffmpeg_path,
                    "-i", url,
                    "-f", "s16le",
                    "-acodec", "pcm_s16le",
                    "-ar", str(AEC_SAMPLE_RATE),
                    "-ac", str(AEC_CHANNELS),
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._decoder_proc = None
            return

        self._decoder_thread = threading.Thread(
            target=self._decoder_reader_loop, daemon=True
        )
        self._decoder_thread.start()

    def _decoder_reader_loop(self) -> None:
        """Read raw PCM from ffmpeg stdout into the reference ring buffer."""
        proc = self._decoder_proc
        if proc is None or proc.stdout is None:
            return

        bytes_per_sample = 2  # S16LE
        chunk_bytes = AEC_READ_CHUNK * bytes_per_sample

        while not self._decoder_stop.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            # Convert S16LE bytes to float32 numpy array normalized to [-1, 1]
            n_samples = len(data) // bytes_per_sample
            samples = np.frombuffer(data[:n_samples * bytes_per_sample], dtype=np.int16)
            samples_float = samples.astype(np.float32) / 32768.0
            
            # Split into 160-sample frames and append each frame
            frame_size = 160
            for i in range(0, len(samples_float), frame_size):
                frame = samples_float[i : i + frame_size]
                if len(frame) == frame_size:  # Only append full frames
                    with self._reference_lock:
                        self._reference_buffer.append(frame)

    def _stop_reference_decoder(self) -> None:
        """Stop the ffmpeg reference decoder subprocess and reader thread."""
        self._decoder_stop.set()
        if self._decoder_proc is not None:
            if self._decoder_proc.poll() is None:
                self._decoder_proc.terminate()
                try:
                    self._decoder_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._decoder_proc.kill()
            self._decoder_proc = None
        if self._decoder_thread is not None:
            self._decoder_thread.join(timeout=2.0)
            self._decoder_thread = None

    def get_reference_chunk(self, num_samples: int) -> Optional[np.ndarray]:
        """Return the oldest `num_samples` of reference audio for AEC.

        Returns None if no reference audio is available (no music playing).
        Returns a float32 numpy array normalized to [-1, 1].
        """
        if num_samples != 160:
            # We expect num_samples to be 160 for webrtcaec
            return None

        with self._reference_lock:
            if not self._reference_buffer:
                return None
            return self._reference_buffer.popleft()

    def close(self) -> None:
        self.stop()
