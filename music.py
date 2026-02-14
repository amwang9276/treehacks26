"""
Simple local audio playback for generated track URLs.

Uses ffplay (FFmpeg) to stream and play audio URLs.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class MusicPlaybackError(RuntimeError):
    """Raised when audio download or playback fails."""


class MusicPlayer:
    def __init__(self, ffplay_path: Optional[str] = None) -> None:
        self.ffplay_path = ffplay_path or shutil.which("ffplay")
        if not self.ffplay_path:
            raise MusicPlaybackError(
                "ffplay was not found on PATH. Install FFmpeg and ensure `ffplay` is available."
            )
        self._proc: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def play_url(self, url: str) -> str:
        self.stop()
        try:
            self._proc = subprocess.Popen(
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
        except Exception as err:
            raise MusicPlaybackError(f"Failed to launch ffplay for url {url}: {err}") from err
        return url

    def close(self) -> None:
        self.stop()
