from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AudioFetchError(RuntimeError):
    """Raised when audio preview downloads fail."""


def _download_once(url: str, destination: Path, timeout_s: float) -> None:
    req = Request(url, method="GET", headers={"User-Agent": "treehacks26/0.1"})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    destination.write_bytes(data)


def fetch_preview_audio(
    track_id: str,
    preview_url: str,
    *,
    cache_dir: str = ".cache/audio",
    retries: int = 2,
    timeout_s: float = 30.0,
) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{track_id}.mp3"
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    last_err: Optional[Exception] = None
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            _download_once(preview_url, destination, timeout_s=timeout_s)
            if destination.stat().st_size <= 0:
                raise AudioFetchError(f"Empty download for track {track_id}")
            return destination
        except (HTTPError, URLError, OSError, AudioFetchError) as err:
            last_err = err
            if attempt + 1 < attempts:
                time.sleep(0.75 * (attempt + 1))

    raise AudioFetchError(
        f"Failed to download preview for {track_id} after {attempts} attempts: {last_err}"
    )
