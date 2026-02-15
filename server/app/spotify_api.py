from __future__ import annotations

import json
import logging
import time
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings


class SpotifyApiError(RuntimeError):
    """Raised when Spotify resource API calls fail."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

logger = logging.getLogger(__name__)


def _http_json(
    url: str,
    access_token: str,
    *,
    timeout_s: float = 30.0,
    debug: bool = False,
    retries: int = 2,
) -> Dict | List[Dict]:
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        if debug:
            logger.info("Spotify API request: GET %s (attempt %s/%s)", url, attempt + 1, attempts)
        req = Request(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if debug:
                    logger.info("Spotify API success: %s", url)
                return payload
        except HTTPError as err:
            details = err.read().decode("utf-8", errors="replace")
            retryable = err.code in {429, 500, 502, 503, 504}
            if debug:
                logger.warning("Spotify API HTTP %s for %s: %s", err.code, url, details)
            if retryable and attempt + 1 < attempts:
                retry_after_header = err.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        delay_s = max(0.5, float(retry_after_header))
                    except ValueError:
                        delay_s = 1.0 + attempt
                else:
                    delay_s = 1.0 + attempt
                time.sleep(delay_s)
                continue
            last_exc = SpotifyApiError(
                f"Spotify HTTP {err.code}: {details}", status_code=err.code
            )
            break
        except URLError as err:
            if debug:
                logger.warning("Spotify API connection error for %s: %s", url, err)
            if attempt + 1 < attempts:
                time.sleep(0.5 + attempt)
                continue
            last_exc = SpotifyApiError(f"Spotify connection error: {err}")
            break
    if isinstance(last_exc, SpotifyApiError):
        raise last_exc
    raise SpotifyApiError("Spotify request failed for unknown reason.")


def get_current_user_profile(settings: Settings, access_token: str) -> Dict:
    return _http_json(
        f"{settings.spotify_api_base}/me", access_token, debug=settings.spotify_debug
    )


def get_user_playlists(
    settings: Settings, access_token: str, *, limit: int = 50, offset: int = 0
) -> Dict:
    params = urlencode({"limit": limit, "offset": offset})
    payload = _http_json(
        f"{settings.spotify_api_base}/me/playlists?{params}",
        access_token,
        debug=settings.spotify_debug,
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    normalized = []
    for p in items:
        items_obj = p.get("items") or {}
        tracks_obj = p.get("tracks") or {}
        normalized.append(
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "owner": (p.get("owner") or {}).get("display_name"),
                "owner_id": (p.get("owner") or {}).get("id"),
                "images": p.get("images") or [],
                "tracks_total": (
                    items_obj.get("total")
                    if items_obj.get("total") is not None
                    else (tracks_obj.get("total") or 0)
                ),
                "tracks_href": items_obj.get("href") or tracks_obj.get("href"),
                "is_public": p.get("public"),
            }
        )
    return {
        "items": normalized,
        "limit": payload.get("limit", limit) if isinstance(payload, dict) else limit,
        "offset": payload.get("offset", offset) if isinstance(payload, dict) else offset,
        "total": payload.get("total", len(normalized)) if isinstance(payload, dict) else len(normalized),
    }


def get_playlist_tracks(
    settings: Settings,
    access_token: str,
    playlist_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> Dict:
    params = urlencode({"limit": limit, "offset": offset})
    payload = _http_json(
        f"{settings.spotify_api_base}/playlists/{playlist_id}/items?{params}",
        access_token,
        debug=settings.spotify_debug,
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    normalized = []
    for item in items:
        # Spotify migrated playlist payloads from track-centric fields to item-centric fields.
        track = item.get("item") or item.get("track") or {}
        artists = [a.get("name") for a in (track.get("artists") or []) if a.get("name")]
        normalized.append(
            {
                "track_id": track.get("id"),
                "name": track.get("name"),
                "artists": artists,
                "album": (track.get("album") or {}).get("name"),
                "duration_ms": track.get("duration_ms"),
                "preview_url": track.get("preview_url"),
                "spotify_url": (track.get("external_urls") or {}).get("spotify"),
            }
        )
    return {
        "items": normalized,
        "limit": payload.get("limit", limit) if isinstance(payload, dict) else limit,
        "offset": payload.get("offset", offset) if isinstance(payload, dict) else offset,
        "total": payload.get("total", len(normalized)) if isinstance(payload, dict) else len(normalized),
    }
