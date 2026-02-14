from __future__ import annotations

import json
import logging
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
) -> Dict | List[Dict]:
    if debug:
        logger.info("Spotify API request: GET %s", url)
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
        if debug:
            logger.warning("Spotify API HTTP %s for %s: %s", err.code, url, details)
        raise SpotifyApiError(
            f"Spotify HTTP {err.code}: {details}", status_code=err.code
        ) from err
    except URLError as err:
        if debug:
            logger.warning("Spotify API connection error for %s: %s", url, err)
        raise SpotifyApiError(f"Spotify connection error: {err}") from err


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
