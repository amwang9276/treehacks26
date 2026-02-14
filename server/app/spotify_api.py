from __future__ import annotations

import json
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings


class SpotifyApiError(RuntimeError):
    """Raised when Spotify resource API calls fail."""


def _http_json(
    url: str, access_token: str, *, timeout_s: float = 30.0
) -> Dict | List[Dict]:
    req = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise SpotifyApiError(f"Spotify HTTP {err.code}: {details}") from err
    except URLError as err:
        raise SpotifyApiError(f"Spotify connection error: {err}") from err


def get_current_user_profile(settings: Settings, access_token: str) -> Dict:
    return _http_json(f"{settings.spotify_api_base}/me", access_token)


def get_user_playlists(
    settings: Settings, access_token: str, *, limit: int = 50, offset: int = 0
) -> Dict:
    params = urlencode({"limit": limit, "offset": offset})
    payload = _http_json(f"{settings.spotify_api_base}/me/playlists?{params}", access_token)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    normalized = []
    for p in items:
        normalized.append(
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "owner": (p.get("owner") or {}).get("display_name"),
                "images": p.get("images") or [],
                "tracks_total": ((p.get("tracks") or {}).get("total") or 0),
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
        f"{settings.spotify_api_base}/playlists/{playlist_id}/tracks?{params}", access_token
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    normalized = []
    for item in items:
        track = item.get("track") or {}
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
