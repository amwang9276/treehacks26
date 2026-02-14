from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_PLAYLIST_API = "https://api.spotify.com/v1/playlists/{playlist_id}/tracks"


class SpotifyIngestError(RuntimeError):
    """Raised when Spotify auth or ingest requests fail."""


@dataclass(frozen=True)
class SpotifyTrack:
    track_id: str
    name: str
    artists: List[str]
    album: str
    popularity: int
    preview_url: Optional[str]
    spotify_url: Optional[str]
    duration_ms: int
    playlist_id: str


def parse_playlist_id(playlist_url_or_id: str) -> str:
    raw = playlist_url_or_id.strip()
    if not raw:
        raise SpotifyIngestError("Playlist URL or ID is empty.")
    if "/" not in raw:
        return raw
    m = re.search(r"playlist/([A-Za-z0-9]+)", raw)
    if not m:
        raise SpotifyIngestError(f"Could not parse playlist ID from: {raw}")
    return m.group(1)


def get_spotify_credentials(
    explicit_client_id: Optional[str], explicit_client_secret: Optional[str]
) -> Tuple[str, str]:
    client_id = explicit_client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = explicit_client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SpotifyIngestError(
            "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET or pass --spotify-client-id/--spotify-client-secret."
        )
    return client_id, client_secret


def _http_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
) -> Dict:
    data = None
    if body is not None:
        data = "&".join([f"{quote(k)}={quote(v)}" for k, v in body.items()]).encode(
            "utf-8"
        )
    req = Request(url, method=method, data=data, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise SpotifyIngestError(f"Spotify HTTP error {err.code}: {details}") from err
    except URLError as err:
        raise SpotifyIngestError(f"Spotify connection error: {err}") from err


def get_access_token(client_id: str, client_secret: str) -> str:
    token = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode(
        "utf-8"
    )
    payload = _http_json(
        "POST",
        SPOTIFY_TOKEN_URL,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body={"grant_type": "client_credentials"},
    )
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise SpotifyIngestError(f"Spotify token response missing access_token: {payload}")
    return access_token


def _normalize_track(raw_track: Dict, playlist_id: str) -> Optional[SpotifyTrack]:
    t = raw_track.get("track") or {}
    track_id = t.get("id")
    if not track_id:
        return None
    artists = [a.get("name", "") for a in (t.get("artists") or []) if a.get("name")]
    album = ((t.get("album") or {}).get("name")) or ""
    ext_urls = t.get("external_urls") or {}
    return SpotifyTrack(
        track_id=str(track_id),
        name=str(t.get("name") or ""),
        artists=artists,
        album=str(album),
        popularity=int(t.get("popularity") or 0),
        preview_url=t.get("preview_url"),
        spotify_url=ext_urls.get("spotify"),
        duration_ms=int(t.get("duration_ms") or 0),
        playlist_id=playlist_id,
    )


def fetch_playlist_tracks(
    playlist_url_or_id: str,
    *,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Tuple[List[SpotifyTrack], int]:
    playlist_id = parse_playlist_id(playlist_url_or_id)
    cid, csecret = get_spotify_credentials(client_id, client_secret)
    access_token = get_access_token(cid, csecret)

    tracks: List[SpotifyTrack] = []
    skipped_no_preview = 0
    next_url = SPOTIFY_PLAYLIST_API.format(playlist_id=playlist_id) + "?limit=100"

    while next_url:
        page = _http_json(
            "GET",
            next_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout_s=timeout_s,
        )
        items = page.get("items") or []
        for item in items:
            normalized = _normalize_track(item, playlist_id=playlist_id)
            if normalized is None:
                continue
            if not normalized.preview_url:
                skipped_no_preview += 1
                continue
            tracks.append(normalized)
        next_url = page.get("next")

    return tracks, skipped_no_preview
