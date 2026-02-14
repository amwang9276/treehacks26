from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..config import Settings
from ..session_store import InMemorySessionStore, SessionCookieSigner, UserSession
from ..spotify_api import SpotifyApiError, get_playlist_tracks, get_user_playlists
from ..spotify_oauth import SpotifyAuthError, refresh_access_token


router = APIRouter(tags=["spotify"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> InMemorySessionStore:
    return request.app.state.session_store


def _signer(request: Request) -> SessionCookieSigner:
    return request.app.state.cookie_signer


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message}})


def _get_session(
    request: Request,
    settings: Settings,
    store: InMemorySessionStore,
    signer: SessionCookieSigner,
) -> tuple[str, UserSession]:
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        raise _error(401, "UNAUTHORIZED", "Missing auth session.")
    sid = signer.loads(token)
    if not sid:
        raise _error(401, "UNAUTHORIZED", "Invalid auth session.")
    session = store.get_session(sid)
    if session is None:
        raise _error(401, "UNAUTHORIZED", "Session expired.")
    return sid, session


def _refresh_if_needed(
    sid: str, session: UserSession, settings: Settings, store: InMemorySessionStore
) -> UserSession:
    if time.time() < session.expires_at_s - 30:
        return session
    if not session.refresh_token:
        raise _error(401, "TOKEN_EXPIRED", "Token expired and no refresh token available.")
    try:
        payload = refresh_access_token(settings, session.refresh_token)
    except SpotifyAuthError as err:
        store.delete_session(sid)
        raise _error(401, "TOKEN_EXPIRED", f"Token refresh failed: {err}") from err

    new_session = UserSession(
        access_token=str(payload.get("access_token") or session.access_token),
        refresh_token=str(payload.get("refresh_token") or session.refresh_token),
        expires_at_s=time.time() + int(payload.get("expires_in") or 3600),
        spotify_user_id=session.spotify_user_id,
        display_name=session.display_name,
        avatar_url=session.avatar_url,
    )
    store.update_session(sid, new_session)
    return new_session


@router.get("/me")
def me(
    request: Request,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
) -> Dict[str, Any]:
    sid, session = _get_session(request, settings, store, signer)
    session = _refresh_if_needed(sid, session, settings, store)
    return {
        "id": session.spotify_user_id,
        "display_name": session.display_name,
        "avatar_url": session.avatar_url,
    }


@router.get("/spotify/playlists")
def playlists(
    request: Request,
    limit: int = Query(default=50, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
) -> Dict[str, Any]:
    sid, session = _get_session(request, settings, store, signer)
    session = _refresh_if_needed(sid, session, settings, store)
    try:
        return get_user_playlists(
            settings, session.access_token, limit=limit, offset=offset
        )
    except SpotifyApiError as err:
        raise _error(502, "SPOTIFY_API_ERROR", str(err)) from err


@router.get("/spotify/playlists/{playlist_id}/tracks")
def playlist_tracks(
    playlist_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
) -> Dict[str, Any]:
    sid, session = _get_session(request, settings, store, signer)
    session = _refresh_if_needed(sid, session, settings, store)
    try:
        return get_playlist_tracks(
            settings, session.access_token, playlist_id, limit=limit, offset=offset
        )
    except SpotifyApiError as err:
        raise _error(502, "SPOTIFY_API_ERROR", str(err)) from err
