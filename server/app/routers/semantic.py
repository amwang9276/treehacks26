from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..config import Settings
from ..semantic_service import SemanticService, SemanticServiceError
from ..session_store import InMemorySessionStore, SessionCookieSigner, UserSession
from ..spotify_oauth import SpotifyAuthError, refresh_access_token


router = APIRouter(tags=["semantic"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> InMemorySessionStore:
    return request.app.state.session_store


def _signer(request: Request) -> SessionCookieSigner:
    return request.app.state.cookie_signer


def _semantic(request: Request) -> SemanticService:
    return request.app.state.semantic_service


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
    import time

    if time.time() < session.expires_at_s - 30:
        return session
    if not session.refresh_token:
        raise _error(401, "TOKEN_EXPIRED", "Token expired and no refresh token available.")
    try:
        payload = refresh_access_token(settings, session.refresh_token)
    except SpotifyAuthError as err:
        store.delete_session(sid)
        raise _error(401, "TOKEN_EXPIRED", f"Token refresh failed: {err}") from err

    session.access_token = str(payload.get("access_token") or session.access_token)
    session.refresh_token = str(payload.get("refresh_token") or session.refresh_token)
    session.expires_at_s = time.time() + int(payload.get("expires_in") or 3600)
    store.update_session(sid, session)
    return session


@router.post("/semantic/index/start")
def semantic_index_start(
    request: Request,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
    semantic: SemanticService = Depends(_semantic),
) -> Dict[str, Any]:
    sid, session = _get_session(request, settings, store, signer)
    session = _refresh_if_needed(sid, session, settings, store)
    try:
        job = semantic.start_or_get_job(
            user_id=session.spotify_user_id, session_id=sid, force=True
        )
    except SemanticServiceError as err:
        raise _error(500, "SEMANTIC_INDEX_ERROR", str(err)) from err
    return {
        "job_id": job.job_id,
        "status": job.status,
        "started_at": job.started_at,
        "user_id": job.user_id,
    }


@router.get("/semantic/index/status")
def semantic_index_status(
    request: Request,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
    semantic: SemanticService = Depends(_semantic),
) -> Dict[str, Any]:
    _, session = _get_session(request, settings, store, signer)
    job = semantic.job_as_dict(session.spotify_user_id)
    if job is None:
        return {
            "job_id": None,
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "user_id": session.spotify_user_id,
            "playlists_scanned": 0,
            "tracks_seen": 0,
            "tracks_with_preview": 0,
            "indexed": 0,
            "download_failures": 0,
            "embed_failures": 0,
            "skipped_no_preview": 0,
            "phase": "idle",
            "playlist_total": 0,
            "playlists_processed": 0,
            "total_to_index": 0,
            "last_error": None,
        }
    return job


@router.get("/semantic/search")
def semantic_search(
    request: Request,
    text: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=50),
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
    semantic: SemanticService = Depends(_semantic),
) -> Dict[str, Any]:
    _, session = _get_session(request, settings, store, signer)
    try:
        return semantic.search(user_id=session.spotify_user_id, text=text, top_k=top_k)
    except SemanticServiceError as err:
        raise _error(500, "SEMANTIC_SEARCH_ERROR", str(err)) from err
