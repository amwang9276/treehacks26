from __future__ import annotations

import secrets
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from ..config import Settings
from ..session_store import InMemorySessionStore, OAuthState, SessionCookieSigner, UserSession
from ..spotify_api import SpotifyApiError, get_current_user_profile
from ..spotify_oauth import (
    SpotifyAuthError,
    build_authorize_url,
    create_pkce_pair,
    exchange_code_for_tokens,
)


router = APIRouter(prefix="/auth", tags=["auth"])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> InMemorySessionStore:
    return request.app.state.session_store


def _signer(request: Request) -> SessionCookieSigner:
    return request.app.state.cookie_signer


@router.get("/login")
def login(
    request: Request,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
) -> Response:
    state = secrets.token_urlsafe(24)
    verifier, challenge = create_pkce_pair()
    temp_sid = secrets.token_urlsafe(24)
    store.put_oauth_state(
        temp_sid,
        OAuthState(state=state, code_verifier=verifier, created_at_s=time.time()),
    )
    url = build_authorize_url(settings, state=state, code_challenge=challenge)
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie(
        key=f"{settings.session_cookie_name}_tmp",
        value=temp_sid,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    return resp


@router.get("/callback")
def callback(
    request: Request,
    code: str,
    state: str,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
) -> Response:
    temp_sid = request.cookies.get(f"{settings.session_cookie_name}_tmp")
    if not temp_sid:
        raise HTTPException(status_code=401, detail={"error": {"code": "UNAUTHORIZED", "message": "Missing temp oauth session."}})
    oauth_state = store.pop_oauth_state(temp_sid)
    if oauth_state is None or oauth_state.state != state:
        raise HTTPException(status_code=401, detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid OAuth state."}})
    try:
        tokens = exchange_code_for_tokens(settings, code=code, code_verifier=oauth_state.code_verifier)
        profile = get_current_user_profile(settings, tokens["access_token"])
    except (SpotifyAuthError, SpotifyApiError) as err:
        raise HTTPException(status_code=502, detail={"error": {"code": "SPOTIFY_API_ERROR", "message": str(err)}}) from err

    expires_in = int(tokens.get("expires_in") or 3600)
    session = UserSession(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        expires_at_s=time.time() + expires_in,
        spotify_user_id=str(profile.get("id") or ""),
        display_name=str(profile.get("display_name") or ""),
        avatar_url=((profile.get("images") or [{}])[0] or {}).get("url")
        if profile.get("images")
        else None,
    )
    sid = store.create_session(session)
    cookie_value = signer.dumps(sid)
    resp = RedirectResponse(url=f"{settings.client_origin}/playlists", status_code=302)
    resp.set_cookie(
        key=settings.session_cookie_name,
        value=cookie_value,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    resp.delete_cookie(f"{settings.session_cookie_name}_tmp", path="/")
    return resp


@router.post("/logout")
def logout(
    request: Request,
    settings: Settings = Depends(_settings),
    store: InMemorySessionStore = Depends(_store),
    signer: SessionCookieSigner = Depends(_signer),
) -> Response:
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        sid = signer.loads(token)
        if sid:
            store.delete_session(sid)
    resp = Response(status_code=204)
    resp.delete_cookie(settings.session_cookie_name, path="/")
    return resp
