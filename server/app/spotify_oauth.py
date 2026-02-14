from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings


class SpotifyAuthError(RuntimeError):
    """Raised when Spotify OAuth flow fails."""


def _http_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    form: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
) -> Dict:
    data = None
    if form is not None:
        data = urlencode(form).encode("utf-8")
    req = Request(url, method=method, data=data, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise SpotifyAuthError(f"Spotify HTTP {err.code}: {detail}") from err
    except URLError as err:
        raise SpotifyAuthError(f"Spotify connection error: {err}") from err


def create_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


def build_authorize_url(settings: Settings, state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": settings.spotify_client_id,
        "scope": settings.spotify_scopes,
        "redirect_uri": settings.spotify_redirect_uri,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    }
    return f"{settings.spotify_accounts_base}/authorize?{urlencode(params)}"


def exchange_code_for_tokens(settings: Settings, code: str, code_verifier: str) -> Dict:
    token = base64.b64encode(
        f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode("utf-8")
    ).decode("utf-8")
    return _http_json(
        "POST",
        f"{settings.spotify_accounts_base}/api/token",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        form={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.spotify_redirect_uri,
            "code_verifier": code_verifier,
        },
    )


def refresh_access_token(settings: Settings, refresh_token: str) -> Dict:
    token = base64.b64encode(
        f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode("utf-8")
    ).decode("utf-8")
    return _http_json(
        "POST",
        f"{settings.spotify_accounts_base}/api/token",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        form={"grant_type": "refresh_token", "refresh_token": refresh_token},
    )
