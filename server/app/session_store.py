from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UserSession:
    access_token: str
    refresh_token: str
    expires_at_s: float
    spotify_user_id: str
    display_name: str
    avatar_url: Optional[str]


@dataclass
class OAuthState:
    state: str
    code_verifier: str
    created_at_s: float


class InMemorySessionStore:
    def __init__(self, ttl_s: int = 60 * 60 * 24):
        self._ttl_s = ttl_s
        self._sessions: Dict[str, tuple[UserSession, float]] = {}
        self._oauth_states: Dict[str, OAuthState] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, (_, exp) in self._sessions.items() if exp < now]
        for sid in expired:
            self._sessions.pop(sid, None)

        expired_states = [
            sid
            for sid, state in self._oauth_states.items()
            if now - state.created_at_s > 60 * 10
        ]
        for sid in expired_states:
            self._oauth_states.pop(sid, None)

    def create_session(self, session: UserSession) -> str:
        self._purge_expired()
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = (session, time.time() + self._ttl_s)
        return session_id

    def get_session(self, session_id: str) -> Optional[UserSession]:
        self._purge_expired()
        value = self._sessions.get(session_id)
        if value is None:
            return None
        return value[0]

    def update_session(self, session_id: str, session: UserSession) -> None:
        if session_id in self._sessions:
            self._sessions[session_id] = (session, time.time() + self._ttl_s)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._oauth_states.pop(session_id, None)

    def put_oauth_state(self, session_id: str, state: OAuthState) -> None:
        self._oauth_states[session_id] = state

    def pop_oauth_state(self, session_id: str) -> Optional[OAuthState]:
        return self._oauth_states.pop(session_id, None)


class SessionCookieSigner:
    def __init__(self, secret: str):
        self._secret = secret.encode("utf-8")

    def dumps(self, session_id: str) -> str:
        payload = json.dumps({"sid": session_id}, separators=(",", ":")).encode("utf-8")
        b64 = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
        sig = hmac.new(self._secret, b64.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{b64}.{sig}"

    def loads(self, token: str) -> Optional[str]:
        if "." not in token:
            return None
        b64, sig = token.split(".", 1)
        expected = hmac.new(self._secret, b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            padded = b64 + "=" * (-len(b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
        except Exception:
            return None
        return payload.get("sid")
