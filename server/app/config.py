from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    client_origin: str
    session_secret: str
    session_cookie_name: str = "treehacks_session"
    spotify_scopes: str = "playlist-read-private playlist-read-collaborative"
    spotify_api_base: str = "https://api.spotify.com/v1"
    spotify_accounts_base: str = "https://accounts.spotify.com"
    cookie_secure: bool = False


def _load_dotenv_files() -> None:
    # server/app/config.py -> server/ (parents[1]) -> repo root (parents[2])
    server_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    # Load root first, then allow server/.env to override.
    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(server_dir / ".env", override=True)


def _get_setting(name: str) -> str:
    return os.environ.get(name, "").strip()


def _require_env(name: str) -> str:
    value = _get_setting(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    _load_dotenv_files()
    return Settings(
        spotify_client_id=_require_env("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=_require_env("SPOTIFY_CLIENT_SECRET"),
        spotify_redirect_uri=_require_env("SPOTIFY_REDIRECT_URI"),
        client_origin=_require_env("CLIENT_ORIGIN"),
        session_secret=_require_env("SESSION_SECRET"),
        session_cookie_name=_get_setting("SESSION_COOKIE_NAME") or "treehacks_session",
        spotify_scopes=_get_setting("SPOTIFY_SCOPES")
        or "playlist-read-private playlist-read-collaborative",
        spotify_api_base=_get_setting("SPOTIFY_API_BASE") or "https://api.spotify.com/v1",
        spotify_accounts_base=_get_setting("SPOTIFY_ACCOUNTS_BASE")
        or "https://accounts.spotify.com",
        cookie_secure=_get_setting("COOKIE_SECURE").lower() in {"1", "true", "yes"},
    )
