import os

from fastapi.testclient import TestClient


def _set_required_env() -> None:
    os.environ["SPOTIFY_CLIENT_ID"] = "cid"
    os.environ["SPOTIFY_CLIENT_SECRET"] = "csecret"
    os.environ["SPOTIFY_REDIRECT_URI"] = "http://localhost:8000/auth/callback"
    os.environ["CLIENT_ORIGIN"] = "http://localhost:3000"
    os.environ["SESSION_SECRET"] = "session-secret"


def test_login_redirect_exists() -> None:
    _set_required_env()
    from app.main import create_app

    client = TestClient(create_app())
    res = client.get("/auth/login", follow_redirects=False)
    assert res.status_code == 302
    assert "accounts.spotify.com" in res.headers["location"]


def test_unauthorized_me() -> None:
    _set_required_env()
    from app.main import create_app

    client = TestClient(create_app())
    res = client.get("/me")
    assert res.status_code == 401
