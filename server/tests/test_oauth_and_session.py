from app.session_store import InMemorySessionStore, SessionCookieSigner, UserSession
from app.spotify_oauth import build_authorize_url, create_pkce_pair
from app.config import Settings


def _settings() -> Settings:
    return Settings(
        spotify_client_id="cid",
        spotify_client_secret="csecret",
        spotify_redirect_uri="http://localhost:8000/auth/callback",
        client_origin="http://localhost:3000",
        session_secret="secret",
    )


def test_authorize_url_has_required_fields() -> None:
    settings = _settings()
    verifier, challenge = create_pkce_pair()
    assert verifier
    assert challenge
    url = build_authorize_url(settings, state="abc", code_challenge=challenge)
    assert "response_type=code" in url
    assert "state=abc" in url
    assert "code_challenge=" in url


def test_cookie_sign_and_session_roundtrip() -> None:
    signer = SessionCookieSigner("super-secret")
    store = InMemorySessionStore()
    sid = store.create_session(
        UserSession(
            access_token="a",
            refresh_token="r",
            expires_at_s=9e9,
            spotify_user_id="u1",
            display_name="User",
            avatar_url=None,
        )
    )
    token = signer.dumps(sid)
    sid_back = signer.loads(token)
    assert sid_back == sid
    session = store.get_session(sid_back or "")
    assert session is not None
    assert session.spotify_user_id == "u1"
