from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import ConfigError, load_settings
from .routers.auth import router as auth_router
from .routers.spotify import router as spotify_router
from .session_store import InMemorySessionStore, SessionCookieSigner


def create_app() -> FastAPI:
    app = FastAPI(title="treehacks26 server", version="0.1.0")

    try:
        settings = load_settings()
    except ConfigError as err:
        # App still starts, but surfaces a clear startup configuration error.
        @app.get("/healthz")
        def _healthz_failed() -> JSONResponse:
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "CONFIG_ERROR", "message": str(err)}},
            )

        return app

    app.state.settings = settings
    app.state.session_store = InMemorySessionStore()
    app.state.cookie_signer = SessionCookieSigner(settings.session_secret)
    logging.basicConfig(
        level=logging.INFO if settings.spotify_debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.client_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    app.include_router(auth_router)
    app.include_router(spotify_router)
    return app


app = create_app()
