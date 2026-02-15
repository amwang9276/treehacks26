from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings
from .session_store import InMemorySessionStore
from .spotify_api import SpotifyApiError, get_playlist_tracks, get_user_playlists
from .spotify_oauth import SpotifyAuthError, refresh_access_token


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # server/app/semantic_service.py -> server/ -> repo root
    return Path(__file__).resolve().parents[2]


def _default_es_url() -> str:
    return os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


def _default_model_id() -> str:
    return os.environ.get("MULAN_MODEL_ID", "OpenMuQ/MuQ-MuLan-large")


def _default_cache_dir() -> str:
    return os.environ.get("SEMANTIC_AUDIO_CACHE_DIR", ".cache/audio")


@dataclass
class SemanticJob:
    job_id: str
    user_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    playlists_scanned: int = 0
    tracks_seen: int = 0
    tracks_with_preview: int = 0
    indexed: int = 0
    download_failures: int = 0
    embed_failures: int = 0
    skipped_no_preview: int = 0
    phase: str = "queued"
    playlist_total: int = 0
    playlists_processed: int = 0
    total_to_index: int = 0
    last_error: Optional[str] = None


@dataclass
class TrackForIndex:
    track_id: str
    name: str
    artists: List[str]
    album: str
    duration_ms: int
    preview_url: str
    spotify_url: Optional[str]
    playlist_ids: set[str] = field(default_factory=set)
    playlist_names: set[str] = field(default_factory=set)


class SemanticServiceError(RuntimeError):
    """Raised when semantic indexing/search operations fail."""


class SemanticService:
    def __init__(self, settings: Settings, session_store: InMemorySessionStore) -> None:
        self._settings = settings
        self._session_store = session_store
        self._lock = threading.Lock()
        self._jobs_by_user: Dict[str, SemanticJob] = {}
        self._session_id_by_user: Dict[str, str] = {}
        self._model_id = _default_model_id()
        self._cache_dir = str((_repo_root() / _default_cache_dir()).resolve())
        self._es_url = _default_es_url()
        self._embedder: Any = None
        self._es_client: Any = None
        self._embedding_dim: Optional[int] = None

    def index_name_for_user(self, user_id: str) -> str:
        return f"songs_mulan_{user_id}"

    def latest_job(self, user_id: str) -> Optional[SemanticJob]:
        with self._lock:
            return self._jobs_by_user.get(user_id)

    def start_or_get_job(self, *, user_id: str, session_id: str, force: bool = False) -> SemanticJob:
        with self._lock:
            current = self._jobs_by_user.get(user_id)
            if current and current.status in {"queued", "running"}:
                return current
            if not force:
                persisted = self._get_persisted_index_snapshot(user_id)
                if persisted and persisted["indexed"] > 0:
                    last_indexed_at = str(
                        persisted.get("last_indexed_at") or datetime.now(timezone.utc).isoformat()
                    )
                    job = SemanticJob(
                        job_id=str(uuid.uuid4()),
                        user_id=user_id,
                        status="completed",
                        started_at=last_indexed_at,
                        finished_at=last_indexed_at,
                        indexed=int(persisted["indexed"]),
                        phase="completed",
                        total_to_index=int(persisted["indexed"]),
                    )
                    self._jobs_by_user[user_id] = job
                    self._session_id_by_user[user_id] = session_id
                    logger.info(
                        "Semantic indexing skipped: user_id='%s' existing_docs=%s",
                        user_id,
                        persisted["indexed"],
                    )
                    return job
            job = SemanticJob(
                job_id=str(uuid.uuid4()),
                user_id=user_id,
                status="queued",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._jobs_by_user[user_id] = job
            self._session_id_by_user[user_id] = session_id

        thread = threading.Thread(
            target=self._run_job,
            kwargs={"user_id": user_id, "job_id": job.job_id},
            daemon=True,
        )
        thread.start()
        return job

    def _get_persisted_index_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        index_name = self.index_name_for_user(user_id)
        try:
            es_client = self._get_es_client()
            if not es_client.indices.exists(index=index_name):
                return None
            count_resp = es_client.count(index=index_name)
            indexed = int(count_resp.get("count") or 0)
            if indexed <= 0:
                return {"indexed": 0, "last_indexed_at": None}
            last_resp = es_client.search(
                index=index_name,
                body={
                    "size": 1,
                    "_source": ["last_indexed_at"],
                    "sort": [{"last_indexed_at": {"order": "desc"}}],
                },
            )
            hits = ((last_resp.get("hits") or {}).get("hits") or [])
            last_indexed_at = None
            if hits:
                src = hits[0].get("_source") or {}
                last_indexed_at = src.get("last_indexed_at")
            return {"indexed": indexed, "last_indexed_at": last_indexed_at}
        except Exception:
            logger.exception(
                "Failed reading persisted semantic index snapshot for user_id='%s'", user_id
            )
            return None

    def _set_job_state(self, user_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs_by_user.get(user_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _import_runtime_modules(self) -> tuple[Any, Any, Any]:
        import sys

        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from audio_fetch import AudioFetchError, fetch_preview_audio  # type: ignore
            from mulan_embed import MuLanEmbedder  # type: ignore
        except Exception as err:
            raise SemanticServiceError(
                f"Failed loading embedding runtime modules from repo root: {err}"
            ) from err
        return (AudioFetchError, fetch_preview_audio, MuLanEmbedder)

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        _, _, MuLanEmbedder = self._import_runtime_modules()
        self._embedder = MuLanEmbedder(model_id=self._model_id, device="cpu")
        self._embedding_dim = int(self._embedder.embedding_dim())
        logger.info("MuLan model loaded: model_id='%s' dim=%s", self._model_id, self._embedding_dim)
        return self._embedder

    def _get_es_client(self) -> Any:
        if self._es_client is not None:
            return self._es_client
        try:
            from elasticsearch import Elasticsearch
        except Exception as err:
            raise SemanticServiceError(
                "Elasticsearch dependency is missing. Install server deps including elasticsearch."
            ) from err
        user = os.environ.get("ELASTICSEARCH_USERNAME")
        password = os.environ.get("ELASTICSEARCH_PASSWORD")
        api_key = os.environ.get("ELASTICSEARCH_API_KEY")
        verify_certs = os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        kwargs: Dict[str, Any] = {"verify_certs": verify_certs}
        if api_key:
            kwargs["api_key"] = api_key
        elif user and password:
            kwargs["basic_auth"] = (user, password)
        self._es_client = Elasticsearch(self._es_url, **kwargs)
        if not self._es_client.ping():
            raise SemanticServiceError(f"Elasticsearch ping failed for {self._es_url}")
        return self._es_client

    def _ensure_index(self, user_id: str) -> str:
        client = self._get_es_client()
        embedder = self._get_embedder()
        dim = self._embedding_dim if self._embedding_dim is not None else int(embedder.embedding_dim())
        index_name = self.index_name_for_user(user_id)
        if client.indices.exists(index=index_name):
            return index_name
        mappings = {
            "mappings": {
                "properties": {
                    "track_id": {"type": "keyword"},
                    "spotify_user_id": {"type": "keyword"},
                    "playlist_ids": {"type": "keyword"},
                    "playlist_names": {"type": "keyword"},
                    "name": {"type": "text"},
                    "artists": {"type": "keyword"},
                    "album": {"type": "text"},
                    "duration_ms": {"type": "integer"},
                    "preview_url": {"type": "keyword"},
                    "spotify_url": {"type": "keyword"},
                    "has_preview": {"type": "boolean"},
                    "last_indexed_at": {"type": "date"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        client.indices.create(index=index_name, body=mappings)
        return index_name

    def _get_valid_access_token(self, user_id: str) -> str:
        with self._lock:
            sid = self._session_id_by_user.get(user_id)
        if not sid:
            raise SemanticServiceError("Missing session id for indexing user.")
        session = self._session_store.get_session(sid)
        if session is None:
            raise SemanticServiceError("Session expired before indexing could start.")
        if time.time() < session.expires_at_s - 30:
            return session.access_token
        if not session.refresh_token:
            raise SemanticServiceError("Spotify access token expired and no refresh token exists.")
        try:
            payload = refresh_access_token(self._settings, session.refresh_token)
        except SpotifyAuthError as err:
            raise SemanticServiceError(f"Spotify token refresh failed: {err}") from err
        refreshed = session
        refreshed.access_token = str(payload.get("access_token") or session.access_token)
        refreshed.refresh_token = str(payload.get("refresh_token") or session.refresh_token)
        refreshed.expires_at_s = time.time() + int(payload.get("expires_in") or 3600)
        self._session_store.update_session(sid, refreshed)
        return refreshed.access_token

    def _fetch_all_playlists(self, user_id: str, access_token: str) -> List[Dict[str, Any]]:
        playlists: List[Dict[str, Any]] = []
        offset = 0
        while True:
            try:
                page = get_user_playlists(self._settings, access_token, limit=50, offset=offset)
            except SpotifyApiError as err:
                if err.status_code == 401:
                    access_token = self._get_valid_access_token(user_id)
                    page = get_user_playlists(self._settings, access_token, limit=50, offset=offset)
                else:
                    raise
            items = page.get("items") or []
            playlists.extend(items)
            offset += int(page.get("limit") or 50)
            if len(playlists) >= int(page.get("total") or len(playlists)):
                break
        return playlists

    def _fetch_all_tracks_for_playlist(
        self, user_id: str, access_token: str, playlist_id: str
    ) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        offset = 0
        while True:
            try:
                page = get_playlist_tracks(
                    self._settings, access_token, playlist_id, limit=50, offset=offset
                )
            except SpotifyApiError as err:
                if err.status_code == 401:
                    access_token = self._get_valid_access_token(user_id)
                    page = get_playlist_tracks(
                        self._settings, access_token, playlist_id, limit=50, offset=offset
                    )
                else:
                    raise
            items = page.get("items") or []
            tracks.extend(items)
            offset += int(page.get("limit") or 50)
            if len(tracks) >= int(page.get("total") or len(tracks)):
                break
        return tracks

    def _collect_deduped_tracks(self, user_id: str, access_token: str) -> Dict[str, TrackForIndex]:
        by_track_id: Dict[str, TrackForIndex] = {}
        playlists = self._fetch_all_playlists(user_id, access_token)
        self._set_job_state(
            user_id,
            playlists_scanned=len(playlists),
            playlist_total=len(playlists),
            playlists_processed=0,
        )

        for playlist in playlists:
            pid = str(playlist.get("id") or "")
            pname = str(playlist.get("name") or "")
            if not pid:
                self._increment_job_counter(user_id, "playlists_processed", 1)
                continue
            try:
                items = self._fetch_all_tracks_for_playlist(user_id, access_token, pid)
            except SpotifyApiError as err:
                logger.warning("Skipping playlist '%s' due to Spotify error: %s", pid, err)
                self._increment_job_counter(user_id, "playlists_processed", 1)
                continue
            for item in items:
                self._increment_job_counter(user_id, "tracks_seen", 1)
                track_id = str(item.get("track_id") or "")
                if not track_id:
                    continue
                preview_url = item.get("preview_url")
                if not preview_url:
                    self._increment_job_counter(user_id, "skipped_no_preview", 1)
                    continue
                self._increment_job_counter(user_id, "tracks_with_preview", 1)
                existing = by_track_id.get(track_id)
                if existing is None:
                    existing = TrackForIndex(
                        track_id=track_id,
                        name=str(item.get("name") or ""),
                        artists=list(item.get("artists") or []),
                        album=str(item.get("album") or ""),
                        duration_ms=int(item.get("duration_ms") or 0),
                        preview_url=str(preview_url),
                        spotify_url=item.get("spotify_url"),
                    )
                    by_track_id[track_id] = existing
                existing.playlist_ids.add(pid)
                if pname:
                    existing.playlist_names.add(pname)
            self._increment_job_counter(user_id, "playlists_processed", 1)
        return by_track_id

    def _increment_job_counter(self, user_id: str, field_name: str, amount: int) -> None:
        with self._lock:
            job = self._jobs_by_user.get(user_id)
            if not job:
                return
            current = int(getattr(job, field_name))
            setattr(job, field_name, current + amount)

    def _run_job(self, *, user_id: str, job_id: str) -> None:
        self._set_job_state(user_id, status="running", phase="initializing")
        logger.info("Semantic indexing job started: user_id='%s' job_id='%s'", user_id, job_id)
        try:
            access_token = self._get_valid_access_token(user_id)
            self._set_job_state(user_id, phase="loading_model")
            index_name = self._ensure_index(user_id)
            self._set_job_state(user_id, phase="fetching_spotify")
            deduped = self._collect_deduped_tracks(user_id, access_token)
            self._set_job_state(user_id, phase="embedding", total_to_index=len(deduped))
            AudioFetchError, fetch_preview_audio, _ = self._import_runtime_modules()
            embedder = self._get_embedder()
            es_client = self._get_es_client()

            for track in deduped.values():
                try:
                    audio_path = fetch_preview_audio(
                        track.track_id,
                        track.preview_url,
                        cache_dir=self._cache_dir,
                        retries=2,
                        timeout_s=30.0,
                    )
                except AudioFetchError:
                    self._increment_job_counter(user_id, "download_failures", 1)
                    continue
                try:
                    vector = embedder.embed_audio_file(str(audio_path)).tolist()
                    logger.info(
                        "MuLan embedding generated: track_id='%s' dims=%s vector=%s",
                        track.track_id,
                        len(vector),
                        vector,
                    )
                except Exception:
                    self._increment_job_counter(user_id, "embed_failures", 1)
                    continue

                doc = {
                    "track_id": track.track_id,
                    "spotify_user_id": user_id,
                    "playlist_ids": sorted(track.playlist_ids),
                    "playlist_names": sorted(track.playlist_names),
                    "name": track.name,
                    "artists": track.artists,
                    "album": track.album,
                    "duration_ms": track.duration_ms,
                    "preview_url": track.preview_url,
                    "spotify_url": track.spotify_url,
                    "has_preview": True,
                    "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                    "vector": vector,
                }
                es_client.index(index=index_name, id=track.track_id, document=doc)
                self._increment_job_counter(user_id, "indexed", 1)

            self._set_job_state(
                user_id,
                status="completed",
                phase="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info(
                "Semantic indexing job completed: user_id='%s' job_id='%s' indexed=%s",
                user_id,
                job_id,
                self.latest_job(user_id).indexed if self.latest_job(user_id) else 0,
            )
        except Exception as err:
            logger.exception(
                "Semantic indexing job failed: user_id='%s' job_id='%s' err=%s",
                user_id,
                job_id,
                err,
            )
            self._set_job_state(
                user_id,
                status="failed",
                phase="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_error=str(err),
            )

    def search(self, *, user_id: str, text: str, top_k: int) -> Dict[str, Any]:
        if not text.strip():
            raise SemanticServiceError("Query text is empty.")
        es_client = self._get_es_client()
        embedder = self._get_embedder()
        query_vector = embedder.embed_text(text).tolist()
        index_name = self.index_name_for_user(user_id)
        if not es_client.indices.exists(index=index_name):
            return {"items": []}
        body = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": max(1, top_k),
                "num_candidates": max(20, top_k * 4),
            },
            "_source": [
                "track_id",
                "name",
                "artists",
                "album",
                "preview_url",
                "spotify_url",
                "playlist_ids",
                "playlist_names",
            ],
            "size": max(1, top_k),
        }
        resp = es_client.search(index=index_name, body=body)
        hits = []
        for hit in (resp.get("hits") or {}).get("hits", []):
            src = hit.get("_source") or {}
            hits.append(
                {
                    "track_id": str(src.get("track_id") or ""),
                    "name": str(src.get("name") or ""),
                    "artists": list(src.get("artists") or []),
                    "album": str(src.get("album") or ""),
                    "preview_url": src.get("preview_url"),
                    "spotify_url": src.get("spotify_url"),
                    "playlist_ids": list(src.get("playlist_ids") or []),
                    "playlist_names": list(src.get("playlist_names") or []),
                    "score": float(hit.get("_score") or 0.0),
                }
            )
        return {"items": hits}

    def job_as_dict(self, user_id: str) -> Optional[Dict[str, Any]]:
        job = self.latest_job(user_id)
        if job is None:
            persisted = self._get_persisted_index_snapshot(user_id)
            if not persisted or int(persisted.get("indexed") or 0) <= 0:
                return None
            last_indexed_at = str(
                persisted.get("last_indexed_at") or datetime.now(timezone.utc).isoformat()
            )
            return {
                "job_id": None,
                "status": "completed",
                "started_at": last_indexed_at,
                "finished_at": last_indexed_at,
                "user_id": user_id,
                "playlists_scanned": 0,
                "tracks_seen": 0,
                "tracks_with_preview": 0,
                "indexed": int(persisted["indexed"]),
                "download_failures": 0,
                "embed_failures": 0,
                "skipped_no_preview": 0,
                "phase": "completed",
                "playlist_total": 0,
                "playlists_processed": 0,
                "total_to_index": int(persisted["indexed"]),
                "last_error": None,
            }
        return asdict(job)
