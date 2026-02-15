from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import NotFoundError

from spotify_ingest import SpotifyTrack


class ESIndexError(RuntimeError):
    """Raised when Elasticsearch setup/index/search fails."""


@dataclass
class SearchHit:
    track_id: str
    name: str
    artists: List[str]
    album: str
    preview_url: Optional[str]
    spotify_url: Optional[str]
    lyrics: Optional[str]
    score: float


_LYRICS_CACHE: Optional[Dict[str, str]] = None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _lyrics_dir() -> Path:
    configured = os.environ.get("LYRICS_DIR", "lyrics")
    return Path(configured).resolve()


def _load_lyrics_cache() -> Dict[str, str]:
    global _LYRICS_CACHE
    if _LYRICS_CACHE is not None:
        return _LYRICS_CACHE
    cache: Dict[str, str] = {}
    ldir = _lyrics_dir()
    if not ldir.exists():
        _LYRICS_CACHE = cache
        return cache
    for txt_path in sorted(ldir.glob("*.txt")):
        key = _slugify(txt_path.stem)
        if not key:
            continue
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = txt_path.read_text(encoding="latin-1", errors="ignore").strip()
        cache[key] = content
    _LYRICS_CACHE = cache
    return cache


def _lyrics_candidates(track: SpotifyTrack) -> List[str]:
    first_artist = track.artists[0] if track.artists else ""
    all_artists = "_".join(track.artists) if track.artists else ""
    return [
        _slugify(track.name),
        _slugify(f"{track.name}_{first_artist}"),
        _slugify(f"{track.name}_{all_artists}"),
    ]


def _lookup_lyrics(track: SpotifyTrack) -> Optional[str]:
    lyrics_map = _load_lyrics_cache()
    for key in _lyrics_candidates(track):
        if key and key in lyrics_map:
            return lyrics_map[key]
    return None


def _build_client(es_url: str) -> Elasticsearch:
    user = os.environ.get("ELASTICSEARCH_USERNAME")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")
    api_key = os.environ.get("ELASTICSEARCH_API_KEY")
    verify_certs = os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    kwargs: Dict = {"verify_certs": verify_certs}
    if api_key:
        kwargs["api_key"] = api_key
    elif user and password:
        kwargs["basic_auth"] = (user, password)
    return Elasticsearch(es_url, **kwargs)


def ensure_index(
    es_url: str,
    index_name: str,
    embedding_dim: int,
    *,
    recreate: bool = False,
) -> Elasticsearch:
    client = _build_client(es_url)
    try:
        if recreate and client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)

        if client.indices.exists(index=index_name):
            return client

        mappings = {
            "mappings": {
                "properties": {
                    "track_id": {"type": "keyword"},
                    "playlist_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "artists": {"type": "keyword"},
                    "album": {"type": "text"},
                    "popularity": {"type": "integer"},
                    "duration_ms": {"type": "integer"},
                    "preview_url": {"type": "keyword"},
                    "spotify_url": {"type": "keyword"},
                    "lyrics": {"type": "text"},
                    "created_at": {"type": "date"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        client.indices.create(index=index_name, body=mappings)
        return client
    except ESConnectionError as err:
        raise ESIndexError(
            f"Could not connect to Elasticsearch at {es_url}. Ensure service is running."
        ) from err
    except Exception as err:
        raise ESIndexError(f"Failed creating/ensuring index '{index_name}': {err}") from err


def healthcheck(es_url: str) -> None:
    client = _build_client(es_url)
    try:
        if not client.ping():
            raise ESIndexError(
                f"Elasticsearch ping failed for {es_url}. Check URL/auth and container health."
            )
    except ESConnectionError as err:
        raise ESIndexError(f"Elasticsearch unreachable at {es_url}: {err}") from err


def upsert_track_embedding(
    client: Elasticsearch,
    index_name: str,
    track: SpotifyTrack,
    vector: List[float],
) -> None:
    lyrics = _lookup_lyrics(track)
    doc = {
        "track_id": track.track_id,
        "playlist_id": track.playlist_id,
        "name": track.name,
        "artists": track.artists,
        "album": track.album,
        "popularity": track.popularity,
        "duration_ms": track.duration_ms,
        "preview_url": track.preview_url,
        "spotify_url": track.spotify_url,
        "lyrics": lyrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector": vector,
    }
    client.index(index=index_name, id=track.track_id, document=doc)


def search_by_text_embedding(
    es_url: str,
    index_name: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
) -> List[SearchHit]:
    client = _build_client(es_url)
    # Convert cosine similarity from [-1, 1] into confidence score [0, 1].
    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "(cosineSimilarity(params.q, 'vector') + 1.0) / 2.0",
                    "params": {"q": query_vector},
                },
            }
        },
        "_source": [
            "track_id",
            "name",
            "artists",
            "album",
            "preview_url",
            "spotify_url",
            "lyrics",
        ],
        "size": max(1, top_k),
    }
    try:
        resp = client.search(index=index_name, body=body)
    except NotFoundError as err:
        raise ESIndexError(
            f"Index '{index_name}' not found. Run index-playlist first."
        ) from err
    except Exception as err:
        raise ESIndexError(f"Elasticsearch search failed: {err}") from err

    hits = []
    for hit in (resp.get("hits") or {}).get("hits", []):
        src = hit.get("_source") or {}
        hits.append(
            SearchHit(
                track_id=str(src.get("track_id") or ""),
                name=str(src.get("name") or ""),
                artists=list(src.get("artists") or []),
                album=str(src.get("album") or ""),
                preview_url=src.get("preview_url"),
                spotify_url=src.get("spotify_url"),
                lyrics=src.get("lyrics"),
                score=max(0.0, min(1.0, float(hit.get("_score") or 0.0))),
            )
        )
    return hits
