from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
    score: float


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
    body = {
        "knn": {
            "field": "vector",
            "query_vector": query_vector,
            "k": max(1, top_k),
            "num_candidates": max(20, top_k * 4),
        },
        "_source": ["track_id", "name", "artists", "album", "preview_url", "spotify_url"],
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
                score=float(hit.get("_score") or 0.0),
            )
        )
    return hits
