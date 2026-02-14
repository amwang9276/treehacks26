from __future__ import annotations

import argparse
import os
import sys
from typing import List

from audio_fetch import AudioFetchError, fetch_preview_audio
from es_index import (
    ESIndexError,
    ensure_index,
    healthcheck,
    search_by_text_embedding,
    upsert_track_embedding,
)
from mulan_embed import DEFAULT_MULAN_MODEL_ID, MuLanEmbedError, MuLanEmbedder
from spotify_ingest import SpotifyIngestError, fetch_playlist_tracks


DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_INDEX = "songs_mulan"


def _resolve_es_url(arg_value: str | None) -> str:
    return arg_value or os.environ.get("ELASTICSEARCH_URL", DEFAULT_ES_URL)


def _resolve_index_name(arg_value: str | None) -> str:
    return arg_value or os.environ.get("ELASTICSEARCH_INDEX", DEFAULT_INDEX)


def _resolve_model_id(arg_value: str | None) -> str:
    return arg_value or os.environ.get("MULAN_MODEL_ID", DEFAULT_MULAN_MODEL_ID)


def cmd_index_playlist(args: argparse.Namespace) -> int:
    es_url = _resolve_es_url(args.es_url)
    index_name = _resolve_index_name(args.index)
    model_id = _resolve_model_id(args.model_id)

    print(f"[INDEX] Health-check Elasticsearch at {es_url} ...")
    healthcheck(es_url)

    print(f"[INDEX] Loading MuLan model: {model_id}")
    embedder = MuLanEmbedder(model_id=model_id, device=args.device)
    embedding_dim = embedder.embedding_dim()
    print(f"[INDEX] Embedding dimension: {embedding_dim}")

    client = ensure_index(
        es_url, index_name, embedding_dim, recreate=bool(args.recreate_index)
    )
    print(f"[INDEX] Fetching playlist tracks from {args.playlist_url}")
    tracks, skipped_no_preview = fetch_playlist_tracks(
        args.playlist_url,
        client_id=args.spotify_client_id,
        client_secret=args.spotify_client_secret,
    )
    print(
        f"[INDEX] Tracks with preview: {len(tracks)} "
        f"(skipped without preview: {skipped_no_preview})"
    )

    indexed = 0
    failed_download = 0
    failed_embed = 0
    skipped_short = 0

    for track in tracks:
        try:
            audio_path = fetch_preview_audio(
                track.track_id, track.preview_url or "", cache_dir=args.cache_dir
            )
        except AudioFetchError as err:
            failed_download += 1
            print(f"[INDEX][WARN] download failed for {track.track_id}: {err}")
            continue

        if args.min_preview_duration_ms > 0 and track.duration_ms < args.min_preview_duration_ms:
            skipped_short += 1
            continue

        try:
            vector = embedder.embed_audio_file(str(audio_path)).tolist()
        except MuLanEmbedError as err:
            failed_embed += 1
            print(f"[INDEX][WARN] embedding failed for {track.track_id}: {err}")
            continue

        upsert_track_embedding(client, index_name, track, vector)
        indexed += 1

    print(
        f"[INDEX] Completed: indexed={indexed}, download_failures={failed_download}, "
        f"embed_failures={failed_embed}, skipped_short={skipped_short}"
    )
    if indexed == 0:
        print("[INDEX][WARN] No tracks indexed. Check playlist preview availability.")
    return 0


def cmd_query_vibe(args: argparse.Namespace) -> int:
    es_url = _resolve_es_url(args.es_url)
    index_name = _resolve_index_name(args.index)
    model_id = _resolve_model_id(args.model_id)

    print(f"[QUERY] Health-check Elasticsearch at {es_url} ...")
    healthcheck(es_url)

    print(f"[QUERY] Loading MuLan model: {model_id}")
    embedder = MuLanEmbedder(model_id=model_id, device=args.device)
    query_vec = embedder.embed_text(args.text).tolist()

    hits = search_by_text_embedding(
        es_url, index_name, query_vec, top_k=max(1, args.top_k)
    )
    if not hits:
        print("[QUERY] No results.")
        return 0

    print(f"[QUERY] Top {len(hits)} results for: {args.text}")
    for idx, hit in enumerate(hits, start=1):
        artists = ", ".join(hit.artists)
        print(
            f"{idx:02d}. score={hit.score:.4f} | {hit.name} - {artists}\n"
            f"    spotify={hit.spotify_url}\n"
            f"    preview={hit.preview_url}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuLan + Spotify preview + Elasticsearch semantic vibe search."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index-playlist", help="Ingest playlist and index embeddings.")
    p_index.add_argument("--playlist-url", required=True, help="Spotify playlist URL or ID.")
    p_index.add_argument("--es-url", default=None, help="Elasticsearch URL.")
    p_index.add_argument("--index", default=None, help="Elasticsearch index name.")
    p_index.add_argument("--cache-dir", default=".cache/audio", help="Audio cache directory.")
    p_index.add_argument("--model-id", default=None, help="MuLan-compatible model id.")
    p_index.add_argument("--device", default="cpu", help="cpu or cuda.")
    p_index.add_argument("--spotify-client-id", default=None)
    p_index.add_argument("--spotify-client-secret", default=None)
    p_index.add_argument("--min-preview-duration-ms", type=int, default=0)
    p_index.add_argument("--recreate-index", action="store_true")
    p_index.set_defaults(func=cmd_index_playlist)

    p_query = sub.add_parser("query-vibe", help="Semantic search by vibe text.")
    p_query.add_argument("--text", required=True, help="Free-form vibe query text.")
    p_query.add_argument("--es-url", default=None, help="Elasticsearch URL.")
    p_query.add_argument("--index", default=None, help="Elasticsearch index name.")
    p_query.add_argument("--model-id", default=None, help="MuLan-compatible model id.")
    p_query.add_argument("--device", default="cpu", help="cpu or cuda.")
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.set_defaults(func=cmd_query_vibe)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (SpotifyIngestError, AudioFetchError, MuLanEmbedError, ESIndexError) as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
