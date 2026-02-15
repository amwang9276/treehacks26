from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from elasticsearch import Elasticsearch

from mulan import DEFAULT_MULAN_MODEL_ID, MuLanEmbedError, MuLanEmbedder


def _resolve_env(name: str, default: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_val = line.split("=", 1)
            if key.strip() == name:
                return raw_val.strip().strip("\"' ")
    return default


def _resolve_bool_env(name: str, default: bool) -> bool:
    raw = _resolve_env(name, str(default).lower()).strip().lower()
    return raw in {"1", "true", "yes"}


@dataclass
class QueryCase:
    query: str
    expected: str


@dataclass
class SongVector:
    path: Path
    vector: np.ndarray


def _load_song_files(songs_dir: Path) -> List[Path]:
    patterns = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(songs_dir.glob(pattern)))
    return files


def _load_cases(cases_file: Path | None, songs: List[Path]) -> List[QueryCase]:
    if cases_file is not None:
        raw = json.loads(cases_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Cases file must be a JSON array of {query, expected}.")
        cases: List[QueryCase] = []
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError("Each test case must be an object.")
            query = str(row.get("query") or "").strip()
            expected = str(row.get("expected") or "").strip()
            if not query or not expected:
                raise ValueError("Each test case needs non-empty 'query' and 'expected'.")
            cases.append(QueryCase(query=query, expected=expected))
        return cases

    # Default bootstrap cases from song filenames.
    # Example: songs/happy_pharrell_williams.mp3 -> query "happy song", expected "happy"
    cases = []
    for song in songs:
        token = song.stem.split("_")[0].strip().lower()
        if token:
            cases.append(QueryCase(query=f"{token} song", expected=token))
    return cases


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_key(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _score_to_unit_interval(cosine_score: float) -> float:
    return max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))


def _build_song_lookup(song_vectors: List[SongVector]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for song in song_vectors:
        key = _normalize_key(song.path.stem)
        lookup[key] = song.path
    return lookup


def _mulan_topk_dict(
    qvec: np.ndarray, song_vectors: List[SongVector], top_k: int
) -> Tuple[List[Tuple[float, Path]], Dict[str, float]]:
    ranked = sorted(
        ((_cosine(qvec, song.vector), song.path) for song in song_vectors),
        key=lambda x: x[0],
        reverse=True,
    )
    mulan_scores: Dict[str, float] = {}
    for cosine_score, path in ranked[:top_k]:
        mulan_scores[_normalize_key(path.stem)] = _score_to_unit_interval(cosine_score)
    return ranked, mulan_scores


def _build_es_client(es_url: str) -> Elasticsearch:
    api_key = _resolve_env("ELASTICSEARCH_API_KEY", "")
    username = _resolve_env("ELASTICSEARCH_USERNAME", "")
    password = _resolve_env("ELASTICSEARCH_PASSWORD", "")
    verify_certs = _resolve_bool_env("ELASTICSEARCH_VERIFY_CERTS", True)
    kwargs: Dict[str, object] = {"verify_certs": verify_certs}
    if api_key:
        kwargs["api_key"] = api_key
    elif username and password:
        kwargs["basic_auth"] = (username, password)
    return Elasticsearch(es_url, **kwargs)


def _print_index_snapshot(es_url: str, es_index: str, limit: int = 5) -> None:
    client = _build_es_client(es_url)
    exists = client.indices.exists(index=es_index)
    print(f"[ES] index='{es_index}' exists={exists}")
    if not exists:
        return
    count = int(client.count(index=es_index).get("count") or 0)
    print(f"[ES] doc_count={count}")
    if count <= 0:
        return
    resp = client.search(
        index=es_index,
        body={
            "query": {"match_all": {}},
            "_source": ["song_key", "song_title", "lyric_id"],
            "size": max(1, limit),
        },
    )
    for i, hit in enumerate((resp.get("hits") or {}).get("hits", []), start=1):
        src = hit.get("_source") or {}
        print(
            f"[ES] sample_{i}: id={hit.get('_id')} "
            f"song_key={src.get('song_key')} song_title={src.get('song_title')}"
        )


def _elastic_topk_dict(
    *,
    es_url: str,
    es_index: str,
    query_text: str,
    top_k: int,
) -> Dict[str, float]:
    client = _build_es_client(es_url)
    if not client.indices.exists(index=es_index):
        raise RuntimeError(f"Index '{es_index}' not found.")
    resp = client.search(
        index=es_index,
        body={
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["lyrics", "song_title^2"],
                }
            },
            "_source": ["song_key", "song_title"],
            "size": max(1, top_k),
        },
    )
    hits = (resp.get("hits") or {}).get("hits", [])
    if not hits:
        return {}
    max_score = float(hits[0].get("_score") or 1.0)
    if max_score <= 0:
        max_score = 1.0
    es_scores: Dict[str, float] = {}
    for hit in hits[:top_k]:
        src = hit.get("_source") or {}
        raw_key = str(src.get("song_key") or src.get("song_title") or hit.get("_id") or "")
        key = _normalize_key(raw_key)
        normalized = float(hit.get("_score") or 0.0) / max_score
        es_scores[key] = max(0.0, min(1.0, normalized))
    return es_scores


def weighted_blend_scores(
    mulan_scores: Dict[str, float],
    elastic_scores: Dict[str, float],
    *,
    mulan_weight: float = 0.5,
    elastic_weight: float = 0.5,
) -> Dict[str, float]:
    total = mulan_weight + elastic_weight
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    mw = mulan_weight / total
    ew = elastic_weight / total
    blended: Dict[str, float] = {}
    for key in set(mulan_scores) | set(elastic_scores):
        blended[key] = mw * mulan_scores.get(key, 0.0) + ew * elastic_scores.get(key, 0.0)
    return blended


def _cache_key_for_song(path: Path, model_id: str, sample_rate: int) -> str:
    stat = path.stat()
    key_src = (
        f"path={path.resolve()}|size={stat.st_size}|mtime_ns={stat.st_mtime_ns}"
        f"|model={model_id}|sr={sample_rate}"
    )
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()


def _load_cached_vector(cache_dir: Path, cache_key: str) -> np.ndarray | None:
    cache_file = cache_dir / f"{cache_key}.npy"
    if not cache_file.exists():
        return None
    try:
        vec = np.load(cache_file)
    except Exception:
        return None
    if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.size == 0:
        return None
    return vec.astype(np.float32)


def _save_cached_vector(cache_dir: Path, cache_key: str, vector: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.npy"
    np.save(cache_file, vector.astype(np.float32))


def _embed_songs(
    embedder: MuLanEmbedder,
    song_paths: List[Path],
    *,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[List[SongVector], int, int]:
    vectors: List[SongVector] = []
    cache_hits = 0
    embedded_now = 0
    for path in song_paths:
        cache_key = _cache_key_for_song(path, embedder.model_id, embedder.sample_rate)
        vec: np.ndarray | None = None
        if use_cache:
            vec = _load_cached_vector(cache_dir, cache_key)
            if vec is not None:
                cache_hits += 1
        if vec is None:
            vec = embedder.embed_audio_file(str(path))
            embedded_now += 1
            if use_cache:
                _save_cached_vector(cache_dir, cache_key, vec)
        vectors.append(SongVector(path=path, vector=vec))
    return vectors, cache_hits, embedded_now


def _is_match(song_path: Path, expected: str) -> bool:
    return expected.lower() in song_path.stem.lower()


def _is_key_match(song_key: str, expected: str) -> bool:
    return _normalize_key(expected) in song_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate MuLan text->audio retrieval on local songs."
    )
    parser.add_argument("--songs-dir", default="songs", help="Directory with local audio files.")
    parser.add_argument(
        "--cases-file",
        default=None,
        help="Optional JSON file with array of {query, expected}.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MULAN_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--es-url",
        default=_resolve_env("ELASTICSEARCH_URL", "http://localhost:9200"),
        help="Elasticsearch URL for semantic score blending.",
    )
    parser.add_argument(
        "--es-index",
        default=_resolve_env("ELASTICSEARCH_INDEX", "lyrics"),
        help="Elasticsearch index name for semantic score blending.",
    )
    parser.add_argument(
        "--disable-elastic",
        action="store_true",
        help="Disable Elasticsearch blending and use MuLan-only ranking.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/mulan_song_embeddings",
        help="Directory to store cached song embeddings.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving cached song embeddings.",
    )
    args = parser.parse_args()

    songs_dir = Path(args.songs_dir).resolve()
    if not songs_dir.exists():
        print(f"[ERROR] songs dir not found: {songs_dir}", file=sys.stderr)
        return 1

    song_paths = _load_song_files(songs_dir)
    if not song_paths:
        print(f"[ERROR] no audio files found in {songs_dir}", file=sys.stderr)
        return 1

    cases_file = Path(args.cases_file).resolve() if args.cases_file else None
    try:
        cases = _load_cases(cases_file, song_paths)
    except Exception as err:
        print(f"[ERROR] invalid cases file: {err}", file=sys.stderr)
        return 1
    if not cases:
        print(
            "[ERROR] no test cases found. Add a cases file or use song names with mood prefixes.",
            file=sys.stderr,
        )
        return 1

    try:
        embedder = MuLanEmbedder(model_id=args.model_id, device=args.device)
        cache_dir = Path(args.cache_dir).resolve()
        song_vectors, cache_hits, embedded_now = _embed_songs(
            embedder,
            song_paths,
            cache_dir=cache_dir,
            use_cache=not bool(args.no_cache),
        )
    except MuLanEmbedError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1

    top_k = max(1, int(args.top_k))
    hits_at_1 = 0
    hits_at_k = 0

    print(f"Indexed songs: {len(song_vectors)}")
    print(f"Queries: {len(cases)}")
    print(f"Top-K: {top_k}")
    print(f"Cache hits: {cache_hits}")
    print(f"Embedded now: {embedded_now}")
    if not args.no_cache:
        print(f"Cache dir: {cache_dir}")
    if not args.disable_elastic:
        try:
            _print_index_snapshot(args.es_url, args.es_index, limit=5)
        except Exception as err:
            print(f"[WARN] Failed to print index snapshot: {err}", file=sys.stderr)
    print("")

    song_lookup = _build_song_lookup(song_vectors)
    for i, case in enumerate(cases, start=1):
        qvec = embedder.embed_text(case.query)
        ranked, mulan_scores = _mulan_topk_dict(qvec, song_vectors, top_k=top_k)

        elastic_scores: Dict[str, float] = {}
        if not args.disable_elastic:
            try:
                elastic_scores = _elastic_topk_dict(
                    es_url=args.es_url,
                    es_index=args.es_index,
                    query_text=case.query,
                    top_k=top_k,
                )
            except Exception as err:
                print(
                    f"[WARN] Elasticsearch blend skipped for query '{case.query}': {err}",
                    file=sys.stderr,
                )

        blended_scores = weighted_blend_scores(
            mulan_scores,
            elastic_scores,
            mulan_weight=0.5,
            elastic_weight=0.5,
        )
        blended_top = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top1_key = blended_top[0][0] if blended_top else ""
        top1_path = song_lookup.get(top1_key)
        top1_display = top1_path.name if top1_path else top1_key

        top1_ok = _is_key_match(top1_key, case.expected)
        topk_ok = any(_is_key_match(song_key, case.expected) for song_key, _ in blended_top)
        hits_at_1 += 1 if top1_ok else 0
        hits_at_k += 1 if topk_ok else 0

        print(
            f"[{i:02d}] query='{case.query}' expected~'{case.expected}' "
            f"top1='{top1_display}' top1_ok={top1_ok} top{top_k}_ok={topk_ok}"
        )
        print(f"     mulan_top{top_k}: {mulan_scores}")
        print(f"     elastic_top{top_k}: {elastic_scores}")
        print(f"     blended_top{top_k}: {dict(blended_top)}")
        for rank, (song_key, score) in enumerate(blended_top, start=1):
            song_path = song_lookup.get(song_key)
            display = song_path.name if song_path else song_key
            print(f"     {rank}. blended_score={score:.4f} file={display}")

    total = len(cases)
    recall1 = hits_at_1 / total
    recallk = hits_at_k / total
    print("")
    print(f"Recall@1: {hits_at_1}/{total} = {recall1:.3f}")
    print(f"Recall@{top_k}: {hits_at_k}/{total} = {recallk:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
