from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from mulan import DEFAULT_MULAN_MODEL_ID, MuLanEmbedError, MuLanEmbedder


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
    print("")

    for i, case in enumerate(cases, start=1):
        qvec = embedder.embed_text(case.query)
        ranked = sorted(
            (
                (_cosine(qvec, song.vector), song.path)
                for song in song_vectors
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        top1 = ranked[0][1]
        topk_paths = [p for _, p in ranked[:top_k]]
        top1_ok = _is_match(top1, case.expected)
        topk_ok = any(_is_match(path, case.expected) for path in topk_paths)
        hits_at_1 += 1 if top1_ok else 0
        hits_at_k += 1 if topk_ok else 0

        print(
            f"[{i:02d}] query='{case.query}' expected~'{case.expected}' "
            f"top1='{top1.name}' top1_ok={top1_ok} top{top_k}_ok={topk_ok}"
        )
        for rank, (score, path) in enumerate(ranked[:top_k], start=1):
            print(f"     {rank}. score={score:.4f} file={path.name}")

    total = len(cases)
    recall1 = hits_at_1 / total
    recallk = hits_at_k / total
    print("")
    print(f"Recall@1: {hits_at_1}/{total} = {recall1:.3f}")
    print(f"Recall@{top_k}: {hits_at_k}/{total} = {recallk:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
