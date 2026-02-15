from __future__ import annotations

import argparse
import hashlib
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
from openai import OpenAI
from elasticsearch import Elasticsearch

from camera import FramePacket, build_local_camera_help_text, build_source_from_args
from context_shot import ContextShot
from facial_emotions import EmotionObservation, EmotionProcessor
from fusion import SensorFusion, SensorState
from play_music import MusicPlaybackError, MusicPlayer
from mulan import DEFAULT_MULAN_MODEL_ID, MuLanEmbedError, MuLanEmbedder
from suno_gen_music import SunoError, SunoGenerationResult, generate_from_prompt
from voice import VoiceObservation, VoiceProcessor


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_MAX_TOKENS = 48


def _read_env_file(path: Path = Path(".env")) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def get_openai_api_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    file_key = _read_env_file().get("OPENAI_API_KEY")
    if file_key:
        return file_key
    return None


def generate_suno_prompt_for_emotion(
    emotion: str,
    *,
    client: OpenAI,
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
    timeout_s: float = 30.0,
    context: Optional[str] = None,
) -> str:
    effective_client = client.with_options(timeout=timeout_s)
    print(f"[OPENAI] generating prompt for emotion '{emotion}'")
    user_content = f"Detected emotion: {emotion}"
    if context:
        user_content += f"\nRoom context: {context}"
    try:
        completion = effective_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Output exactly one short Suno music prompt line. "
                        "No markdown, no quotes."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=max(16, max_tokens),
            temperature=0.5,
        )
    except Exception as err:
        raise RuntimeError(f"OpenAI SDK error: {err}") from err

    content = completion.choices[0].message.content if completion.choices else None
    text = (content or "").strip()
    if not text:
        raise RuntimeError("OpenAI completion did not include prompt text.")
    return text


@dataclass
class StableEmotionChangeDetector:
    min_stable_seconds: float = 1.0
    candidate_emotion: Optional[str] = None
    candidate_since_s: Optional[float] = None
    last_triggered_emotion: Optional[str] = None

    def observe(self, observation: EmotionObservation) -> Optional[str]:
        emotion = observation.label
        now = observation.timestamp_s
        if not emotion:
            self.candidate_emotion = None
            self.candidate_since_s = None
            return None

        if emotion != self.candidate_emotion:
            self.candidate_emotion = emotion
            self.candidate_since_s = now
            return None

        if self.candidate_since_s is None:
            self.candidate_since_s = now
            return None

        if now - self.candidate_since_s < self.min_stable_seconds:
            return None

        if emotion == self.last_triggered_emotion:
            return None

        self.last_triggered_emotion = emotion
        return emotion


def _choose_track_url(result: SunoGenerationResult) -> Optional[str]:
    for track in result.tracks:
        if track.stream_url:
            return track.stream_url
        if track.audio_url:
            return track.audio_url
    return None


# Queue items: either "__STOP__" or (emotion, context_or_none)
_MusicQueueItem = Union[str, tuple]


@dataclass
class LocalSongEmbedding:
    key: str
    path: Path
    vector: "np.ndarray"


def _parse_bool_arg(value: str) -> bool:
    lowered = (value or "").strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected boolean value for --generate (true/false)."
    )


def _resolve_setting(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    file_value = _read_env_file().get(name)
    if file_value:
        return file_value
    return default


def _resolve_bool_setting(name: str, default: bool) -> bool:
    raw = _resolve_setting(name, str(default).lower()).strip().lower()
    return raw in {"1", "true", "yes"}


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _score_to_unit_interval(cosine_score: float) -> float:
    return max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))


def _weighted_blend_scores(
    mulan_scores: Dict[str, float],
    elastic_scores: Dict[str, float],
    *,
    mulan_weight: float = 0.7,
    elastic_weight: float = 0.3,
) -> Dict[str, float]:
    total = mulan_weight + elastic_weight
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    mw = mulan_weight / total
    ew = elastic_weight / total
    return {
        key: mw * mulan_scores.get(key, 0.0) + ew * elastic_scores.get(key, 0.0)
        for key in (set(mulan_scores) | set(elastic_scores))
    }


def _build_es_client(es_url: str) -> Elasticsearch:
    api_key = _resolve_setting("ELASTICSEARCH_API_KEY", "")
    username = _resolve_setting("ELASTICSEARCH_USERNAME", "")
    password = _resolve_setting("ELASTICSEARCH_PASSWORD", "")
    verify_certs = _resolve_bool_setting("ELASTICSEARCH_VERIFY_CERTS", True)
    kwargs: Dict[str, object] = {"verify_certs": verify_certs}
    if api_key:
        kwargs["api_key"] = api_key
    elif username and password:
        kwargs["basic_auth"] = (username, password)
    return Elasticsearch(es_url, **kwargs)


def _load_local_song_embeddings(
    *,
    songs_dir: str,
    model_id: str,
    cache_dir: str,
    use_cache: bool,
    device: str = "cpu",
) -> Tuple[MuLanEmbedder, List[LocalSongEmbedding]]:
    import numpy as np

    directory = Path(songs_dir).resolve()
    patterns = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    if not files:
        raise RuntimeError(f"No song files found in {directory}")

    resolved_cache_dir = Path(cache_dir).resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key_for_path(path: Path, sample_rate: int) -> str:
        stat = path.stat()
        key_src = (
            f"path={path.resolve()}|size={stat.st_size}|mtime_ns={stat.st_mtime_ns}"
            f"|model={model_id}|sr={sample_rate}"
        )
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def try_load_cached_vector(path: Path, sample_rate: int) -> Optional["np.ndarray"]:
        if not use_cache:
            return None
        key = cache_key_for_path(path, sample_rate)
        fpath = resolved_cache_dir / f"{key}.npy"
        if not fpath.exists():
            return None
        try:
            vec = np.load(fpath)
        except Exception:
            return None
        if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.size == 0:
            return None
        return vec.astype(np.float32)

    def save_cached_vector(path: Path, sample_rate: int, vector: "np.ndarray") -> None:
        if not use_cache:
            return
        key = cache_key_for_path(path, sample_rate)
        fpath = resolved_cache_dir / f"{key}.npy"
        np.save(fpath, vector.astype(np.float32))

    print(
        f"[RETRIEVAL] starting song embedding calculation: "
        f"model='{model_id}' songs_dir='{directory}' files={len(files)}"
    )
    embedder = MuLanEmbedder(model_id=model_id, device=device)
    embeddings: List[LocalSongEmbedding] = []
    cache_hits = 0
    embedded_now = 0
    for path in files:
        vec = try_load_cached_vector(path, embedder.sample_rate)
        if vec is not None:
            cache_hits += 1
        else:
            vec = embedder.embed_audio_file(str(path))
            embedded_now += 1
            save_cached_vector(path, embedder.sample_rate, vec)
        if not isinstance(vec, np.ndarray):
            continue
        embeddings.append(
            LocalSongEmbedding(key=_normalize_key(path.stem), path=path, vector=vec)
        )
    if not embeddings:
        raise RuntimeError("No song embeddings could be generated.")
    print(
        f"[RETRIEVAL] song embeddings ready: embedded={len(embeddings)} "
        f"model='{model_id}' cache_hits={cache_hits} embedded_now={embedded_now} "
        f"cache_dir='{resolved_cache_dir}'"
    )
    return embedder, embeddings


def _mulan_topk_scores(
    query_vector: "np.ndarray",
    songs: List[LocalSongEmbedding],
    *,
    top_k: int,
) -> Dict[str, float]:
    import numpy as np

    ranked = sorted(
        (
            (
                float(np.dot(query_vector, song.vector))
                / max(float(np.linalg.norm(query_vector) * np.linalg.norm(song.vector)), 1e-8),
                song,
            )
            for song in songs
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    result: Dict[str, float] = {}
    for cosine_score, song in ranked[: max(1, top_k)]:
        result[song.key] = _score_to_unit_interval(cosine_score)
    return result


def _elastic_topk_scores(
    es_client: Elasticsearch,
    *,
    es_index: str,
    query_text: str,
    top_k: int,
) -> Dict[str, float]:
    if not es_client.indices.exists(index=es_index):
        return {}
    resp = es_client.search(
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
    out: Dict[str, float] = {}
    for hit in hits[: max(1, top_k)]:
        src = hit.get("_source") or {}
        raw_key = str(src.get("song_key") or src.get("song_title") or hit.get("_id") or "")
        out[_normalize_key(raw_key)] = max(
            0.0, min(1.0, float(hit.get("_score") or 0.0) / max_score)
        )
    return out


def _select_song_from_query(
    *,
    query_text: str,
    embedder: MuLanEmbedder,
    songs: List[LocalSongEmbedding],
    es_client: Optional[Elasticsearch],
    es_index: str,
    top_k: int = 3,
) -> Tuple[Optional[Path], Dict[str, float], Dict[str, float], Dict[str, float]]:
    print(
        f"[RETRIEVAL] starting query calculation (MuLan + Elasticsearch): "
        f"query='{query_text}' top_k={top_k} es_index='{es_index}'"
    )
    query_vector = embedder.embed_text(query_text)
    mulan_scores = _mulan_topk_scores(query_vector, songs, top_k=top_k)
    elastic_scores: Dict[str, float] = {}
    if es_client is not None:
        try:
            elastic_scores = _elastic_topk_scores(
                es_client, es_index=es_index, query_text=query_text, top_k=top_k
            )
        except Exception as err:
            print(f"[MUSIC] elastic retrieval warning: {err}", file=sys.stderr)
    blended = _weighted_blend_scores(
        mulan_scores, elastic_scores, mulan_weight=0.8, elastic_weight=0.2
    )
    if not blended:
        return None, mulan_scores, elastic_scores, blended
    best_key = max(blended.items(), key=lambda x: x[1])[0]
    path_map = {song.key: song.path for song in songs}
    return path_map.get(best_key), mulan_scores, elastic_scores, blended


def _list_local_song_paths(songs_dir: str) -> List[Path]:
    directory = Path(songs_dir).resolve()
    patterns = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    return files


def _select_song_from_query_elastic_only(
    *,
    query_text: str,
    song_paths: List[Path],
    es_client: Optional[Elasticsearch],
    es_index: str,
    top_k: int = 3,
) -> Tuple[Optional[Path], Dict[str, float], Dict[str, float], Dict[str, float]]:
    print(
        f"[RETRIEVAL] starting query calculation (Elasticsearch-only fallback): "
        f"query='{query_text}' top_k={top_k} es_index='{es_index}'"
    )
    elastic_scores: Dict[str, float] = {}
    if es_client is not None:
        try:
            elastic_scores = _elastic_topk_scores(
                es_client, es_index=es_index, query_text=query_text, top_k=top_k
            )
        except Exception as err:
            print(f"[MUSIC] elastic retrieval warning: {err}", file=sys.stderr)

    if not song_paths:
        return None, {}, elastic_scores, elastic_scores

    path_map = {_normalize_key(path.stem): path for path in song_paths}
    selected_path: Optional[Path] = None
    if elastic_scores:
        best_key = max(elastic_scores.items(), key=lambda x: x[1])[0]
        selected_path = path_map.get(best_key)
    if selected_path is None:
        selected_path = sorted(song_paths)[0]
    return selected_path, {}, elastic_scores, elastic_scores


def _music_worker(
    emotion_queue: "queue.Queue[_MusicQueueItem]",
    *,
    generate: bool,
    openai_api_key: Optional[str],
    openai_model: str,
    openai_max_tokens: int,
    suno_poll_interval_s: float,
    player: Optional[MusicPlayer],
    songs_dir: str,
    es_url: str,
    es_index: str,
    retrieval_top_k: int,
    retrieval_model_id: str,
    retrieval_cache_dir: str,
    retrieval_no_cache: bool,
) -> None:
    last_processed_signature: Optional[Tuple[str, bool]] = None
    client: Optional[OpenAI] = None
    retrieval_embedder: Optional[MuLanEmbedder] = None
    retrieval_songs: List[LocalSongEmbedding] = []
    retrieval_song_paths: List[Path] = []
    retrieval_es_client: Optional[Elasticsearch] = None
    retrieval_fallback_mode = False
    retrieval_fallback_logged = False
    retrieval_initialized = False
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)

    def _initialize_retrieval() -> None:
        nonlocal retrieval_embedder
        nonlocal retrieval_songs
        nonlocal retrieval_song_paths
        nonlocal retrieval_es_client
        nonlocal retrieval_fallback_mode
        nonlocal retrieval_initialized
        if retrieval_initialized:
            return
        try:
            retrieval_embedder, retrieval_songs = _load_local_song_embeddings(
                songs_dir=songs_dir,
                model_id=retrieval_model_id,
                cache_dir=retrieval_cache_dir,
                use_cache=not retrieval_no_cache,
                device="cpu",
            )
            retrieval_es_client = _build_es_client(es_url)
            print(
                f"[MUSIC] retrieval mode ready: songs={len(retrieval_songs)} "
                f"es_index='{es_index}'"
            )
            retrieval_initialized = True
        except (MuLanEmbedError, RuntimeError) as err:
            print(f"[MUSIC] retrieval setup error: {err}", file=sys.stderr)
            retrieval_song_paths = _list_local_song_paths(songs_dir)
            retrieval_es_client = _build_es_client(es_url)
            if retrieval_song_paths:
                retrieval_fallback_mode = True
                print(
                    f"[MUSIC] retrieval fallback enabled (elastic-only): "
                    f"songs={len(retrieval_song_paths)} es_index='{es_index}'"
                )
            retrieval_initialized = True
        except Exception as err:
            print(f"[MUSIC] retrieval setup warning: {err}", file=sys.stderr)
            retrieval_initialized = True

    if not generate:
        _initialize_retrieval()

    while True:
        item = emotion_queue.get()
        if item == "__STOP__":
            return
        item_generate = generate
        if isinstance(item, tuple):
            if len(item) >= 3:
                emotion, context, item_generate = item[0], item[1], bool(item[2])
            elif len(item) == 2:
                emotion, context = item
            else:
                emotion, context = item[0], None
        else:
            emotion, context = item, None
        use_generate = bool(item_generate)
        signature = (str(emotion), use_generate)
        if signature == last_processed_signature:
            continue
        try:
            # If fusion/context text exists, pass it directly to Suno/retrieval.
            # This avoids shortening/paraphrasing via an extra OpenAI prompt step.
            if context and str(context).strip():
                prompt = str(context).strip()
            elif client is not None:
                prompt = generate_suno_prompt_for_emotion(
                    emotion,
                    client=client,
                    model=openai_model,
                    max_tokens=openai_max_tokens,
                    context=context,
                )
            else:
                prompt = f"{emotion} mood music"

            if use_generate:
                result = generate_from_prompt(
                    prompt,
                    wait=True,
                    poll_interval_s=max(0.5, suno_poll_interval_s),
                )
                url = _choose_track_url(result)
                if not url:
                    print(
                        f"[MUSIC] task_id={result.task_id} returned no playable url.",
                        file=sys.stderr,
                    )
                    continue
                if player is None:
                    print(
                        f"[MUSIC] emotion={emotion} prompt='{prompt}' "
                        f"task_id={result.task_id} playable_url={url} (playback disabled)"
                    )
                else:
                    played_file = player.play_url(url)
                    print(
                        f"[MUSIC] emotion={emotion} prompt='{prompt}' "
                        f"task_id={result.task_id} playing={played_file}"
                    )
            else:
                if not retrieval_initialized:
                    print("[MUSIC] initializing retrieval on first Spotify-mode event...")
                    _initialize_retrieval()
                if retrieval_embedder is not None and retrieval_songs:
                    selected_path, mulan_scores, elastic_scores, blended_scores = _select_song_from_query(
                        query_text=prompt,
                        embedder=retrieval_embedder,
                        songs=retrieval_songs,
                        es_client=retrieval_es_client,
                        es_index=es_index,
                        top_k=max(1, retrieval_top_k),
                    )
                elif retrieval_song_paths:
                    if retrieval_fallback_mode and not retrieval_fallback_logged:
                        print("[MUSIC] using elastic-only retrieval fallback (MuLan unavailable).")
                        retrieval_fallback_logged = True
                    selected_path, mulan_scores, elastic_scores, blended_scores = (
                        _select_song_from_query_elastic_only(
                            query_text=prompt,
                            song_paths=retrieval_song_paths,
                            es_client=retrieval_es_client,
                            es_index=es_index,
                            top_k=max(1, retrieval_top_k),
                        )
                    )
                else:
                    raise RuntimeError(
                        "Retrieval mode not initialized; cannot select local songs."
                    )
                print(f"[MUSIC] prompt='{prompt}'")
                print(f"[MUSIC] mulan_top{retrieval_top_k}: {mulan_scores}")
                print(f"[MUSIC] elastic_top{retrieval_top_k}: {elastic_scores}")
                print(f"[MUSIC] blended_top{retrieval_top_k}: {blended_scores}")
                if selected_path is None:
                    print("[MUSIC] retrieval returned no playable local song.", file=sys.stderr)
                    continue
                if player is None:
                    print(
                        f"[MUSIC] emotion={emotion} prompt='{prompt}' "
                        f"selected_song={selected_path} (playback disabled)"
                    )
                else:
                    played_file = player.play_url(str(selected_path))
                    print(
                        f"[MUSIC] emotion={emotion} prompt='{prompt}' "
                        f"selected_song={selected_path} playing={played_file}"
                    )
            last_processed_signature = signature
        except MusicPlaybackError as err:
            print(f"[MUSIC] playback error for emotion '{emotion}': {err}", file=sys.stderr)
        except SunoError as err:
            print(f"[MUSIC] Suno error for emotion '{emotion}': {err}", file=sys.stderr)
        except Exception as err:
            print(
                f"[MUSIC] generation error for emotion '{emotion}': {err}",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run camera + emotion recognition and auto-generate/play Suno music "
            "when emotion changes and is stable for >= N seconds."
        )
    )
    parser.add_argument(
        "--generate",
        type=_parse_bool_arg,
        default=True,
        help=(
            "If true, generate a new song with Suno. "
            "If false, retrieve and play a local song using MuLan+Elasticsearch blend."
        ),
    )
    parser.add_argument(
        "--source-type",
        choices=["local", "url", "file", "pi"],
        default="local",
        help="Camera backend type.",
    )
    parser.add_argument("--index", type=int, help="Local camera index.")
    parser.add_argument("--url", type=str, help="RTSP/HTTP stream URL.")
    parser.add_argument("--file", type=str, help="Video file path.")
    parser.add_argument(
        "--max-probe-index",
        type=int,
        default=8,
        help="Maximum local camera index to probe for interactive selection.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Emotion + Music Orchestrator",
        help="Preview window name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Hugging Face emotion model ID or local model path.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum face detection confidence.",
    )
    parser.add_argument(
        "--inference-fps",
        type=float,
        default=10.0,
        help="Emotion model inference rate.",
    )
    parser.add_argument(
        "--face-padding-ratio",
        type=float,
        default=0.15,
        help="Relative padding around each detected face crop.",
    )
    parser.add_argument(
        "--face-detector-model",
        type=str,
        default=None,
        help="Optional local path to MediaPipe face detector .tflite model.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=1.0,
        help="Required stable duration before triggering music generation.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used to convert emotion to Suno prompt.",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=DEFAULT_OPENAI_MAX_TOKENS,
        help="Max tokens for OpenAI prompt generation (lower is usually faster).",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key override.",
    )
    parser.add_argument(
        "--ffplay-path",
        type=str,
        default=None,
        help="Optional explicit path to ffplay executable.",
    )
    parser.add_argument(
        "--suno-poll-interval",
        type=float,
        default=2.5,
        help="Suno status polling interval in seconds (2-3s recommended).",
    )
    parser.add_argument(
        "--songs-dir",
        type=str,
        default="songs",
        help="Directory containing local songs used in retrieval mode.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=3,
        help="Top-k candidates used when blending MuLan and Elasticsearch scores.",
    )
    parser.add_argument(
        "--es-url",
        type=str,
        default=None,
        help="Elasticsearch URL (defaults to ELASTICSEARCH_URL env/.env).",
    )
    parser.add_argument(
        "--es-index",
        type=str,
        default=None,
        help="Elasticsearch index for retrieval mode (defaults to ELASTICSEARCH_INDEX or 'lyrics').",
    )
    parser.add_argument(
        "--mulan-model-id",
        type=str,
        default=None,
        help="MuLan model id for retrieval mode.",
    )
    parser.add_argument(
        "--retrieval-cache-dir",
        type=str,
        default=".cache/mulan_song_embeddings_main",
        help="Cache directory for local song embeddings in retrieval mode.",
    )
    parser.add_argument(
        "--retrieval-no-cache",
        action="store_true",
        default=False,
        help="Disable embedding cache in retrieval mode.",
    )
    parser.add_argument(
        "--context-interval",
        type=float,
        default=7200,
        help="Interval in seconds between room context captures (default: 10800 = 30 min).",
    )
    parser.add_argument(
        "--enable-voice",
        action="store_true",
        default=True,
        help="Enable voice capture and analysis.",
    )
    parser.add_argument(
        "--disable-voice",
        action="store_true",
        default=False,
        help="Disable voice capture and analysis.",
    )
    parser.add_argument(
        "--voice-chunk-duration",
        type=float,
        default=5.0,
        help="Duration in seconds of each voice processing chunk.",
    )
    parser.add_argument(
        "--voice-silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold below which audio is considered silence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = None
    processor: Optional[EmotionProcessor] = None
    player: Optional[MusicPlayer] = None
    voice_processor: Optional[VoiceProcessor] = None

    openai_api_key = get_openai_api_key(args.openai_api_key)
    es_url = args.es_url or _resolve_setting("ELASTICSEARCH_URL", "http://localhost:9200")
    es_index = args.es_index or _resolve_setting("ELASTICSEARCH_INDEX", "lyrics")
    mulan_model_id = args.mulan_model_id or _resolve_setting(
        "MULAN_MODEL_ID", DEFAULT_MULAN_MODEL_ID
    )

    # Shared OpenAI client for fusion and context shot modules
    openai_client: Optional[OpenAI] = None
    context_shot: Optional[ContextShot] = None
    fusion: Optional[SensorFusion] = None
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        context_shot = ContextShot(
            client=openai_client, interval_s=args.context_interval
        )
        fusion = SensorFusion(client=openai_client)

    # Voice shared state
    latest_voice_observation: Optional[VoiceObservation] = None
    voice_lock = threading.Lock()

    def on_voice_observation(obs: VoiceObservation) -> None:
        nonlocal latest_voice_observation
        with voice_lock:
            latest_voice_observation = obs
        if not obs.is_speech:
            print(
                f"[VOICE] t={obs.timestamp_s:.1f} speech=False rms={obs.energy_rms:.4f}"
            )
            return
        print(
            f"[VOICE] t={obs.timestamp_s:.1f} speech=True rms={obs.energy_rms:.4f} "
            f"prosody={obs.prosody_emotion} vocal_mood={obs.vocal_mood} "
            f"vocal_mood_score={obs.vocal_mood_score:.2f} speech_rate={obs.speech_rate}"
        )
        if obs.transcript:
            print(f"[VOICE] transcript: {obs.transcript}")
        if obs.text_emotion:
            print(
                f"[VOICE] text_emotion: {obs.text_emotion} "
                f"({obs.text_emotion_score:.2f})"
            )
        if obs.topics:
            print(f"[VOICE] topics: {obs.topics}")
        if obs.keywords:
            print(f"[VOICE] keywords: {obs.keywords}")


    detector = StableEmotionChangeDetector(min_stable_seconds=args.stable_seconds)
    emotion_queue: "queue.Queue[_MusicQueueItem]" = queue.Queue()
    try:
        player = MusicPlayer(ffplay_path=args.ffplay_path)
    except MusicPlaybackError as err:
        print(f"[MUSIC] playback disabled: {err}", file=sys.stderr)
        player = None
    # Initialize voice processor if enabled
    voice_enabled = bool(args.enable_voice) and not bool(args.disable_voice)
    if voice_enabled:
        voice_processor = VoiceProcessor(
            chunk_duration_s=args.voice_chunk_duration,
            silence_threshold_rms=args.voice_silence_threshold,
            openai_api_key=openai_api_key,
            music_player=player,
            on_observation=on_voice_observation,
        )
        voice_processor.start()

    worker = threading.Thread(
        target=_music_worker,
        kwargs={
            "emotion_queue": emotion_queue,
            "generate": bool(args.generate),
            "openai_api_key": openai_api_key,
            "openai_model": args.openai_model,
            "openai_max_tokens": args.openai_max_tokens,
            "suno_poll_interval_s": args.suno_poll_interval,
            "player": player,
            "songs_dir": args.songs_dir,
            "es_url": es_url,
            "es_index": es_index,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_model_id": mulan_model_id,
            "retrieval_cache_dir": args.retrieval_cache_dir,
            "retrieval_no_cache": args.retrieval_no_cache,
        },
        daemon=True,
    )
    worker.start()

    last_enqueued_emotion: Optional[str] = None

    def on_observation(observation: EmotionObservation) -> None:
        nonlocal last_enqueued_emotion
        triggered = detector.observe(observation)
        if triggered and triggered != last_enqueued_emotion:
            print(
                f"[EMOTION] stable change detected: {triggered} "
                f"(>= {args.stable_seconds:.1f}s)"
            )

            # Build fusion context if available
            fusion_context: Optional[str] = None
            if fusion is not None:
                # Read latest voice observation thread-safely
                with voice_lock:
                    voice_obs = latest_voice_observation

                voice_kwargs = {}
                if voice_obs is not None and voice_obs.is_speech:
                    voice_kwargs = dict(
                        prosody_emotion=voice_obs.prosody_emotion,
                        vocal_mood=voice_obs.vocal_mood,
                        vocal_mood_score=voice_obs.vocal_mood_score,
                        speech_rate=voice_obs.speech_rate,
                        transcript=voice_obs.transcript,
                        text_emotion=voice_obs.text_emotion,
                        text_emotion_score=voice_obs.text_emotion_score,
                        topics=voice_obs.topics,
                        keywords=voice_obs.keywords,
                    )

                state = SensorState(
                    emotion=triggered,
                    emotion_score=observation.score,
                    face_count=observation.face_count,
                    room_description=(
                        context_shot.last_description if context_shot else None
                    ),
                    timestamp=observation.timestamp_s,
                    **voice_kwargs,
                )
                try:
                    fusion_context = fusion.generate_description(state)
                except Exception as err:
                    print(f"[FUSION] error: {err}", file=sys.stderr)

            # Keep only the newest pending transition to avoid over-querying OpenAI.
            while True:
                try:
                    pending = emotion_queue.get_nowait()
                    if pending == "__STOP__":
                        emotion_queue.put("__STOP__")
                        break
                except queue.Empty:
                    break
            emotion_queue.put((triggered, fusion_context))
            last_enqueued_emotion = triggered

    try:
        source = build_source_from_args(args)
        processor = EmotionProcessor(
            model_name=args.model_name,
            min_detection_confidence=args.min_detection_confidence,
            inference_fps=args.inference_fps,
            face_padding_ratio=args.face_padding_ratio,
            face_detector_model_path=args.face_detector_model,
            on_observation=on_observation,
        )

        source.open()
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        print(f"Streaming from: {source.label}")
        print("Controls: press 'q' or ESC to quit.")

        frame_index = 0
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("Frame read failed; stopping stream.")
                break

            now = time.time()
            packet = FramePacket(
                frame=frame, timestamp_s=now, frame_index=frame_index
            )

            # Periodic room context capture
            if context_shot is not None:
                try:
                    context_shot.maybe_capture(frame, now)
                except Exception as err:
                    print(f"[CONTEXT] error: {err}", file=sys.stderr)

            out_frame = processor.process(packet)
            cv2.imshow(args.window_name, out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            frame_index += 1

    except RuntimeError as err:
        if args.source_type == "local":
            print(build_local_camera_help_text(), file=sys.stderr)
            print(f"\nDetails: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        raise
    finally:
        emotion_queue.put("__STOP__")
        worker.join(timeout=1.0)
        if voice_processor is not None:
            voice_processor.close()
        if processor is not None:
            processor.close()
        if source is not None:
            source.release()
        if player is not None:
            player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
