from __future__ import annotations

import queue
import re
import sys
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque, Dict, Iterator, Optional, Union

import cv2
from openai import OpenAI

from .config import Settings


_MusicQueueItem = Union[str, tuple]
_TAG_LINE_RE = re.compile(r"^\[(?P<tag>[A-Za-z0-9_]+)\]\s*(?P<msg>.*)$")
_DEFAULT_TAG = "SYSTEM"


class _TaggedOutputRouter:
    def __init__(self, runtime: "DashboardRuntime", fallback_tag: str = _DEFAULT_TAG):
        self._runtime = runtime
        self._fallback_tag = fallback_tag
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit(line.rstrip("\r"))
        return len(data)

    def flush(self) -> None:
        with self._lock:
            if self._buf:
                self._emit(self._buf.rstrip("\r"))
                self._buf = ""

    def _emit(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        match = _TAG_LINE_RE.match(stripped)
        if match:
            tag = str(match.group("tag")).upper()
            msg = str(match.group("msg")).strip()
            self._runtime._log(tag, msg)
            return
        self._runtime._log(self._fallback_tag, stripped)


class DashboardRuntime:
    """Runs a headless main.py-style pipeline and exposes frames + categorized logs."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._logs: Dict[str, Deque[str]] = {}
        self._max_logs_per_tag = 120

    def _repo_root(self) -> Path:
        # server/app/dashboard_runtime.py -> server/app -> server -> repo root
        return Path(__file__).resolve().parents[2]

    def _import_root_runtime(self):
        root = str(self._repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from camera import FramePacket, OpenCVCaptureSource  # type: ignore
        from facial_emotions import EmotionObservation, EmotionProcessor  # type: ignore
        from context_shot import ContextShot  # type: ignore
        from fusion import SensorFusion, SensorState  # type: ignore
        from main import (  # type: ignore
            DEFAULT_OPENAI_MAX_TOKENS,
            StableEmotionChangeDetector,
            _music_worker,
            _read_env_file,
        )
        from es_index import ESIndexError, upload_lyrics_folder_to_index  # type: ignore
        from play_music import MusicPlaybackError, MusicPlayer  # type: ignore
        from voice import VoiceObservation, VoiceProcessor  # type: ignore

        return {
            "FramePacket": FramePacket,
            "OpenCVCaptureSource": OpenCVCaptureSource,
            "EmotionObservation": EmotionObservation,
            "EmotionProcessor": EmotionProcessor,
            "ContextShot": ContextShot,
            "SensorFusion": SensorFusion,
            "SensorState": SensorState,
            "DEFAULT_OPENAI_MAX_TOKENS": DEFAULT_OPENAI_MAX_TOKENS,
            "StableEmotionChangeDetector": StableEmotionChangeDetector,
            "_music_worker": _music_worker,
            "_read_env_file": _read_env_file,
            "ESIndexError": ESIndexError,
            "upload_lyrics_folder_to_index": upload_lyrics_folder_to_index,
            "MusicPlaybackError": MusicPlaybackError,
            "MusicPlayer": MusicPlayer,
            "VoiceObservation": VoiceObservation,
            "VoiceProcessor": VoiceProcessor,
        }

    def _log(self, tag: str, message: str) -> None:
        key = tag.upper()
        with self._lock:
            if key not in self._logs:
                self._logs[key] = deque(maxlen=self._max_logs_per_tag)
            self._logs[key].appendleft(message)

    def _set_frame(self, frame_bgr) -> None:
        ok, enc = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = enc.tobytes()

    def start(self, *, generate: bool = True) -> Dict[str, object]:
        with self._lock:
            if self._running:
                return {"running": True, "message": "already running"}
            self._running = True
            self._logs.clear()
            self._latest_jpeg = None
            self._stop_event.clear()
        self._log("SYSTEM", f"starting dashboard runtime (generate={generate})")
        self._thread = threading.Thread(
            target=self._run_pipeline,
            kwargs={"generate": generate},
            daemon=True,
        )
        self._thread.start()
        return {"running": True, "message": "started"}

    def stop(self) -> Dict[str, object]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._running = False
        return {"running": False, "message": "stopped"}

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def logs_snapshot(self) -> Dict[str, object]:
        with self._lock:
            out = {k: list(v) for k, v in self._logs.items()}
            out["running"] = self._running
        return out

    @contextmanager
    def _capture_tagged_output(self) -> Iterator[None]:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        router_stdout = _TaggedOutputRouter(self, fallback_tag=_DEFAULT_TAG)
        router_stderr = _TaggedOutputRouter(self, fallback_tag="SYSTEM")
        sys.stdout = router_stdout  # type: ignore[assignment]
        sys.stderr = router_stderr  # type: ignore[assignment]
        try:
            yield
        finally:
            try:
                router_stdout.flush()
            except Exception:
                pass
            try:
                router_stderr.flush()
            except Exception:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def mjpeg_stream(self, fps: float = 15.0) -> Iterator[bytes]:
        delay_s = 1.0 / max(1.0, fps)
        while True:
            with self._lock:
                payload = self._latest_jpeg
                running = self._running
            if payload is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
            if not running and payload is None:
                break
            time.sleep(delay_s)

    def _run_pipeline(self, *, generate: bool) -> None:
        self._log("SYSTEM", "loading runtime modules...")
        with self._capture_tagged_output():
            rt = self._import_root_runtime()
            self._log("SYSTEM", "runtime modules loaded")
            FramePacket = rt["FramePacket"]
            OpenCVCaptureSource = rt["OpenCVCaptureSource"]
            EmotionProcessor = rt["EmotionProcessor"]
            ContextShot = rt["ContextShot"]
            SensorFusion = rt["SensorFusion"]
            SensorState = rt["SensorState"]
            StableEmotionChangeDetector = rt["StableEmotionChangeDetector"]
            _music_worker = rt["_music_worker"]
            _read_env_file = rt["_read_env_file"]
            DEFAULT_OPENAI_MAX_TOKENS = rt["DEFAULT_OPENAI_MAX_TOKENS"]
            ESIndexError = rt["ESIndexError"]
            upload_lyrics_folder_to_index = rt["upload_lyrics_folder_to_index"]
            MusicPlayer = rt["MusicPlayer"]
            MusicPlaybackError = rt["MusicPlaybackError"]
            VoiceProcessor = rt["VoiceProcessor"]

            source = None
            processor = None
            player = None
            voice_processor = None
            worker = None
            emotion_queue: "queue.Queue[_MusicQueueItem]" = queue.Queue()

            # Env/settings shared with main defaults.
            # Read repo-root .env explicitly because server cwd is often `server/`.
            repo_env = _read_env_file(self._repo_root() / ".env")
            cwd_env = _read_env_file()
            openai_api_key = (
                os.environ.get("OPENAI_API_KEY")
                or repo_env.get("OPENAI_API_KEY")
                or cwd_env.get("OPENAI_API_KEY")
            )
            es_url = (
                os.environ.get("ELASTICSEARCH_URL")
                or repo_env.get("ELASTICSEARCH_URL")
                or cwd_env.get("ELASTICSEARCH_URL")
                or "http://localhost:9200"
            )
            es_index = (
                os.environ.get("ELASTICSEARCH_INDEX")
                or repo_env.get("ELASTICSEARCH_INDEX")
                or cwd_env.get("ELASTICSEARCH_INDEX")
                or "lyrics"
            )
            mulan_model_id = (
                os.environ.get("MULAN_MODEL_ID")
                or repo_env.get("MULAN_MODEL_ID")
                or cwd_env.get("MULAN_MODEL_ID")
                or "OpenMuQ/MuQ-MuLan-large"
            )
            ffplay_path = (
                os.environ.get("FFPLAY_PATH")
                or repo_env.get("FFPLAY_PATH")
                or cwd_env.get("FFPLAY_PATH")
            )
            songs_dir_path = self._repo_root() / "songs"
            retrieval_cache_dir_path = (
                self._repo_root() / ".cache" / "mulan_song_embeddings_main"
            )
            lyrics_dir_path = self._repo_root() / "lyrics"

            self._log("SYSTEM", "Dashboard runtime started")
            self._log("RETRIEVAL", f"songs_dir='{songs_dir_path}'")
            self._log("RETRIEVAL", "syncing Elasticsearch lyrics index...")
            try:
                uploaded, deleted = upload_lyrics_folder_to_index(
                    es_url,
                    index_name="lyrics",
                    lyrics_dir=str(lyrics_dir_path),
                    sync_delete=True,
                )
                self._log(
                    "RETRIEVAL",
                    f"lyrics index synced: uploaded_or_updated={uploaded} deleted_stale={deleted}",
                )
            except ESIndexError as err:
                self._log("RETRIEVAL", f"lyrics index sync failed: {err}")
            except Exception as err:
                self._log("RETRIEVAL", f"lyrics index sync warning: {err}")

            openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
            context_shot = (
                ContextShot(client=openai_client, interval_s=600)
                if openai_client
                else None
            )
            fusion = SensorFusion(client=openai_client) if openai_client else None
            if openai_client is None:
                self._log("CONTEXT", "disabled: missing OPENAI_API_KEY")
                self._log("FUSION", "disabled: missing OPENAI_API_KEY")
            else:
                self._log(
                    "CONTEXT",
                    "enabled: capturing scene every 10 minutes (and once at startup)",
                )
                self._log(
                    "FUSION",
                    "enabled: running once at startup, then on stable emotion changes",
                )
                self._log("FUSION", "waiting for first emotion observation")

            latest_voice_observation = None
            voice_lock = threading.Lock()

            def on_voice_observation(obs):
                nonlocal latest_voice_observation
                with voice_lock:
                    latest_voice_observation = obs
                if obs.is_speech:
                    self._log(
                        "VOICE",
                        f"speech=True rms={obs.energy_rms:.4f} mood={obs.vocal_mood} "
                        f"text_emotion={obs.text_emotion} transcript={obs.transcript or ''}",
                    )
                else:
                    self._log("VOICE", f"speech=False rms={obs.energy_rms:.4f}")

            detector = StableEmotionChangeDetector(min_stable_seconds=1.0)
            last_enqueued_emotion: Optional[str] = None
            bootstrap_fusion_done = False

            def on_observation(observation):
                nonlocal last_enqueued_emotion, bootstrap_fusion_done
                triggered = detector.observe(observation)

                fusion_context = None
                if fusion is not None:
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

                    effective_emotion = triggered or observation.label
                    state = SensorState(
                        emotion=effective_emotion,
                        emotion_score=observation.score,
                        face_count=observation.face_count,
                        room_description=(
                            context_shot.last_description if context_shot else None
                        ),
                        timestamp=observation.timestamp_s,
                        **voice_kwargs,
                    )
                    if not bootstrap_fusion_done:
                        try:
                            fusion_context = fusion.generate_description(state)
                            self._log("FUSION", f"startup: {fusion_context}")
                        except Exception as err:
                            self._log("FUSION", f"startup error: {err}")
                        finally:
                            bootstrap_fusion_done = True

                if not triggered or triggered == last_enqueued_emotion:
                    return
                self._log("EMOTION", f"stable change detected: {triggered}")

                if fusion is not None:
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
                        self._log("FUSION", fusion_context)
                    except Exception as err:
                        self._log("FUSION", f"error: {err}")

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
                try:
                    player = MusicPlayer(ffplay_path=ffplay_path)
                except MusicPlaybackError as err:
                    self._log("MUSIC", f"playback disabled: {err}")
                    player = None

                voice_processor = VoiceProcessor(
                    chunk_duration_s=5.0,
                    silence_threshold_rms=0.01,
                    openai_api_key=openai_api_key,
                    music_player=player,
                    on_observation=on_voice_observation,
                )
                voice_processor.start()
                self._log("VOICE", "voice processor started")

                worker = threading.Thread(
                    target=_music_worker,
                    kwargs={
                        "emotion_queue": emotion_queue,
                        "generate": bool(generate),
                        "openai_api_key": openai_api_key,
                        "openai_model": "gpt-4o-mini",
                        "openai_max_tokens": DEFAULT_OPENAI_MAX_TOKENS,
                        "suno_poll_interval_s": 2.5,
                        "player": player,
                        "songs_dir": str(songs_dir_path),
                        "es_url": es_url,
                        "es_index": es_index,
                        "retrieval_top_k": 3,
                        "retrieval_model_id": mulan_model_id,
                        "retrieval_cache_dir": str(retrieval_cache_dir_path),
                        "retrieval_no_cache": False,
                    },
                    daemon=True,
                )
                worker.start()
                self._log("MUSIC", f"music worker started generate={generate}")

                source = OpenCVCaptureSource(0)
                source.open()
                self._log("SYSTEM", "camera source opened: local index 0")
                processor = EmotionProcessor(
                    model_name=None,
                    min_detection_confidence=0.5,
                    inference_fps=10.0,
                    face_padding_ratio=0.15,
                    on_observation=on_observation,
                )
                frame_index = 0
                while not self._stop_event.is_set():
                    ok, frame = source.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    now = time.time()
                    if context_shot is not None:
                        try:
                            desc = context_shot.maybe_capture(frame, now)
                            if desc:
                                self._log("CONTEXT", desc)
                        except Exception as err:
                            self._log("CONTEXT", f"error: {err}")
                    packet = FramePacket(
                        frame=frame, timestamp_s=now, frame_index=frame_index
                    )
                    out = processor.process(packet)
                    self._set_frame(out)
                    frame_index += 1
            except Exception as err:
                self._log("SYSTEM", f"runtime error: {err}")
            finally:
                emotion_queue.put("__STOP__")
                if worker is not None:
                    worker.join(timeout=1.0)
                if voice_processor is not None:
                    voice_processor.close()
                if processor is not None:
                    try:
                        processor.close()
                    except Exception:
                        pass
                if source is not None:
                    try:
                        source.release()
                    except Exception:
                        pass
                if player is not None:
                    try:
                        player.close()
                    except Exception:
                        pass
                with self._lock:
                    self._running = False
                self._log("SYSTEM", "Dashboard runtime stopped")
