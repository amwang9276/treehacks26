from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
from openai import OpenAI

from camera import FramePacket, build_local_camera_help_text, build_source_from_args
from context_shot import ContextShot
from emotions import EmotionObservation, EmotionProcessor
from fusion import SensorFusion, SensorState
from music import MusicPlaybackError, MusicPlayer
from suno import SunoError, SunoGenerationResult, generate_from_prompt


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


def _music_worker(
    emotion_queue: "queue.Queue[_MusicQueueItem]",
    *,
    openai_api_key: Optional[str],
    openai_model: str,
    openai_max_tokens: int,
    suno_poll_interval_s: float,
    player: Optional[MusicPlayer],
) -> None:
    last_processed_emotion: Optional[str] = None
    client: Optional[OpenAI] = None
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)

    while True:
        item = emotion_queue.get()
        if item == "__STOP__":
            return
        if isinstance(item, tuple):
            emotion, context = item
        else:
            emotion, context = item, None
        if emotion == last_processed_emotion:
            continue
        try:
            if client is None:
                raise RuntimeError("OPENAI_API_KEY is required to generate prompts.")
            prompt = generate_suno_prompt_for_emotion(
                emotion,
                client=client,
                model=openai_model,
                max_tokens=openai_max_tokens,
                context=context,
            )
            prompt_source = "openai"

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
                    f"[MUSIC] emotion={emotion} prompt_source={prompt_source} "
                    f"task_id={result.task_id} playable_url={url} (playback disabled)"
                )
            else:
                played_file = player.play_url(url)
                print(
                    f"[MUSIC] emotion={emotion} prompt_source={prompt_source} "
                    f"task_id={result.task_id} playing={played_file}"
                )
            last_processed_emotion = emotion
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
        default=3.0,
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
        "--context-interval",
        type=float,
        default=1800,
        help="Interval in seconds between room context captures (default: 1800 = 30 min).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = None
    processor: Optional[EmotionProcessor] = None
    player: Optional[MusicPlayer] = None

    openai_api_key = get_openai_api_key(args.openai_api_key)

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

    detector = StableEmotionChangeDetector(min_stable_seconds=args.stable_seconds)
    emotion_queue: "queue.Queue[_MusicQueueItem]" = queue.Queue()
    try:
        player = MusicPlayer(ffplay_path=args.ffplay_path)
    except MusicPlaybackError as err:
        print(f"[MUSIC] playback disabled: {err}", file=sys.stderr)
        player = None
    worker = threading.Thread(
        target=_music_worker,
        kwargs={
            "emotion_queue": emotion_queue,
            "openai_api_key": openai_api_key,
            "openai_model": args.openai_model,
            "openai_max_tokens": args.openai_max_tokens,
            "suno_poll_interval_s": args.suno_poll_interval,
            "player": player,
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
                state = SensorState(
                    emotion=triggered,
                    emotion_score=observation.score,
                    face_count=observation.face_count,
                    current_track_info=None,
                    room_description=(
                        context_shot.last_description if context_shot else None
                    ),
                    timestamp=observation.timestamp_s,
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
        if processor is not None:
            processor.close()
        if source is not None:
            source.release()
        if player is not None:
            player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
