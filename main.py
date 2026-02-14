from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2

from camera import FramePacket, build_local_camera_help_text, build_source_from_args
from emotions import EmotionObservation, EmotionProcessor
from suno import SunoError, generate_from_prompt


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


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


def get_openai_api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    file_key = _read_env_file().get("OPENAI_API_KEY")
    if file_key:
        return file_key
    raise RuntimeError("OPENAI_API_KEY not found (arg/env/.env).")


def generate_suno_prompt_for_emotion(
    emotion: str,
    *,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    timeout_s: float = 30.0,
) -> str:
    system_text = (
        "You generate concise music-generation prompts for Suno.\n"
        "Output only a single prompt line (no quotes, no markdown).\n"
        "Length: 14-24 words.\n"
        "Include mood, tempo, instrumentation, and production style."
    )
    user_text = (
        "Detected emotion: "
        f"{emotion}\n"
        "Create one prompt for a song that matches this feeling."
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "max_output_tokens": 120,
    }
    req = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP error {err.code}: {detail}") from err
    except URLError as err:
        raise RuntimeError(f"OpenAI connection error: {err}") from err

    text = (data.get("output_text") or "").strip()
    if text:
        return text

    output = data.get("output") or []
    for item in output:
        for content in item.get("content", []):
            candidate = (content.get("text") or "").strip()
            if candidate:
                return candidate
    raise RuntimeError("OpenAI response did not include output text.")


@dataclass
class StableEmotionChangeDetector:
    min_stable_seconds: float = 3.0
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

        stable_for = now - self.candidate_since_s
        if stable_for < self.min_stable_seconds:
            return None

        if emotion == self.last_triggered_emotion:
            return None

        self.last_triggered_emotion = emotion
        return emotion


def _music_worker(
    emotion_queue: "queue.Queue[str]",
    *,
    openai_api_key: str,
    openai_model: str,
    suno_wait: bool,
) -> None:
    while True:
        emotion = emotion_queue.get()
        if emotion == "__STOP__":
            return
        try:
            prompt = generate_suno_prompt_for_emotion(
                emotion, api_key=openai_api_key, model=openai_model
            )
            result = generate_from_prompt(prompt, wait=suno_wait)
            print(f"[MUSIC] emotion={emotion} prompt={prompt}")
            if isinstance(result, str):
                print(f"[MUSIC] created task_id={result}")
            else:
                print(f"[MUSIC] task_id={result.task_id} tracks={len(result.tracks)}")
        except SunoError as err:
            print(f"[MUSIC] Suno error for emotion '{emotion}': {err}", file=sys.stderr)
        except Exception as err:
            print(
                f"[MUSIC] Generation error for emotion '{emotion}': {err}",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run camera + emotion recognition and trigger song generation when "
            "emotion changes and is stable for >= 3 seconds."
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
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model used to convert emotion to Suno prompt.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key override.",
    )
    parser.add_argument(
        "--suno-wait",
        action="store_true",
        help="Wait for Suno completion instead of returning only task IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = None
    processor: Optional[EmotionProcessor] = None
    openai_api_key = get_openai_api_key(args.openai_api_key)

    detector = StableEmotionChangeDetector(min_stable_seconds=args.stable_seconds)
    emotion_queue: "queue.Queue[str]" = queue.Queue()
    worker = threading.Thread(
        target=_music_worker,
        kwargs={
            "emotion_queue": emotion_queue,
            "openai_api_key": openai_api_key,
            "openai_model": args.openai_model,
            "suno_wait": args.suno_wait,
        },
        daemon=True,
    )
    worker.start()

    def on_observation(observation: EmotionObservation) -> None:
        triggered = detector.observe(observation)
        if triggered:
            print(
                f"[EMOTION] stable change detected: {triggered} "
                f"(>= {args.stable_seconds:.1f}s)"
            )
            emotion_queue.put(triggered)

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

            packet = FramePacket(
                frame=frame, timestamp_s=time.time(), frame_index=frame_index
            )
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
