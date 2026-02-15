"""
Emotion recognition stream built on top of camera.py abstractions.

Design goal:
- Keep camera ingestion concerns in camera.py.
- emotions.py only implements CV processing and runtime options.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from camera import (
    CameraStreamer,
    FramePacket,
    FrameProcessor,
    build_local_camera_help_text,
    build_source_from_args,
)

try:
    import mediapipe as mp
except ImportError as err:
    raise ImportError(
        "mediapipe is required for face detection. Install with: pip install mediapipe"
    ) from err


@dataclass(frozen=True)
class FacePrediction:
    box: Tuple[int, int, int, int]
    label: str
    score: float


@dataclass(frozen=True)
class EmotionObservation:
    timestamp_s: float
    label: Optional[str]
    score: float
    face_count: int


MP_FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
DEFAULT_EMOTION_MODEL_CANDIDATES = [
    "trpakov/vit-face-expression",
    "dima806/facial_emotions_image_detection",
]


class EmotionProcessor(FrameProcessor):
    """MediaPipe face detection + FER+ MobileNetV2 emotion classification."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        inference_fps: float = 10.0,
        face_padding_ratio: float = 0.15,
        face_detector_model_path: Optional[str] = None,
        on_observation: Optional[Callable[[EmotionObservation], None]] = None,
    ) -> None:
        self.inference_period_s = 1.0 / max(inference_fps, 1.0)
        self.face_padding_ratio = max(0.0, face_padding_ratio)
        self.last_infer_ts: float = -1.0
        self.last_predictions: List[FacePrediction] = []
        self.last_observation: EmotionObservation = EmotionObservation(
            timestamp_s=0.0, label=None, score=0.0, face_count=0
        )
        self.on_observation = on_observation
        self._face_detector_mode: str = ""
        self.detector = self._init_face_detector(
            min_detection_confidence=min_detection_confidence,
            face_detector_model_path=face_detector_model_path,
        )

        self.device = torch.device("cpu")
        (
            self.processor,
            self.model,
            self.loaded_model_name,
        ) = self._load_emotion_model(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def _load_emotion_model(
        self, requested_model_name: Optional[str]
    ) -> Tuple[AutoImageProcessor, AutoModelForImageClassification, str]:
        if requested_model_name:
            candidates = [requested_model_name]
        else:
            candidates = DEFAULT_EMOTION_MODEL_CANDIDATES

        last_err: Optional[Exception] = None
        for candidate in candidates:
            try:
                processor = AutoImageProcessor.from_pretrained(candidate)
                model = AutoModelForImageClassification.from_pretrained(candidate)
                print(f"Loaded emotion model: {candidate}")
                return processor, model, candidate
            except Exception as err:
                print(f"Could not load model '{candidate}': {err}", file=sys.stderr)
                last_err = err

        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
        proxy_state = {k: os.environ.get(k) for k in proxy_keys if os.environ.get(k)}
        raise RuntimeError(
            "Failed to load any emotion model from Hugging Face. "
            f"Tried: {', '.join(candidates)}. "
            f"Proxy env: {proxy_state if proxy_state else 'not set'}. "
            "If you're offline, pre-download a model and pass --model-name <local_path>."
        ) from last_err

    def _resolve_face_detector_model_path(
        self, face_detector_model_path: Optional[str]
    ) -> str:
        if face_detector_model_path:
            given = Path(face_detector_model_path).expanduser().resolve()
            if not given.exists():
                raise FileNotFoundError(
                    f"MediaPipe face detector model file not found: {given}"
                )
            return str(given)

        cache_path = (
            Path(__file__).resolve().parent
            / ".cache"
            / "mediapipe"
            / "blaze_face_short_range.tflite"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return str(cache_path)

        try:
            urllib.request.urlretrieve(MP_FACE_DETECTOR_MODEL_URL, str(cache_path))
            return str(cache_path)
        except Exception as err:
            raise RuntimeError(
                "Unable to download MediaPipe face detector model. "
                "Download it manually and pass --face-detector-model <path>. "
                f"URL: {MP_FACE_DETECTOR_MODEL_URL}"
            ) from err

    def _init_face_detector(
        self,
        min_detection_confidence: float,
        face_detector_model_path: Optional[str],
    ):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self._face_detector_mode = "solutions"
            return mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_detection_confidence,
            )

        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import face_detector as mp_face_detector
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
        except Exception as err:
            raise RuntimeError(
                "Your mediapipe build does not expose a compatible face detector API."
            ) from err

        model_path = self._resolve_face_detector_model_path(face_detector_model_path)
        options = mp_face_detector.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._face_detector_mode = "tasks"
        return mp_face_detector.FaceDetector.create_from_options(options)

    def _clip_box(
        self, xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        return (
            max(0, min(width - 1, xmin)),
            max(0, min(height - 1, ymin)),
            max(0, min(width, xmax)),
            max(0, min(height, ymax)),
        )

    def _detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes: List[Tuple[int, int, int, int]] = []

        if self._face_detector_mode == "solutions":
            result = self.detector.process(rgb)
            detections = result.detections if result and result.detections else []
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                pad_x = int(bw * self.face_padding_ratio)
                pad_y = int(bh * self.face_padding_ratio)
                xmin, ymin = x - pad_x, y - pad_y
                xmax, ymax = x + bw + pad_x, y + bh + pad_y
                box = self._clip_box(xmin, ymin, xmax, ymax, width=w, height=h)
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
            return boxes

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        detections = result.detections if result and result.detections else []
        for det in detections:
            bbox = det.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)

            pad_x = int(bw * self.face_padding_ratio)
            pad_y = int(bh * self.face_padding_ratio)
            xmin, ymin = x - pad_x, y - pad_y
            xmax, ymax = x + bw + pad_x, y + bh + pad_y

            box = self._clip_box(xmin, ymin, xmax, ymax, width=w, height=h)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes

    def _predict_emotion(self, face_rgb: np.ndarray) -> Tuple[str, float]:
        pil_image = Image.fromarray(face_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            score = float(probs[pred_idx].item())
        return str(self.id2label[pred_idx]), score

    def _run_inference(
        self, frame_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> List[FacePrediction]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictions: List[FacePrediction] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            label, score = self._predict_emotion(face)
            predictions.append(FacePrediction(box=box, label=label, score=score))
        return predictions

    def process(self, packet: FramePacket) -> np.ndarray:
        frame = packet.frame.copy()
        boxes = self._detect_faces(frame)

        should_infer = (
            self.last_infer_ts < 0
            or (packet.timestamp_s - self.last_infer_ts) >= self.inference_period_s
        )
        if not boxes:
            self.last_predictions = []
        elif should_infer:
            self.last_predictions = self._run_inference(frame, boxes)
            self.last_infer_ts = packet.timestamp_s

        if self.last_predictions:
            best = max(self.last_predictions, key=lambda pred: pred.score)
            observation = EmotionObservation(
                timestamp_s=packet.timestamp_s,
                label=best.label,
                score=best.score,
                face_count=len(self.last_predictions),
            )
        else:
            observation = EmotionObservation(
                timestamp_s=packet.timestamp_s, label=None, score=0.0, face_count=0
            )
        self.last_observation = observation
        if self.on_observation is not None:
            self.on_observation(observation)

        for pred in self.last_predictions:
            x1, y1, x2, y2 = pred.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 60), 2)
            text = f"{pred.label} ({pred.score:.2f})"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 60),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Emotion inference: {1.0 / self.inference_period_s:.0f} FPS (CPU)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        self.detector.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emotion recognition stream using camera.py source abstraction."
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
        default="Emotion Stream",
        help="Preview window name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Hugging Face model ID or local model path. If omitted, tries built-in fallbacks.",
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
        help="Optional local path to a MediaPipe face detector .tflite model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor: Optional[EmotionProcessor] = None
    try:
        source = build_source_from_args(args)
        processor = EmotionProcessor(
            model_name=args.model_name,
            min_detection_confidence=args.min_detection_confidence,
            inference_fps=args.inference_fps,
            face_padding_ratio=args.face_padding_ratio,
            face_detector_model_path=args.face_detector_model,
        )
        streamer = CameraStreamer(
            source=source,
            processor=processor,
            window_name=args.window_name,
        )
        streamer.run()
    except RuntimeError as err:
        if args.source_type == "local":
            print(build_local_camera_help_text(), file=sys.stderr)
            print(f"\nDetails: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        raise
    finally:
        if processor is not None:
            processor.close()


if __name__ == "__main__":
    main()
