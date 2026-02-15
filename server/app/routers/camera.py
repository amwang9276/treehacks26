from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse


router = APIRouter(tags=["camera"])


def _repo_root() -> Path:
    # server/app/routers/camera.py -> server/app/routers -> server/app -> server -> repo root
    return Path(__file__).resolve().parents[3]


def _import_emotion_processor():
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from camera import FramePacket  # type: ignore
    from facial_emotions import EmotionProcessor  # type: ignore

    return FramePacket, EmotionProcessor


def _mjpeg_frames(index: int, fps: float, overlay: bool) -> Iterator[bytes]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    frame_packet_cls = None
    emotion_processor = None
    if overlay:
        try:
            FramePacket, EmotionProcessor = _import_emotion_processor()
            frame_packet_cls = FramePacket
            emotion_processor = EmotionProcessor(
                model_name=None,
                min_detection_confidence=0.5,
                inference_fps=min(10.0, max(2.0, fps)),
                face_padding_ratio=0.15,
            )
        except Exception as err:
            # Keep stream alive with raw frames if emotion stack fails.
            emotion_processor = None
            frame_packet_cls = None
            print(f"[CAMERA] emotion overlay disabled: {err}")

    delay_s = 1.0 / max(1.0, fps)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(delay_s)
                continue

            if emotion_processor is not None and frame_packet_cls is not None:
                try:
                    packet = frame_packet_cls(
                        frame=frame,
                        timestamp_s=time.time(),
                        frame_index=frame_index,
                    )
                    frame = emotion_processor.process(packet)
                except Exception as err:
                    cv2.putText(
                        frame,
                        f"Emotion overlay error: {err}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(delay_s)
                continue
            payload = encoded.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
                + payload
                + b"\r\n"
            )
            frame_index += 1
            time.sleep(delay_s)
    finally:
        if emotion_processor is not None:
            try:
                emotion_processor.close()
            except Exception:
                pass
        cap.release()


@router.get("/camera/stream")
def camera_stream(
    index: int = Query(default=0, ge=0),
    fps: float = Query(default=15.0, ge=1.0, le=30.0),
    overlay: bool = Query(default=True),
) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_frames(index=index, fps=fps, overlay=overlay),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
