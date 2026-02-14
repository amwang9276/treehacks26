"""
Modular camera streaming utility.

Features:
- Popup preview window via OpenCV.
- Camera source abstraction for local webcams, RTSP/HTTP streams, and files.
- Interactive camera selection for local devices.
- Frame processing hook to plug in CV models (e.g., emotion recognition).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2


@dataclass(frozen=True)
class FramePacket:
    """Container passed through the CV pipeline."""

    frame: np.ndarray
    timestamp_s: float
    frame_index: int


class FrameProcessor(ABC):
    """Base class for CV processing modules."""

    @abstractmethod
    def process(self, packet: FramePacket) -> np.ndarray:
        """Return an output frame for visualization."""


class IdentityProcessor(FrameProcessor):
    """Default no-op processor."""

    def process(self, packet: FramePacket) -> np.ndarray:
        return packet.frame


class CameraSource(ABC):
    """Base camera input interface."""

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        pass


class OpenCVCaptureSource(CameraSource):
    """OpenCV-backed source from index, URL, or file path."""

    def __init__(self, source: int | str, api_preference: int = cv2.CAP_ANY):
        self._source = source
        self._api_preference = api_preference
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def label(self) -> str:
        return f"OpenCVSource({self._source})"

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self._source, self._api_preference)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._source}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Camera source not opened.")
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RaspberryPiCameraSource(CameraSource):
    """
    Placeholder source for Raspberry Pi camera integration.

    You can later wire this to Picamera2/libcamera and keep the same interface.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def label(self) -> str:
        return "RaspberryPiCameraSource(unimplemented)"

    def open(self) -> None:
        raise NotImplementedError(
            "RaspberryPiCameraSource is a stub. Replace with Picamera2 integration."
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


def is_wsl() -> bool:
    """Best-effort WSL detection."""
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


@contextmanager
def temporary_opencv_log_level(level: int):
    """Set OpenCV log level temporarily when supported."""
    prev_level = None
    can_set = hasattr(cv2, "setLogLevel")
    can_get = hasattr(cv2, "getLogLevel")
    try:
        if can_set and can_get:
            prev_level = cv2.getLogLevel()
            cv2.setLogLevel(level)
        elif can_set:
            cv2.setLogLevel(level)
        yield
    finally:
        if prev_level is not None and can_set:
            cv2.setLogLevel(prev_level)


def probe_local_cameras(max_index: int = 8) -> List[int]:
    """Probe webcam indices that can be opened."""
    available: List[int] = []
    with temporary_opencv_log_level(getattr(cv2, "LOG_LEVEL_ERROR", 2)):
        for idx in range(max_index + 1):
            # Prefer V4L2 probing on Linux to avoid noisy FFMPEG fallback logs.
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
            cap.release()
    return available


def choose_camera_index(max_index: int = 8) -> int:
    """
    Prompt the user to select a local camera index.
    Falls back to first available index on bad input.
    """
    available = probe_local_cameras(max_index=max_index)
    if not available:
        raise RuntimeError("No local camera devices were found.")

    print("Available camera indices:")
    for idx in available:
        print(f"  [{idx}]")

    try:
        raw = input(f"Select camera index [{available[0]}]: ").strip()
        if not raw:
            return available[0]
        chosen = int(raw)
        if chosen not in available:
            print(f"Index {chosen} was not probed successfully, using {available[0]}.")
            return available[0]
        return chosen
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"Invalid input. Using {available[0]}.")
        return available[0]


def build_local_camera_help_text() -> str:
    base_lines = [
        "Could not open a local camera.",
        "",
        "Try one of these:",
        "  1) Check camera permissions and whether another app is using it.",
        "  2) Pass an explicit index: python camera.py --source-type local --index 0",
        "  3) Use a network stream: python camera.py --source-type url --url <rtsp_or_http_url>",
        "  4) Use a video file: python camera.py --source-type file --file <path>",
    ]
    if is_wsl():
        base_lines.extend(
            [
                "",
                "WSL note:",
                "  Local USB cameras are often not exposed to WSL by default.",
                "  Either run this script in native Windows Python, or attach the USB camera to WSL",
                "  (e.g., via usbipd-win), or stream from another source via --source-type url.",
            ]
        )
    return "\n".join(base_lines)


class CameraStreamer:
    """Owns the stream loop and popup window."""

    def __init__(
        self,
        source: CameraSource,
        processor: Optional[FrameProcessor] = None,
        window_name: str = "Camera Stream",
    ):
        self.source = source
        self.processor = processor or IdentityProcessor()
        self.window_name = window_name

    def run(self) -> None:
        frame_index = 0
        self.source.open()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print(f"Streaming from: {self.source.label}")
        print("Controls: press 'q' or ESC to quit.")

        try:
            while True:
                ok, frame = self.source.read()
                if not ok or frame is None:
                    print("Frame read failed; stopping stream.")
                    break

                packet = FramePacket(
                    frame=frame,
                    timestamp_s=time.time(),
                    frame_index=frame_index,
                )
                out_frame = self.processor.process(packet)
                cv2.imshow(self.window_name, out_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                frame_index += 1
        finally:
            self.source.release()
            cv2.destroyWindow(self.window_name)


def build_source_from_args(args: argparse.Namespace) -> CameraSource:
    if args.source_type == "local":
        if args.index is None:
            index = choose_camera_index(max_index=args.max_probe_index)
        else:
            index = args.index
        return OpenCVCaptureSource(index)
    if args.source_type == "url":
        if not args.url:
            raise ValueError("--url is required when --source-type=url")
        return OpenCVCaptureSource(args.url)
    if args.source_type == "file":
        if not args.file:
            raise ValueError("--file is required when --source-type=file")
        return OpenCVCaptureSource(args.file)
    if args.source_type == "pi":
        return RaspberryPiCameraSource()
    raise ValueError(f"Unsupported source type: {args.source_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a camera popup and stream frames with a modular CV pipeline."
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
        "--window-name", type=str, default="Camera Stream", help="Preview window name."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        source = build_source_from_args(args)
        streamer = CameraStreamer(
            source=source,
            processor=IdentityProcessor(),
            window_name=args.window_name,
        )
        streamer.run()
    except RuntimeError as err:
        if args.source_type == "local":
            print(build_local_camera_help_text(), file=sys.stderr)
            print(f"\nDetails: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        raise


if __name__ == "__main__":
    main()
