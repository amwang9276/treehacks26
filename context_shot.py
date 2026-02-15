"""Periodic room photo capture + GPT-4o vision scene description."""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI


class ContextShot:
    def __init__(
        self,
        client: OpenAI,
        interval_s: float = 1800,
        model: str = "gpt-4o",
    ) -> None:
        self.client = client
        self.interval_s = interval_s
        self.model = model
        self.last_capture_time: float = 0.0
        self.last_description: Optional[str] = None

    def maybe_capture(self, frame: np.ndarray, now: float) -> Optional[str]:
        """Return a new scene description if interval has elapsed, else None."""
        if now - self.last_capture_time < self.interval_s:
            return None
        self.last_capture_time = now
        description = self._describe_image(frame)
        self.last_description = description
        print(f"[CONTEXT] {description}")
        return description

    def _describe_image(self, frame: np.ndarray) -> str:
        """Encode frame as JPEG base64 and send to GPT-4o vision."""
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to JPEG-encode frame for context shot.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Describe this scene in exactly 4 concise bullet points. "
                        "Do not identify, name, or infer identity of any person. "
                        "Focus on the setting/environment, approximate number of people, "
                        "what is happening, and observable group mood/emotions plus lighting."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=150,
        )

        content = completion.choices[0].message.content if completion.choices else None
        return (content or "").strip()
