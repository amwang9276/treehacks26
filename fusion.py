"""Sensor fusion: combines face emotion, audio state, and room context into a
natural language description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class SensorState:
    emotion: Optional[str]
    emotion_score: float
    face_count: int
    current_track_info: Optional[str]
    room_description: Optional[str]
    timestamp: float


class SensorFusion:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = client
        self.model = model
        self.last_description: Optional[str] = None

    def generate_description(self, state: SensorState) -> str:
        """Build a natural language summary from all available sensor data."""
        prompt = self._build_prompt(state)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You receive sensor data about a room. Produce a 1-2 sentence "
                        "natural language description of the scene and mood. "
                        "No markdown, no bullet points."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        content = completion.choices[0].message.content if completion.choices else None
        text = (content or "").strip()
        self.last_description = text
        print(f"[FUSION] {text}")
        return text

    def _build_prompt(self, state: SensorState) -> str:
        parts: list[str] = []
        if state.room_description:
            parts.append(f"Room context: {state.room_description}")
        if state.emotion:
            parts.append(
                f"Emotion: The person appears {state.emotion} "
                f"(confidence: {state.emotion_score:.2f})."
            )
        if state.face_count:
            parts.append(
                f"There {'is' if state.face_count == 1 else 'are'} "
                f"{state.face_count} {'person' if state.face_count == 1 else 'people'} visible."
            )
        if state.current_track_info:
            parts.append(f"Currently playing: {state.current_track_info}")
        return " ".join(parts) if parts else "No sensor data available."
