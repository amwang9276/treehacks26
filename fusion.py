"""Sensor fusion: combines face emotion, voice analysis, and room context into a
natural language description."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI


@dataclass
class SensorState:
    emotion: Optional[str]
    emotion_score: float
    face_count: int
    room_description: Optional[str]
    timestamp: float
    # Voice fields (all optional so fusion works with partial data)
    prosody_emotion: Optional[str] = None
    vocal_mood: Optional[str] = None
    vocal_mood_score: float = 0.0
    speech_rate: Optional[float] = None
    transcript: Optional[str] = None
    text_emotion: Optional[str] = None
    text_emotion_score: float = 0.0
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


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
                        "You receive sensor data about a room. Produce exactly 5 sentences "
                        "describing the event or occasion, the mood of the collective, and the atmosphere"
                        "combining all available sensor inputs. No bullet points"
                        "keep the descriptions succinct, informative, and to the point."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
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
        # Voice mood / prosody
        if state.vocal_mood or state.prosody_emotion:
            mood_label = state.vocal_mood or state.prosody_emotion
            rate_desc = "normal"
            if state.speech_rate is not None:
                if state.speech_rate > 160:
                    rate_desc = "rapid"
                elif state.speech_rate < 100:
                    rate_desc = "slow"
            parts.append(
                f"Voice mood: {mood_label} (confidence: {state.vocal_mood_score:.2f}), "
                f"speech rate: {rate_desc}."
            )
        # Transcript snippet
        if state.transcript:
            snippet = state.transcript[:100]
            if len(state.transcript) > 100:
                snippet += "..."
            parts.append(f"Transcript snippet: \"{snippet}\"")
        # Text emotion from speech content
        if state.text_emotion:
            parts.append(
                f"Text emotion: {state.text_emotion} "
                f"(confidence: {state.text_emotion_score:.2f})."
            )
        # Conversation topics
        if state.topics:
            parts.append(f"Topics: {', '.join(state.topics[:5])}.")
        return " ".join(parts) if parts else "No sensor data available."
