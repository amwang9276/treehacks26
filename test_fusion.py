"""Quick smoke test for SensorFusion — exercises generate_description() with
realistic mock sensor data so we can inspect the GPT-4o-mini output without
needing a camera, microphone, or Suno API.

Usage:
    python test_fusion.py

Requires OPENAI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from fusion import SensorFusion, SensorState


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'").strip('"')
    print("ERROR: OPENAI_API_KEY not found in environment or .env", file=sys.stderr)
    sys.exit(1)


SCENARIOS: list[tuple[str, SensorState]] = [
    (
        "Face only (happy, 1 person)",
        SensorState(
            emotion="happy",
            emotion_score=0.92,
            face_count=1,
            room_description=None,
            timestamp=time.time(),
        ),
    ),
    (
        "Face + room context (sad, 2 people)",
        SensorState(
            emotion="sad",
            emotion_score=0.78,
            face_count=2,
            room_description=(
                "A dimly lit living room with two people sitting on a couch. "
                "The curtains are drawn and a single lamp provides warm light."
            ),
            timestamp=time.time(),
        ),
    ),
    (
        "Full sensors (angry, speaking fast)",
        SensorState(
            emotion="angry",
            emotion_score=0.85,
            face_count=1,
            room_description="Bright office space with one person at a desk, multiple monitors.",
            timestamp=time.time(),
            prosody_emotion="angry",
            vocal_mood="frustrated",
            vocal_mood_score=0.88,
            speech_rate=185.0,
            transcript="I can't believe this happened again, this is the third time today",
            text_emotion="anger",
            text_emotion_score=0.91,
            topics=["work frustration", "repeated issues"],
            keywords=["believe", "happened", "third time"],
        ),
    ),
    (
        "Full sensors (calm, speaking slowly)",
        SensorState(
            emotion="neutral",
            emotion_score=0.70,
            face_count=1,
            room_description="Cozy bedroom with soft lighting, plants on the windowsill, rain outside.",
            timestamp=time.time(),
            prosody_emotion="calm",
            vocal_mood="relaxed",
            vocal_mood_score=0.82,
            speech_rate=85.0,
            transcript="Yeah I think I'm just going to read for a bit and then head to sleep",
            text_emotion="neutral",
            text_emotion_score=0.65,
            topics=["reading", "sleep"],
            keywords=["read", "sleep"],
        ),
    ),
    (
        "Quiet room (face but no speech)",
        SensorState(
            emotion="surprise",
            emotion_score=0.60,
            face_count=3,
            room_description="A conference room with three people, whiteboard with diagrams.",
            timestamp=time.time(),
        ),
    ),
]


def main() -> None:
    api_key = _load_api_key()
    client = OpenAI(api_key=api_key)
    fuser = SensorFusion(client=client)

    for name, state in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"  Scenario: {name}")
        print(f"{'=' * 60}")

        # Show the raw prompt that will be sent to the LLM
        raw_prompt = fuser._build_prompt(state)
        print(f"\n[PROMPT]\n{raw_prompt}\n")

        # Call the actual fusion
        try:
            description = fuser.generate_description(state)
        except Exception as err:
            print(f"[ERROR] {err}", file=sys.stderr)
            continue

        print(f"\n[DESCRIPTION]\n{description}")
        print()


if __name__ == "__main__":
    main()
