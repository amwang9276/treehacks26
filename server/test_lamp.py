"""Test script to change the lamp to each mood color one at a time.

Mood index reference (from colorchanging.ino):
  0 = focus    (all LEDs full brightness)
  1 = sad      (blue-heavy)
  2 = calm     (blue/green tones)
  3 = happy    (warm mix)
  4 = angry    (strong red/blue)
  5 = romantic  (red/pink tones)
"""

import serial
import time
import sys

MOODS = {
    0: "focus",
    1: "sad",
    2: "calm",
    3: "happy",
    4: "angry",
    5: "romantic",
}

DEFAULT_PORT = "COM4"
SECONDS_PER_MOOD = 5


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    print(f"Opening serial port {port} at 9600 baud...")
    try:
        ser = serial.Serial(port, 9600, timeout=1)
    except Exception as e:
        print(f"Failed to open {port}: {e}")
        sys.exit(1)

    # Wait for Arduino reset after serial open
    print("Waiting 2s for Arduino reset...\n")
    time.sleep(2)

    print(f"Cycling through all 6 moods ({SECONDS_PER_MOOD}s each):\n")
    for mood_index, mood_name in MOODS.items():
        message = f"{mood_index}\n"
        ser.write(message.encode("utf-8"))
        ser.flush()
        print(f"  >>> Mood {mood_index}: {mood_name} — watching for {SECONDS_PER_MOOD}s...")
        time.sleep(SECONDS_PER_MOOD)

    print("\nAll moods tested! Closing serial connection.")
    ser.close()


if __name__ == "__main__":
    main()
