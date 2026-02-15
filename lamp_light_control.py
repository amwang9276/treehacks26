"""
rgb_controller.py

Control three PWM LEDs (R, G, B) connected to an Arduino over serial.

Protocol:
    "R,G,B\\n"

Author: Your Name
"""

import serial
import time
from typing import Tuple


class RGBController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize serial connection to Arduino.

        :param port: COM port (e.g., "COM5")
        :param baudrate: Must match Arduino Serial.begin()
        :param timeout: Serial timeout
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

        # Arduino resets when serial opens
        time.sleep(2)

    def _clamp(self, value: int) -> int:
        """Clamp value to 0–255."""
        return max(0, min(255, int(value)))

    def set_rgb(self, r: int, g: int, b: int, brightness: float = 1.0):
        """
        Set LED color with optional brightness scaling.

        :param r: Red value (0–255)
        :param g: Green value (0–255)
        :param b: Blue value (0–255)
        :param brightness: 0.0–1.0 brightness scaling
        """

        brightness = max(0.0, min(1.0, brightness))

        r = self._clamp(r * brightness)
        g = self._clamp(g * brightness)
        b = self._clamp(b * brightness)

        message = f"{int(r)},{int(g)},{int(b)}\n"
        self.ser.write(message.encode("utf-8"))
        self.ser.flush()

    def off(self):
        """Turn all LEDs off."""
        self.set_rgb(0, 0, 0)

    def close(self):
        """Close serial connection safely."""
        self.off()
        time.sleep(0.1)
        self.ser.close()


# =============================
# CLI / Test Harness
# =============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control RGB LEDs via Arduino")
    parser.add_argument("--port", required=True, help="COM port (e.g., COM5)")
    parser.add_argument("--r", type=int, default=255, help="Red (0-255)")
    parser.add_argument("--g", type=int, default=0, help="Green (0-255)")
    parser.add_argument("--b", type=int, default=0, help="Blue (0-255)")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness (0.0-1.0)")

    args = parser.parse_args()

    controller = RGBController(port=args.port)

    try:
        controller.set_rgb(args.r, args.g, args.b, brightness=args.brightness)
        print(f"Set RGB to ({args.r}, {args.g}, {args.b}) at brightness {args.brightness}")
    finally:
        controller.close()
