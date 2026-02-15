const int NUM_LEDS = 9;

const int ledPins[NUM_LEDS] = {
  2, 3, 4,
  5, 6, 7,
  8, 9, 10
};

/*
Grid layout (your color arrangement reference)
g, b, r,
r, g, b,
b, r, g
*/

// ---------- MOOD SYSTEM ----------
// 0..5 = six moods
const int NUM_MOODS = 6;
int currentMood = 0;

// Put your brightness settings (0–255) for each mood here.
// Each row is one mood, with 9 values for the 9 LEDs (left-to-right, top-to-bottom).
int moodBrightness[NUM_MOODS][NUM_LEDS] = {
  // Mood 0 focus
  {255, 255, 255,
   255, 255, 255,
   255, 255, 255},

  // Mood 1 sad
  {0, 255, 0,
   50, 0, 255,
   255, 50, 0},

  // Mood 2 calm
  {150, 255, 50,
   50, 100, 255,
   255, 50, 100},

  // Mood 3 happy
  {100, 70, 255,
   255, 100, 70,
   70, 255, 100},

  // Mood 4 angry
  {0, 0, 255,
   255, 0, 0,
   0, 255, 0},

  // Mood 5 romantic
  {0, 100, 255,
   255, 0, 100,
   130, 255, 0},
};

// Copy of the active mood into the working array
int ledBrightness[NUM_LEDS];

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < NUM_LEDS; i++) {
    pinMode(ledPins[i], OUTPUT);
  }

  // Start on Mood 0
  setMood(0);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int mood = input.toInt();
    if (mood >= 0 && mood < NUM_MOODS) {
      setMood(mood);
    }
  }
}

void setMood(int moodIndex) {
  if (moodIndex < 0) moodIndex = 0;
  if (moodIndex >= NUM_MOODS) moodIndex = NUM_MOODS - 1;

  Serial.print("[LAMP] Switching to mood ");
  Serial.print(moodIndex);
  Serial.print(" (from ");
  Serial.print(currentMood);
  Serial.println(")");

  currentMood = moodIndex;

  // Copy mood settings into ledBrightness
  for (int i = 0; i < NUM_LEDS; i++) {
    ledBrightness[i] = moodBrightness[currentMood][i];
  }

  applyBrightness();
}

void nextMood() {
  int next = currentMood + 1;
  if (next >= NUM_MOODS) next = 0;
  setMood(next);
}

void applyBrightness() {
  for (int i = 0; i < NUM_LEDS; i++) {
    analogWrite(ledPins[i], ledBrightness[i]);
  }
}
