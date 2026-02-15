import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Import modules under test
from voice import VoiceProcessor, VoiceObservation, SAMPLE_RATE

class LiveVocalTestApp:
    def __init__(self, master):
        self.master = master
        master.title("Live Vocal Tracking Test")
        master.geometry("800x600")

        self.voice_processor: VoiceProcessor | None = None
        self.is_recording = False
        self.observation_queue = queue.Queue()

        self.create_widgets()
        self.master.after(100, self.process_observations_from_queue) # Start checking the queue

    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        self.start_button = tk.Button(control_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Scrolled Text for Output
        self.output_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, state='disabled', width=90, height=30)
        self.output_text.pack(pady=10)

        self.log_message("Welcome to Live Vocal Tracking Test.")
        self.log_message("Ensure your microphone is connected and working.")
        self.log_message("Press 'Start Recording' to begin capturing and analyzing audio.")
        self.log_message("Note: Model loading can take some time on first start.")
        self.log_message(f"Current working directory: {os.getcwd()}")


    def log_message(self, message):
        self.output_text.config(state='normal')
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.output_text.insert(tk.END, f"{timestamp} {message}\n")
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

    def on_voice_observation(self, obs: VoiceObservation):
        # Put observation in a queue to update GUI from main thread
        self.observation_queue.put(obs)

    def process_observations_from_queue(self):
        try:
            while True:
                obs = self.observation_queue.get_nowait()
                self.display_observation(obs)
        except queue.Empty:
            pass
        self.master.after(100, self.process_observations_from_queue) # Schedule next check

    def display_observation(self, obs: VoiceObservation):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, f"\n{'='*60}\n")
        timestamp_str = datetime.fromtimestamp(obs.timestamp_s).strftime("%H:%M:%S.%f")[:-3]
        self.output_text.insert(tk.END, f"[{timestamp_str}] Speech={obs.is_speech} RMS={obs.energy_rms:.4f}\n")

        if obs.is_speech:
            self.output_text.insert(tk.END, f"  Prosody Emotion: {obs.prosody_emotion} (Scores: {obs.prosody_scores})\n")
            self.output_text.insert(tk.END, f"  Inferred Vocal Mood: {obs.vocal_mood} (Score: {obs.vocal_mood_score:.2f})\n")
            self.output_text.insert(tk.END, (f"  Speech Rate: {obs.speech_rate:.2f} WPM\n" if obs.speech_rate is not None else "  Speech Rate: N/A\n"))

            if obs.vocal_dynamics:
                self.output_text.insert(tk.END, f"  Vocal Dynamics:\n")
                for k, v in obs.vocal_dynamics.items():
                    self.output_text.insert(tk.END, f"    - {k}: {v:.4f}\n")

            if obs.inflection_patterns:
                self.output_text.insert(tk.END, f"  Inflection Patterns:\n")
                for k, v in obs.inflection_patterns.items():
                    self.output_text.insert(tk.END, f"    - {k}: {v:.4f}\n")

            self.output_text.insert(tk.END, f"  Transcript: {obs.transcript or 'N/A'}\n")
            if obs.text_emotion:
                self.output_text.insert(tk.END, f"  Text Emotion: {obs.text_emotion} (Score: {obs.text_emotion_score:.2f})\n")
            if obs.topics:
                self.output_text.insert(tk.END, f"  Topics: {', '.join(obs.topics)}\n")
            if obs.keywords:
                self.output_text.insert(tk.END, f"  Keywords: {', '.join(obs.keywords)}\n")

        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

    def start_recording(self):
        if self.is_recording:
            return

        self.log_message("Loading models... (GUI will remain responsive)")
        self.start_button.config(state=tk.DISABLED)
        self.is_recording = True

        def _init_and_start():
            openai_key = os.environ.get("OPENAI_API_KEY")
            print(openai_key)
            if not openai_key:
                self.master.after(0, lambda: self.log_message("WARNING: OPENAI_API_KEY environment variable not found."))
                self.master.after(0, lambda: self.log_message("Transcription and text emotion analysis will be skipped."))
                self.master.after(0, lambda: self.log_message("To enable, set OPENAI_API_KEY before running this script."))

            try:
                self.voice_processor = VoiceProcessor(
                    chunk_duration_s=5.0,
                    sample_rate=SAMPLE_RATE,
                    silence_threshold_rms=0.001,
                    openai_api_key=openai_key,
                    on_observation=self.on_voice_observation
                )
                self.voice_processor.start()
                self.master.after(0, lambda: self.stop_button.config(state=tk.NORMAL))
                self.master.after(0, lambda: self.log_message("Recording started. Speak into the microphone."))
            except Exception as e:
                self.master.after(0, lambda: self.log_message(f"ERROR starting VoiceProcessor: {e}"))
                self.master.after(0, self.stop_recording)

        threading.Thread(target=_init_and_start, daemon=True).start()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.log_message("Stopping recording and analysis...")
        self.is_recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if self.voice_processor:
            self.voice_processor.close()
            self.voice_processor = None
        self.log_message("Recording stopped.")

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveVocalTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
