"""
Voice capture and analysis module for room mood detection.

Captures microphone audio continuously, applies echo cancellation to remove
music the app is playing, then runs:
  1. Speech prosody analysis (wav2vec2 — how people sound)
  2. Speech-to-text (OpenAI Whisper API)
  3. Text NLP (spaCy — topics and keywords)
  4. Text emotion classification (distilroberta — what emotion the words convey)
"""

from __future__ import annotations

import io
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import librosa

try:
    import sounddevice as sd
except ImportError as err:
    raise ImportError(
        "sounddevice is required for voice capture. Install with: pip install sounddevice"
    ) from err


SAMPLE_RATE = 16000
CHANNELS = 1

# Label mapping: text emotion model → common label set
TEXT_EMOTION_LABEL_MAP = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
}


@dataclass(frozen=True)
class VoiceObservation:
    """Observation from one voice processing cycle."""

    timestamp_s: float
    # Prosody (how people sound)
    prosody_emotion: Optional[str] = None
    prosody_scores: Dict[str, float] = field(default_factory=dict)
    energy_rms: float = 0.0
    is_speech: bool = False
    # New vocal features
    vocal_mood: Optional[str] = None
    vocal_mood_score: float = 0.0 # Add score for vocal mood
    speech_rate: Optional[float] = None  # e.g., words per minute or syllables per second
    vocal_dynamics: Optional[Dict[str, float]] = field(default_factory=dict)  # e.g., avg_volume, volume_variance, silence_duration
    inflection_patterns: Optional[Dict[str, float]] = field(default_factory=dict)  # e.g., rising_tone_ratio, falling_tone_ratio
    # Transcript NLP (what people say)
    transcript: Optional[str] = None
    text_emotion: Optional[str] = None
    text_emotion_score: float = 0.0
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)






def _nlms_aec(
    mic: np.ndarray,
    ref: np.ndarray,
    filter_len: int = 1024,
    mu: float = 0.5,
) -> np.ndarray:
    """Normalized Least Mean Squares adaptive filter for echo cancellation.

    Subtracts the estimated echo (derived from the reference signal) from
    the microphone signal to isolate speech.
    """
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    w = np.zeros(filter_len, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)

    for i in range(filter_len, n):
        x = ref[i - filter_len : i][::-1]
        est = np.dot(w, x)
        err = mic[i] - est
        norm = np.dot(x, x) + 1e-8
        w += (mu / norm) * err * x
        output[i] = err

    return output


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 numpy audio to in-memory WAV bytes for Whisper API."""
    int16_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()


class VoiceProcessor:
    """Continuous microphone capture with voice analysis pipeline."""

    def __init__(
        self,
        *,
        chunk_duration_s: float = 5.0,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold_rms: float = 0.01,
        openai_api_key: Optional[str] = None,
        music_player=None,  # MusicPlayer instance for AEC reference
        on_observation: Optional[Callable[[VoiceObservation], None]] = None,
    ) -> None:
        self.chunk_duration_s = chunk_duration_s
        self.sample_rate = sample_rate
        self.silence_threshold_rms = silence_threshold_rms
        self.music_player = music_player
        self.on_observation = on_observation

        # Ring buffer for mic audio
        self._mic_buffer: deque[np.ndarray] = deque(
            maxlen=int(chunk_duration_s * sample_rate / 1600) + 10
        )
        self._mic_lock = threading.Lock()

        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Lazy-loaded models
        self._prosody_pipeline = None
        self._text_emotion_pipeline = None
        self._nlp = None
        self._openai_client = None

        if openai_api_key:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=openai_api_key)

    def _load_models(self) -> None:
        """Lazy-load ML models on first use (avoids slow startup if --no-voice)."""
        if self._prosody_pipeline is None:
            print("[VOICE] loading prosody model (wav2vec2)...")
            from transformers import pipeline

            self._prosody_pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=-1,  # CPU
            )

        if self._text_emotion_pipeline is None:
            print("[VOICE] loading text emotion model (distilroberta)...")
            from transformers import pipeline

            self._text_emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
            )

        if self._nlp is None:
            try:
                print("[VOICE] loading spaCy NLP model...")
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception as exc:
                print(f"[VOICE] spaCy model unavailable ({exc}); using basic keyword extraction")
                self._nlp = "basic"

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """sounddevice callback — append mic audio to ring buffer."""
        if status:
            print(f"[VOICE] audio status: {status}", file=sys.stderr)
        chunk = indata[:, 0].copy().astype(np.float32)
        with self._mic_lock:
            self._mic_buffer.append(chunk)

    def start(self) -> None:
        """Start microphone capture and processing threads."""
        self._load_models()
        self._stop_event.clear()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=512,  # A common block size for audio processing
            callback=self._audio_callback,
        )
        self._stream.start()
        print(f"[VOICE] microphone capture started ({self.sample_rate} Hz)")

        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

    def _get_mic_audio(self) -> np.ndarray:
        """Assemble the current ring buffer into a single numpy array."""
        with self._mic_lock:
            if not self._mic_buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._mic_buffer))

    def _apply_aec(self, mic_audio: np.ndarray) -> np.ndarray:
        """Apply echo cancellation using reference audio from MusicPlayer."""
        if self.music_player is None:
            return mic_audio
        ref = self.music_player.get_reference_chunk(len(mic_audio))
        if ref is None:
            return mic_audio
        return _nlms_aec(mic_audio, ref)



    def _extract_acoustic_features(self, audio: np.ndarray) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """Extracts core acoustic features using librosa."""
        features = {}
        f0_contour: Optional[np.ndarray] = None
        if len(audio) == 0:
            return features, f0_contour

        # Pitch (f0)
        # Using pyin for more robust pitch estimation
        try:
            f0, *_ = librosa.pyin(
                y=audio,
                sr=self.sample_rate,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C5'),
                            frame_length=2048,
                            hop_length=2048 // 4  # Overlap for better f0 tracking
            ) # Typo fixed: removed extra closing parenthesis
            f0_contour = f0  # Store the full f0 contour
            features["avg_f0"] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
            features["f0_variance"] = float(np.nanvar(f0)) if not np.all(np.isnan(f0)) else 0.0
        except Exception as e:
            print(f"[VOICE] Error extracting f0: {e}", file=sys.stderr)
            features["avg_f0"] = 0.0
            features["f0_variance"] = 0.0

        # Energy (RMS) - already calculated for VAD, but can re-calculate for finer granularity
        rms_frames = librosa.feature.rms(y=audio, frame_length=512, hop_length=512 // 2)[0]
        features["avg_rms"] = float(np.mean(rms_frames))
        features["rms_variance"] = float(np.var(rms_frames))
        features["max_rms"] = float(np.max(rms_frames))

        # MFCCs (Mel-frequency cepstral coefficients)
        # Taking mean of MFCCs over the segment as a summary
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=512 // 2)
        for i, mfcc_val in enumerate(np.mean(mfccs, axis=1)):
            features[f"mfcc_{i}"] = float(mfcc_val)

        return features, f0_contour

    def _analyze_inflection_patterns(self, f0: np.ndarray) -> Dict[str, float]:
        """Analyzes pitch contours for rising/falling inflection patterns."""
        patterns = {"rising_tone_ratio": 0.0, "falling_tone_ratio": 0.0}
        if len(f0) < 2 or np.all(np.isnan(f0)):
            return patterns

        # Remove NaNs for analysis
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) < 2:
            return patterns

        # Simple approach: compare start and end of voiced segments
        # More sophisticated methods would involve contour analysis over utterances

        # Rising inflection: pitch generally goes up from start to end
        # Falling inflection: pitch generally goes down from start to end

        # Let's consider a simple comparison of first and last 25% of voiced pitch
        quarter_len = len(voiced_f0) // 4
        if quarter_len < 1: # Ensure at least one sample for comparison
            return patterns

        start_avg_f0 = np.mean(voiced_f0[:quarter_len])
        end_avg_f0 = np.mean(voiced_f0[-quarter_len:])

        if end_avg_f0 > start_avg_f0 * 1.05:  # 5% increase
            patterns["rising_tone_ratio"] = 1.0
        elif end_avg_f0 < start_avg_f0 * 0.95: # 5% decrease
            patterns["falling_tone_ratio"] = 1.0
        
        # A more detailed approach would involve looking at individual pitch changes
        # For a simple ratio, we can count rising/falling segments
        rising_changes = np.sum(np.diff(voiced_f0) > 0)
        falling_changes = np.sum(np.diff(voiced_f0) < 0)
        total_changes = rising_changes + falling_changes
        
        if total_changes > 0:
            patterns["rising_tone_ratio"] = rising_changes / total_changes
            patterns["falling_tone_ratio"] = falling_changes / total_changes

        return patterns

    def _calculate_speech_rate(self, transcript: str, audio_duration_s: float) -> Optional[float]:
        """Calculates words per minute (WPM) from transcript and audio duration."""
        if not transcript or audio_duration_s == 0:
            return None
        words = transcript.split()
        num_words = len(words)
        wpm = (num_words / audio_duration_s) * 60
        return wpm

    def _infer_vocal_mood_from_features(
        self,
        prosody_emotion: Optional[str],
        prosody_scores: Dict[str, float],
        speech_rate: Optional[float],
        vocal_dynamics: Optional[Dict[str, float]],
        inflection_patterns: Optional[Dict[str, float]],
    ) -> Tuple[Optional[str], float]:
        """
        Combines prosody emotion, speech rate, vocal dynamics, and inflection patterns
        to infer a more refined vocal mood using heuristics.
        """
        # Start with the raw prosody scores
        mood_scores = prosody_scores.copy()
        dominant_mood = prosody_emotion or "neutral" # Default to neutral if no prosody
        dominant_mood_score = mood_scores.get(dominant_mood, 0.0)

        # Ensure all possible emotions are in mood_scores, even if 0
        all_emotions = set(mood_scores.keys())
        # Example emotions from the wav2vec2 model: 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
        # Ensure we have common emotions from TEXT_EMOTION_LABEL_MAP
        common_emotions = {"happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"}
        for emotion in common_emotions:
            if emotion not in mood_scores:
                mood_scores[emotion] = 0.0
        
        # --- Heuristics for Modulation ---
        AVG_WPM = 140.0
        HIGH_WPM_THRESHOLD = 160.0
        LOW_WPM_THRESHOLD = 100.0

        # Speech Rate Modulation
        if speech_rate is not None:
            if speech_rate > HIGH_WPM_THRESHOLD:
                # Boost active/positive/negative emotions, dampen calm/neutral
                mood_scores["happy"] = min(1.0, mood_scores.get("happy", 0.0) + 0.15 * (speech_rate / AVG_WPM))
                mood_scores["happy"] = min(1.0, mood_scores.get("happy", 0.0) + 0.15 * (speech_rate / AVG_WPM)) # Assuming 'excited' is a possible label
                mood_scores["angry"] = min(1.0, mood_scores.get("angry", 0.0) + 0.10 * (speech_rate / AVG_WPM))
                mood_scores["sad"] = max(0.0, mood_scores.get("sad", 0.0) - 0.05 * (speech_rate / AVG_WPM))
                mood_scores["calm"] = max(0.0, mood_scores.get("calm", 0.0) - 0.10 * (speech_rate / AVG_WPM))
                mood_scores["neutral"] = max(0.0, mood_scores.get("neutral", 0.0) - 0.05 * (speech_rate / AVG_WPM))
            elif speech_rate < LOW_WPM_THRESHOLD:
                # Boost calm/neutral/sad, dampen active/positive/negative
                mood_scores["sad"] = min(1.0, mood_scores.get("sad", 0.0) + 0.10 * (1 - speech_rate / AVG_WPM))
                mood_scores["calm"] = min(1.0, mood_scores.get("calm", 0.0) + 0.15 * (1 - speech_rate / AVG_WPM))
                mood_scores["neutral"] = min(1.0, mood_scores.get("neutral", 0.0) + 0.05 * (1 - speech_rate / AVG_WPM))
                mood_scores["happy"] = max(0.0, mood_scores.get("happy", 0.0) - 0.05 * (1 - speech_rate / AVG_WPM))
                mood_scores["angry"] = max(0.0, mood_scores.get("angry", 0.0) - 0.05 * (1 - speech_rate / AVG_WPM))

        # Vocal Dynamics Modulation (using avg_rms for loudness and rms_variance for expressiveness)
        if vocal_dynamics:
            avg_rms = vocal_dynamics.get("avg_rms", 0.0)
            rms_variance = vocal_dynamics.get("rms_variance", 0.0)

            # Assuming avg_rms is normalized 0-1 (from previous librosa scaling or just a relative scale)
            # A simple rule: louder -> more energetic, higher variance -> more expressive
            loudness_factor = np.clip(avg_rms / 0.1, 0.0, 1.0) # Assume 0.1 is a typical max RMS for normalized audio
            expressiveness_factor = np.clip(rms_variance / 0.001, 0.0, 1.0) # Assume 0.001 is a typical max variance

            mood_scores["angry"] = min(1.0, mood_scores.get("angry", 0.0) + 0.10 * loudness_factor)
            mood_scores["excited"] = min(1.0, mood_scores.get("excited", 0.0) + 0.10 * expressiveness_factor)
            mood_scores["happy"] = min(1.0, mood_scores.get("happy", 0.0) + 0.05 * expressiveness_factor)
            mood_scores["calm"] = max(0.0, mood_scores.get("calm", 0.0) - 0.05 * loudness_factor)


        # Inflection Patterns Modulation
        if inflection_patterns:
            rising = inflection_patterns.get("rising_tone_ratio", 0.0)
            falling = inflection_patterns.get("falling_tone_ratio", 0.0)

            # Rising tone: often associated with questioning, uncertainty, excitement, or positive surprise
            if rising > falling * 1.2: # Significantly more rising
                mood_scores["surprise"] = min(1.0, mood_scores.get("surprise", 0.0) + 0.10 * rising)
                mood_scores["happy"] = min(1.0, mood_scores.get("happy", 0.0) + 0.05 * rising)
                mood_scores["neutral"] = max(0.0, mood_scores.get("neutral", 0.0) - 0.05 * rising)
                mood_scores["sad"] = max(0.0, mood_scores.get("sad", 0.0) - 0.05 * rising)
            # Falling tone: often associated with certainty, finality, sadness, or seriousness
            elif falling > rising * 1.2: # Significantly more falling
                mood_scores["sad"] = min(1.0, mood_scores.get("sad", 0.0) + 0.10 * falling)
                mood_scores["neutral"] = min(1.0, mood_scores.get("neutral", 0.0) + 0.05 * falling)
                mood_scores["happy"] = max(0.0, mood_scores.get("happy", 0.0) - 0.05 * falling)
                mood_scores["excited"] = max(0.0, mood_scores.get("excited", 0.0) - 0.05 * falling)
        
        # Re-normalize scores if they exceed 1.0 due to boosts, or just clip.
        for emotion in mood_scores:
            mood_scores[emotion] = np.clip(mood_scores[emotion], 0.0, 1.0)

        # Find the new dominant mood and its confidence
        if mood_scores:
            new_dominant_mood = max(mood_scores, key=mood_scores.get)
            new_dominant_mood_score = mood_scores[new_dominant_mood]
            return new_dominant_mood, new_dominant_mood_score
        
        return "neutral", 0.0 # Fallback

    def _analyze_prosody(
        self, audio: np.ndarray
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Run wav2vec2 speech emotion recognition on audio."""
        if self._prosody_pipeline is None or len(audio) < self.sample_rate:
            return None, {}
        try:
            results = self._prosody_pipeline(
                audio, sampling_rate=self.sample_rate
            )
            scores = {r["label"].lower(): r["score"] for r in results}
            top_label = max(scores, key=scores.get) if scores else None
            return top_label, scores
        except Exception as err:
            print(f"[VOICE] prosody error: {err}", file=sys.stderr)
            return None, {}

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper API."""
        if self._openai_client is None:
            return None
        wav_bytes = _audio_to_wav_bytes(audio, self.sample_rate)
        try:
            result = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_bytes, "audio/wav"),
            )
            text = result.text.strip()
            return text if text else None
        except Exception as err:
            print(f"[VOICE] transcription error: {err}", file=sys.stderr)
            return None

    def _analyze_text_emotion(
        self, text: str
    ) -> Tuple[Optional[str], float]:
        """Run text emotion classification."""
        if self._text_emotion_pipeline is None or not text:
            return None, 0.0
        try:
            results = self._text_emotion_pipeline(text[:512])
            if results and results[0]:
                top = max(results[0], key=lambda r: r["score"])
                label = TEXT_EMOTION_LABEL_MAP.get(
                    top["label"].lower(), top["label"].lower()
                )
                return label, top["score"]
        except Exception as err:
            print(f"[VOICE] text emotion error: {err}", file=sys.stderr)
        return None, 0.0

    # Common English stop words for the basic keyword fallback
    _STOP_WORDS = frozenset(
        "a an the is are was were be been being have has had do does did will "
        "would shall should may might can could i me my we our you your he him "
        "his she her it its they them their this that these those am not no nor "
        "so if or but and to of in for on with at by from as into about between "
        "through after before above below up down out off over under again then "
        "once here there when where why how all each every both few more most "
        "other some such than too very just also now get got".split()
    )

    def _extract_nlp(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract topics (noun chunks + entities) and keywords from text."""
        if self._nlp is None or not text:
            return [], []

        # Basic fallback: split on whitespace, filter stop words and short tokens
        if self._nlp == "basic":
            words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text[:1000].split()]
            keywords = list(
                {w for w in words if len(w) > 2 and w.isalpha() and w not in self._STOP_WORDS}
            )
            return [], keywords[:10]

        try:
            doc = self._nlp(text[:1000])
            topics = list(
                {chunk.text.lower() for chunk in doc.noun_chunks}
                | {ent.text.lower() for ent in doc.ents}
            )
            keywords = list(
                {
                    token.lemma_.lower()
                    for token in doc
                    if token.pos_ in ("NOUN", "VERB", "ADJ")
                    and not token.is_stop
                    and len(token.text) > 2
                }
            )
            return topics[:10], keywords[:10]
        except Exception as err:
            print(f"[VOICE] NLP error: {err}", file=sys.stderr)
            return [], []

    def _processing_loop(self) -> None:
        """Main processing loop — runs every chunk_duration_s seconds."""
        while not self._stop_event.wait(timeout=self.chunk_duration_s):
            try:
                self._process_chunk()
            except Exception as err:
                print(f"[VOICE] processing error: {err}", file=sys.stderr)

    def _process_chunk(self) -> None:
        """Process one chunk of audio through the full pipeline."""
        now = time.time()
        mic_audio = self._get_mic_audio()
        if len(mic_audio) == 0:
            return

        # Echo cancellation
        clean_audio = self._apply_aec(mic_audio)

        # Silence detection - check RMS of the entire clean_audio segment
        rms = float(np.sqrt(np.mean(clean_audio**2)))
        if rms < self.silence_threshold_rms:
            obs = VoiceObservation(timestamp_s=now, energy_rms=rms, is_speech=False)
            if self.on_observation:
                self.on_observation(obs)
            return

        # Treat the entire chunk as speech if above threshold (simplified VAD)
        speech_audio = clean_audio
        is_speech_detected = True

        # Extract acoustic features
        acoustic_features, f0_contour = self._extract_acoustic_features(speech_audio)
        
        # Extract f0 from acoustic_features for inflection analysis
        inflection_patterns = self._analyze_inflection_patterns(f0_contour)

        # Prosody analysis - use speech_audio
        prosody_emotion, prosody_scores = self._analyze_prosody(speech_audio)

        # Speech-to-text - use speech_audio
        transcript = self._transcribe(speech_audio)
        
        speech_rate = None
        audio_duration_s = len(speech_audio) / self.sample_rate
        if transcript:
            speech_rate = self._calculate_speech_rate(transcript, audio_duration_s)

        
        # Infer a refined vocal mood from all features
        inferred_vocal_mood, inferred_vocal_mood_score = self._infer_vocal_mood_from_features(
            prosody_emotion=prosody_emotion,
            prosody_scores=prosody_scores,
            speech_rate=speech_rate,
            vocal_dynamics=acoustic_features, # vocal_dynamics now contains acoustic_features
            inflection_patterns=inflection_patterns,
        )

        # Text analysis (only if we got a transcript)
        text_emotion, text_emotion_score = None, 0.0
        topics, keywords = [], []
        if transcript:
            text_emotion, text_emotion_score = self._analyze_text_emotion(
                transcript
            )
            topics, keywords = self._extract_nlp(transcript)

        obs = VoiceObservation(
            timestamp_s=now,
            prosody_emotion=prosody_emotion,
            prosody_scores=prosody_scores,
            energy_rms=rms,
            is_speech=True,
            vocal_mood=inferred_vocal_mood,
            vocal_mood_score=inferred_vocal_mood_score,
            speech_rate=speech_rate,
            vocal_dynamics=acoustic_features,
            inflection_patterns=inflection_patterns,
            transcript=transcript,
            text_emotion=text_emotion,
            text_emotion_score=text_emotion_score,
            topics=topics,
            keywords=keywords,
        )
        if self.on_observation:
            self.on_observation(obs)

    def stop(self) -> None:
        """Stop capture and processing."""
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=3.0)
            self._processing_thread = None

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test voice capture and analysis.")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument(
        "--silence-threshold", type=float, default=0.01
    )
    args = parser.parse_args()

    import os

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

    def print_observation(obs: VoiceObservation) -> None:
        print(f"\n{'='*60}")
        print(f"[{obs.timestamp_s:.1f}] speech={obs.is_speech} rms={obs.energy_rms:.4f}")
        if obs.is_speech:
            print(f"  prosody: {obs.prosody_emotion} {obs.prosody_scores}")
            print(f"  transcript: {obs.transcript}")
            print(f"  text_emotion: {obs.text_emotion} ({obs.text_emotion_score:.2f})")
            print(f"  topics: {obs.topics}")
            print(f"  keywords: {obs.keywords}")

    vp = VoiceProcessor(
        chunk_duration_s=args.chunk_seconds,
        silence_threshold_rms=args.silence_threshold,
        openai_api_key=api_key,
        on_observation=print_observation,
    )
    try:
        vp.start()
        print("Listening... Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        vp.close()
        print("\nStopped.")
