import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import collections
import io
import wave
import threading
import time
import sys # Import sys

# Import modules under test
# Assuming the test file is in the same directory as the source files
from music import MusicPlayer, AEC_SAMPLE_RATE, AEC_READ_CHUNK, MusicPlaybackError
from voice import VoiceProcessor, VoiceObservation, SAMPLE_RATE, _audio_to_wav_bytes



# --- Helper for creating dummy audio ---
def create_dummy_audio(duration_s, sample_rate=SAMPLE_RATE, frequency=440, amplitude=0.5):
    """Creates a dummy sine wave audio numpy array."""
    t = np.linspace(0., duration_s, int(duration_s * sample_rate), endpoint=False)
    audio = amplitude * np.sin(2. * np.pi * frequency * t)
    return audio.astype(np.float32)

def create_dummy_speech_audio(duration_s, sample_rate=SAMPLE_RATE, wpm=150):
    """Creates dummy speech-like audio (random noise) for VAD/feature tests."""
    return np.random.rand(int(duration_s * sample_rate)).astype(np.float32) * 0.1 # low amplitude noise

def create_dummy_f0_contour(length, start_f0=100, end_f0=200):
    """Creates a dummy F0 contour (pitch) array for inflection tests."""
    return np.linspace(start_f0, end_f0, length).astype(np.float32)

# --- Test Cases for music.py ---
class TestMusicPlayer(unittest.TestCase):
    def setUp(self):
        # Mock shutil.which to prevent actual ffplay/ffmpeg lookup
        with patch('shutil.which', return_value='/usr/bin/ffplay'):
            self.player = MusicPlayer()
            # Manually set the _decoder_proc and _decoder_thread to avoid actual subprocess creation
            self.player._decoder_proc = MagicMock()
            self.player._decoder_thread = MagicMock()
            self.player._reference_buffer = collections.deque(maxlen=(AEC_SAMPLE_RATE * 30) // AEC_READ_CHUNK)

    def tearDown(self):
        self.player.close()

    def test_get_reference_chunk_consumption(self):
        """
        Test that get_reference_chunk correctly returns a 160-sample frame
        and removes it from the buffer.

        How to change for architectural updates:
        - If AEC_READ_CHUNK changes, adjust the size of appended frames and expected returned chunk.
        - If _reference_buffer storage mechanism changes (e.g., becomes a raw samples deque),
          this test needs to be rewritten to reflect that.
        """
        # Populate buffer with a few 160-sample frames
        frame1 = np.full(AEC_READ_CHUNK, 0.1, dtype=np.float32)
        frame2 = np.full(AEC_READ_CHUNK, 0.2, dtype=np.float32)
        frame3 = np.full(AEC_READ_CHUNK, 0.3, dtype=np.float32)
        
        with self.player._reference_lock:
            self.player._reference_buffer.append(frame1)
            self.player._reference_buffer.append(frame2)
            self.player._reference_buffer.append(frame3)

        # First call
        chunk = self.player.get_reference_chunk(AEC_READ_CHUNK)
        self.assertIsNotNone(chunk)
        np.testing.assert_array_equal(chunk, frame1)
        self.assertEqual(len(self.player._reference_buffer), 2)

        # Second call
        chunk = self.player.get_reference_chunk(AEC_READ_CHUNK)
        self.assertIsNotNone(chunk)
        np.testing.assert_array_equal(chunk, frame2)
        self.assertEqual(len(self.player._reference_buffer), 1)

    def test_get_reference_chunk_empty_buffer(self):
        """
        Test that get_reference_chunk returns None when the buffer is empty.

        How to change for architectural updates:
        - Logic for handling empty buffer should remain similar, but the exact
          condition might change if buffer implementation is different.
        """
        self.assertIsNone(self.player.get_reference_chunk(AEC_READ_CHUNK))
        self.assertEqual(len(self.player._reference_buffer), 0)

    def test_get_reference_chunk_incorrect_size(self):
        """
        Test that get_reference_chunk returns None if num_samples is not AEC_READ_CHUNK.

        How to change for architectural updates:
        - If the expected frame size for AEC changes, update AEC_READ_CHUNK accordingly.
        - If get_reference_chunk is designed to handle variable sizes in the future,
          this test might need to be removed or adapted.
        """
        self.assertIsNone(self.player.get_reference_chunk(100)) # Incorrect size
        self.assertIsNone(self.player.get_reference_chunk(AEC_READ_CHUNK + 1)) # Incorrect size
    
    @patch('subprocess.Popen')
    def test_start_reference_decoder_populates_buffer(self, mock_popen):
        """
        Test that _decoder_reader_loop correctly populates the _reference_buffer
        with 160-sample frames when fed dummy ffmpeg stdout.

        How to change for architectural updates:
        - If the ffmpeg command or parsing logic changes, the mock_popen setup needs
          to reflect the new expected stdout/stderr.
        - If AEC_READ_CHUNK or AEC_FRAME_SIZE changes, adjust dummy data and expected
          frame size accordingly.
        - If the underlying reference buffer implementation changes, update assertions.
        """
        # Mock ffmpeg stdout to return specific raw PCM data
        mock_stdout = io.BytesIO()
        # Create some dummy 16-bit PCM data (equivalent to float32 values 0.1, 0.2, ... after conversion)
        # We need data for at least two 160-sample frames (160 * 2 bytes/sample * 2 frames)
        raw_audio_int16 = np.array([0.1, 0.2] * AEC_READ_CHUNK, dtype=np.float32) * 32767
        raw_audio_int16 = raw_audio_int16.astype(np.int16)
        mock_stdout.write(raw_audio_int16.tobytes())
        mock_stdout.seek(0) # Rewind to start for reading

        mock_popen.return_value.stdout = mock_stdout
        mock_popen.return_value.poll.return_value = None # Simulate running process

        # Start the decoder (it will run in a separate thread)
        self.player.ffmpeg_path = '/usr/bin/ffmpeg' # Ensure ffmpeg_path is set
        self.player._start_reference_decoder("http://dummy.url")

        # Wait a moment for the thread to process
        self.player._decoder_thread.join(timeout=0.1)
        self.player._stop_reference_decoder() # Stop the thread gracefully

        self.assertGreater(len(self.player._reference_buffer), 0)
        # Check if frames are 160 samples and float32
        for frame in self.player._reference_buffer:
            self.assertEqual(len(frame), AEC_READ_CHUNK) # AEC_READ_CHUNK is 160
            self.assertEqual(frame.dtype, np.float32)


# --- Test Cases for voice.py ---
class TestVoiceProcessor(unittest.TestCase):
    def setUp(self):
        # Mock external dependencies for VoiceProcessor
        self.mock_music_player = MagicMock()
        self.mock_openai_client = MagicMock()
        self.mock_prosody_pipeline = MagicMock()
        self.mock_text_emotion_pipeline = MagicMock()
        self.mock_nlp = MagicMock()

        # Patch dependencies that VoiceProcessor lazy-loads
        self.mock_spacy_module = MagicMock()
        self.mock_spacy_module.load.return_value = self.mock_nlp
        patcher_modules = patch.dict('sys.modules', {
            'openai': MagicMock(),
            'spacy': self.mock_spacy_module,
        })
        patcher_prosody = patch('transformers.pipeline', side_effect=[
            self.mock_prosody_pipeline, # First call to pipeline
            self.mock_text_emotion_pipeline # Second call to pipeline
        ])

        patcher_modules.start()
        sys.modules['openai'].OpenAI.return_value = self.mock_openai_client # Configure the mocked OpenAI class
        self.addCleanup(patcher_modules.stop)

        self.mock_pipeline = patcher_prosody.start()
        self.addCleanup(patcher_prosody.stop)

        # Initialize VoiceProcessor with mocked dependencies
        self.processor = VoiceProcessor(
            chunk_duration_s=1.0, # Short chunk for testing
            sample_rate=SAMPLE_RATE,
            silence_threshold_rms=0.001,
            openai_api_key="dummy_key",
            music_player=self.mock_music_player,
            on_observation=MagicMock()
        )
        # Manually load models as start() calls _load_models()
        self.processor._load_models() 

    def tearDown(self):
        self.processor.close()

    def test_apply_aec_with_ref(self):
        """
        Test _apply_aec correctly processes mic_audio with a reference signal.
        This is a smoke test, as _nlms_aec's internal logic is not tested directly.

        How to change for architectural updates:
        - If AEC library or integration changes, this test needs to be adapted.
        - If input/output audio formats change, update numpy array types/ranges.
        """
        mic_audio = create_dummy_audio(AEC_READ_CHUNK / SAMPLE_RATE) # Mic audio same length as ref frame
        ref_audio_frame = create_dummy_audio(AEC_READ_CHUNK / SAMPLE_RATE, frequency=1000) # One frame of ref audio
        
        # Mock music_player to provide reference frames
        self.mock_music_player.get_reference_chunk.return_value = ref_audio_frame

        clean_audio = self.processor._apply_aec(mic_audio)
        self.assertEqual(clean_audio.shape, mic_audio.shape)
        self.assertIsInstance(clean_audio, np.ndarray)
        # A more rigorous test would check for actual echo reduction, but that's complex without a real AEC setup.
        # For now, we assume _nlms_aec works and check if the method runs without error and returns correct shape/type.

    def test_apply_aec_no_ref(self):
        """
        Test _apply_aec returns original mic_audio if no reference is available.

        How to change for architectural updates:
        - If reference handling changes, verify the fallback behavior.
        """
        mic_audio = create_dummy_audio(1.0)
        self.mock_music_player.get_reference_chunk.return_value = None # No reference available
        clean_audio = self.processor._apply_aec(mic_audio)
        np.testing.assert_array_equal(clean_audio, mic_audio) # Should return original if no ref

    def test_extract_acoustic_features_speech(self):
        """
        Test _extract_acoustic_features extracts expected features from speech audio.

        How to change for architectural updates:
        - If librosa functions or feature extraction logic changes, update expected features.
        - If AEC_FRAME_SIZE or hop_length changes, verify pyin/rms/mfcc calls.
        - If the return type of this function changes, update assertions.
        """
        speech_audio = create_dummy_speech_audio(2.0) # 2 seconds of speech
        features, f0_contour = self.processor._extract_acoustic_features(speech_audio)
        
        self.assertIsInstance(features, dict)
        self.assertIn("avg_f0", features)
        self.assertIn("avg_rms", features)
        self.assertIn("mfcc_0", features)
        self.assertIsInstance(f0_contour, np.ndarray)
        self.assertGreater(len(f0_contour), 0) # Should have some F0 values

    def test_extract_acoustic_features_silence(self):
        """
        Test _extract_acoustic_features handles silence (or empty audio).

        How to change for architectural updates:
        - Verify how silence is represented in features if the logic changes.
        """
        silence_audio = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        features, f0_contour = self.processor._extract_acoustic_features(silence_audio)
        self.assertIsInstance(features, dict)
        self.assertEqual(features["avg_f0"], 0.0) # Should be 0 for silence or NaN f0
        self.assertIsInstance(f0_contour, np.ndarray) # Still returns an array, possibly all NaN
        self.assertTrue(np.all(np.isnan(f0_contour)) or len(f0_contour) == 0)

    def test_analyze_inflection_patterns(self):
        """
        Test _analyze_inflection_patterns identifies rising/falling tones.

        How to change for architectural updates:
        - If inflection detection algorithm changes (e.g., uses more complex contour analysis),
          update dummy f0 and expected ratios.
        - If the output dictionary keys change, update assertions.
        """
        # Rising F0 contour
        rising_f0 = create_dummy_f0_contour(100, 100, 150)
        patterns = self.processor._analyze_inflection_patterns(rising_f0)
        self.assertGreater(patterns["rising_tone_ratio"], 0.5)
        self.assertLess(patterns["falling_tone_ratio"], 0.5)

        # Falling F0 contour
        falling_f0 = create_dummy_f0_contour(100, 150, 100)
        patterns = self.processor._analyze_inflection_patterns(falling_f0)
        self.assertLess(patterns["rising_tone_ratio"], 0.5)
        self.assertGreater(patterns["falling_tone_ratio"], 0.5)

        # Flat F0 contour
        flat_f0 = np.full(100, 120.0)
        patterns = self.processor._analyze_inflection_patterns(flat_f0)
        self.assertEqual(patterns["rising_tone_ratio"], 0.0)
        self.assertEqual(patterns["falling_tone_ratio"], 0.0)
        
        # All NaN F0 contour
        nan_f0 = np.full(100, np.nan)
        patterns = self.processor._analyze_inflection_patterns(nan_f0)
        self.assertEqual(patterns["rising_tone_ratio"], 0.0)
        self.assertEqual(patterns["falling_tone_ratio"], 0.0)

    def test_calculate_speech_rate(self):
        """
        Test _calculate_speech_rate correctly computes WPM.

        How to change for architectural updates:
        - If WPM calculation method changes (e.g., syllables per minute), update logic.
        - If average word length or assumed speaking speed changes, update expected values.
        """
        transcript = "This is a test sentence with several words."
        duration = 5.0 # seconds
        expected_wpm = (len(transcript.split()) / duration) * 60
        self.assertAlmostEqual(self.processor._calculate_speech_rate(transcript, duration), expected_wpm)

        self.assertIsNone(self.processor._calculate_speech_rate("", 5.0)) # Empty transcript
        self.assertIsNone(self.processor._calculate_speech_rate("hello", 0.0)) # Zero duration

    def test_infer_vocal_mood_high_wpm(self):
        """
        Test _infer_vocal_mood_from_features with high WPM boosting active emotions.
        """
        prosody_scores = {"neutral": 0.3, "happy": 0.4, "sad": 0.1, "angry": 0.2}
        prosody_emotion = "happy"
        speech_rate = 200.0 # High WPM
        vocal_dynamics = {"avg_rms": 0.05, "rms_variance": 0.0005}
        inflection_patterns = {"rising_tone_ratio": 0.5, "falling_tone_ratio": 0.5}

        vocal_mood, score = self.processor._infer_vocal_mood_from_features(
            prosody_emotion, prosody_scores, speech_rate, vocal_dynamics, inflection_patterns
        )
        self.assertIn(vocal_mood, ["happy", "angry"]) # Expect happy or angry to be boosted
        self.assertGreater(score, 0.4) # Should be boosted from initial happy score

    def test_infer_vocal_mood_low_wpm(self):
        """
        Test _infer_vocal_mood_from_features with low WPM boosting calm/sad emotions.
        """
        prosody_scores = {"happy": 0.3, "neutral": 0.4, "calm": 0.2, "sad": 0.1}
        prosody_emotion = "neutral"
        speech_rate = 70.0 # Low WPM
        vocal_dynamics = {"avg_rms": 0.01, "rms_variance": 0.00001}
        inflection_patterns = {"rising_tone_ratio": 0.2, "falling_tone_ratio": 0.8}

        vocal_mood, score = self.processor._infer_vocal_mood_from_features(
            prosody_emotion, prosody_scores, speech_rate, vocal_dynamics, inflection_patterns
        )
        self.assertIn(vocal_mood, ["sad", "calm", "neutral"]) # Should shift towards these
        self.assertGreater(score, 0.4) # Should be boosted from initial neutral score

    def test_infer_vocal_mood_combined_modulation(self):
        """
        Test _infer_vocal_mood_from_features with a combination of modulations.
        """
        prosody_scores = {"neutral": 0.3, "surprise": 0.4, "happy": 0.2, "angry": 0.1}
        prosody_emotion = "surprise"
        speech_rate = 150.0 # Slightly above avg
        vocal_dynamics = {"avg_rms": 0.08, "rms_variance": 0.0008} # Loud and dynamic
        inflection_patterns = {"rising_tone_ratio": 0.7, "falling_tone_ratio": 0.3} # Predominantly rising

        vocal_mood, score = self.processor._infer_vocal_mood_from_features(
            prosody_emotion, prosody_scores, speech_rate, vocal_dynamics, inflection_patterns
        )
        # Expected: Neutral should be reduced, surprise/happy/angry boosted due to dynamics and rising inflection
        self.assertIn(vocal_mood, ["surprise", "happy", "angry"])
        self.assertGreater(score, 0.4) # Should be boosted from neutral/surprise/happy base


if __name__ == '__main__':
    unittest.main()
