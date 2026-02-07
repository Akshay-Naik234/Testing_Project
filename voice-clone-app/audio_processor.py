import numpy as np
import soundfile as sf

from config import EmotionConfig


class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def apply_emotion(self, audio: np.ndarray, config: EmotionConfig) -> np.ndarray:
        audio = self._apply_pitch_shift(audio, config.pitch_shift_semitones)
        audio = self._apply_speed_change(audio, config.speed_factor)
        audio = self._apply_volume(audio, config.volume_factor)
        if config.tremolo_depth > 0:
            audio = self._apply_tremolo(audio, config.tremolo_depth, config.tremolo_rate)
        if config.breathiness > 0:
            audio = self._apply_breathiness(audio, config.breathiness)
        return self.normalize(audio)

    def _apply_pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.01:
            return audio
        factor = 2.0 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]

    def _apply_speed_change(self, audio: np.ndarray, factor: float) -> np.ndarray:
        if abs(factor - 1.0) < 0.01:
            return audio
        length = len(audio)
        new_length = int(length / factor)
        indices = np.linspace(0, length - 1, new_length).astype(int)
        return audio[indices]

    def _apply_volume(self, audio: np.ndarray, factor: float) -> np.ndarray:
        return audio * factor

    def _apply_tremolo(self, audio: np.ndarray, depth: float, rate: float) -> np.ndarray:
        t = np.arange(len(audio)) / self.sample_rate
        modulation = 1.0 + depth * np.sin(2.0 * np.pi * rate * t)
        return audio * modulation

    def _apply_breathiness(self, audio: np.ndarray, amount: float) -> np.ndarray:
        noise = np.random.normal(0, 0.02, len(audio))
        return audio * (1 - amount) + noise * amount

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        return audio

    def concatenate(self, segments: list[np.ndarray], crossfade_ms: int = 30) -> np.ndarray:
        if not segments:
            return np.array([], dtype=np.float32)
        if len(segments) == 1:
            return segments[0]

        crossfade_samples = int(self.sample_rate * crossfade_ms / 1000)
        result = segments[0].copy()

        for seg in segments[1:]:
            if len(result) < crossfade_samples or len(seg) < crossfade_samples:
                result = np.concatenate([result, seg])
                continue

            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)

            overlap = (result[-crossfade_samples:] * fade_out + seg[:crossfade_samples] * fade_in)
            result = np.concatenate([result[:-crossfade_samples], overlap, seg[crossfade_samples:]])

        return result

    def generate_silence(self, duration_ms: int) -> np.ndarray:
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)

    def save_audio(self, audio: np.ndarray, filepath: str):
        sf.write(filepath, audio, self.sample_rate)

    def load_audio(self, filepath: str) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(filepath, dtype="float32")
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)
        return audio, self.sample_rate

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length).astype(int)
        return audio[indices]
