import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    output_format: str = "wav"


@dataclass
class EmotionConfig:
    pitch_shift_semitones: float = 0.0
    speed_factor: float = 1.0
    volume_factor: float = 1.0
    tremolo_depth: float = 0.0
    tremolo_rate: float = 0.0
    breathiness: float = 0.0


EMOTION_PRESETS: dict[str, EmotionConfig] = {
    "neutral": EmotionConfig(),
    "happy": EmotionConfig(
        pitch_shift_semitones=2.0,
        speed_factor=1.1,
        volume_factor=1.1,
        tremolo_depth=0.05,
        tremolo_rate=5.0,
    ),
    "sad": EmotionConfig(
        pitch_shift_semitones=-2.0,
        speed_factor=0.85,
        volume_factor=0.85,
        tremolo_depth=0.02,
        tremolo_rate=3.0,
    ),
    "angry": EmotionConfig(
        pitch_shift_semitones=1.5,
        speed_factor=1.15,
        volume_factor=1.3,
        tremolo_depth=0.08,
        tremolo_rate=6.0,
    ),
    "fearful": EmotionConfig(
        pitch_shift_semitones=3.0,
        speed_factor=1.2,
        volume_factor=0.9,
        tremolo_depth=0.1,
        tremolo_rate=7.0,
        breathiness=0.3,
    ),
    "surprised": EmotionConfig(
        pitch_shift_semitones=4.0,
        speed_factor=1.05,
        volume_factor=1.15,
    ),
    "disgusted": EmotionConfig(
        pitch_shift_semitones=-1.0,
        speed_factor=0.9,
        volume_factor=1.1,
        breathiness=0.2,
    ),
    "calm": EmotionConfig(
        pitch_shift_semitones=-1.0,
        speed_factor=0.85,
        volume_factor=0.75,
    ),
    "excited": EmotionConfig(
        pitch_shift_semitones=3.0,
        speed_factor=1.2,
        volume_factor=1.2,
        tremolo_depth=0.06,
        tremolo_rate=5.5,
    ),
    "whisper": EmotionConfig(
        pitch_shift_semitones=0.0,
        speed_factor=0.8,
        volume_factor=0.3,
        breathiness=0.7,
    ),
    "authoritative": EmotionConfig(
        pitch_shift_semitones=-2.5,
        speed_factor=0.9,
        volume_factor=1.25,
    ),
}


@dataclass
class AppConfig:
    base_dir: Path = field(default_factory=lambda: Path(os.path.dirname(os.path.abspath(__file__))))
    input_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cpu"

    def __post_init__(self):
        if self.input_dir is None:
            self.input_dir = self.base_dir / "input"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
