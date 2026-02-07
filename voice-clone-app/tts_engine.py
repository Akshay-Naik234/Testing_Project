import logging
import os
import uuid
from pathlib import Path

import numpy as np

from audio_processor import AudioProcessor
from config import AppConfig
from emotion_parser import EmotionParser, ScriptSegment
from voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cloner = VoiceCloner(
            model_name=config.tts_model,
            device=config.device,
        )
        self.processor = AudioProcessor(sample_rate=config.audio.sample_rate)
        self.parser = EmotionParser()

    def generate(
        self,
        script: str,
        speaker_wav: str,
        output_filename: str | None = None,
        language: str = "en",
    ) -> str:
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"Speaker WAV not found: {speaker_wav}")

        segments = self.parser.parse(script)
        logger.info("Parsed %d segment(s) from script.", len(segments))

        if not segments:
            raise ValueError("Script produced no valid segments after parsing.")

        self.cloner.load_model()
        actual_sr = self.cloner.output_sample_rate
        self.processor.sample_rate = actual_sr

        audio_segments: list[np.ndarray] = []
        for i, seg in enumerate(segments):
            logger.info(
                "Processing segment %d/%d — emotion=%s, text='%s'",
                i + 1, len(segments), seg.emotion, seg.text[:60],
            )
            audio = self._process_segment(seg, speaker_wav, language)
            audio_segments.append(audio)

        final_audio = self.processor.concatenate(audio_segments, crossfade_ms=30)
        final_audio = self.processor.normalize(final_audio)

        if output_filename is None:
            output_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
        output_path = str(self.config.output_dir / output_filename)
        self.processor.save_audio(final_audio, output_path)
        logger.info("Final audio saved to: %s", output_path)
        return output_path

    def _process_segment(
        self,
        segment: ScriptSegment,
        speaker_wav: str,
        language: str,
    ) -> np.ndarray:
        if segment.break_ms > 0:
            return self.processor.generate_silence(segment.break_ms)

        if not segment.text.strip():
            return self.processor.generate_silence(100)

        audio = self.cloner.synthesize_to_array(
            text=segment.text,
            speaker_wav=speaker_wav,
            language=language,
        )

        if segment.emotion != "neutral":
            audio = self.processor.apply_emotion(audio, segment.emotion_config)

        return audio

    def list_available_models(self) -> list[str]:
        try:
            from TTS.api import TTS
            return list(TTS().list_models())
        except Exception:
            return [
                "tts_models/multilingual/multi-dataset/xtts_v2",
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/vits",
            ]
