import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VoiceCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tts = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return
        logger.info("Loading TTS model: %s (this may take a while on first run)...", self.model_name)
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name).to(self.device)
            self._loaded = True
            logger.info("TTS model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load TTS model: %s", e)
            raise RuntimeError(
                f"Could not load TTS model '{self.model_name}'. "
                f"Make sure you have installed the TTS package: pip install TTS\n"
                f"Error: {e}"
            ) from e

    def clone_and_speak(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
        language: str = "en",
    ) -> str:
        self.load_model()
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav}")

        logger.info("Generating speech for: '%s'", text[:80] + ("..." if len(text) > 80 else ""))
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )
        logger.info("Audio saved to: %s", output_path)
        return output_path

    def synthesize_to_array(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
    ) -> np.ndarray:
        self.load_model()
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav}")

        logger.info("Synthesizing: '%s'", text[:80] + ("..." if len(text) > 80 else ""))
        wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
        )
        return np.array(wav, dtype=np.float32)

    @property
    def output_sample_rate(self) -> int:
        self.load_model()
        if hasattr(self.tts, "synthesizer") and self.tts.synthesizer is not None:
            return self.tts.synthesizer.output_sample_rate
        return 22050
