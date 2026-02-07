#!/usr/bin/env python3
"""
Voice Clone Application
=======================
A local voice cloning and text-to-speech application with emotion tag support.
No API keys required — uses Coqui TTS (open-source) under the hood.

Usage:
    python main.py --script "Hello world" --voice input/sample.wav
    python main.py --script-file script.txt --voice input/sample.wav --output my_audio.wav
    python main.py --script '<happy>Great news!</happy> <sad>But also some bad news.</sad>' --voice input/sample.wav

Emotion Tags:
    <emotion type="happy">text</emotion>
    <sad>text</sad>
    <angry intensity="0.5">text</angry>
    <break time="500ms"/>

Supported emotions:
    neutral, happy, sad, angry, fearful, surprised,
    disgusted, calm, excited, whisper, authoritative
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from config import AppConfig
from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-clone")


def find_speaker_wav(voice_arg: str, config: AppConfig) -> str:
    path = Path(voice_arg)
    if path.is_file():
        return str(path.resolve())

    in_input = config.input_dir / voice_arg
    if in_input.is_file():
        return str(in_input.resolve())

    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        candidate = config.input_dir / (voice_arg + ext)
        if candidate.is_file():
            return str(candidate.resolve())

    wav_files = list(config.input_dir.glob("*.wav"))
    if wav_files:
        logger.warning(
            "Voice file '%s' not found. Using first WAV in input/: %s",
            voice_arg, wav_files[0].name,
        )
        return str(wav_files[0].resolve())

    raise FileNotFoundError(
        f"Could not find voice file '{voice_arg}'. "
        f"Place a .wav file in the input/ directory or provide a full path."
    )


def read_script(args) -> str:
    if args.script:
        return args.script
    if args.script_file:
        p = Path(args.script_file)
        if not p.is_file():
            raise FileNotFoundError(f"Script file not found: {args.script_file}")
        return p.read_text(encoding="utf-8").strip()
    raise ValueError("Provide --script or --script-file")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Clone — local TTS with voice cloning and emotion tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--script", type=str, default=None,
        help="Inline script text (supports emotion tags)",
    )
    parser.add_argument(
        "--script-file", type=str, default=None,
        help="Path to a text file containing the script",
    )
    parser.add_argument(
        "--voice", type=str, required=True,
        help="Path to the speaker WAV sample (or filename in input/ dir)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (saved in output/ dir). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="TTS model name (default: xtts_v2)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--list-emotions", action="store_true",
        help="Print supported emotions and exit",
    )

    args = parser.parse_args()

    if args.list_emotions:
        from config import EMOTION_PRESETS
        print("\nSupported emotion tags:\n")
        for name, cfg in EMOTION_PRESETS.items():
            print(f"  <{name}>text</{name}>")
            print(f"      pitch={cfg.pitch_shift_semitones:+.1f}st  "
                  f"speed={cfg.speed_factor:.2f}x  "
                  f"volume={cfg.volume_factor:.2f}x")
        print("\nYou can also set intensity: <happy intensity=\"0.5\">text</happy>")
        print("And insert pauses: <break time=\"500ms\"/>")
        return

    config = AppConfig(device=args.device)
    if args.model:
        config.tts_model = args.model

    script = read_script(args)
    speaker_wav = find_speaker_wav(args.voice, config)

    logger.info("Script length: %d chars", len(script))
    logger.info("Speaker WAV: %s", speaker_wav)
    logger.info("Device: %s", config.device)

    engine = TTSEngine(config)
    start = time.time()
    output_path = engine.generate(
        script=script,
        speaker_wav=speaker_wav,
        output_filename=args.output,
        language=args.language,
    )
    elapsed = time.time() - start

    print(f"\nDone! Audio saved to: {output_path}")
    print(f"Generation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
