import re
from dataclasses import dataclass, field

from config import EMOTION_PRESETS, EmotionConfig


@dataclass
class ScriptSegment:
    text: str
    emotion: str = "neutral"
    intensity: float = 1.0
    emotion_config: EmotionConfig = field(default_factory=EmotionConfig)
    break_ms: int = 0


class EmotionParser:
    EMOTION_PATTERN = re.compile(
        r'<emotion\s+type=["\'](\w+)["\']\s*(?:intensity=["\']([0-9.]+)["\'])?\s*>'
        r'(.*?)'
        r'</emotion>',
        re.DOTALL | re.IGNORECASE,
    )

    SHORT_EMOTION_PATTERN = re.compile(
        r'<(\w+)\s*(?:intensity=["\']([0-9.]+)["\'])?\s*>'
        r'(.*?)'
        r'</\1>',
        re.DOTALL | re.IGNORECASE,
    )

    BREAK_PATTERN = re.compile(
        r'<break\s+time=["\'](\d+)(ms|s)["\']\s*/?>',
        re.IGNORECASE,
    )

    PROSODY_PATTERN = re.compile(
        r'<prosody\s+(.*?)>(.*?)</prosody>',
        re.DOTALL | re.IGNORECASE,
    )

    VALID_EMOTIONS = set(EMOTION_PRESETS.keys())

    def parse(self, script: str) -> list[ScriptSegment]:
        segments: list[ScriptSegment] = []
        script = self._normalize_whitespace(script)
        self._parse_recursive(script, "neutral", 1.0, segments)
        return [s for s in segments if s.text.strip() or s.break_ms > 0]

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def _parse_recursive(
        self,
        text: str,
        current_emotion: str,
        current_intensity: float,
        segments: list[ScriptSegment],
    ):
        combined_pattern = re.compile(
            r'<emotion\s+type=["\'](\w+)["\']\s*(?:intensity=["\']([0-9.]+)["\'])?\s*>(.*?)</emotion>'
            r'|<(\w+)\s*(?:intensity=["\']([0-9.]+)["\'])?\s*>(.*?)</\4>'
            r'|<break\s+time=["\'](\d+)(ms|s)["\']\s*/?>',
            re.DOTALL | re.IGNORECASE,
        )

        last_end = 0
        for match in combined_pattern.finditer(text):
            start = match.start()

            if start > last_end:
                plain_text = text[last_end:start].strip()
                if plain_text:
                    segments.append(self._create_segment(plain_text, current_emotion, current_intensity))

            if match.group(1) is not None:
                emotion = match.group(1).lower()
                intensity = float(match.group(2)) if match.group(2) else 1.0
                inner_text = match.group(3).strip()
                if emotion in self.VALID_EMOTIONS:
                    self._parse_recursive(inner_text, emotion, intensity, segments)
                else:
                    self._parse_recursive(inner_text, current_emotion, current_intensity, segments)

            elif match.group(4) is not None:
                emotion = match.group(4).lower()
                intensity = float(match.group(5)) if match.group(5) else 1.0
                inner_text = match.group(6).strip()
                if emotion in self.VALID_EMOTIONS:
                    self._parse_recursive(inner_text, emotion, intensity, segments)
                else:
                    self._parse_recursive(inner_text, current_emotion, current_intensity, segments)

            elif match.group(7) is not None:
                duration = int(match.group(7))
                unit = match.group(8).lower()
                if unit == "s":
                    duration *= 1000
                segments.append(ScriptSegment(text="", break_ms=duration))

            last_end = match.end()

        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append(self._create_segment(remaining, current_emotion, current_intensity))

    def _create_segment(self, text: str, emotion: str, intensity: float) -> ScriptSegment:
        base_config = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])
        scaled_config = self._scale_emotion(base_config, intensity)
        return ScriptSegment(
            text=text,
            emotion=emotion,
            intensity=intensity,
            emotion_config=scaled_config,
        )

    def _scale_emotion(self, config: EmotionConfig, intensity: float) -> EmotionConfig:
        if intensity == 1.0:
            return config
        neutral = EMOTION_PRESETS["neutral"]
        return EmotionConfig(
            pitch_shift_semitones=neutral.pitch_shift_semitones + (config.pitch_shift_semitones - neutral.pitch_shift_semitones) * intensity,
            speed_factor=neutral.speed_factor + (config.speed_factor - neutral.speed_factor) * intensity,
            volume_factor=neutral.volume_factor + (config.volume_factor - neutral.volume_factor) * intensity,
            tremolo_depth=neutral.tremolo_depth + (config.tremolo_depth - neutral.tremolo_depth) * intensity,
            tremolo_rate=neutral.tremolo_rate + (config.tremolo_rate - neutral.tremolo_rate) * intensity,
            breathiness=neutral.breathiness + (config.breathiness - neutral.breathiness) * intensity,
        )
