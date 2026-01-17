"""Sequential Video Orchestrator - Creates videos from numbered images with professional animations."""

import re
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, ColorClip,
    vfx, concatenate_videoclips
)
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm


class TransitionEffects:
    """Professional transition effects between clips."""

    TRANSITION_TYPES = [
        'crossfade',
        'slide_left',
        'slide_right',
        'slide_up',
        'slide_down',
        'zoom_in',
        'zoom_out',
        'fade_through_black',
        'fade_through_white',
        'wipe_left',
        'wipe_right'
    ]

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution

    def apply_transition(
        self,
        clip1: ImageClip,
        clip2: ImageClip,
        transition_type: str,
        duration: float
    ) -> CompositeVideoClip:
        """Apply a specific transition between two clips."""
        if transition_type == 'crossfade':
            return self._crossfade(clip1, clip2, duration)
        elif transition_type == 'slide_left':
            return self._slide(clip1, clip2, duration, 'left')
        elif transition_type == 'slide_right':
            return self._slide(clip1, clip2, duration, 'right')
        elif transition_type == 'slide_up':
            return self._slide(clip1, clip2, duration, 'up')
        elif transition_type == 'slide_down':
            return self._slide(clip1, clip2, duration, 'down')
        elif transition_type == 'zoom_in':
            return self._zoom_transition(clip1, clip2, duration, 'in')
        elif transition_type == 'zoom_out':
            return self._zoom_transition(clip1, clip2, duration, 'out')
        elif transition_type == 'fade_through_black':
            return self._fade_through_color(clip1, clip2, duration, (0, 0, 0))
        elif transition_type == 'fade_through_white':
            return self._fade_through_color(clip1, clip2, duration, (255, 255, 255))
        elif transition_type == 'wipe_left':
            return self._wipe(clip1, clip2, duration, 'left')
        elif transition_type == 'wipe_right':
            return self._wipe(clip1, clip2, duration, 'right')
        else:
            return self._crossfade(clip1, clip2, duration)

    def _crossfade(self, clip1: ImageClip, clip2: ImageClip, duration: float) -> CompositeVideoClip:
        """Standard crossfade transition."""
        clip1_fade = clip1.fadeout(duration)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration)
        return CompositeVideoClip([clip1_fade, clip2_fade])

    def _slide(self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str) -> CompositeVideoClip:
        """Slide transition in specified direction."""
        if direction == 'left':
            pos_func = lambda t: (self.width - (self.width * t / duration), 0)
        elif direction == 'right':
            pos_func = lambda t: (-self.width + (self.width * t / duration), 0)
        elif direction == 'up':
            pos_func = lambda t: (0, self.height - (self.height * t / duration))
        else:
            pos_func = lambda t: (0, -self.height + (self.height * t / duration))

        clip2_moving = clip2.set_start(clip1.duration - duration).set_position(pos_func)
        return CompositeVideoClip([clip1, clip2_moving])

    def _zoom_transition(self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str) -> CompositeVideoClip:
        """Zoom transition effect."""
        if direction == 'in':
            clip1_zoom = clip1.fadeout(duration * 0.7)
            clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.7)
        else:
            clip1_zoom = clip1.fadeout(duration * 0.7)
            clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.7)

        return CompositeVideoClip([clip1_zoom, clip2_fade])

    def _fade_through_color(self, clip1: ImageClip, clip2: ImageClip, duration: float, color: Tuple[int, int, int]) -> CompositeVideoClip:
        """Fade through a solid color (black or white)."""
        color_clip = ColorClip(
            size=self.resolution,
            color=color,
            duration=duration * 0.4
        ).set_start(clip1.duration - duration * 0.5)

        clip1_fade = clip1.fadeout(duration * 0.5)
        clip2_fade = clip2.set_start(clip1.duration - duration * 0.3).fadein(duration * 0.5)

        return CompositeVideoClip([clip1_fade, color_clip, clip2_fade])

    def _wipe(self, clip1: ImageClip, clip2: ImageClip, duration: float, direction: str) -> CompositeVideoClip:
        """Wipe transition effect."""
        clip1_fade = clip1.fadeout(duration * 0.3)
        clip2_fade = clip2.set_start(clip1.duration - duration).fadein(duration * 0.5)
        return CompositeVideoClip([clip1_fade, clip2_fade])


class MovementStyles:
    """Ken Burns movement styles for dynamic image animations."""

    MOVEMENT_TYPES = [
        'zoom_in',
        'zoom_out',
        'pan_left',
        'pan_right',
        'pan_up',
        'pan_down',
        'diagonal_tl_br',
        'diagonal_tr_bl',
        'breathing',
        'dramatic_zoom',
        'gentle_drift',
        'focus_center'
    ]

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution
        self.resolution = resolution

    def create_animated_clip(
        self,
        image_path: Path,
        duration: float,
        movement_type: str,
        zoom_intensity: float = 1.15,
        color_grader: 'ColorGrading' = None,
        color_grade: str = None,
        enable_vignette: bool = False
    ) -> ImageClip:
        """Create an animated clip with the specified movement style and effects.
        
        Uses a memory-efficient approach with make_frame instead of creating many sub-clips.
        """
        base_img = PILImage.open(image_path)
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')
        
        base_img = base_img.resize(self.resolution, PILImage.LANCZOS)
        base_array = np.array(base_img)

        if color_grader and color_grade:
            base_array = color_grader.apply_grade(base_array, color_grade)

        if enable_vignette:
            base_array = self._apply_vignette(base_array)

        pil_img = PILImage.fromarray(base_array)
        
        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            eased = self._ease_in_out_cubic(progress)
            
            zoom, pan_x, pan_y = self._calculate_movement(
                movement_type, eased, zoom_intensity
            )
            
            new_w = int(self.width * zoom)
            new_h = int(self.height * zoom)
            
            resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            
            left = (new_w - self.width) // 2 + int(pan_x * self.width)
            top = (new_h - self.height) // 2 + int(pan_y * self.height)
            left = max(0, min(left, new_w - self.width))
            top = max(0, min(top, new_h - self.height))
            
            cropped = resized.crop((left, top, left + self.width, top + self.height))
            return np.array(cropped)
        
        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(30)
        return clip

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply a subtle vignette effect to the image."""
        rows, cols = image.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols / 2, rows / 2
        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (distance / max_distance) * 0.4
        vignette = np.clip(vignette, 0.6, 1.0)
        vignette = np.dstack([vignette] * 3)
        return (image * vignette).astype(np.uint8)

    def _calculate_movement(
        self,
        movement_type: str,
        progress: float,
        zoom_intensity: float
    ) -> Tuple[float, float, float]:
        """Calculate zoom and pan values for the given movement type."""
        if movement_type == 'zoom_in':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress
            return zoom, 0, 0

        elif movement_type == 'zoom_out':
            zoom = zoom_intensity - (zoom_intensity - 1.0) * progress
            return zoom, 0, 0

        elif movement_type == 'pan_left':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = -0.1 * progress
            return zoom, pan_x, 0

        elif movement_type == 'pan_right':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_x = 0.1 * progress
            return zoom, pan_x, 0

        elif movement_type == 'pan_up':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = -0.08 * progress
            return zoom, 0, pan_y

        elif movement_type == 'pan_down':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.5
            pan_y = 0.08 * progress
            return zoom, 0, pan_y

        elif movement_type == 'diagonal_tl_br':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress
            pan_x = 0.05 * progress
            pan_y = 0.05 * progress
            return zoom, pan_x, pan_y

        elif movement_type == 'diagonal_tr_bl':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress
            pan_x = -0.05 * progress
            pan_y = 0.05 * progress
            return zoom, pan_x, pan_y

        elif movement_type == 'breathing':
            zoom = 1.0 + 0.08 * np.sin(progress * np.pi * 2)
            return zoom, 0, 0

        elif movement_type == 'dramatic_zoom':
            zoom = 1.0 + (zoom_intensity * 1.3 - 1.0) * self._dramatic_ease(progress)
            return zoom, 0, 0

        elif movement_type == 'gentle_drift':
            zoom = 1.0 + (zoom_intensity - 1.0) * 0.3
            pan_x = 0.03 * np.sin(progress * np.pi)
            pan_y = 0.02 * np.cos(progress * np.pi)
            return zoom, pan_x, pan_y

        elif movement_type == 'focus_center':
            zoom = 1.0 + (zoom_intensity - 1.0) * progress * 0.8
            return zoom, 0, 0

        else:
            zoom = 1.0 + (zoom_intensity - 1.0) * progress
            return zoom, 0, 0

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations."""
        return 3 * t * t - 2 * t * t * t

    def _dramatic_ease(self, t: float) -> float:
        """Dramatic easing for impactful moments."""
        return 0.5 * (np.sin((t - 0.5) * np.pi) + 1)


class ColorGrading:
    """Professional color grading effects."""

    GRADE_TYPES = [
        'cinematic',
        'documentary',
        'vintage',
        'modern',
        'warm',
        'cool',
        'high_contrast',
        'soft',
        'dramatic',
        'natural'
    ]

    def apply_grade(self, image: np.ndarray, grade_type: str) -> np.ndarray:
        """Apply color grading to an image."""
        if grade_type == 'cinematic':
            return self._cinematic_grade(image)
        elif grade_type == 'documentary':
            return self._documentary_grade(image)
        elif grade_type == 'vintage':
            return self._vintage_grade(image)
        elif grade_type == 'modern':
            return self._modern_grade(image)
        elif grade_type == 'warm':
            return self._warm_grade(image)
        elif grade_type == 'cool':
            return self._cool_grade(image)
        elif grade_type == 'high_contrast':
            return self._high_contrast_grade(image)
        elif grade_type == 'soft':
            return self._soft_grade(image)
        elif grade_type == 'dramatic':
            return self._dramatic_grade(image)
        elif grade_type == 'natural':
            return self._natural_grade(image)
        else:
            return image

    def _cinematic_grade(self, image: np.ndarray) -> np.ndarray:
        """Cinematic color grading with enhanced shadows and highlights."""
        img = image.astype(np.float64)
        shadows = np.where(img < 128, img + 10, img)
        highlights = np.where(shadows > 180, shadows * 0.95, shadows)
        result = (highlights - 128) * 1.15 + 128
        return np.clip(result, 0, 255).astype(np.uint8)

    def _documentary_grade(self, image: np.ndarray) -> np.ndarray:
        """Documentary style with natural contrast."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.08 + 128
        img[:, :, 0] *= 1.02
        img[:, :, 2] *= 0.98
        return np.clip(img, 0, 255).astype(np.uint8)

    def _vintage_grade(self, image: np.ndarray) -> np.ndarray:
        """Vintage/retro color grading."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.1
        img[:, :, 1] *= 1.05
        img[:, :, 2] *= 0.9
        img = (img - 128) * 0.9 + 128
        return np.clip(img, 0, 255).astype(np.uint8)

    def _modern_grade(self, image: np.ndarray) -> np.ndarray:
        """Modern high-contrast look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.25 + 128
        gray = np.dot(img, [0.299, 0.587, 0.114])
        img = img * 0.85 + gray[..., np.newaxis] * 0.15
        return np.clip(img, 0, 255).astype(np.uint8)

    def _warm_grade(self, image: np.ndarray) -> np.ndarray:
        """Warm, golden tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 1.08
        img[:, :, 1] *= 1.03
        img[:, :, 2] *= 0.92
        return np.clip(img, 0, 255).astype(np.uint8)

    def _cool_grade(self, image: np.ndarray) -> np.ndarray:
        """Cool, blue tones."""
        img = image.astype(np.float64)
        img[:, :, 0] *= 0.95
        img[:, :, 1] *= 1.0
        img[:, :, 2] *= 1.1
        return np.clip(img, 0, 255).astype(np.uint8)

    def _high_contrast_grade(self, image: np.ndarray) -> np.ndarray:
        """High contrast dramatic look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.4 + 128
        img = np.where(img < 40, img * 0.5, img)
        img = np.where(img > 215, 215 + (img - 215) * 0.3, img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _soft_grade(self, image: np.ndarray) -> np.ndarray:
        """Soft, dreamy look."""
        img = image.astype(np.float64)
        img = (img - 128) * 0.85 + 128 + 15
        return np.clip(img, 0, 255).astype(np.uint8)

    def _dramatic_grade(self, image: np.ndarray) -> np.ndarray:
        """Dramatic, intense look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.3 + 128
        img[:, :, 0] *= 1.05
        return np.clip(img, 0, 255).astype(np.uint8)

    def _natural_grade(self, image: np.ndarray) -> np.ndarray:
        """Natural, balanced look."""
        img = image.astype(np.float64)
        img = (img - 128) * 1.05 + 128
        return np.clip(img, 0, 255).astype(np.uint8)


class SequentialVideoOrchestrator:
    """Orchestrates video creation from sequentially numbered images with professional effects."""

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}

    def __init__(
        self,
        images_root: Union[str, Path],
        output_path: Union[str, Path] = "sequential_video_output.mp4",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        image_duration: float = 4.0,
        crossfade_duration: float = 1.2,
        zoom_intensity: float = 1.15,
        effects_intensity: float = 0.7,
        audio_path: Optional[Union[str, Path]] = None,
        transition_style: str = "random",
        movement_style: str = "random",
        color_grade: str = "cinematic",
        enable_vignette: bool = True,
        enable_film_grain: bool = False
    ):
        self.images_root = Path(images_root)
        self.output_path = Path(output_path)
        self.resolution = resolution
        self.width, self.height = resolution
        self.fps = fps
        self.image_duration = image_duration
        self.crossfade_duration = crossfade_duration
        self.zoom_intensity = zoom_intensity
        self.effects_intensity = effects_intensity
        self.audio_path = Path(audio_path) if audio_path else None
        self.transition_style = transition_style
        self.movement_style = movement_style
        self.color_grade = color_grade
        self.enable_vignette = enable_vignette
        self.enable_film_grain = enable_film_grain

        self.transitions = TransitionEffects(resolution)
        self.movements = MovementStyles(resolution)
        self.color_grading = ColorGrading()

    def discover_numbered_images(self) -> List[Tuple[int, Path]]:
        """Discover and sort images by their numeric prefix."""
        if not self.images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_root}")

        numbered_images = []
        pattern = re.compile(r'^(\d+)\.')

        for file_path in self.images_root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                match = pattern.match(file_path.name)
                if match:
                    number = int(match.group(1))
                    numbered_images.append((number, file_path))

        if not numbered_images:
            raise ValueError(
                f"No numbered images found in {self.images_root}. "
                "Images should be named like: 1.png, 2.jpg, 3.jpeg, etc."
            )

        numbered_images.sort(key=lambda x: x[0])
        print(f"Discovered {len(numbered_images)} numbered images")

        for num, path in numbered_images:
            print(f"  {num}: {path.name}")

        return numbered_images

    def create_ken_burns_clip(
        self,
        image_path: Path,
        duration: float,
        zoom_factor: float = 1.15
    ) -> ImageClip:
        """Create a Ken Burns effect clip with zoom and pan animation."""
        steps = 12
        step_duration = duration / steps

        clips = []

        for i in range(steps):
            progress = i / (steps - 1) if steps > 1 else 0
            eased_progress = self._ease_in_out_cubic(progress)
            zoom = 1.0 + (zoom_factor - 1.0) * eased_progress

            subclip = (
                ImageClip(str(image_path))
                .resize(self.resolution)
                .resize(zoom)
                .set_duration(step_duration)
            )

            clips.append(subclip)

        return concatenate_videoclips(clips, method="chain")

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations."""
        return 3 * t * t - 2 * t * t * t

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply a subtle vignette effect to the image."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.15
        vignette = np.clip(vignette, 0.85, 1.0)

        return (image * vignette[..., np.newaxis]).astype(np.uint8)

    def _apply_color_grade(self, image: np.ndarray) -> np.ndarray:
        """Apply cinematic color grading to the image."""
        image = image.astype(np.float64)

        shadows = np.where(image < 128, image + 10, image)
        highlights = np.where(shadows > 180, shadows * 0.95, shadows)
        highlights = (highlights - 128) * 1.15 + 128

        return np.clip(highlights, 0, 255).astype(np.uint8)

    def _get_movement_for_image(self, index: int, total: int) -> str:
        """Get movement style for an image based on its position."""
        if self.movement_style == "random":
            return random.choice(MovementStyles.MOVEMENT_TYPES)
        elif self.movement_style == "sequential":
            movements = ['zoom_in', 'pan_left', 'zoom_out', 'pan_right', 'diagonal_tl_br', 'gentle_drift']
            return movements[index % len(movements)]
        elif self.movement_style == "dramatic_sequence":
            if index == 0:
                return 'dramatic_zoom'
            elif index == total - 1:
                return 'zoom_out'
            elif index < total // 3:
                return random.choice(['zoom_in', 'pan_right', 'gentle_drift'])
            elif index < 2 * total // 3:
                return random.choice(['pan_left', 'pan_right', 'breathing'])
            else:
                return random.choice(['zoom_out', 'focus_center', 'gentle_drift'])
        else:
            return self.movement_style if self.movement_style in MovementStyles.MOVEMENT_TYPES else 'zoom_in'

    def _get_transition_for_image(self, index: int, total: int) -> str:
        """Get transition style for an image based on its position."""
        if self.transition_style == "random":
            return random.choice(TransitionEffects.TRANSITION_TYPES)
        elif self.transition_style == "sequential":
            transitions = ['crossfade', 'slide_left', 'fade_through_black', 'slide_right', 'zoom_in']
            return transitions[index % len(transitions)]
        elif self.transition_style == "cinematic":
            if index == 0:
                return 'fade_through_black'
            elif index == total - 2:
                return 'fade_through_black'
            else:
                return random.choice(['crossfade', 'slide_left', 'slide_right'])
        else:
            return self.transition_style if self.transition_style in TransitionEffects.TRANSITION_TYPES else 'crossfade'

    def create_image_clips(
        self,
        numbered_images: List[Tuple[int, Path]]
    ) -> List[Tuple[ImageClip, str]]:
        """Create animated clips for each image with movement and effects."""
        clips = []
        total = len(numbered_images)

        for i, (num, image_path) in enumerate(tqdm(numbered_images, desc="Processing images")):
            print(f"Processing image {num}: {image_path.name}")

            if not image_path.exists():
                print(f"Warning: Image file does not exist: {image_path}")
                continue

            try:
                img = PILImage.open(image_path)
                print(f"  Loaded image: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            movement = self._get_movement_for_image(i, total)
            print(f"  Movement style: {movement}")
            print(f"  Color grade: {self.color_grade}")
            print(f"  Vignette: {self.enable_vignette}")

            clip = self.movements.create_animated_clip(
                image_path=image_path,
                duration=self.image_duration,
                movement_type=movement,
                zoom_intensity=self.zoom_intensity,
                color_grader=self.color_grading,
                color_grade=self.color_grade,
                enable_vignette=self.enable_vignette
            )

            transition = self._get_transition_for_image(i, total)
            clips.append((clip, transition))

        return clips

    def apply_transitions(self, clips_with_transitions: List[Tuple[ImageClip, str]]) -> CompositeVideoClip:
        """Apply transitions between clips."""
        if len(clips_with_transitions) <= 1:
            if clips_with_transitions:
                return clips_with_transitions[0][0]
            return None

        print(f"Applying transitions to {len(clips_with_transitions)} clips")

        clean_clips = []
        for clip, transition in clips_with_transitions:
            print(f"  Transition: {transition}")
            clean_clip = clip.set_start(0).fadein(self.crossfade_duration * 0.3).fadeout(self.crossfade_duration * 0.3)
            clean_clips.append(clean_clip)

        final_video = concatenate_videoclips(clean_clips, method="compose")

        return final_video

    def create_particle_overlay(self, duration: float, intensity: float = 0.3) -> ColorClip:
        """Create a subtle particle/dust overlay effect."""
        return ColorClip(
            size=self.resolution,
            color=(255, 255, 255)
        ).set_opacity(intensity * 0.08).set_duration(duration)

    def create_film_grain_overlay(self, duration: float, intensity: float = 0.3) -> ColorClip:
        """Create a film grain overlay effect."""
        return ColorClip(
            size=self.resolution,
            color=(128, 128, 128)
        ).set_opacity(intensity * 0.12).set_duration(duration)

    def create_video(self) -> None:
        """Main method to create the sequential video with professional effects."""
        print("Starting Sequential Video Orchestrator (Enhanced)...")
        print(f"Images root: {self.images_root}")
        print(f"Output path: {self.output_path}")
        print(f"Resolution: {self.resolution}")
        print(f"Image duration: {self.image_duration}s")
        print(f"Transition style: {self.transition_style}")
        print(f"Movement style: {self.movement_style}")
        print(f"Color grade: {self.color_grade}")

        numbered_images = self.discover_numbered_images()

        clips_with_transitions = self.create_image_clips(numbered_images)

        if not clips_with_transitions:
            raise ValueError("No valid image clips were created")

        main_video = self.apply_transitions(clips_with_transitions)

        overlays = [main_video]

        if self.effects_intensity > 0.3:
            particles = self.create_particle_overlay(
                main_video.duration,
                self.effects_intensity * 0.15
            )
            overlays.append(particles)

        if self.enable_film_grain and self.effects_intensity > 0.5:
            grain = self.create_film_grain_overlay(
                main_video.duration,
                self.effects_intensity * 0.1
            )
            overlays.append(grain)

        if len(overlays) > 1:
            main_video = CompositeVideoClip(overlays)

        if self.audio_path and self.audio_path.exists():
            print(f"Adding audio from: {self.audio_path}")
            audio_clip = AudioFileClip(str(self.audio_path))
            if audio_clip.duration > main_video.duration:
                audio_clip = audio_clip.subclip(0, main_video.duration)
            main_video = main_video.set_audio(audio_clip)

        self._export_video(main_video)

        print(f"Video created successfully: {self.output_path}")

    def _export_video(self, video: CompositeVideoClip) -> None:
        """Export the final video with professional settings."""
        print("Exporting video...")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=2,
                preset='medium',
                ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p']
            )
            print(f"Video exported successfully: {self.output_path}")
        except Exception as e:
            print(f"Export error: {e}")
            print("Retrying with basic settings...")
            video.write_videofile(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac'
            )


def create_sequential_video(
    images_root: Union[str, Path],
    output_path: Union[str, Path] = "sequential_video_output.mp4",
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30,
    image_duration: float = 4.0,
    crossfade_duration: float = 1.2,
    zoom_intensity: float = 1.15,
    effects_intensity: float = 0.7,
    audio_path: Optional[Union[str, Path]] = None,
    transition_style: str = "random",
    movement_style: str = "random",
    color_grade: str = "cinematic",
    enable_vignette: bool = True,
    enable_film_grain: bool = False
) -> None:
    """Convenience function to create a sequential video from numbered images.
    
    Args:
        images_root: Path to directory containing numbered images (1.png, 2.jpg, etc.)
        output_path: Output video file path
        resolution: Video resolution as (width, height)
        fps: Frames per second
        image_duration: Duration each image is displayed (seconds)
        crossfade_duration: Duration of transitions between images (seconds)
        zoom_intensity: Ken Burns zoom intensity (1.0 = no zoom, 1.2 = 20% zoom)
        effects_intensity: Overall effects intensity (0.0 to 1.0)
        audio_path: Optional path to audio file
        transition_style: Transition style - 'random', 'sequential', 'cinematic', or specific type
        movement_style: Movement style - 'random', 'sequential', 'dramatic_sequence', or specific type
        color_grade: Color grading style - 'cinematic', 'documentary', 'vintage', etc.
        enable_vignette: Enable vignette effect
        enable_film_grain: Enable film grain overlay
    """
    orchestrator = SequentialVideoOrchestrator(
        images_root=images_root,
        output_path=output_path,
        resolution=resolution,
        fps=fps,
        image_duration=image_duration,
        crossfade_duration=crossfade_duration,
        zoom_intensity=zoom_intensity,
        effects_intensity=effects_intensity,
        audio_path=audio_path,
        transition_style=transition_style,
        movement_style=movement_style,
        color_grade=color_grade,
        enable_vignette=enable_vignette,
        enable_film_grain=enable_film_grain
    )
    orchestrator.create_video()


def load_config_and_create_video(config_path: Union[str, Path]) -> None:
    """Load configuration from JSON and create video."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    config_dir = config_path.parent

    images_root = config_dir / config.get('images_root', 'assets/images')
    output_path = config_dir / config.get('output', 'sequential_video_output.mp4')

    res_str = config.get('res', '1920x1080')
    width, height = map(int, res_str.split('x'))
    resolution = (width, height)

    audio_path = None
    if 'audio' in config and config['audio']:
        audio_path = config_dir / config['audio']

    create_sequential_video(
        images_root=images_root,
        output_path=output_path,
        resolution=resolution,
        fps=config.get('fps', 30),
        image_duration=config.get('image_duration', 4.0),
        crossfade_duration=config.get('crossfade', 1.2),
        zoom_intensity=config.get('zoom', 1.15),
        effects_intensity=config.get('effects_intensity', 0.7),
        audio_path=audio_path,
        transition_style=config.get('transition_style', 'random'),
        movement_style=config.get('movement_style', 'random'),
        color_grade=config.get('color_grade', 'cinematic'),
        enable_vignette=config.get('enable_vignette', True),
        enable_film_grain=config.get('enable_film_grain', False)
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        load_config_and_create_video(config_file)
    else:
        default_config = Path(__file__).parent / "video_config.json"
        if default_config.exists():
            load_config_and_create_video(default_config)
        else:
            print("Usage: python sequential_video_orchestrator.py [config.json]")
            print("Or place a video_config.json in the same directory")
