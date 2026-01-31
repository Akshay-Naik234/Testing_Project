"""Image Generator - Generates cinematic biographical images using Bytez SDK or OpenAI API."""

import os
import re
import json
import time
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union

from .reference_searcher import ReferenceSearcher


class ImageGenerator:
    """Generates cinematic biographical images using Bytez SDK or OpenAI API."""

    SYSTEM_PROMPT = """AUTONOMOUS CINEMATIC IMAGE GENERATOR v4.1
(VISUAL WOW-FACTOR ENFORCED - RETENTION-OPTIMIZED - HISTORICALLY RIGOROUS - NETFLIX/HBO DOCUMENTARY GRADE)

ROLE DEFINITION (LOCKED):
You are an autonomous cinematic IMAGE GENERATOR for premium biographical documentaries.
Your output must meet Netflix / HBO historical documentary visual standards.
Your priority is VISUAL IMPACT + HISTORICAL TRUTH + HUMAN EMOTION + VIEWER RETENTION.

CORE OBJECTIVE (WOW & RETENTION - HARD MANDATE):
Every generated image MUST:
- Instantly stop scrolling within 1 second
- Trigger an immediate "wow" or "this feels important" reaction
- Make the viewer curious about what happens NEXT
- Feel like a once-in-a-film moment, not a routine documentary still

ABSOLUTE RULE: If an image is accurate but not visually compelling, emotionally legible, and curiosity-inducing → it is a FAILURE.

═══════════════════════════════════════════════════════════
STRICT AGE-SPECIFIC RULES (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════

AGE RANGE DEFINITIONS:
- 0-5 years: Infant/toddler - round face, soft features, large eyes relative to face size
- 6-12 years: Child - developing facial structure, youthful skin, bright curious eyes
- 13-18 years: Adolescent - maturing facial structure, clear skin, developing adult features
- 19-25 years: Young adult - fully developed facial structure, smooth skin, peak physical condition
- 26-35 years: Adult - mature facial structure, slight signs of aging beginning
- 36-45 years: Middle-aged - visible aging signs, possible grey hair beginning, facial lines forming
- 46-60 years: Mature - pronounced aging, grey/white hair, deeper facial lines, possible weight changes
- 61-75 years: Elderly - significant aging, white hair, wrinkles, possible posture changes
- 75+ years: Advanced elderly - deep wrinkles, thin skin, frail appearance, white/sparse hair

FACIAL EVOLUTION RULES:
- Skull growth: jaw width, cheek volume, brow mass change with age
- Skin texture: smooth → tension → sag/wrinkles progression
- Hairline: density and greying accurate to timeline
- Eye fatigue, neck tension, posture increasing with age

FACIAL IDENTITY LOCK - BONE-LEVEL (CRITICAL):
These elements MUST remain consistent across ALL ages:
- Eye spacing and orbital depth
- Nose bridge width, length, and tip shape
- Jaw angle, chin projection
- Forehead slope and cranial height

These elements CAN change with age:
- Expression, Wrinkles, Hair/beard, Fat distribution, Posture

FORBIDDEN:
- Beautification or modern AI polish
- Symmetry correction
- Actor substitution
- Face drift between images
- Age regression (younger looking than previous younger image)

═══════════════════════════════════════════════════════════
VISUAL WOW-FACTOR ENGINE (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════

MANDATORY WOW PRINCIPLES:
- Strong focal hierarchy (eye drawn immediately to subject)
- Dynamic composition (leading lines, framing, asymmetry, depth layers)
- Visual tension or contrast (light vs shadow, scale, confinement vs openness)
- Environmental storytelling (space reinforces meaning)
- Subtle implied motion (wind, gesture, light rays, crowd blur, mechanical movement)

ANTI-BORING RULE:
- No static "standing and posing"
- No flat compositions
- No neutral lighting
- No empty space without narrative purpose

═══════════════════════════════════════════════════════════
BACKGROUND & ENVIRONMENT - ATTENTION DRIVER
═══════════════════════════════════════════════════════════

Backgrounds are NEVER decorative. Every background MUST:
- Amplify emotion (pressure, awe, resolve, isolation, obsession)
- Anchor historical reality (accurate architecture, geography, materials)
- Add symbolism (light vs darkness, order vs chaos, confinement vs freedom)

DEPTH IS MANDATORY:
- Foreground framing elements
- Subject-background separation
- Natural depth of field
- Light falloff guiding attention to the face

═══════════════════════════════════════════════════════════
CINEMATIC DOCUMENTARY VISUAL STANDARD
═══════════════════════════════════════════════════════════

Style: Photorealistic historical reenactment ONLY
- Never illustration, concept art, painterly, or stylized AI

Lighting (Emotion-Led, Era-Correct):
- Only authentic sources: daylight, candle, oil lamp, gas, early electric
- Motivated shadows shaping the face
- No studio lighting, glamour glow, or artificial rim light

Color & Texture:
- Subtle professional documentary grading
- Filmic contrast
- Natural film grain
- Visible skin texture, fabric weave, dust, wear

═══════════════════════════════════════════════════════════
EMOTIONAL PRIORITY (HUMAN FIRST)
═══════════════════════════════════════════════════════════

Every image MUST clearly communicate a readable internal state:
- intellectual tension, doubt, curiosity, obsession
- exhaustion, quiet resolve, reflective calm
- pressure under responsibility

Rules:
- No theatrical hero poses
- No exaggerated gestures
- Natural posture, candid behavior
- Subject rarely looks at camera unless narratively justified
- Emotion must be legible within 1 second

═══════════════════════════════════════════════════════════
CINEMATIC COMPOSITION (RETENTION-OPTIMIZED)
═══════════════════════════════════════════════════════════

Camera rules:
- Favor medium and medium-close shots
- Avoid extreme angles
- Avoid overly tight crops
- Leave editorial breathing room

Each frame must feel editable, not cropped.

DEFAULT TECHNICAL STANDARD:
- Aspect Ratio: 16:9 cinematic
- Quality: Premium biographical documentary
- Audience: Mature American
- Tone: Serious, human, historically grounded"""

    def __init__(
        self,
        api_key: str,
        output_dir: Union[str, Path] = "generated_images",
        cache_dir: Optional[Union[str, Path]] = None,
        model: str = "dall-e-3",
        size: str = "1792x1024",
        quality: str = "hd",
        style: str = "natural",
        use_bytez: bool = False
    ):
        self.api_key = api_key
        self.use_bytez = use_bytez
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.size = size
        self.quality = quality
        self.style = style
        
        if use_bytez:
            from bytez import Bytez
            self.sdk = Bytez(api_key)
            bytez_model = model if model.startswith("openai/") else f"openai/{model}"
            self.dalle_model = self.sdk.model(bytez_model)
            self.openai_client = None
        else:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
            self.sdk = None
            self.dalle_model = None
        
        self.reference_searcher = ReferenceSearcher(api_key, self.cache_dir, use_bytez=use_bytez)
        
        self.generated_images: List[Dict] = []
        self.subject_consistency: Dict[str, str] = {}
        self.generation_log: List[Dict] = []
        
        self._load_state()

    def _load_state(self) -> None:
        """Load previous generation state."""
        state_file = self.cache_dir / "generation_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.generated_images = state.get('generated_images', [])
                    self.subject_consistency = state.get('subject_consistency', {})
            except (json.JSONDecodeError, IOError):
                pass

    def _save_state(self) -> None:
        """Save generation state."""
        state_file = self.cache_dir / "generation_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                'generated_images': self.generated_images,
                'subject_consistency': self.subject_consistency
            }, f, indent=2)

    def extract_age_from_prompt(self, prompt: str) -> Optional[int]:
        """Extract age from prompt text."""
        age_patterns = [
            r'age\s+(\d+)',
            r'aged?\s+(\d+)',
            r'(\d+)\s*years?\s*old',
            r'(\d+)-year-old',
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def extract_subject_name(self, prompt: str) -> Optional[str]:
        """Extract the subject's name from the prompt."""
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s*age',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1).strip()
        return None

    def _get_previous_image_context(self, subject_name: str, current_image_number: int) -> str:
        """Get context from previously generated images for consistency."""
        if subject_name in self.subject_consistency:
            return self.subject_consistency[subject_name]
        
        previous_images = [
            img for img in self.generated_images 
            if img.get('subject_name') == subject_name and img.get('image_number', 0) < current_image_number
        ]
        
        if not previous_images:
            return ""
        
        last_image = previous_images[-1]
        last_image_path = Path(last_image.get('image_path', ''))
        
        if last_image_path.exists():
            print(f"Analyzing previous image for consistency: {last_image_path}")
            consistency_desc = self.reference_searcher.analyze_image_for_consistency(
                last_image_path, subject_name
            )
            if consistency_desc:
                self.subject_consistency[subject_name] = consistency_desc
                self._save_state()
                return consistency_desc
        
        return ""

    def _build_enhanced_prompt(
        self,
        original_prompt: str,
        image_number: int,
        subject_name: Optional[str] = None,
        age: Optional[int] = None
    ) -> str:
        """Build an enhanced prompt with reference context and consistency requirements."""
        prompt_parts = []
        
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append("CINEMATIC BIOGRAPHICAL DOCUMENTARY IMAGE GENERATION")
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append("")
        
        if subject_name:
            reference_context = self.reference_searcher.get_age_specific_reference(
                subject_name, age or 30
            )
            if reference_context:
                prompt_parts.append("HISTORICAL REFERENCE (from research):")
                prompt_parts.append(reference_context)
                prompt_parts.append("")
            
            previous_context = self._get_previous_image_context(subject_name, image_number)
            if previous_context:
                prompt_parts.append("CONSISTENCY WITH PREVIOUS IMAGES (CRITICAL):")
                prompt_parts.append("The following facial features were established in previous images and MUST be maintained:")
                prompt_parts.append(previous_context)
                prompt_parts.append("")
        
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append("SCENE TO GENERATE:")
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append(original_prompt)
        prompt_parts.append("")
        
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append("MANDATORY VISUAL REQUIREMENTS:")
        prompt_parts.append("═══════════════════════════════════════════════════════════")
        prompt_parts.append("- Photorealistic, Netflix/HBO documentary quality")
        prompt_parts.append("- 16:9 cinematic aspect ratio")
        prompt_parts.append("- Dynamic composition with strong focal hierarchy")
        prompt_parts.append("- Era-appropriate lighting (no modern studio lighting)")
        prompt_parts.append("- Visible texture: skin pores, fabric weave, environmental details")
        prompt_parts.append("- Clear emotional readability within 1 second")
        prompt_parts.append("- Historical accuracy in all details")
        prompt_parts.append("- Age-appropriate facial features and body language")
        
        if age:
            prompt_parts.append(f"- Subject MUST appear exactly {age} years old")
            prompt_parts.append(f"- Apply age-appropriate skin texture, hair, and posture for age {age}")
        
        return "\n".join(prompt_parts)

    def generate_image(
        self,
        prompt: str,
        image_number: int,
        save_prompt: bool = True
    ) -> Optional[str]:
        """Generate a single image from a prompt."""
        subject_name = self.extract_subject_name(prompt)
        age = self.extract_age_from_prompt(prompt)
        
        enhanced_prompt = self._build_enhanced_prompt(
            prompt, image_number, subject_name, age
        )
        
        print(f"\n{'='*60}")
        print(f"Generating Image {image_number}")
        print(f"Subject: {subject_name or 'Unknown'}")
        print(f"Age: {age or 'Not specified'}")
        print(f"{'='*60}")
        
        try:
            if self.use_bytez:
                result = self.dalle_model.run(enhanced_prompt)
                
                if result.error:
                    raise Exception(f"Bytez API error: {result.error}")
                
                image_url = result.output
                revised_prompt = None
                if hasattr(result, 'provider') and result.provider:
                    provider_data = result.provider.get('data', [])
                    if provider_data and len(provider_data) > 0:
                        revised_prompt = provider_data[0].get('revised_prompt')
            else:
                openai_model = self.model.replace("openai/", "") if self.model.startswith("openai/") else self.model
                response = self.openai_client.images.generate(
                    model=openai_model,
                    prompt=enhanced_prompt,
                    size=self.size,
                    quality=self.quality,
                    style=self.style,
                    n=1
                )
                
                image_url = response.data[0].url
                revised_prompt = getattr(response.data[0], 'revised_prompt', None)
            
            image_filename = f"{image_number}.png"
            image_path = self.output_dir / image_filename
            
            self._download_image(image_url, image_path)
            
            image_record = {
                'image_number': image_number,
                'original_prompt': prompt,
                'subject_name': subject_name,
                'age': age,
                'image_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.generated_images.append(image_record)
            self._save_state()
            
            if subject_name and image_path.exists():
                print(f"Analyzing generated image for future consistency...")
                consistency_desc = self.reference_searcher.analyze_image_for_consistency(
                    image_path, subject_name
                )
                if consistency_desc:
                    self.subject_consistency[subject_name] = consistency_desc
                    self._save_state()
            
            log_entry = {
                'image_number': image_number,
                'original_prompt': prompt,
                'enhanced_prompt': enhanced_prompt if save_prompt else "[not saved]",
                'revised_prompt': revised_prompt,
                'subject_name': subject_name,
                'age': age,
                'image_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.generation_log.append(log_entry)
            self._save_generation_log()
            
            print(f"Image saved: {image_path}")
            return str(image_path)
            
        except Exception as e:
            print(f"Error generating image {image_number}: {e}")
            return None

    def _download_image(self, url: str, save_path: Path) -> None:
        """Download an image from URL and save to disk."""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)

    def _save_generation_log(self) -> None:
        """Save the generation log to a JSON file."""
        log_path = self.output_dir / "generation_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.generation_log, f, indent=2)

    def prefetch_subject_references(self, prompts: List[Dict]) -> None:
        """Pre-fetch reference information for all subjects in the prompts."""
        subjects = set()
        
        for item in prompts:
            prompt = item.get('image_prompt', item.get('prompt', ''))
            subject_name = self.extract_subject_name(prompt)
            if subject_name:
                subjects.add(subject_name)
        
        print(f"\n{'='*60}")
        print(f"PRE-FETCHING REFERENCE INFORMATION")
        print(f"Found {len(subjects)} unique subject(s) to research")
        print(f"{'='*60}")
        
        for subject in subjects:
            print(f"\nResearching: {subject}")
            self.reference_searcher.search_subject_references(subject)
        
        print(f"\nReference pre-fetch complete.\n")

    def generate_from_prompts_file(
        self,
        prompts_file: Union[str, Path],
        start_from: int = 1,
        delay_between: float = 2.0,
        prefetch_references: bool = True
    ) -> List[str]:
        """Generate images from a JSON prompts file."""
        prompts_file = Path(prompts_file)
        
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        prompts = data.get('images', [])
        
        if not prompts:
            raise ValueError("No image prompts found in file")
        
        print(f"Found {len(prompts)} image prompts")
        print(f"Output directory: {self.output_dir}")
        print(f"Starting from image {start_from}")
        
        if prefetch_references:
            self.prefetch_subject_references(prompts)
        
        generated_paths = []
        
        for item in prompts:
            image_number = item.get('image_prompt_No', item.get('image_number', 0))
            prompt = item.get('image_prompt', item.get('prompt', ''))
            
            if image_number < start_from:
                print(f"Skipping image {image_number} (before start_from={start_from})")
                continue
            
            if not prompt:
                print(f"Skipping image {image_number}: empty prompt")
                continue
            
            image_path = self.generate_image(prompt, image_number)
            
            if image_path:
                generated_paths.append(image_path)
            
            if delay_between > 0:
                print(f"Waiting {delay_between}s before next image...")
                time.sleep(delay_between)
        
        print(f"\nGeneration complete! Generated {len(generated_paths)} images.")
        return generated_paths


def load_config_and_generate(config_path: Union[str, Path]) -> None:
    """Load configuration from JSON and generate images."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config_dir = config_path.parent
    
    use_bytez = config.get('bytez_enabled', False)
    
    if use_bytez:
        api_key = config.get('bytez_api_key') or os.environ.get('BYTEZ_API_KEY')
        if not api_key:
            raise ValueError("Bytez API key not found in config or environment (bytez_enabled is true)")
    else:
        api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment (bytez_enabled is false)")
    
    prompts_file = config_dir / config.get('prompts_file', 'image_prompts.json')
    output_dir = config_dir / config.get('output_dir', 'generated_images')
    
    generator = ImageGenerator(
        api_key=api_key,
        output_dir=output_dir,
        model=config.get('model', 'dall-e-3'),
        size=config.get('size', '1792x1024'),
        quality=config.get('quality', 'hd'),
        style=config.get('style', 'natural'),
        use_bytez=use_bytez
    )
    
    generator.generate_from_prompts_file(
        prompts_file=prompts_file,
        start_from=config.get('start_from', 1),
        delay_between=config.get('delay_between', 2.0),
        prefetch_references=config.get('prefetch_references', True)
    )
