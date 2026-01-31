"""Age Tracker - Manages age ranges and facial consistency for biographical image generation."""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AgeTracker:
    """Tracks age ranges and maintains facial consistency across generated images."""

    AGE_RANGES = [
        (0, 5, "0-5"),
        (6, 12, "6-12"),
        (13, 18, "13-18"),
        (19, 25, "19-25"),
        (26, 35, "26-35"),
        (36, 45, "36-45"),
        (46, 60, "46-60"),
        (61, 75, "60-75"),
        (76, 150, "75+")
    ]

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self.subject_cache: Dict[str, Dict] = {}
        self.generated_images: List[Dict] = []
        
        if cache_file and cache_file.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Load cached subject data from file."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.subject_cache = data.get('subjects', {})
                self.generated_images = data.get('generated_images', [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cache: {e}")

    def save_cache(self) -> None:
        """Save subject data to cache file."""
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'subjects': self.subject_cache,
                    'generated_images': self.generated_images
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

    def get_age_range(self, age: int) -> str:
        """Get the age range category for a given age."""
        for min_age, max_age, label in self.AGE_RANGES:
            if min_age <= age <= max_age:
                return label
        return "75+"

    def get_age_range_description(self, age: int) -> str:
        """Get a detailed description of physical characteristics for an age range."""
        if age <= 5:
            return "infant/toddler features: round face, soft features, large eyes relative to face"
        elif age <= 12:
            return "child features: developing facial structure, youthful skin, bright eyes"
        elif age <= 18:
            return "adolescent features: maturing facial structure, clear skin, developing adult features"
        elif age <= 25:
            return "young adult features: fully developed facial structure, smooth skin, peak physical condition"
        elif age <= 35:
            return "adult features: mature facial structure, slight signs of aging beginning"
        elif age <= 45:
            return "middle-aged features: visible aging signs, possible grey hair beginning, facial lines forming"
        elif age <= 60:
            return "mature features: pronounced aging, grey/white hair, deeper facial lines, possible weight changes"
        elif age <= 75:
            return "elderly features: significant aging, white hair, wrinkles, possible posture changes"
        else:
            return "advanced elderly features: deep wrinkles, thin skin, frail appearance, white/sparse hair"

    def register_subject(self, name: str, base_description: str = "") -> None:
        """Register a new subject for tracking."""
        if name not in self.subject_cache:
            self.subject_cache[name] = {
                'name': name,
                'base_description': base_description,
                'age_range_images': {},
                'facial_features': {
                    'eye_spacing': 'standard',
                    'nose_shape': 'standard',
                    'jaw_shape': 'standard',
                    'forehead': 'standard',
                    'distinctive_features': []
                }
            }

    def update_subject_features(self, name: str, features: Dict) -> None:
        """Update the facial features for a subject."""
        if name in self.subject_cache:
            self.subject_cache[name]['facial_features'].update(features)
            self.save_cache()

    def record_generated_image(
        self,
        image_number: int,
        prompt: str,
        subject_name: str,
        age: int,
        image_path: str
    ) -> None:
        """Record a generated image for reference."""
        age_range = self.get_age_range(age)
        
        record = {
            'image_number': image_number,
            'prompt': prompt,
            'subject_name': subject_name,
            'age': age,
            'age_range': age_range,
            'image_path': image_path
        }
        
        self.generated_images.append(record)
        
        if subject_name in self.subject_cache:
            if age_range not in self.subject_cache[subject_name]['age_range_images']:
                self.subject_cache[subject_name]['age_range_images'][age_range] = []
            self.subject_cache[subject_name]['age_range_images'][age_range].append(record)
        
        self.save_cache()

    def get_previous_images_for_subject(
        self,
        subject_name: str,
        current_age: Optional[int] = None
    ) -> List[Dict]:
        """Get previously generated images for a subject."""
        if subject_name not in self.subject_cache:
            return []
        
        all_images = []
        for age_range, images in self.subject_cache[subject_name]['age_range_images'].items():
            all_images.extend(images)
        
        return sorted(all_images, key=lambda x: x['image_number'])

    def get_consistency_context(
        self,
        subject_name: str,
        current_age: int,
        current_image_number: int
    ) -> str:
        """Generate context for maintaining facial consistency."""
        if subject_name not in self.subject_cache:
            return ""
        
        subject = self.subject_cache[subject_name]
        current_age_range = self.get_age_range(current_age)
        age_description = self.get_age_range_description(current_age)
        
        context_parts = []
        
        context_parts.append(f"Subject: {subject_name}")
        context_parts.append(f"Current age: {current_age} (age range: {current_age_range})")
        context_parts.append(f"Age-appropriate features: {age_description}")
        
        if subject.get('base_description'):
            context_parts.append(f"Base appearance: {subject['base_description']}")
        
        features = subject.get('facial_features', {})
        if features.get('distinctive_features'):
            context_parts.append(
                f"Distinctive features to maintain: {', '.join(features['distinctive_features'])}"
            )
        
        previous_images = self.get_previous_images_for_subject(subject_name, current_age)
        if previous_images:
            recent = previous_images[-3:]
            context_parts.append(
                f"Previously generated {len(previous_images)} images. "
                f"Most recent ages: {[img['age'] for img in recent]}"
            )
            
            same_range_images = [
                img for img in previous_images 
                if img['age_range'] == current_age_range
            ]
            if same_range_images:
                context_parts.append(
                    f"Images in same age range ({current_age_range}): {len(same_range_images)}. "
                    "Maintain exact facial consistency with these."
                )
        
        return "\n".join(context_parts)

    def get_facial_evolution_guidance(self, subject_name: str, from_age: int, to_age: int) -> str:
        """Get guidance for facial evolution between ages."""
        if from_age >= to_age:
            return "No age progression needed."
        
        changes = []
        
        if from_age < 18 and to_age >= 18:
            changes.append("Transition from adolescent to adult facial structure")
        
        if from_age < 35 and to_age >= 35:
            changes.append("Begin showing subtle aging signs")
        
        if from_age < 45 and to_age >= 45:
            changes.append("More pronounced facial lines, possible grey hair")
        
        if from_age < 60 and to_age >= 60:
            changes.append("Significant aging: deeper wrinkles, white/grey hair, skin texture changes")
        
        if from_age < 75 and to_age >= 75:
            changes.append("Advanced aging: thin skin, pronounced wrinkles, frail features")
        
        if not changes:
            changes.append(f"Gradual aging from {from_age} to {to_age}")
        
        return "; ".join(changes)
