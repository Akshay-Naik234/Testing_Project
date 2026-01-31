"""Reference Image Searcher - Searches for and downloads reference images of subjects."""

import os
import re
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI


class ReferenceSearcher:
    """Searches for reference images of biographical subjects at different ages."""

    AGE_RANGES = ["young (childhood)", "teenager", "young adult (20s-30s)", "middle-aged (40s-50s)", "elderly (60+)"]

    def __init__(self, api_key: str, cache_dir: Path):
        self.client = OpenAI(api_key=api_key)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.references_cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached reference data."""
        cache_file = self.cache_dir / "references_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.references_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.references_cache = {}

    def _save_cache(self) -> None:
        """Save reference data to cache."""
        cache_file = self.cache_dir / "references_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.references_cache, f, indent=2)

    def search_subject_references(self, subject_name: str) -> Dict:
        """Search for reference information about a subject using GPT-4."""
        if subject_name in self.references_cache:
            print(f"Using cached references for {subject_name}")
            return self.references_cache[subject_name]

        print(f"Searching for reference information about {subject_name}...")

        prompt = f"""I need detailed physical appearance descriptions of {subject_name} at different life stages for generating historically accurate biographical documentary images.

For each age range, provide:
1. Approximate years/age range
2. Detailed facial features (bone structure, eye shape, nose, jaw, distinctive features)
3. Hair description (color, style, hairline)
4. Body type and posture
5. Typical clothing/attire of that era
6. Notable physical changes from previous stage
7. Key historical context (what was happening in their life)

Age ranges to cover:
- Childhood (0-12 years)
- Teenager (13-19 years)  
- Young Adult (20-35 years)
- Middle Age (36-55 years)
- Later Years (56+ years)

Be specific about facial bone structure that remains constant (eye spacing, nose bridge, jaw angle, forehead shape) vs features that change with age.

Format as JSON with this structure:
{{
  "subject_name": "{subject_name}",
  "birth_year": <year>,
  "death_year": <year or null if alive>,
  "distinctive_features": ["feature1", "feature2"],
  "bone_structure": {{
    "eye_spacing": "description",
    "nose": "description", 
    "jaw": "description",
    "forehead": "description"
  }},
  "age_ranges": [
    {{
      "range": "childhood",
      "years": "birth-12",
      "age_span": "0-12",
      "facial_description": "detailed description",
      "hair": "description",
      "body_type": "description",
      "attire": "description",
      "historical_context": "what was happening"
    }}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a historical research assistant specializing in biographical details and physical appearances of historical figures. Provide accurate, detailed information based on historical records, photographs, and documented descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            content = response.choices[0].message.content
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                reference_data = json.loads(json_match.group())
            else:
                reference_data = {"subject_name": subject_name, "raw_description": content}

            self.references_cache[subject_name] = reference_data
            self._save_cache()

            print(f"Found reference information for {subject_name}")
            return reference_data

        except Exception as e:
            print(f"Error searching for references: {e}")
            return {"subject_name": subject_name, "error": str(e)}

    def get_age_specific_reference(self, subject_name: str, age: int) -> str:
        """Get reference description for a specific age."""
        references = self.search_subject_references(subject_name)
        
        if "error" in references:
            return f"Historical figure: {subject_name}, age {age}"

        bone_structure = references.get("bone_structure", {})
        distinctive = references.get("distinctive_features", [])
        
        description_parts = [
            f"Subject: {subject_name}, age {age}",
            f"Distinctive features that MUST be maintained: {', '.join(distinctive)}" if distinctive else "",
            f"Bone structure (LOCKED - must remain consistent):",
            f"  - Eye spacing: {bone_structure.get('eye_spacing', 'standard')}",
            f"  - Nose: {bone_structure.get('nose', 'standard')}",
            f"  - Jaw: {bone_structure.get('jaw', 'standard')}",
            f"  - Forehead: {bone_structure.get('forehead', 'standard')}"
        ]

        age_ranges = references.get("age_ranges", [])
        for age_range in age_ranges:
            age_span = age_range.get("age_span", "")
            if "-" in age_span:
                try:
                    min_age, max_age = map(int, age_span.split("-"))
                    if min_age <= age <= max_age:
                        description_parts.extend([
                            f"\nAge-specific appearance ({age_range.get('range', '')}):",
                            f"  - Facial features: {age_range.get('facial_description', '')}",
                            f"  - Hair: {age_range.get('hair', '')}",
                            f"  - Body type: {age_range.get('body_type', '')}",
                            f"  - Era-appropriate attire: {age_range.get('attire', '')}",
                            f"  - Historical context: {age_range.get('historical_context', '')}"
                        ])
                        break
                except ValueError:
                    continue

        return "\n".join(filter(None, description_parts))

    def analyze_image_for_consistency(self, image_path: Path, subject_name: str) -> str:
        """Analyze a generated image to extract features for consistency."""
        if not image_path.exists():
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this image of {subject_name} and describe the key facial features that should remain consistent in future images:
1. Eye shape and spacing
2. Nose shape and size
3. Jaw and chin structure
4. Forehead shape
5. Any distinctive features (scars, moles, etc.)
6. Overall face shape

Provide a concise description that can be used to maintain consistency in future image generations."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error analyzing image: {e}")
            return ""
