"""Main entry point for the image generator module."""

import sys
from pathlib import Path

from .image_generator import load_config_and_generate


def main():
    """Run the image generator with default config."""
    project_root = Path(__file__).parent.parent
    default_config = project_root / "examples" / "input" / "image_generator_config.json"
    
    if default_config.exists():
        print(f"Loading config from: {default_config}")
        load_config_and_generate(default_config)
    else:
        print(f"Error: Config file not found at {default_config}")
        print("Please create a config file at examples/input/image_generator_config.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
