"""Sequential Video Composer - Creates videos from numbered images with animations."""

from .sequential_video_orchestrator import (
    SequentialVideoOrchestrator,
    create_sequential_video,
    load_config_and_create_video
)

__version__ = "1.0.0"
__author__ = "Sequential Video Systems"

__all__ = [
    'SequentialVideoOrchestrator',
    'create_sequential_video',
    'load_config_and_create_video'
]
