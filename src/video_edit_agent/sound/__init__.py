"""Sound direction: intents, profiles, a small SFX registry, a deterministic resolver."""
from video_edit_agent.sound.intent import EventType, SoundIntent
from video_edit_agent.sound.profile import SoundProfile, get_profile
from video_edit_agent.sound.registry import SfxRegistry, load_registry
from video_edit_agent.sound.resolver import Resolution, resolve

__all__ = [
    "EventType", "Resolution", "SfxRegistry", "SoundIntent", "SoundProfile", "get_profile", "load_registry", "resolve",
]
