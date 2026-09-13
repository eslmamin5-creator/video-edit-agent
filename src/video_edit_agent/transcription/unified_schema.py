"""Re-export of the unified transcript schema for import convenience.

The canonical definitions live in `video_edit_agent.core.schemas` so that
non-transcription code (editorial, captions, QA) doesn't need to import
through the `transcription` package.
"""
from video_edit_agent.core.schemas import Segment, Transcript, Word

__all__ = ["Transcript", "Segment", "Word"]
