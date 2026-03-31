"""Core types for tts-tools."""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Engine(str, Enum):
    """TTS engine to use for synthesis."""

    GOOGLE_CLOUD = "google_cloud"
    GEMINI = "gemini"


class AudioFormat(str, Enum):
    """Output audio format."""

    MP3 = "mp3"
    WAV = "wav"


@dataclass
class SynthesisResult:
    """Result of a TTS synthesis call."""

    audio_bytes: bytes
    format: AudioFormat
    sample_rate: int
    duration: float  # seconds
    text: str

    @property
    def file_size(self) -> int:
        return len(self.audio_bytes)

    def save(self, path: str | Path) -> Path:
        """Write audio bytes to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.audio_bytes)
        return path

    def as_segment(self) -> "AudioSegment":
        """Return a pydub AudioSegment for further processing."""
        from pydub import AudioSegment

        return AudioSegment.from_file(io.BytesIO(self.audio_bytes), format=self.format.value)
