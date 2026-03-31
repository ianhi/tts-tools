"""Tests for core types."""

import tempfile
from pathlib import Path

from tts_tools._types import AudioFormat, Engine, SynthesisResult


def test_synthesis_result_save():
    result = SynthesisResult(
        audio_bytes=b"fake audio data",
        format=AudioFormat.MP3,
        sample_rate=24000,
        duration=1.0,
        text="hello",
    )
    with tempfile.TemporaryDirectory() as td:
        path = result.save(Path(td) / "subdir" / "test.mp3")
        assert path.exists()
        assert path.read_bytes() == b"fake audio data"


def test_synthesis_result_file_size():
    result = SynthesisResult(
        audio_bytes=b"x" * 1000,
        format=AudioFormat.WAV,
        sample_rate=24000,
        duration=0.5,
        text="test",
    )
    assert result.file_size == 1000


def test_engine_enum():
    assert Engine.GOOGLE_CLOUD.value == "google_cloud"
    assert Engine.GEMINI.value == "gemini"


def test_audio_format_enum():
    assert AudioFormat.MP3.value == "mp3"
    assert AudioFormat.WAV.value == "wav"
