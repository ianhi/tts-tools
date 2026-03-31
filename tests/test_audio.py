"""Tests for audio processing functions (no API calls needed)."""

import io

import numpy as np
import pytest
import soundfile as sf

from tts_tools._audio import (
    pcm_to_wav,
    samples_to_int16,
    to_mp3,
    to_wav,
    trim_silence,
    validate_audio,
)


def test_trim_silence_removes_padding(sine_samples):
    sr, samples = sine_samples
    # Pad with silence on both sides
    padded = np.concatenate([np.zeros(sr, dtype=np.int16), samples, np.zeros(sr, dtype=np.int16)])
    trimmed = trim_silence(padded)
    # Trimmed should be shorter than padded
    assert len(trimmed) < len(padded)
    # But should still contain audio
    assert len(trimmed) > 0


def test_trim_silence_preserves_audio(sine_samples):
    sr, samples = sine_samples
    trimmed = trim_silence(samples)
    # Pure tone with no silence should be mostly preserved
    assert len(trimmed) > len(samples) * 0.8


def test_samples_to_int16_from_float():
    float_samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    result = samples_to_int16(float_samples)
    assert result.dtype == np.int16
    assert result[0] == 0
    assert result[3] == 32767


def test_samples_to_int16_passthrough():
    int_samples = np.array([0, 100, -100], dtype=np.int16)
    result = samples_to_int16(int_samples)
    assert result is int_samples  # Should be the exact same array


def test_to_mp3(sine_samples):
    sr, samples = sine_samples
    try:
        mp3_bytes = to_mp3(samples, sr)
    except FileNotFoundError:
        pytest.skip("ffmpeg not installed")
    assert len(mp3_bytes) > 0
    # MP3 files start with ID3 tag or sync word
    assert mp3_bytes[:3] == b"ID3" or mp3_bytes[:2] == b"\xff\xfb"


def test_to_wav(sine_samples):
    sr, samples = sine_samples
    wav_bytes = to_wav(samples, sr)
    assert wav_bytes[:4] == b"RIFF"
    # Should be readable by soundfile
    data, read_sr = sf.read(io.BytesIO(wav_bytes))
    assert read_sr == sr
    assert len(data) == len(samples)


def test_pcm_to_wav():
    pcm = b"\x00" * 48000  # 1 second of silence at 24kHz, 16-bit mono
    wav = pcm_to_wav(pcm, sample_rate=24000)
    assert wav[:4] == b"RIFF"


def test_validate_audio_valid(sine_samples):
    sr, samples = sine_samples
    wav_bytes = to_wav(samples, sr)
    result = validate_audio(wav_bytes)
    assert result["valid"] is True


def test_validate_audio_too_small():
    result = validate_audio(b"tiny", min_file_size=100)
    assert result["valid"] is False
    assert "Too small" in result["reason"]


def test_validate_audio_silent(silent_samples):
    sr, samples = silent_samples
    wav_bytes = to_wav(samples, sr)
    result = validate_audio(wav_bytes)
    assert result["valid"] is False
    assert "Silent" in result["reason"]


def test_validate_audio_too_short(short_samples):
    sr, samples = short_samples
    wav_bytes = to_wav(samples, sr)
    result = validate_audio(wav_bytes, min_duration=0.3)
    assert result["valid"] is False
    # May fail on size or duration depending on how short it is
    assert "Too short" in result["reason"] or "Too small" in result["reason"]
