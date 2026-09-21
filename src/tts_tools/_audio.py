"""Audio processing: silence trimming, format conversion, validation."""

from __future__ import annotations

import io
import shutil
import wave

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def _check_ffmpeg() -> None:
    """Raise a clear error if ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for MP3 conversion but was not found on PATH. "
            "Install it with: sudo apt-get install ffmpeg (Linux), "
            "brew install ffmpeg (macOS), or download from https://ffmpeg.org/"
        )


def trim_silence(samples: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence from audio samples."""
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
    return trimmed


def samples_to_int16(samples: np.ndarray) -> np.ndarray:
    """Normalize audio samples to int16 for pydub/export."""
    if samples.dtype == np.int16:
        return samples
    if samples.dtype in (np.float32, np.float64):
        return (samples * 32767).astype(np.int16)
    return samples.astype(np.int16)


def to_mp3(samples: np.ndarray, sample_rate: int, bitrate: str = "192k") -> bytes:
    """Convert PCM samples to MP3 bytes. Requires ffmpeg on PATH."""
    _check_ffmpeg()
    int_samples = samples_to_int16(samples)
    segment = AudioSegment(
        int_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert PCM samples to WAV bytes."""
    int_samples = samples_to_int16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def validate_audio(
    audio_bytes: bytes,
    min_file_size: int = 5000,
    min_duration: float = 0.3,
) -> dict:
    """Validate that audio bytes contain real audio content.

    Returns dict with 'valid' (bool), 'reason' (str), 'duration', 'file_size'.
    """
    size = len(audio_bytes)
    if size < min_file_size:
        return {
            "valid": False,
            "reason": f"Too small ({size} bytes)",
            "duration": 0,
            "file_size": size,
        }

    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / sr
    except Exception as e:
        return {"valid": False, "reason": f"Unreadable: {e}", "duration": 0, "file_size": size}

    if duration < min_duration:
        return {
            "valid": False,
            "reason": f"Too short ({duration:.2f}s)",
            "duration": duration,
            "file_size": size,
        }

    if len(data) > 0:
        rms = float((data**2).mean() ** 0.5)
        if rms < 1e-6:
            return {
                "valid": False,
                "reason": "Silent audio",
                "duration": duration,
                "file_size": size,
            }

    return {"valid": True, "reason": "OK", "duration": duration, "file_size": size}
