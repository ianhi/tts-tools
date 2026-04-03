"""Edge TTS engine using Microsoft Edge's online TTS service.

Free, no API key required. Natively async; sync wrapper uses asyncio.run().
"""

from __future__ import annotations

import asyncio
import io

from ._retry import retry_async, retry_sync
from ._types import AudioFormat, SynthesisResult


def _default_voice(language: str) -> str:
    """Pick a reasonable default Edge voice for a language code."""
    defaults = {
        "en-US": "en-US-AriaNeural",
        "en-GB": "en-GB-SoniaNeural",
        "es-ES": "es-ES-ElviraNeural",
        "es-US": "es-US-PalomaNeural",
        "fr-FR": "fr-FR-DeniseNeural",
        "de-DE": "de-DE-KatjaNeural",
        "ja-JP": "ja-JP-NanamiNeural",
        "ko-KR": "ko-KR-SunHiNeural",
        "zh-CN": "zh-CN-XiaoxiaoNeural",
        "pt-BR": "pt-BR-FranciscaNeural",
        "bn-IN": "bn-IN-TanishaaNeural",
        "hi-IN": "hi-IN-SwaraNeural",
    }
    if language in defaults:
        return defaults[language]
    # Fallback: try constructing a plausible name
    return f"{language}-AriaNeural" if language.startswith("en") else f"en-US-AriaNeural"


async def _edge_tts_request(text: str, voice: str) -> bytes:
    """Single edge-tts request. Returns MP3 bytes."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "Edge TTS requires the edge-tts package. "
            "Install it with: pip install tts-tools[edge]"
        ) from None

    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError("Edge TTS returned no audio data")
    return b"".join(chunks)


def synthesize_edge(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    max_retries: int = 3,
) -> SynthesisResult:
    """Synthesize speech using Edge TTS (synchronous)."""
    edge_voice = voice or _default_voice(language)

    def _call():
        return asyncio.run(_edge_tts_request(text, edge_voice))

    mp3_bytes = retry_sync(_call, max_retries=max_retries)
    return _mp3_to_result(mp3_bytes, text=text, format=format, trim=trim)


async def synthesize_edge_async(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    max_retries: int = 3,
) -> SynthesisResult:
    """Synthesize speech using Edge TTS (async)."""
    edge_voice = voice or _default_voice(language)

    async def _call():
        return await _edge_tts_request(text, edge_voice)

    mp3_bytes = await retry_async(_call, max_retries=max_retries)
    return _mp3_to_result(mp3_bytes, text=text, format=format, trim=trim)


def _mp3_to_result(
    mp3_bytes: bytes,
    *,
    text: str,
    format: AudioFormat,
    trim: bool,
) -> SynthesisResult:
    """Convert MP3 bytes from edge-tts to a SynthesisResult."""
    from pydub import AudioSegment

    segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

    if trim:
        from ._audio import trim_silence
        import numpy as np

        samples = np.array(segment.get_array_of_samples(), dtype=np.int16)
        samples = trim_silence(samples)
        trimmed = segment._spawn(samples.tobytes())
        trimmed = trimmed.set_frame_rate(segment.frame_rate)
        trimmed = trimmed.set_channels(segment.channels)
        trimmed = trimmed.set_sample_width(segment.sample_width)
        segment = trimmed

    sample_rate = segment.frame_rate
    duration = len(segment) / 1000.0

    if format == AudioFormat.WAV:
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        audio_bytes = buf.getvalue()
    else:
        buf = io.BytesIO()
        segment.export(buf, format="mp3", bitrate="192k")
        audio_bytes = buf.getvalue()

    return SynthesisResult(
        audio_bytes=audio_bytes,
        format=format,
        sample_rate=sample_rate,
        duration=duration,
        text=text,
    )
