"""tts-tools: Simple, high-quality text-to-speech synthesis.

>>> from tts_tools import synthesize
>>> result = synthesize("hello", language="en-US")
>>> result.save("hello.mp3")
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ._types import AudioFormat, Engine, SynthesisResult
from ._voices import VoiceInfo, list_voices

if TYPE_CHECKING:
    pass

__all__ = [
    "AudioFormat",
    "Engine",
    "SynthesisResult",
    "VoiceInfo",
    "list_voices",
    "synthesize",
    "synthesize_async",
    "synthesize_batch",
]


def synthesize(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    engine: Engine = Engine.GOOGLE_CLOUD,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    volume_gain_db: float = 0.0,
    sample_rate: int = 24000,
    timeout: int = 30,
    max_retries: int = 3,
) -> SynthesisResult:
    """Synthesize speech from text. This is the main entry point.

    Args:
        text: The text to speak.
        language: BCP-47 language code (e.g. "en-US", "bn-IN").
        voice: Voice name. If None, uses a sensible default for the language.
        engine: TTS engine to use (GOOGLE_CLOUD or GEMINI).
        format: Output audio format (MP3 or WAV).
        trim: Whether to trim leading/trailing silence.
        volume_gain_db: Volume adjustment in dB.
        sample_rate: Output sample rate in Hz.
        timeout: API timeout in seconds.
        max_retries: Number of retry attempts on transient failures.
    """
    if engine == Engine.GEMINI:
        from ._gemini import synthesize_gemini

        return synthesize_gemini(
            text,
            voice=voice or "Kore",
            format=format,
            max_retries=max_retries,
        )

    from ._google_cloud import synthesize_google_cloud

    return synthesize_google_cloud(
        text,
        language=language,
        voice=voice,
        format=format,
        trim=trim,
        sample_rate=sample_rate,
        volume_gain_db=volume_gain_db,
        timeout=timeout,
        max_retries=max_retries,
    )


async def synthesize_async(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    engine: Engine = Engine.GOOGLE_CLOUD,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    volume_gain_db: float = 0.0,
    sample_rate: int = 24000,
    timeout: int = 30,
    max_retries: int = 3,
) -> SynthesisResult:
    """Async version of synthesize(). Same arguments."""
    if engine == Engine.GEMINI:
        # Gemini SDK is sync-only; run in thread
        from ._gemini import synthesize_gemini

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: synthesize_gemini(
                text, voice=voice or "Kore", format=format, max_retries=max_retries,
            ),
        )

    from ._google_cloud import synthesize_google_cloud_async

    return await synthesize_google_cloud_async(
        text,
        language=language,
        voice=voice,
        format=format,
        trim=trim,
        sample_rate=sample_rate,
        volume_gain_db=volume_gain_db,
        timeout=timeout,
        max_retries=max_retries,
    )


async def synthesize_batch(
    texts: list[str],
    *,
    language: str,
    voice: str | None = None,
    engine: Engine = Engine.GOOGLE_CLOUD,
    format: AudioFormat = AudioFormat.MP3,
    max_concurrent: int = 10,
    **kwargs,
) -> list[SynthesisResult]:
    """Synthesize a batch of texts concurrently.

    For Gemini with short texts, automatically uses the efficient
    batch-and-split strategy.
    """
    if engine == Engine.GEMINI:
        from ._gemini import SHORT_TEXT_THRESHOLD, synthesize_gemini, synthesize_gemini_batch

        all_short = all(len(t.strip()) < SHORT_TEXT_THRESHOLD for t in texts)
        if all_short and len(texts) > 1:
            return synthesize_gemini_batch(
                texts, voice=voice or "Kore", format=format,
            )
        # Fall back to individual calls for longer texts
        results = []
        for t in texts:
            results.append(synthesize_gemini(t, voice=voice or "Kore", format=format))
        return results

    sem = asyncio.Semaphore(max_concurrent)

    async def _one(text: str) -> SynthesisResult:
        async with sem:
            return await synthesize_async(
                text, language=language, voice=voice, engine=engine, format=format, **kwargs,
            )

    return await asyncio.gather(*[_one(t) for t in texts])
