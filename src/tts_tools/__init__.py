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
    api_key: str | None = None,
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
        api_key: Google Cloud TTS API key. If set, uses REST API instead of ADC.
                 Can also be set via GOOGLE_CLOUD_TTS_API_KEY env var.
    """
    if engine == Engine.GEMINI:
        from ._gemini import synthesize_gemini

        return synthesize_gemini(
            text,
            voice=voice or "Kore",
            format=format,
            max_retries=max_retries,
        )

    if engine == Engine.EDGE:
        from ._edge import synthesize_edge

        return synthesize_edge(
            text,
            language=language,
            voice=voice,
            format=format,
            trim=trim,
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
        api_key=api_key,
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
    api_key: str | None = None,
) -> SynthesisResult:
    """Async version of synthesize(). Same arguments."""
    if engine == Engine.GEMINI:
        # Gemini SDK is sync-only; run in thread
        from ._gemini import synthesize_gemini

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: synthesize_gemini(
                text,
                voice=voice or "Kore",
                format=format,
                max_retries=max_retries,
            ),
        )

    if engine == Engine.EDGE:
        from ._edge import synthesize_edge_async

        return await synthesize_edge_async(
            text,
            language=language,
            voice=voice,
            format=format,
            trim=trim,
            max_retries=max_retries,
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
        api_key=api_key,
    )


async def synthesize_batch(
    texts: list[str],
    *,
    language: str,
    voice: str | None = None,
    engine: Engine = Engine.GOOGLE_CLOUD,
    format: AudioFormat = AudioFormat.MP3,
    max_concurrent: int = 50,
    **kwargs,
) -> list[SynthesisResult]:
    """Synthesize a batch of texts concurrently.

    Concurrency is controlled by *max_concurrent* (default 50). For Google
    Cloud TTS, 50 concurrent requests stays well within the 300 req/min
    quota while keeping throughput high. For Gemini TTS, short texts are
    automatically batched into a single API call.

    Args:
        texts: List of texts to synthesize.
        language: BCP-47 language code.
        max_concurrent: Max parallel requests. Default 50 for Google Cloud
            TTS. For Gemini, rate limiting is handled automatically.
    """
    if engine == Engine.GEMINI:
        return await _batch_gemini(texts, voice=voice, format=format)

    if engine == Engine.EDGE:
        from ._edge import synthesize_edge_async

        sem = asyncio.Semaphore(max_concurrent)

        async def _one_edge(text: str) -> SynthesisResult:
            async with sem:
                return await synthesize_edge_async(
                    text,
                    language=language,
                    voice=voice,
                    format=format,
                )

        return await asyncio.gather(*[_one_edge(t) for t in texts])

    sem = asyncio.Semaphore(max_concurrent)

    async def _one(text: str) -> SynthesisResult:
        async with sem:
            return await synthesize_async(
                text,
                language=language,
                voice=voice,
                engine=engine,
                format=format,
                **kwargs,
            )

    return await asyncio.gather(*[_one(t) for t in texts])


async def _batch_gemini(
    texts: list[str],
    *,
    voice: str | None = None,
    format: AudioFormat = AudioFormat.MP3,
) -> list[SynthesisResult]:
    """Batch Gemini TTS with smart short-word batching and rate limiting.

    - Short texts (< 8 chars) are batched into single API calls
    - Long texts are synthesized individually with rate limiting (10 RPM)
    """
    from ._gemini import SHORT_TEXT_THRESHOLD, synthesize_gemini, synthesize_gemini_batch

    gemini_voice = voice or "Kore"

    # Separate short and long texts, preserving original indices
    short_indices = []
    short_texts = []
    long_indices = []
    long_texts = []

    for i, t in enumerate(texts):
        if len(t.strip()) < SHORT_TEXT_THRESHOLD:
            short_indices.append(i)
            short_texts.append(t)
        else:
            long_indices.append(i)
            long_texts.append(t)

    results: list[SynthesisResult | None] = [None] * len(texts)

    # Batch all short texts in one API call
    if short_texts:
        short_results = synthesize_gemini_batch(
            short_texts,
            voice=gemini_voice,
            format=format,
        )
        for idx, result in zip(short_indices, short_results):
            results[idx] = result

    # Process long texts sequentially with rate limiting (10 RPM = 6s between)
    for i, (idx, t) in enumerate(zip(long_indices, long_texts)):
        if i > 0:
            await asyncio.sleep(6)  # Gemini TTS: 10 RPM limit
        results[idx] = synthesize_gemini(t, voice=gemini_voice, format=format)

    return results  # type: ignore[return-value]
