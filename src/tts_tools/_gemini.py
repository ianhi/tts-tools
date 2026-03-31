"""Gemini TTS engine with short-word batching."""

from __future__ import annotations

import io

from ._audio import pcm_to_wav
from ._retry import retry_sync
from ._types import AudioFormat, SynthesisResult

# Below this character count, Gemini often returns empty audio.
# The caller should use batch mode for short texts.
SHORT_TEXT_THRESHOLD = 8


def _get_client():
    """Lazily create a Gemini client (requires GOOGLE_API_KEY env var)."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "Gemini TTS requires the google-genai package. "
            "Install it with: pip install tts-tools[gemini]"
        ) from None
    return genai.Client()


def _gemini_tts_request(text: str, voice: str, client) -> bytes:
    """Single Gemini TTS request. Returns WAV bytes."""
    from google import genai

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=genai.types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai.types.SpeechConfig(
                voice_config=genai.types.VoiceConfig(
                    prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )

    candidate = response.candidates[0]
    if candidate.content is None:
        raise RuntimeError(f"Empty response (finish_reason={candidate.finish_reason})")

    audio_data = candidate.content.parts[0].inline_data.data
    mime_type = candidate.content.parts[0].inline_data.mime_type

    if "L16" in mime_type or "pcm" in mime_type.lower():
        return pcm_to_wav(audio_data, sample_rate=24000)
    return audio_data


def _split_audio(wav_bytes: bytes, expected_count: int) -> list[bytes]:
    """Split a WAV containing multiple words at silence gaps."""
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    chunks = split_on_silence(
        audio,
        min_silence_len=200,
        silence_thresh=audio.dBFS - 16,
        keep_silence=80,
    )
    if len(chunks) != expected_count:
        raise RuntimeError(f"Expected {expected_count} segments, got {len(chunks)}")

    results = []
    for chunk in chunks:
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        results.append(buf.getvalue())
    return results


def synthesize_gemini(
    text: str,
    *,
    voice: str = "Kore",
    format: AudioFormat = AudioFormat.MP3,
    max_retries: int = 3,
    client=None,
) -> SynthesisResult:
    """Synthesize speech using Gemini TTS.

    For short texts (< 8 chars), automatically appends a period to help
    Gemini produce output. Use synthesize_gemini_batch() for better
    results with multiple short words.
    """
    if client is None:
        client = _get_client()

    effective_text = text
    if len(text.strip()) < SHORT_TEXT_THRESHOLD:
        effective_text = text.strip() + "."

    def _call():
        return _gemini_tts_request(effective_text, voice, client)

    wav_bytes = retry_sync(_call, max_retries=max_retries)
    return _wav_to_result(wav_bytes, text=text, format=format)


def synthesize_gemini_batch(
    texts: list[str],
    *,
    voice: str = "Kore",
    format: AudioFormat = AudioFormat.MP3,
    max_retries: int = 3,
    client=None,
) -> list[SynthesisResult]:
    """Synthesize multiple short words in one API call.

    Joins texts with ' ... ', synthesizes once, splits at silence boundaries.
    Much more reliable than individual calls for short words.
    """
    if client is None:
        client = _get_client()

    combined = " ... ".join(texts)

    def _call():
        return _gemini_tts_request(combined, voice, client)

    wav_bytes = retry_sync(_call, max_retries=max_retries)
    chunks = _split_audio(wav_bytes, len(texts))
    return [_wav_to_result(chunk, text=t, format=format) for t, chunk in zip(texts, chunks)]


def _wav_to_result(
    wav_bytes: bytes, *, text: str, format: AudioFormat
) -> SynthesisResult:
    """Convert WAV bytes to a SynthesisResult, optionally converting to MP3."""
    import soundfile as sf

    if format == AudioFormat.MP3:
        from pydub import AudioSegment

        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        buf = io.BytesIO()
        segment.export(buf, format="mp3", bitrate="192k")
        audio_bytes = buf.getvalue()
        sample_rate = segment.frame_rate
        duration = len(segment) / 1000.0
    else:
        audio_bytes = wav_bytes
        data, sample_rate = sf.read(io.BytesIO(wav_bytes))
        duration = len(data) / sample_rate

    return SynthesisResult(
        audio_bytes=audio_bytes,
        format=format,
        sample_rate=sample_rate,
        duration=duration,
        text=text,
    )
