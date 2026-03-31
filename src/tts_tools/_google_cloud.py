"""Google Cloud Text-to-Speech engine (sync + async)."""

from __future__ import annotations

import asyncio
import io

from google.cloud import texttospeech, texttospeech_v1
from scipy.io import wavfile

from ._audio import to_mp3, to_wav, trim_silence
from ._punctuation import ensure_sentence_stop
from ._retry import retry_async, retry_sync
from ._types import AudioFormat, SynthesisResult


def synthesize_google_cloud(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    sample_rate: int = 24000,
    volume_gain_db: float = 0.0,
    effects_profile_id: str | None = None,
    add_punctuation: bool = True,
    timeout: int = 30,
    max_retries: int = 3,
) -> SynthesisResult:
    """Synthesize speech using Google Cloud TTS (synchronous)."""
    if add_punctuation:
        text = ensure_sentence_stop(text, language)

    voice_name = voice or _default_voice(language)

    def _call():
        client = texttospeech.TextToSpeechClient()
        response = client.synthesize_speech(
            request={
                "input": texttospeech.SynthesisInput(text=text),
                "voice": texttospeech.VoiceSelectionParams(
                    language_code=language, name=voice_name,
                ),
                "audio_config": texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    volume_gain_db=volume_gain_db,
                    effects_profile_id=[effects_profile_id] if effects_profile_id else [],
                ),
            },
            timeout=timeout if timeout > 0 else None,
        )
        return response.audio_content

    raw_wav = retry_sync(_call, max_retries=max_retries)
    return _process_raw(raw_wav, text=text, format=format, trim=trim)


async def synthesize_google_cloud_async(
    text: str,
    *,
    language: str,
    voice: str | None = None,
    format: AudioFormat = AudioFormat.MP3,
    trim: bool = True,
    sample_rate: int = 24000,
    volume_gain_db: float = 0.0,
    effects_profile_id: str | None = None,
    add_punctuation: bool = True,
    timeout: int = 30,
    max_retries: int = 3,
) -> SynthesisResult:
    """Synthesize speech using Google Cloud TTS (async)."""
    if add_punctuation:
        text = ensure_sentence_stop(text, language)

    voice_name = voice or _default_voice(language)

    async def _call():
        client = texttospeech_v1.TextToSpeechAsyncClient()
        response = await asyncio.wait_for(
            client.synthesize_speech(
                request={
                    "input": texttospeech_v1.SynthesisInput(text=text),
                    "voice": texttospeech_v1.VoiceSelectionParams(
                        language_code=language, name=voice_name,
                    ),
                    "audio_config": texttospeech_v1.AudioConfig(
                        audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                        sample_rate_hertz=sample_rate,
                        volume_gain_db=volume_gain_db,
                        effects_profile_id=[effects_profile_id] if effects_profile_id else [],
                    ),
                },
            ),
            timeout=timeout,
        )
        return response.audio_content

    raw_wav = await retry_async(_call, max_retries=max_retries)
    return _process_raw(raw_wav, text=text, format=format, trim=trim)


def _process_raw(
    raw_wav: bytes,
    *,
    text: str,
    format: AudioFormat,
    trim: bool,
) -> SynthesisResult:
    """Shared post-processing: trim silence, convert format."""
    sr, samples = wavfile.read(io.BytesIO(raw_wav))
    if trim:
        samples = trim_silence(samples)
    if format == AudioFormat.MP3:
        audio_bytes = to_mp3(samples, sr)
    else:
        audio_bytes = to_wav(samples, sr)
    duration = len(samples) / sr
    return SynthesisResult(
        audio_bytes=audio_bytes,
        format=format,
        sample_rate=sr,
        duration=duration,
        text=text,
    )


def _default_voice(language: str) -> str:
    """Pick a reasonable default voice for a language code."""
    prefix = language  # e.g. "bn-IN"
    return f"{prefix}-Chirp3-HD-Kore"
