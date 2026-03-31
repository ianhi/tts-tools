"""Google Cloud Text-to-Speech engine (sync + async).

Supports two auth modes:
  1. ADC / service account (default) — uses the google-cloud-texttospeech SDK
  2. API key — uses the REST API directly via httpx (no SDK auth needed)

Pass `api_key="..."` to use mode 2.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os

from scipy.io import wavfile

from ._audio import pcm_to_wav, to_mp3, to_wav, trim_silence
from ._punctuation import ensure_sentence_stop
from ._retry import retry_async, retry_sync
from ._types import AudioFormat, SynthesisResult

CLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


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
    api_key: str | None = None,
) -> SynthesisResult:
    """Synthesize speech using Google Cloud TTS (synchronous).

    Args:
        api_key: If provided, uses the REST API with this key instead of ADC.
                 Can also be set via GOOGLE_CLOUD_TTS_API_KEY env var.
    """
    if add_punctuation:
        text = ensure_sentence_stop(text, language)

    voice_name = voice or _default_voice(language)
    key = api_key or os.environ.get("GOOGLE_CLOUD_TTS_API_KEY")

    if key:
        raw_wav = retry_sync(
            lambda: _rest_synthesize(text, language, voice_name, sample_rate, volume_gain_db, effects_profile_id, key, timeout),
            max_retries=max_retries,
        )
    else:
        raw_wav = retry_sync(
            lambda: _sdk_synthesize(text, language, voice_name, sample_rate, volume_gain_db, effects_profile_id, timeout),
            max_retries=max_retries,
        )
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
    api_key: str | None = None,
) -> SynthesisResult:
    """Synthesize speech using Google Cloud TTS (async).

    Args:
        api_key: If provided, uses the REST API with this key instead of ADC.
                 Can also be set via GOOGLE_CLOUD_TTS_API_KEY env var.
    """
    if add_punctuation:
        text = ensure_sentence_stop(text, language)

    voice_name = voice or _default_voice(language)
    key = api_key or os.environ.get("GOOGLE_CLOUD_TTS_API_KEY")

    if key:
        raw_wav = await retry_async(
            lambda: _async_rest_synthesize(text, language, voice_name, sample_rate, volume_gain_db, effects_profile_id, key, timeout),
            max_retries=max_retries,
        )
    else:
        async def _call():
            from google.cloud import texttospeech_v1

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


# --- SDK-based synthesis (ADC / service account) ---


def _sdk_synthesize(
    text: str,
    language: str,
    voice_name: str,
    sample_rate: int,
    volume_gain_db: float,
    effects_profile_id: str | None,
    timeout: int,
) -> bytes:
    """Synthesize using the google-cloud-texttospeech SDK (requires ADC)."""
    from google.cloud import texttospeech

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


# --- REST API synthesis (API key, no SDK auth) ---


def _rest_synthesize(
    text: str,
    language: str,
    voice_name: str,
    sample_rate: int,
    volume_gain_db: float,
    effects_profile_id: str | None,
    api_key: str,
    timeout: int,
) -> bytes:
    """Synthesize using the REST API with an API key. Returns WAV bytes."""
    import httpx

    payload = {
        "input": {"text": text},
        "voice": {"languageCode": language, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": sample_rate,
            "volumeGainDb": volume_gain_db,
        },
    }
    if effects_profile_id:
        payload["audioConfig"]["effectsProfileId"] = [effects_profile_id]

    resp = httpx.post(CLOUD_TTS_URL, params={"key": api_key}, json=payload, timeout=timeout)
    resp.raise_for_status()
    pcm_data = base64.b64decode(resp.json()["audioContent"])
    return pcm_to_wav(pcm_data, sample_rate=sample_rate)


async def _async_rest_synthesize(
    text: str,
    language: str,
    voice_name: str,
    sample_rate: int,
    volume_gain_db: float,
    effects_profile_id: str | None,
    api_key: str,
    timeout: int,
) -> bytes:
    """Async REST API synthesis with an API key. Returns WAV bytes."""
    import httpx

    payload = {
        "input": {"text": text},
        "voice": {"languageCode": language, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": sample_rate,
            "volumeGainDb": volume_gain_db,
        },
    }
    if effects_profile_id:
        payload["audioConfig"]["effectsProfileId"] = [effects_profile_id]

    async with httpx.AsyncClient() as client:
        resp = await client.post(CLOUD_TTS_URL, params={"key": api_key}, json=payload, timeout=timeout)
    resp.raise_for_status()
    pcm_data = base64.b64decode(resp.json()["audioContent"])
    return pcm_to_wav(pcm_data, sample_rate=sample_rate)


# --- Shared ---


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
    return f"{language}-Chirp3-HD-Kore"
