"""Voice discovery via Google Cloud TTS API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class VoiceInfo:
    """A single TTS voice."""

    name: str
    language_codes: list[str]
    ssml_gender: str
    natural_sample_rate: int


def list_voices(language: str | None = None, *, api_key: str | None = None) -> list[VoiceInfo]:
    """List available Google Cloud TTS voices, optionally filtered by language.

    Queries the API live — no hardcoded voice lists.

    Args:
        language: BCP-47 language code to filter by (e.g. "bn-IN").
        api_key: If provided, uses the REST API with this key instead of ADC.
                 Can also be set via GOOGLE_CLOUD_TTS_API_KEY env var.
    """
    key = api_key or os.environ.get("GOOGLE_CLOUD_TTS_API_KEY")

    if key:
        return _list_voices_rest(language, key)
    return _list_voices_sdk(language)


def _list_voices_sdk(language: str | None) -> list[VoiceInfo]:
    """List voices using the google-cloud-texttospeech SDK (requires ADC)."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    response = client.list_voices(language_code=language or "")

    voices = []
    for v in response.voices:
        voices.append(
            VoiceInfo(
                name=v.name,
                language_codes=list(v.language_codes),
                ssml_gender=texttospeech.SsmlVoiceGender(v.ssml_gender).name,
                natural_sample_rate=v.natural_sample_rate_hertz,
            )
        )
    return voices


def _list_voices_rest(language: str | None, api_key: str) -> list[VoiceInfo]:
    """List voices using the REST API with an API key."""
    import httpx

    params = {"key": api_key}
    if language:
        params["languageCode"] = language

    resp = httpx.get("https://texttospeech.googleapis.com/v1/voices", params=params, timeout=30)
    resp.raise_for_status()

    voices = []
    for v in resp.json().get("voices", []):
        voices.append(
            VoiceInfo(
                name=v["name"],
                language_codes=v.get("languageCodes", []),
                ssml_gender=v.get("ssmlGender", "SSML_VOICE_GENDER_UNSPECIFIED"),
                natural_sample_rate=v.get("naturalSampleRateHertz", 0),
            )
        )
    return voices
