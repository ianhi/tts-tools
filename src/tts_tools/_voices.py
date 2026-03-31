"""Voice discovery via Google Cloud TTS API."""

from __future__ import annotations

from dataclasses import dataclass

from google.cloud import texttospeech


@dataclass
class VoiceInfo:
    """A single TTS voice."""

    name: str
    language_codes: list[str]
    ssml_gender: str
    natural_sample_rate: int


def list_voices(language: str | None = None) -> list[VoiceInfo]:
    """List available Google Cloud TTS voices, optionally filtered by language.

    Queries the API live — no hardcoded voice lists.
    """
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
