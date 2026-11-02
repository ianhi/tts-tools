"""Thin CLI for tts-tools.

Designed for both humans and AI agents. Every option has a clear description
and sensible defaults. All output goes to stderr (progress) or stdout (results).

Examples:
    # Synthesize text to a file
    tts-tools synthesize "hello world" -l en-US -o hello.mp3

    # Use an API key instead of ADC
    tts-tools synthesize "হ্যালো" -l bn-IN --api-key YOUR_KEY -o hello.mp3

    # Use Gemini TTS
    tts-tools synthesize "hello" -l en-US --engine gemini -o hello.mp3

    # List available voices for a language
    tts-tools voices -l bn-IN

    # Output as JSON (for piping to other tools)
    tts-tools voices -l en-US --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import AudioFormat, Engine, list_voices, synthesize


@click.group()
@click.version_option(package_name="tts-tools")
def main():
    """tts-tools: Text-to-speech synthesis via Google Cloud TTS, Gemini TTS, and Edge TTS.

    Generates audio files from text with silence trimming, audio validation,
    and format conversion. Supports two auth modes for Google:

    \b
      1. Application Default Credentials (gcloud auth application-default login)
      2. API key (--api-key or GOOGLE_CLOUD_TTS_API_KEY env var)

    For Gemini TTS, set GOOGLE_API_KEY env var and use --engine gemini.
    For Edge TTS, use --engine edge (free, no API key required).
    """


@main.command()
@click.argument("text")
@click.option(
    "-l", "--language", required=True, help="BCP-47 language code (e.g. en-US, bn-IN, es-US)."
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(),
    help="Output file path (e.g. hello.mp3, hello.wav).",
)
@click.option(
    "--voice",
    default=None,
    help="Voice name (e.g. en-US-Chirp3-HD-Kore). Defaults to {language}-Chirp3-HD-Kore.",
)
@click.option(
    "--engine",
    type=click.Choice(["google_cloud", "gemini", "edge"], case_sensitive=False),
    default="google_cloud",
    help="TTS engine. 'google_cloud' (default), 'gemini', or 'edge'.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["mp3", "wav"], case_sensitive=False),
    default=None,
    help="Audio format. Auto-detected from output file extension if not set.",
)
@click.option(
    "--api-key",
    default=None,
    envvar="GOOGLE_CLOUD_TTS_API_KEY",
    help="Google Cloud TTS API key (alternative to ADC). Env: GOOGLE_CLOUD_TTS_API_KEY.",
)
@click.option(
    "--sample-rate", default=24000, type=int, show_default=True, help="Output sample rate in Hz."
)
@click.option("--no-trim", is_flag=True, help="Disable silence trimming.")
@click.option("--json-output", is_flag=True, help="Print result metadata as JSON to stdout.")
def synthesize_cmd(
    text, language, output, voice, engine, fmt, api_key, sample_rate, no_trim, json_output
):
    """Synthesize TEXT to an audio file.

    \b
    Examples:
      tts-tools synthesize "hello world" -l en-US -o hello.mp3
      tts-tools synthesize "হ্যালো" -l bn-IN -o hello.mp3 --api-key KEY
      tts-tools synthesize "hola" -l es-US --engine gemini -o hola.wav
    """
    # Auto-detect format from extension
    if fmt is None:
        ext = Path(output).suffix.lower().lstrip(".")
        fmt = ext if ext in ("mp3", "wav") else "mp3"

    result = synthesize(
        text,
        language=language,
        voice=voice,
        engine=Engine(engine),
        format=AudioFormat(fmt),
        trim=not no_trim,
        sample_rate=sample_rate,
        api_key=api_key,
    )
    path = result.save(output)

    if json_output:
        print(
            json.dumps(
                {
                    "path": str(path),
                    "format": result.format.value,
                    "duration": round(result.duration, 3),
                    "file_size": result.file_size,
                    "sample_rate": result.sample_rate,
                    "text": result.text,
                }
            )
        )
    else:
        click.echo(
            f"Saved {result.format.value} ({result.duration:.2f}s, {result.file_size:,} bytes) → {path}",
            err=True,
        )


@main.command()
@click.option(
    "-l", "--language", default=None, help="Filter voices by BCP-47 language code (e.g. bn-IN)."
)
@click.option(
    "--api-key",
    default=None,
    envvar="GOOGLE_CLOUD_TTS_API_KEY",
    help="Google Cloud TTS API key. Env: GOOGLE_CLOUD_TTS_API_KEY.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Output as JSON array (for piping to jq, scripts, etc)."
)
def voices(language, api_key, as_json):
    """List available Google Cloud TTS voices.

    \b
    Examples:
      tts-tools voices -l bn-IN
      tts-tools voices -l en-US --json
      tts-tools voices -l es-US --json | jq '.[].name'
    """
    results = list_voices(language, api_key=api_key)

    if as_json:
        print(
            json.dumps(
                [
                    {
                        "name": v.name,
                        "language_codes": v.language_codes,
                        "gender": v.ssml_gender,
                        "sample_rate": v.natural_sample_rate,
                    }
                    for v in results
                ],
                indent=2,
            )
        )
    else:
        if not results:
            click.echo("No voices found.", err=True)
            sys.exit(1)
        for v in results:
            langs = ", ".join(v.language_codes)
            click.echo(f"{v.name:<40} {v.ssml_gender:<10} {v.natural_sample_rate:>6}Hz  [{langs}]")
        click.echo(f"\n{len(results)} voices found.", err=True)
