# TTS Tools

Thin Python library over Google Cloud TTS, Gemini TTS, and Edge TTS. Handles silence trimming, audio validation, format conversion, and retries.

## Structure

```
src/tts_tools/
├── __init__.py          # Public API: synthesize, synthesize_async, synthesize_batch, list_voices
├── _types.py            # SynthesisResult, AudioFormat, Engine
├── _audio.py            # trim_silence, validate_audio, to_mp3, to_wav, pcm_to_wav
├── _google_cloud.py     # Google Cloud TTS (sync + async)
├── _gemini.py           # Gemini TTS with short-word batching
├── _edge.py             # Edge TTS (free, no API key)
├── _voices.py           # list_voices() via live API
├── _punctuation.py      # Language-aware sentence stop
├── _retry.py            # Retry decorator (sync + async)
└── py.typed             # PEP 561 marker
tests/
├── conftest.py          # Synthetic audio fixtures
├── test_audio.py        # Audio processing tests
├── test_punctuation.py  # Punctuation tests
├── test_types.py        # Type tests
└── test_retry.py        # Retry logic tests
```

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -o "addopts="
```

## Dependencies

Core: `google-cloud-texttospeech`, `pydub`, `librosa`, `soundfile`, `scipy`
Optional: `google-genai` (for Gemini engine), `edge-tts` (for Edge engine)
