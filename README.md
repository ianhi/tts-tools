# tts-tools

Simple, high-quality text-to-speech synthesis. A thin layer over Google Cloud TTS, Gemini TTS, and Edge TTS that handles silence trimming, audio validation, format conversion, and retries.

## Install

```bash
uv sync

# With Gemini TTS support
uv sync --extra gemini

# With Edge TTS support (free, no API key needed)
uv sync --extra edge
```

Requires ffmpeg on PATH (for MP3 conversion). Auth depends on engine:
- **Google Cloud**: ADC (`gcloud auth application-default login`) or API key (`GOOGLE_CLOUD_TTS_API_KEY`)
- **Gemini**: `GOOGLE_API_KEY` env var
- **Edge**: No auth required (free)

## CLI

```bash
# Synthesize text to a file
tts-tools synthesize "hello world" -l en-US -o hello.mp3

# Use an API key instead of ADC
tts-tools synthesize "হ্যালো" -l bn-IN --api-key YOUR_KEY -o hello.mp3

# Use Gemini TTS (requires GOOGLE_API_KEY env var)
tts-tools synthesize "hello" -l en-US --engine gemini -o hello.mp3

# Use Edge TTS (free, no API key)
tts-tools synthesize "hello" -l en-US --engine edge -o hello.mp3

# Get JSON metadata (for piping to other tools / agents)
tts-tools synthesize "hello" -l en-US -o hello.mp3 --json-output

# List available voices
tts-tools voices -l bn-IN
tts-tools voices -l en-US --json | jq '.[].name'
```

Run `tts-tools --help`, `tts-tools synthesize --help`, or `tts-tools voices --help` for full options.

## Python API

```python
from tts_tools import synthesize

# Basic usage
result = synthesize("hello", language="en-US")
result.save("hello.mp3")

# Pick a voice
result = synthesize("হ্যালো", language="bn-IN", voice="bn-IN-Chirp3-HD-Kore")

# With API key (no ADC needed)
result = synthesize("hello", language="en-US", api_key="YOUR_KEY")

# WAV output, no silence trimming
from tts_tools import AudioFormat
result = synthesize("hola", language="es-US", format=AudioFormat.WAV, trim=False)

# Use Gemini TTS (requires GOOGLE_API_KEY env var)
from tts_tools import Engine
result = synthesize("hello", language="en-US", engine=Engine.GEMINI)

# Use Edge TTS (free, no API key)
result = synthesize("hello", language="en-US", engine=Engine.EDGE)
```

### Async & Batch

```python
from tts_tools import synthesize_async, synthesize_batch

# Single async call
result = await synthesize_async("hello", language="en-US")

# Batch with concurrency control
results = await synthesize_batch(
    ["cat", "bat", "hat"],
    language="en-US",
    max_concurrent=10,
)
for r in results:
    r.save(f"{r.text}.mp3")
```

For Gemini TTS, `synthesize_batch` automatically uses the efficient batch-and-split strategy for short texts (joins with `" ... "`, synthesizes once, splits at silence boundaries).

### Voice Discovery

```python
from tts_tools import list_voices

# Live API query (no hardcoded lists)
voices = list_voices(language="bn-IN")

# Works with API key too
voices = list_voices(language="bn-IN", api_key="YOUR_KEY")
```

### SynthesisResult

Every call returns a `SynthesisResult` with:

- `audio_bytes` - the final audio (MP3 or WAV)
- `format` - `AudioFormat.MP3` or `AudioFormat.WAV`
- `sample_rate` - output sample rate in Hz
- `duration` - audio duration in seconds
- `text` - the input text
- `file_size` - length of `audio_bytes`
- `save(path)` - write to file (creates parent dirs)
- `as_segment()` - get a pydub `AudioSegment` for further processing

### Audio Validation

```python
from tts_tools._audio import validate_audio

result = validate_audio(audio_bytes, min_duration=0.3, min_file_size=5000)
if not result["valid"]:
    print(result["reason"])  # "Too short (0.12s)", "Silent audio", etc.
```

## What This Does (and Doesn't Do)

This library handles the ~150 lines of boilerplate you'd otherwise copy between projects:

- **Silence trimming** via librosa (Google Cloud TTS returns variable padding)
- **Audio validation** (RMS energy, duration, file size checks)
- **Format conversion** (API returns LINEAR16 PCM; you want MP3). Requires ffmpeg.
- **Retry with backoff** for transient API failures
- **Gemini short-word batching** (Gemini fails on <8 char inputs; batches them with `" ... "` separators)
- **Dual auth** — ADC or simple API key, your choice

It does NOT include input file parsers, manifest management, or speech-to-text verification. Those are application-level concerns.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -o "addopts="
```
