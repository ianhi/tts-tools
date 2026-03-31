# tts-tools

Simple, high-quality text-to-speech synthesis. A thin layer over Google Cloud TTS and Gemini TTS that handles silence trimming, audio validation, format conversion, and retries.

## Install

```bash
pip install -e .

# With Gemini TTS support
pip install -e ".[gemini]"
```

Requires Google Cloud credentials (`gcloud auth application-default login`) and the Text-to-Speech API enabled.

## Usage

```python
from tts_tools import synthesize

# Basic usage
result = synthesize("hello", language="en-US")
result.save("hello.mp3")

# Pick a voice
result = synthesize("হ্যালো", language="bn-IN", voice="bn-IN-Chirp3-HD-Kore")
result.save("hello_bn.mp3")

# WAV output, no silence trimming
result = synthesize("hola", language="es-US", format=AudioFormat.WAV, trim=False)

# Use Gemini TTS
from tts_tools import Engine
result = synthesize("hello", language="en-US", engine=Engine.GEMINI)
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

# List all Bengali voices (live API query, no hardcoded lists)
voices = list_voices(language="bn-IN")
for v in voices:
    print(f"{v.name} ({v.ssml_gender})")
```

### Audio Validation

```python
from tts_tools._audio import validate_audio

result = validate_audio(audio_bytes, min_duration=0.3, min_file_size=5000)
if not result["valid"]:
    print(result["reason"])  # "Too short (0.12s)", "Silent audio", etc.
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

## What This Does (and Doesn't Do)

This library handles the ~150 lines of boilerplate you'd otherwise copy between projects:

- **Silence trimming** via librosa (Google Cloud TTS returns variable padding)
- **Audio validation** (RMS energy, duration, file size checks)
- **Format conversion** (API returns LINEAR16 PCM; you want MP3)
- **Retry with backoff** for transient API failures
- **Gemini short-word batching** (Gemini fails on <8 char inputs; batches them with `" ... "` separators)

It does NOT include CLI tools, input file parsers, manifest management, or speech-to-text verification. Those are application-level concerns.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -o "addopts="
```
