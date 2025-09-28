# TTS Tools

A flexible, multi-voice text-to-speech generation toolkit designed for language learning applications, with special focus on Anki deck integration. Generate high-quality audio from arbitrary text sources using Google Cloud Text-to-Speech.

## Features

- **Multi-Voice TTS**: Generate audio using multiple Google Cloud TTS voices per text item
- **Multiple Input Sources**:
  - Minimal pairs JSON (backward compatibility)
  - Plain text files, CSV, JSON
  - Anki deck files (.apkg/.anki2)
- **High-Quality Audio**: MP3 output with silence trimming and quality validation
- **Language Support**: Bengali (bn-IN), Spanish (es-US), extensible to other languages
- **Async Processing**: Parallel generation for performance
- **Audio Verification**: Speech-to-text verification of generated audio
- **Manifest Management**: Track all generated audio files for easy integration

## Prerequisites

### Google Cloud Setup

This tool uses Google Cloud Text-to-Speech API. You need to set up authentication:

#### Option 1: Using gcloud CLI (Recommended)

1. **Install Google Cloud CLI**
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate with Google Cloud**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set your project**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Enable required APIs**
   ```bash
   gcloud services enable texttospeech.googleapis.com
   gcloud services enable speech.googleapis.com
   ```

#### Option 2: Using Service Account (Alternative)

1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

**Security Note**: Never commit credential files to git. The `.gitignore` file already excludes common credential file patterns.

## Installation

```bash
# Clone the repository
git clone https://github.com/ianhi/tts-tools.git
cd tts-tools

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

### Command Line Interface

The package provides a CLI tool called `tts-tools`:

```bash
# See all available commands
tts-tools --help
```

### Quick Start Examples

#### Single File Generation

Generate a single audio file from text (great for testing):

```bash
# Generate Bengali audio
tts-tools generate-single "হ্যালো" hello.mp3 --language bn-IN

# Generate Spanish audio with specific voice
tts-tools generate-single "Hola mundo" hello.wav --language es-US --voice es-US-Neural2-A

# Test with dry run first
tts-tools generate-single "কিন্তু" test.mp3 --language bn-IN --dry-run
```

#### Generate from Text Files

```bash
# From plain text file (one word/phrase per line)
tts-tools generate-from-text words.txt --language bn-IN --limit-voices 2

# From CSV file with custom columns
tts-tools generate-from-text data.csv --language bn-IN \
    --text-column "bengali_text" \
    --identifier-column "word_id"

# Generate clean filenames without source metadata
tts-tools generate-from-text words.csv --language bn-IN --clean-filenames

# Preview without generating
tts-tools generate-from-text words.txt --language bn-IN --dry-run
```

#### Generate from Anki Decks

```bash
# Extract text from specific fields in Anki deck
tts-tools generate-from-anki deck.apkg --language bn-IN \
    --text-fields "Front" --text-fields "Back"

# Filter by deck name and use custom identifier
tts-tools generate-from-anki collection.anki2 --language bn-IN \
    --text-fields "Bengali" \
    --identifier-field "English" \
    --deck-name "Bengali Vocabulary"
```

#### Batch Processing

Create a JSON configuration file for multiple jobs:

```json
{
  "jobs": [
    {
      "name": "Bengali Words",
      "source_type": "text_list",
      "source_path": "bengali_words.txt",
      "language_code": "bn-IN",
      "voice_options": {
        "voice_type": "chirp",
        "limit_voices": 2
      }
    },
    {
      "name": "Spanish Phrases",
      "source_type": "text_list",
      "source_path": "spanish_phrases.csv",
      "language_code": "es-US",
      "voice_options": {
        "voice_type": "neural2",
        "limit_voices": 1
      }
    }
  ]
}
```

```bash
# Run batch processing
tts-tools generate-batch batch_config.json

# Preview batch jobs
tts-tools generate-batch batch_config.json --dry-run
```

### Legacy Commands (Minimal Pairs)

For backward compatibility with existing minimal pairs workflows:

```bash
# Generate audio for minimal pairs
tts-tools generate --limit-voices 3 --voice-type chirp

# Full pipeline (generate + manifest + verification)
tts-tools full-pipeline --language bn-IN
```

#### Verify Pronunciations

```bash
# Verify all audio files using Chirp2 model
tts-tools verify

# Verify specific words or categories
tts-tools verify \
    --model chirp2 \
    --words "kata" --words "khata" \
    --max-files 50 \
    --output verification_results.json
```

#### Generate Audio Manifest

```bash
# Generate manifest for audio files
tts-tools manifest

# Verify manifest and fix missing files
tts-tools manifest \
    --verify-files \
    --fix-missing
```

#### Clean Invalid Audio Files

```bash
# Remove invalid audio files
tts-tools clean --min-size 5000

# Dry run to see what would be cleaned
tts-tools clean --dry-run
```

### Python API

```python
from tts_tools import AudioGenerator, PronunciationVerifier, Chirp2Model

# Generate audio
generator = AudioGenerator(
    base_output_path="public/audio/bn-IN",
    overwrite=False
)
results = generator.generate_all_audio()

# Verify pronunciations
model = Chirp2Model()
verifier = PronunciationVerifier(model)
results = verifier.verify_all_audio(max_files=100)
```

## Troubleshooting

### Authentication Issues

**Error: "Could not automatically determine credentials"**
```bash
# Solution: Authenticate with gcloud
gcloud auth application-default login
```

**Error: "This API method requires billing to be enabled"**
```bash
# Solution: Enable billing and APIs
gcloud services enable texttospeech.googleapis.com
gcloud services enable speech.googleapis.com
# Also enable billing in Google Cloud Console
```

**Error: "Google Cloud project ID not found"**
```bash
# Solution: Set your project
gcloud config set project YOUR_PROJECT_ID
# or set environment variable
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

### Common Issues

**Files not being skipped/overwritten properly**
- Use `--overwrite` flag to regenerate existing files
- Check file permissions in output directory

**Audio quality issues**
- Increase `--volume-gain` for quiet audio (e.g., `--volume-gain 2.0`)
- Try different voice types with `--voice-type chirp` or `--voice-type wavenet`

**Large batch processing fails**
- Use `--limit-voices` to reduce concurrent requests
- Process in smaller batches using JSON configuration

### Getting Help

```bash
# See all available commands
tts-tools --help

# Get help for specific command
tts-tools generate-single --help
tts-tools generate-from-text --help
```

## Requirements

- Python 3.11+
- Google Cloud credentials configured
- Google Cloud Text-to-Speech and Speech-to-Text APIs enabled
- Billing enabled on Google Cloud project

## File Formats Supported

### Input Formats
- **TXT**: One text item per line
- **CSV**: Columns for text and identifier
- **JSON**: Array of objects or structured data
- **Anki**: .apkg and .anki2 deck files

### Output Formats
- **MP3**: Default, optimized for web browsers
- **WAV**: Uncompressed, use with `generate-single` command

### Example Input Files

**words.txt**
```
হ্যালো
নমস্কার
ধন্যবাদ
```

**words.csv**
```csv
text,identifier
হ্যালো,hello
নমস্কার,namaskar
ধন্যবাদ,thank_you
```

**words.json**
```json
[
  {"text": "হ্যালো", "identifier": "hello"},
  {"text": "নমস্কার", "identifier": "namaskar"}
]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
