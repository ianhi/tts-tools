# TTS Tools Project Instructions

## Project Overview

**TTS Tools** is a flexible, multi-voice text-to-speech generation toolkit designed for language learning applications, with special focus on Anki deck integration. This project evolved from a minimal-pairs-specific tool into a comprehensive system that can generate high-quality audio from arbitrary text sources using Google Cloud Text-to-Speech.

## Project Structure

```
tts-tools/
├── src/tts_tools/              # Main package
│   ├── cli.py                  # Command-line interface
│   ├── generator.py            # Core audio generation (minimal pairs specific)
│   ├── input_adapters.py       # NEW: Flexible input adapters
│   ├── config.py              # Configuration management
│   ├── models.py              # Speech recognition models
│   ├── verifier.py            # Audio quality verification
│   ├── manifest.py            # Audio manifest generation
│   ├── utils.py               # Utility functions
│   └── async_generator.py     # Async audio generation
├── pyproject.toml             # Project configuration and dependencies
├── README.md                  # Project documentation
└── uv.lock                    # Dependency lock file
```

## Key Features

### Current Capabilities
- **Multi-voice TTS**: Generate audio using multiple Google Cloud TTS voices per text item
- **Multiple Input Sources**:
  - Minimal pairs JSON (backward compatibility)
  - Plain text files, CSV, JSON
  - Anki deck files (.apkg/.anki2)
- **High-Quality Audio**: MP3 output with silence trimming and quality validation
- **Language Support**: Bengali (bn-IN), Spanish (es-US), extensible to other languages
- **Async Processing**: Parallel generation for performance
- **Audio Verification**: Speech-to-text verification of generated audio

### Architecture Highlights

#### Input Adapters System
The `input_adapters.py` module provides a flexible abstraction for reading text from various sources:

- **TextItem**: Standardized representation of text with metadata
- **InputAdapter**: Abstract base class for all input sources
- **MinimalPairsAdapter**: Backward compatibility with existing minimal pairs format
- **TextListAdapter**: Support for TXT, CSV, JSON files
- **AnkiDeckAdapter**: Extract text from Anki decks using ankipandas

#### Audio Generation Pipeline
1. **Text Input**: Read from various sources via adapters
2. **TTS Synthesis**: Google Cloud TTS with multiple voice models
3. **Audio Processing**: Silence trimming with librosa
4. **Format Conversion**: Convert to MP3 for browser compatibility
5. **Quality Validation**: File size and duration checks
6. **Manifest Generation**: Track all generated audio files

## Development Guidelines

### Code Organization
- Keep existing minimal pairs functionality intact for backward compatibility
- Build new generic functionality alongside existing code
- Use input adapters for all new text input sources
- Maintain rich console output for user feedback
- Follow existing error handling and retry patterns

### Testing Approach
- Unit tests for each input adapter type
- Integration tests with sample files (text lists, Anki decks)
- Audio quality validation tests
- Performance tests for batch generation

### Key Dependencies
- **google-cloud-texttospeech**: Core TTS functionality
- **ankipandas**: Reading Anki deck files
- **librosa**: Audio processing and silence trimming
- **pydub**: Audio format conversion
- **rich**: Console output and progress tracking
- **click**: Command-line interface

## Current Development Status

### ✅ Completed
- Input adapters module with comprehensive text source support
- Project rename from minimal-pairs-audio to tts-tools
- ankipandas dependency added for Anki deck reading
- Repository extracted with full git history

### 🚧 In Progress
- Generic audio generator module for arbitrary text input

### 📋 Planned
- Enhanced configuration system for new input sources
- New CLI commands for generic text generation
- Anki deck integration tools
- Test suite and example files

## Usage Examples

### Current CLI (Minimal Pairs)
```bash
# Generate audio for minimal pairs
tts-tools generate --overwrite --limit-voices 10

# Full pipeline with verification
tts-tools full-pipeline --language bn-IN
```

### Planned CLI (Generic Text)
```bash
# Generate from text file
tts-tools generate-from-list --input words.txt --language bn-IN

# Generate from Anki deck
tts-tools generate-from-anki --deck my_deck.apkg --fields "Front,Back"

# Batch processing
tts-tools generate-batch --config batch_config.json
```

## Future Vision

### Phase 1: Core Enhancement (Current)
- Generic audio generation beyond minimal pairs
- Flexible input system for various text sources
- Enhanced CLI with new commands

### Phase 2: Anki Integration (Future)
- Tools to inject generated audio back into Anki decks
- JavaScript templates for random audio selection in cards
- AnkiConnect API integration
- Audio deduplication across decks

### Phase 3: Advanced Features (Future)
- Web interface for non-technical users
- Cloud deployment options
- Additional TTS providers (Azure, AWS)
- Voice cloning capabilities

## Important Notes

### Package Naming
The main package has been renamed from `minimal_pairs_audio` to `tts_tools` to better reflect its expanded scope and functionality.

### Configuration
The tool uses Google Cloud TTS, so proper authentication and project setup are required:
- Set `GOOGLE_CLOUD_PROJECT` environment variable
- Ensure gcloud CLI is configured
- Enable Text-to-Speech API in your GCP project

### Audio Output
- All audio is generated as MP3 files for maximum browser compatibility
- Files are organized in a hierarchical structure: `language/word/word_voice.mp3`
- Manifests track all generated files for easy integration

### Performance
- Use async generation for large batches
- Implement rate limiting to respect Google Cloud quotas
- Cache and reuse audio files when possible

## Development Commands

```bash
# Install in development mode
uv pip install -e .

# Run tool
tts-tools --help

# Install additional dependencies
uv add package_name

# Run tests (when implemented)
uv run pytest
```

This project represents a significant evolution from a specialized minimal pairs tool to a comprehensive language learning audio generation system, designed to be both powerful for advanced users and accessible for integration into various language learning workflows.
