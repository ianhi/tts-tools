# TTS Tools Development Plan

## Project Vision

Transform the minimal-pairs-specific audio tool into a flexible system that can:
1. Accept arbitrary text lists (words, phrases, sentences) from multiple sources
2. Generate audio using multiple Google Cloud TTS voice models per text item
3. Read from existing Anki decks to identify text needing audio
4. Prepare audio files for Anki integration with proper naming conventions

## Architecture Overview

### Core Design Principles

- **Backward Compatibility**: Maintain existing minimal pairs functionality
- **Flexible Input**: Support multiple text sources through adapter pattern
- **Scalable Processing**: Handle large batches efficiently with async processing
- **Quality Assurance**: Maintain audio validation and verification capabilities
- **Anki Integration**: Prepare for seamless Anki deck enhancement workflows

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Processing     │    │   Output        │
│                 │    │  Layer          │    │   Layer         │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Text Files   │ │───▶│ │Generic Audio│ │───▶│ │MP3 Files    │ │
│ │CSV/JSON     │ │    │ │Generator    │ │    │ │Manifests    │ │
│ │Anki Decks   │ │    │ │             │ │    │ │Verification │ │
│ │Minimal Pairs│ │    │ │Multi-Voice  │ │    │ │Results      │ │
│ └─────────────┘ │    │ │TTS Pipeline │ │    │ └─────────────┘ │
└─────────────────┘    │ └─────────────┘ │    └─────────────────┘
                       └─────────────────┘
```

## Implementation Plan

### Phase 1: Core Audio Generation Enhancement

#### ✅ Completed Tasks

1. **Input Adapters Module** (`input_adapters.py`)
   - Abstract `InputAdapter` base class
   - `TextItem` standardized data structure
   - `MinimalPairsAdapter` for backward compatibility
   - `TextListAdapter` for TXT/CSV/JSON files
   - `AnkiDeckAdapter` for .apkg/.anki2 files
   - Factory function for adapter creation

2. **Project Infrastructure**
   - Renamed from minimal-pairs-audio to lang-audio-generator
   - Added ankipandas and pandas dependencies
   - Extracted repository with full git history
   - Updated project metadata and descriptions

#### 🚧 Current Task: Generic Audio Generator

**Create `generic_generator.py`** - New module for arbitrary text processing

**Key Components:**
```python
class GenericAudioGenerator:
    def __init__(self, config, voices_config):
        # Support arbitrary language codes
        # Configurable voice selection strategies
        # Output path templates

    def generate_audio_for_text(self, text, identifier, metadata=None):
        # Generate with all configured voices
        # Store with flexible naming scheme
        # Track metadata for Anki integration

    def batch_generate(self, text_items, progress_callback=None):
        # Parallel processing support
        # Progress tracking
        # Error recovery
```

**Features to Implement:**
- Support for arbitrary text lengths (words, phrases, sentences)
- Flexible output organization strategies
- Metadata preservation for source tracking
- Integration with existing audio processing pipeline
- Voice selection strategies (all, random subset, specific types)

#### 📋 Remaining Phase 1 Tasks

3. **Enhanced Configuration System**
   - Extend `AudioToolsConfig` for new input sources
   - Add voice selection configurations
   - Support multiple output organization schemes
   - Environment-based configuration overrides

4. **New CLI Commands**
   - `generate-from-list`: Process text files/CSV/JSON
   - `generate-from-anki`: Extract and generate from Anki decks
   - `generate-batch`: Multi-source batch processing
   - Maintain existing commands for backward compatibility

5. **Testing and Validation**
   - Unit tests for all input adapters
   - Integration tests with sample files
   - Performance tests for batch operations
   - Audio quality validation tests

### Output Organization Strategy

#### Flexible Directory Structure
```
output/
├── by_language/
│   ├── bn-IN/
│   │   ├── word1/
│   │   │   ├── word1_voice1.mp3
│   │   │   └── word1_voice2.mp3
│   │   └── phrase1/
│   │       ├── phrase1_voice1.mp3
│   │       └── phrase1_voice2.mp3
│   └── es-US/
│       └── sentence1/
├── by_source/
│   ├── anki_deck_name/
│   │   ├── note_123_front/
│   │   └── note_456_back/
│   └── text_list_name/
│       ├── item_001/
│       └── item_002/
└── manifests/
    ├── audio_manifest_bn-IN.json
    ├── audio_manifest_es-US.json
    └── source_manifest_anki_deck.json
```

#### Manifest Enhancement
```json
{
  "language": "bn-IN",
  "total_items": 150,
  "total_files": 1500,
  "sources": {
    "anki_deck_name": {
      "item_count": 50,
      "file_count": 500,
      "notes": [
        {
          "note_id": 123,
          "field": "Front",
          "text": "নমস্কার",
          "identifier": "nomoshkar_123_front",
          "audio_files": ["nomoshkar_123_front_voice1.mp3", "..."]
        }
      ]
    }
  },
  "voices_used": ["bn-IN-Chirp3-HD-Aoede", "..."],
  "generation_metadata": {
    "timestamp": "2024-09-27T10:00:00Z",
    "total_duration": "45 minutes",
    "success_rate": 0.98
  }
}
```

### Phase 2: Advanced Features (Future)

#### Anki Integration Tools
1. **Audio Injection Module**
   - Read existing Anki decks
   - Add [sound:] tags to appropriate fields
   - Update media collection with new audio files
   - Create new .apkg with embedded audio

2. **JavaScript Templates**
   - Random audio selection for variety
   - Configurable playback behavior
   - Card template generation

3. **AnkiConnect Integration**
   - Live sync with running Anki instance
   - Real-time audio addition
   - Deck monitoring and auto-generation

#### Enhanced Processing
1. **Voice Management**
   - Voice quality scoring
   - Automatic voice selection based on text characteristics
   - Voice caching and reuse strategies

2. **Text Analysis**
   - Language detection
   - Text normalization
   - Pronunciation hints integration

3. **Quality Assurance**
   - Advanced audio validation
   - Automatic regeneration of poor quality audio
   - User feedback integration

### Technical Specifications

#### Performance Requirements
- **Throughput**: 100+ audio files per minute (async processing)
- **Quality**: >95% successful generation rate
- **Memory**: Efficient processing of large text lists (10,000+ items)
- **Storage**: Organized output with deduplication capabilities

#### Voice Selection Strategies
1. **All Voices**: Generate with every available voice
2. **Voice Limit**: Configurable maximum number of voices
3. **Voice Types**: Filter by voice type (chirp, wavenet, neural2)
4. **Quality Selection**: Prioritize high-quality voices
5. **Random Sampling**: Random subset for variety

#### Error Handling
- Graceful degradation on TTS API failures
- Automatic retry with exponential backoff
- Detailed error reporting and logging
- Resume capability for interrupted batch operations

#### Integration Points
- **Google Cloud TTS**: Primary audio generation
- **ankipandas**: Anki deck reading
- **librosa**: Audio processing
- **pydub**: Format conversion
- **rich**: User interface and progress tracking

### Migration Strategy

1. **Gradual Enhancement**
   - Keep existing minimal pairs functionality intact
   - Build new generic system alongside existing code
   - Refactor common components into shared utilities

2. **Backward Compatibility**
   - Maintain existing CLI commands
   - Support existing configuration files
   - Preserve output format compatibility

3. **Testing Strategy**
   - Comprehensive test suite for new functionality
   - Regression tests for existing features
   - Performance benchmarks for optimization

4. **Documentation**
   - Update README with new capabilities
   - Create usage examples for each input type
   - API documentation for programmatic use

### Success Metrics

#### Functionality
- [ ] Support for 5+ input formats (minimal pairs, TXT, CSV, JSON, Anki)
- [ ] Multi-language generation (bn-IN, es-US, extensible)
- [ ] Batch processing of 1000+ items efficiently
- [ ] Audio quality validation with >95% success rate

#### Usability
- [ ] Intuitive CLI interface for all input types
- [ ] Clear error messages and help documentation
- [ ] Progress tracking for long-running operations
- [ ] Example files and tutorials

#### Performance
- [ ] 10x improvement in batch processing speed (async)
- [ ] Memory usage optimization for large datasets
- [ ] Resumable operations for interrupted processes
- [ ] Efficient storage organization and deduplication

This plan provides a comprehensive roadmap for transforming the minimal pairs tool into a flexible, powerful audio generation system suitable for diverse language learning applications while maintaining the robustness and quality of the existing system.