# Minimal Pairs Audio Tools (Node.js)

A Node.js wrapper for the minimal pairs audio generation and verification tools.

## Quick Start

```bash
# Use directly with npx (no installation required)
npx @minimal-pairs/audio-tools generate --help

# Or install globally
npm install -g @minimal-pairs/audio-tools
minimal-pairs-audio --help
```

## Commands

### Generate Audio
```bash
npx @minimal-pairs/audio-tools generate --output-dir public/audio/bn-IN
```

### Verify Audio Quality
```bash
npx @minimal-pairs/audio-tools verify --model chirp2 --max-files 50
```

### Create Audio Manifest
```bash
npx @minimal-pairs/audio-tools manifest --audio-dir public/audio/bn-IN
```

### Run Full Pipeline
```bash
npx @minimal-pairs/audio-tools full-pipeline --voice-type chirp
```

### Regenerate Single Word
```bash
npx @minimal-pairs/audio-tools regenerate --word kata
```

## Requirements

- Node.js 18+
- Python 3.9+ with `uv` tool installed
- Google Cloud credentials configured

## Installation

This package wraps the Python-based audio tools and requires `uv` to be installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# The Python tools will be automatically managed by uv
```

## Development

```bash
git clone <your-repo>
cd audio-tools-node
npm install
npm link  # for local development
```
