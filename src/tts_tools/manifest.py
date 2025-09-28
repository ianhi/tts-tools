"""Audio manifest generation for minimal pairs."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import AudioToolsConfig
from .utils import ensure_directory_exists

console = Console()


class ManifestGenerator:
    """Generate and manage audio file manifests."""

    def __init__(self, config: AudioToolsConfig):
        self.config = config
        self.audio_base_path = self.config.base_audio_dir / self.config.language_code

    def generate_manifest(self, output_path: Path | None = None) -> dict[str, Any]:
        """Generate manifest of all audio files in the audio directory."""
        if output_path is None:
            output_path = (
                self.config.base_audio_dir / f"audio_manifest_{self.config.language_code}.json"
            )

        manifest = {"words": {}}
        stats = {
            "total_words": 0,
            "total_files": 0,
            "files_per_word": defaultdict(int),
            "voice_distribution": defaultdict(int),
        }

        # Scan audio directory
        if not self.audio_base_path.exists():
            console.print(f"[red]Audio directory not found: {self.audio_base_path}[/red]")
            return manifest

        # Process each word directory
        for word_dir in sorted(self.audio_base_path.iterdir()):
            if not word_dir.is_dir():
                continue

            word_name = word_dir.name
            voices = []
            extension = None

            # Scan audio files in word directory
            for audio_file in sorted(word_dir.iterdir()):
                if audio_file.suffix in [".wav", ".mp3"]:
                    # Extract voice name from filename
                    # Format: word_voice.ext
                    filename = audio_file.stem
                    if filename.startswith(f"{word_name}_"):
                        voice_name = filename[len(word_name) + 1 :]
                        voices.append(voice_name)

                        if extension is None:
                            extension = audio_file.suffix[1:]  # Remove dot

                        # Update stats
                        stats["total_files"] += 1
                        stats["voice_distribution"][voice_name] += 1

            if voices:
                manifest["words"][word_name] = {"voices": voices, "extension": extension}
                stats["total_words"] += 1
                stats["files_per_word"][len(voices)] += 1

        # Save manifest
        ensure_directory_exists(output_path.parent)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

    def verify_manifest(
        self, manifest_path: Path | None = None, fix_missing: bool = False
    ) -> dict[str, Any]:
        """Verify that all files in manifest actually exist."""
        if manifest_path is None:
            manifest_path = (
                self.config.base_audio_dir / f"audio_manifest_{self.config.language_code}.json"
            )

        if not manifest_path.exists():
            console.print(f"[red]Manifest not found: {manifest_path}[/red]")
            return {"error": "Manifest not found"}

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        results = {"total_files": 0, "missing_files": [], "existing_files": 0}

        # Check each file
        for word, word_data in manifest.get("words", {}).items():
            voices = word_data.get("voices", [])
            extension = word_data.get("extension", "wav")

            for voice in voices:
                file_path = self.audio_base_path / word / f"{word}_{voice}.{extension}"
                results["total_files"] += 1

                if file_path.exists():
                    results["existing_files"] += 1
                else:
                    results["missing_files"].append(str(file_path))

        # Print results
        console.print("\n[bold]Manifest Verification Results[/bold]")
        console.print(f"Total files in manifest: {results['total_files']}")
        console.print(f"Existing files: {results['existing_files']}")
        console.print(f"Missing files: {len(results['missing_files'])}")

        if results["missing_files"]:
            console.print("\n[red]Missing files:[/red]")
            for path in results["missing_files"][:10]:
                console.print(f"  • {path}")
            if len(results["missing_files"]) > 10:
                console.print(f"  ... and {len(results['missing_files']) - 10} more")

        # Fix manifest if requested
        if fix_missing and results["missing_files"]:
            self._fix_manifest(manifest, results["missing_files"], manifest_path)

        return results

    def _fix_manifest(
        self, manifest: dict[str, Any], missing_files: list, manifest_path: Path
    ) -> None:
        """Remove missing files from manifest."""
        console.print("\n[yellow]Fixing manifest by removing missing files...[/yellow]")

        # Convert missing files to set for faster lookup
        missing_set = set(missing_files)

        # Update manifest
        words_to_remove = []
        for word, word_data in manifest.get("words", {}).items():
            voices = word_data.get("voices", [])
            extension = word_data.get("extension", "wav")

            # Keep only existing voices
            existing_voices = []
            for voice in voices:
                file_path = str(self.audio_base_path / word / f"{word}_{voice}.{extension}")
                if file_path not in missing_set:
                    existing_voices.append(voice)

            if existing_voices:
                word_data["voices"] = existing_voices
            else:
                words_to_remove.append(word)

        # Remove words with no voices
        for word in words_to_remove:
            del manifest["words"][word]

        # Save updated manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        console.print(
            f"[green]Manifest updated. Removed {len(missing_files)} missing files.[/green]"
        )
