"""Generic audio generation functionality for arbitrary text input."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from .generator import (
    get_all_voice_names,
    get_minimal_voice_name,
    synthesize_raw_audio,
    validate_audio_file,
)
from .input_adapters import InputAdapter, TextItem
from .utils import ensure_directory_exists

console = Console()


class VoiceConfig:
    """Configuration for voice selection and audio generation."""

    def __init__(
        self,
        voice_type: str = "all",
        limit_voices: int | None = None,
        voice_names: list[str] | None = None,
        volume_gain_db: float = 0.0,
        effects_profile: str = "headphone-class-device",
    ):
        """Initialize voice configuration.

        Args:
            voice_type: Type of voices to use ("all", "chirp", "wavenet", "neural2").
            limit_voices: Maximum number of voices to use per text item.
            voice_names: Specific voice names to use (overrides voice_type).
            volume_gain_db: Volume gain in decibels.
            effects_profile: Audio effects profile for TTS.
        """
        self.voice_type = voice_type
        self.limit_voices = limit_voices
        self.voice_names = voice_names
        self.volume_gain_db = volume_gain_db
        self.effects_profile = effects_profile


class OutputConfig:
    """Configuration for output organization and file naming."""

    def __init__(
        self,
        base_output_dir: Path,
        organization_strategy: str = "by_language",
        include_metadata: bool = True,
        file_format: str = "mp3",
        overwrite: bool = False,
    ):
        """Initialize output configuration.

        Args:
            base_output_dir: Base directory for audio output.
            organization_strategy: How to organize files ("by_language", "by_source", "flat").
            include_metadata: Whether to include metadata in filenames.
            file_format: Audio file format ("mp3", "wav").
            overwrite: Whether to overwrite existing files.
        """
        self.base_output_dir = Path(base_output_dir)
        self.organization_strategy = organization_strategy
        self.include_metadata = include_metadata
        self.file_format = file_format
        self.overwrite = overwrite


class GenerationResult:
    """Result of audio generation for a single text item."""

    def __init__(
        self,
        text_item: TextItem,
        success_count: int = 0,
        failed_count: int = 0,
        skipped_count: int = 0,
        generated_files: list[Path] | None = None,
        errors: list[str] | None = None,
    ):
        """Initialize generation result.

        Args:
            text_item: The original text item.
            success_count: Number of successful audio generations.
            failed_count: Number of failed audio generations.
            skipped_count: Number of skipped audio generations.
            generated_files: List of successfully generated audio files.
            errors: List of error messages for failed generations.
        """
        self.text_item = text_item
        self.success_count = success_count
        self.failed_count = failed_count
        self.skipped_count = skipped_count
        self.generated_files = generated_files or []
        self.errors = errors or []

    @property
    def total_attempted(self) -> int:
        """Total number of audio generations attempted."""
        return self.success_count + self.failed_count + self.skipped_count

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_attempted == 0:
            return 0.0
        return (self.success_count / self.total_attempted) * 100


class BatchResult:
    """Result of batch audio generation."""

    def __init__(self, results: list[GenerationResult]):
        """Initialize batch result.

        Args:
            results: List of individual generation results.
        """
        self.results = results

    @property
    def total_items(self) -> int:
        """Total number of text items processed."""
        return len(self.results)

    @property
    def total_files_generated(self) -> int:
        """Total number of audio files successfully generated."""
        return sum(result.success_count for result in self.results)

    @property
    def total_files_failed(self) -> int:
        """Total number of audio file generations that failed."""
        return sum(result.failed_count for result in self.results)

    @property
    def total_files_skipped(self) -> int:
        """Total number of audio file generations that were skipped."""
        return sum(result.skipped_count for result in self.results)

    @property
    def success_rate(self) -> float:
        """Overall success rate as a percentage."""
        total_attempted = (
            self.total_files_generated + self.total_files_failed + self.total_files_skipped
        )
        if total_attempted == 0:
            return 0.0
        return (self.total_files_generated / total_attempted) * 100

    # Aliases for CLI compatibility
    @property
    def successful(self) -> int:
        """Alias for total_files_generated for CLI compatibility."""
        return self.total_files_generated

    @property
    def failed(self) -> int:
        """Alias for total_files_failed for CLI compatibility."""
        return self.total_files_failed

    @property
    def skipped(self) -> int:
        """Alias for total_files_skipped for CLI compatibility."""
        return self.total_files_skipped


class GenericAudioGenerator:
    """Generic audio generator for arbitrary text input sources."""

    def __init__(
        self,
        voice_config: VoiceConfig | None = None,
        output_config: OutputConfig | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        # Simplified constructor parameters for CLI compatibility
        base_output_path: Path | None = None,
        language_code: str | None = None,
        overwrite: bool = False,
        clean_filenames: bool = False,
    ):
        """Initialize the generic audio generator.

        Args:
            voice_config: Configuration for voice selection and audio settings.
            output_config: Configuration for output organization.
            max_retries: Maximum number of retry attempts for failed generations.
            timeout_seconds: Timeout for individual TTS requests.
            base_output_path: Simple output path (for CLI compatibility).
            language_code: Language code (for CLI compatibility).
            overwrite: Whether to overwrite existing files (for CLI compatibility).
        """
        # Handle simplified constructor for CLI usage
        if voice_config is None:
            voice_config = VoiceConfig()
        if output_config is None:
            if base_output_path is None:
                raise ValueError("Either output_config or base_output_path must be provided")
            output_config = OutputConfig(
                base_output_dir=base_output_path,
                organization_strategy="by_language",
                include_metadata=not clean_filenames,  # Clean filenames = no metadata
            )

        self.voice_config = voice_config
        self.output_config = output_config
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.overwrite = overwrite
        self.language_code = language_code

        # Ensure output directory exists
        ensure_directory_exists(self.output_config.base_output_dir)

    def get_voices_for_language(self, language_code: str) -> list[str]:
        """Get list of voices to use for a specific language.

        Args:
            language_code: Language code (e.g., "bn-IN", "es-US").

        Returns:
            List of voice names to use.
        """
        # Use specific voice names if provided
        if self.voice_config.voice_names:
            return self.voice_config.voice_names

        # Get all available voices for the language
        try:
            all_voices = get_all_voice_names(language_code)
        except ValueError:
            console.print(f"[yellow]Warning: No voices found for language {language_code}[/yellow]")
            return []

        # Filter by voice type
        voices = []
        if self.voice_config.voice_type == "all":
            for voice_list in all_voices.values():
                voices.extend(voice_list)
        elif self.voice_config.voice_type in all_voices:
            voices = all_voices[self.voice_config.voice_type]
        else:
            # Try to find voice type in any category
            for voice_type, voice_list in all_voices.items():
                if self.voice_config.voice_type.lower() in voice_type.lower():
                    voices.extend(voice_list)

        # Apply voice limit
        if self.voice_config.limit_voices and len(voices) > self.voice_config.limit_voices:
            voices = voices[: self.voice_config.limit_voices]

        return voices

    def get_output_path(self, text_item: TextItem, voice_name: str) -> Path:
        """Generate output path for a text item and voice combination.

        Args:
            text_item: The text item being processed.
            voice_name: Name of the TTS voice.

        Returns:
            Path where the audio file should be saved.
        """
        base_dir = self.output_config.base_output_dir
        minimal_voice = get_minimal_voice_name(voice_name)

        # Build filename
        filename_parts = [text_item.identifier]
        if self.output_config.include_metadata and text_item.metadata:
            source = text_item.metadata.get("source", "")
            if source:
                filename_parts.append(source[:8])  # Truncate for readability

        filename_parts.append(minimal_voice)
        filename = "_".join(filename_parts) + f".{self.output_config.file_format}"

        # Organize by strategy
        if self.output_config.organization_strategy == "by_language":
            return base_dir / text_item.language_code / text_item.identifier / filename

        elif self.output_config.organization_strategy == "by_source":
            source = (
                text_item.metadata.get("source", "unknown") if text_item.metadata else "unknown"
            )
            return base_dir / source / text_item.identifier / filename

        elif self.output_config.organization_strategy == "flat":
            return base_dir / filename

        else:
            # Default to by_language
            return base_dir / text_item.language_code / text_item.identifier / filename

    def generate_audio_for_item(
        self, text_item: TextItem, progress_callback: Callable | None = None
    ) -> GenerationResult:
        """Generate audio for a single text item using all configured voices.

        Args:
            text_item: The text item to generate audio for.
            progress_callback: Optional callback for progress updates.

        Returns:
            GenerationResult with details of the generation process.
        """
        voices = self.get_voices_for_language(text_item.language_code)
        if not voices:
            return GenerationResult(
                text_item=text_item,
                failed_count=1,
                errors=[f"No voices available for language {text_item.language_code}"],
            )

        result = GenerationResult(text_item)

        for voice_name in voices:
            if progress_callback:
                progress_callback(
                    f"Generating {text_item.identifier} with {get_minimal_voice_name(voice_name)}"
                )

            try:
                file_result = self._generate_single_audio_file(text_item, voice_name)
                if file_result["status"] == "success":
                    result.success_count += 1
                    result.generated_files.append(file_result["path"])
                elif file_result["status"] == "skipped":
                    result.skipped_count += 1
                    # Skipped files are still valid, just not regenerated
                else:
                    result.failed_count += 1
                    result.errors.append(
                        f"{get_minimal_voice_name(voice_name)}: {file_result.get('reason', 'Unknown error')}"
                    )

            except Exception as e:
                result.failed_count += 1
                result.errors.append(f"{get_minimal_voice_name(voice_name)}: {str(e)}")

        return result

    def _generate_single_audio_file(self, text_item: TextItem, voice_name: str) -> dict[str, Any]:
        """Generate a single audio file for a text item and voice.

        Args:
            text_item: The text item to generate audio for.
            voice_name: Name of the TTS voice to use.

        Returns:
            Dictionary with generation result details.
        """
        output_path = self.get_output_path(text_item, voice_name)

        # Check if file already exists and skip if not overwriting
        if output_path.exists() and not self.overwrite:
            validation = validate_audio_file(
                output_path,
                min_file_size=5000,  # Same defaults as existing generator
                min_duration=0.3,
            )
            if validation["valid"]:
                return {
                    "status": "skipped",
                    "path": output_path,
                    "file_size": validation["file_size"],
                    "duration": validation["duration"],
                }
            else:
                # File exists but is invalid, regenerate
                pass

        # Ensure output directory exists
        ensure_directory_exists(output_path.parent)

        # Add punctuation for natural speech based on language
        text_with_punctuation = self._add_punctuation(text_item.text, text_item.language_code)

        # Retry logic for failed recordings
        for attempt in range(self.max_retries):
            try:
                # Generate audio with timeout
                sample_rate, samples = synthesize_raw_audio(
                    text_with_punctuation,
                    volume_gain_db=self.voice_config.volume_gain_db,
                    effects_profile_id=self.voice_config.effects_profile,
                    voice_name=voice_name,
                    language_code=text_item.language_code,
                    timeout=self.timeout_seconds,
                )

                # Process and save audio
                return self._process_and_save_audio(samples, sample_rate, output_path)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    return {"status": "failed", "reason": str(e)}

        return {"status": "failed", "reason": f"Failed after {self.max_retries} attempts"}

    def _add_punctuation(self, text: str, language_code: str) -> str:
        """Add appropriate punctuation for natural speech.

        Args:
            text: The text to add punctuation to.
            language_code: Language code for punctuation selection.

        Returns:
            Text with appropriate punctuation.
        """
        text = text.strip()

        # Don't add punctuation if it already ends with punctuation
        if text and text[-1] in ".!?।":
            return text

        # Add language-appropriate punctuation
        if language_code == "bn-IN":
            return f"{text}।"  # Bengali full stop
        elif language_code.startswith("es"):
            return f"{text}."  # Spanish period
        else:
            return f"{text}."  # Default period

    def _process_and_save_audio(
        self, samples: Any, sample_rate: int, output_path: Path
    ) -> dict[str, Any]:
        """Process audio samples and save to file.

        Args:
            samples: Raw audio samples from TTS.
            sample_rate: Sample rate of the audio.
            output_path: Path to save the processed audio.

        Returns:
            Dictionary with processing result details.
        """
        import librosa
        import numpy as np
        from pydub import AudioSegment

        try:
            # Trim silence from beginning and end
            trimmed_samples = librosa.effects.trim(samples, top_db=30)[0]

            # Convert to 16-bit integers for pydub
            if trimmed_samples.dtype != np.int16:
                if trimmed_samples.dtype in [np.float32, np.float64]:
                    # Samples are in [-1, 1] range
                    trimmed_samples = (trimmed_samples * 32767).astype(np.int16)
                else:
                    trimmed_samples = trimmed_samples.astype(np.int16)

            # Create AudioSegment and export
            audio_segment = AudioSegment(
                trimmed_samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=1,  # mono
            )

            # Export in the specified format
            if self.output_config.file_format.lower() == "mp3":
                audio_segment.export(str(output_path), format="mp3", bitrate="192k")
            else:
                audio_segment.export(str(output_path), format=self.output_config.file_format)

            # Validate the generated audio file
            validation = validate_audio_file(output_path)

            if validation["valid"]:
                return {
                    "status": "success",
                    "path": output_path,
                    "file_size": validation["file_size"],
                    "duration": validation["duration"],
                }
            else:
                # Delete invalid file
                if output_path.exists():
                    output_path.unlink()
                return {"status": "failed", "reason": f"Validation failed: {validation['reason']}"}

        except Exception as e:
            # Clean up on error
            if output_path.exists():
                output_path.unlink()
            return {"status": "failed", "reason": f"Audio processing failed: {str(e)}"}

    def generate_from_adapter(
        self, adapter: InputAdapter, progress_callback: Callable | None = None
    ) -> BatchResult:
        """Generate audio for all text items from an input adapter.

        Args:
            adapter: Input adapter providing text items.
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult with details of the batch generation.
        """
        text_items = adapter.get_text_items()
        total_items = len(text_items)

        if total_items == 0:
            console.print("[yellow]No text items found in input source[/yellow]")
            return BatchResult([])

        console.print(f"[bold blue]🎵 Generating audio for {total_items} text items...[/bold blue]")

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating audio...", total=total_items)

            for i, text_item in enumerate(text_items):
                if progress_callback:
                    progress_callback(
                        f"Processing item {i+1}/{total_items}: {text_item.identifier}"
                    )

                progress.update(task, description=f"Processing {text_item.identifier}")

                result = self.generate_audio_for_item(text_item)
                results.append(result)

                progress.advance(task)

        # Display summary
        batch_result = BatchResult(results)
        self._display_summary(batch_result)

        return batch_result

    def _display_summary(self, batch_result: BatchResult) -> None:
        """Display a summary of the batch generation results.

        Args:
            batch_result: The batch result to summarize.
        """
        table = Table(title="Audio Generation Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Items", str(batch_result.total_items))
        table.add_row("Files Generated", str(batch_result.total_files_generated))
        table.add_row("Files Skipped", str(batch_result.total_files_skipped))
        table.add_row("Files Failed", str(batch_result.total_files_failed))
        table.add_row("Success Rate", f"{batch_result.success_rate:.1f}%")

        console.print(table)

        # Show errors if any
        failed_results = [r for r in batch_result.results if r.errors]
        if failed_results and len(failed_results) <= 10:  # Don't spam with too many errors
            console.print("\n[bold red]Errors encountered:[/bold red]")
            for result in failed_results:
                console.print(f"[red]• {result.text_item.identifier}:[/red]")
                for error in result.errors[:3]:  # Limit to first 3 errors per item
                    console.print(f"  - {error}")
