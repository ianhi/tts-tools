"""Audio generation functionality for minimal pairs."""

import io
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from google.cloud import texttospeech
from pydub import AudioSegment
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from scipy.io import wavfile

from .config import AudioToolsConfig
from .utils import ensure_directory_exists, get_unique_words, load_minimal_pairs_data

console = Console()


def get_minimal_voice_name(full_voice_name: str) -> str:
    """Extract minimal voice name from full name.

    Examples:
        'bn-IN-Chirp3-HD-Aoede' -> 'chirp3-hd-aoede'
    """
    parts = full_voice_name.split("-")
    if (
        len(parts) >= 3
        and parts[0] == full_voice_name[:2].lower()
        and parts[1] == full_voice_name[3:5].lower()
    ):
        return "-".join(parts[2:]).lower()
    return full_voice_name.lower()


def validate_audio_file(
    file_path: Path, min_file_size: int = 5000, min_duration: float = 0.3
) -> dict[str, Any]:
    """Validate that an audio file contains actual audio content.

    Args:
        file_path: Path to the audio file
        min_file_size: Minimum file size in bytes (default: 5KB)
        min_duration: Minimum duration in seconds (default: 0.3s)

    Returns:
        Dictionary with validation results
    """
    try:
        # Check file size first (quick check)
        file_size = file_path.stat().st_size
        if file_size < min_file_size:
            return {
                "valid": False,
                "reason": f"File too small ({file_size} bytes, minimum: {min_file_size})",
                "file_size": file_size,
                "duration": 0,
            }

        # Check audio duration and content
        data, sample_rate = sf.read(str(file_path))
        duration = len(data) / sample_rate

        if duration < min_duration:
            return {
                "valid": False,
                "reason": f"Audio too short ({duration:.2f}s, minimum: {min_duration}s)",
                "file_size": file_size,
                "duration": duration,
            }

        # Check if audio contains actual content (not just silence)
        if len(data) > 0:
            rms = (data**2).mean() ** 0.5
            if rms < 1e-6:
                return {
                    "valid": False,
                    "reason": f"Audio appears to be silent (RMS: {rms:.2e})",
                    "file_size": file_size,
                    "duration": duration,
                }

        return {
            "valid": True,
            "reason": "Audio validation passed",
            "file_size": file_size,
            "duration": duration,
        }

    except Exception as e:
        return {
            "valid": False,
            "reason": f"Error reading audio file: {str(e)}",
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "duration": 0,
        }


def synthesize_raw_audio(
    text: str,
    volume_gain_db: float = 0.0,
    effects_profile_id: str = "headphone-class-device",
    voice_name: str = "bn-IN-Chirp3-HD-Aoede",
    language_code: str = "bn-IN",
    timeout: int = 30,
) -> tuple[int, Any]:
    """Synthesize speech from Google TTS and return audio data."""
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Keep WAV for processing
        sample_rate_hertz=44100,  # Standard sample rate for better browser compatibility
        volume_gain_db=volume_gain_db,
        effects_profile_id=[effects_profile_id] if effects_profile_id else [],
    )

    # Use the timeout parameter in the API call
    response = client.synthesize_speech(
        request={
            "input": input_text,
            "voice": voice,
            "audio_config": audio_config,
        },
        timeout=timeout if timeout > 0 else None,
    )

    # Convert response to audio samples for processing
    sample_rate, samples = wavfile.read(io.BytesIO(response.audio_content))
    return sample_rate, samples


def get_all_voice_names(language_code: str = "bn-IN") -> dict[str, list[str]]:
    """Get all available voice names for the specified language."""
    if language_code == "bn-IN":
        return {
            "chirp3_hd": [
                "bn-IN-Chirp3-HD-Achernar",
                "bn-IN-Chirp3-HD-Achird",
                "bn-IN-Chirp3-HD-Algenib",
                "bn-IN-Chirp3-HD-Algieba",
                "bn-IN-Chirp3-HD-Alnilam",
                "bn-IN-Chirp3-HD-Aoede",
                "bn-IN-Chirp3-HD-Autonoe",
                "bn-IN-Chirp3-HD-Callirrhoe",
                "bn-IN-Chirp3-HD-Charon",
                "bn-IN-Chirp3-HD-Despina",
                "bn-IN-Chirp3-HD-Enceladus",
                "bn-IN-Chirp3-HD-Erinome",
                "bn-IN-Chirp3-HD-Fenrir",
                "bn-IN-Chirp3-HD-Gacrux",
                "bn-IN-Chirp3-HD-Iapetus",
                "bn-IN-Chirp3-HD-Kore",
                "bn-IN-Chirp3-HD-Laomedeia",
                "bn-IN-Chirp3-HD-Leda",
                "bn-IN-Chirp3-HD-Orus",
                "bn-IN-Chirp3-HD-Puck",
                "bn-IN-Chirp3-HD-Pulcherrima",
                "bn-IN-Chirp3-HD-Rasalgethi",
                "bn-IN-Chirp3-HD-Sadachbia",
                "bn-IN-Chirp3-HD-Sadaltager",
                "bn-IN-Chirp3-HD-Schedar",
                "bn-IN-Chirp3-HD-Sulafat",
                "bn-IN-Chirp3-HD-Umbriel",
            ],
            "wavenet": [
                "bn-IN-Wavenet-A",  # Female
                "bn-IN-Wavenet-B",  # Male
                "bn-IN-Wavenet-C",  # Female
                "bn-IN-Wavenet-D",  # Male
            ],
        }
    elif language_code == "es-US":
        return {
            "chirp_hd": [
                "es-US-Chirp-HD-D",  # Male
                "es-US-Chirp-HD-F",  # Female
                "es-US-Chirp-HD-O",  # Female
            ],
            "chirp3_hd": [
                "es-US-Chirp3-HD-Achernar",  # Female
                "es-US-Chirp3-HD-Achird",  # Male
                "es-US-Chirp3-HD-Algenib",  # Male
                "es-US-Chirp3-HD-Algieba",  # Male
                "es-US-Chirp3-HD-Alnilam",  # Male
                "es-US-Chirp3-HD-Aoede",  # Female
                "es-US-Chirp3-HD-Autonoe",  # Female
                "es-US-Chirp3-HD-Callirrhoe",  # Female
                "es-US-Chirp3-HD-Charon",  # Male
                "es-US-Chirp3-HD-Despina",  # Female
                "es-US-Chirp3-HD-Enceladus",  # Male
                "es-US-Chirp3-HD-Erinome",  # Female
                "es-US-Chirp3-HD-Fenrir",  # Male
                "es-US-Chirp3-HD-Gacrux",  # Female
                "es-US-Chirp3-HD-Iapetus",  # Male
                "es-US-Chirp3-HD-Kore",  # Female
                "es-US-Chirp3-HD-Laomedeia",  # Female
                "es-US-Chirp3-HD-Leda",  # Female
                "es-US-Chirp3-HD-Orus",  # Male
                "es-US-Chirp3-HD-Puck",  # Male
                "es-US-Chirp3-HD-Pulcherrima",  # Female
                "es-US-Chirp3-HD-Rasalgethi",  # Male
                "es-US-Chirp3-HD-Sadachbia",  # Male
                "es-US-Chirp3-HD-Sadaltager",  # Male
                "es-US-Chirp3-HD-Schedar",  # Male
                "es-US-Chirp3-HD-Sulafat",  # Female
                "es-US-Chirp3-HD-Umbriel",  # Male
                "es-US-Chirp3-HD-Vindemiatrix",  # Female
                "es-US-Chirp3-HD-Zephyr",  # Female
                "es-US-Chirp3-HD-Zubenelgenubi",  # Male
            ],
            "neural2": [
                "es-US-Neural2-A",  # Female
                "es-US-Neural2-B",  # Male
                "es-US-Neural2-C",  # Male
            ],
            "news": [
                "es-US-News-D",  # Male
                "es-US-News-E",  # Male
                "es-US-News-F",  # Female
                "es-US-News-G",  # Female
            ],
            "polyglot": [
                "es-US-Polyglot-1",  # Male
            ],
            "studio": [
                "es-US-Studio-B",  # Male
            ],
            "wavenet": [
                "es-US-Wavenet-A",  # Female
                "es-US-Wavenet-B",  # Male
                "es-US-Wavenet-C",  # Male
            ],
            "standard": [
                "es-US-Standard-A",  # Female
                "es-US-Standard-B",  # Male
                "es-US-Standard-C",  # Male
            ],
        }
    else:
        raise ValueError(f"Unsupported language code: {language_code}")


class AudioGenerator:
    """Generate audio files for minimal pairs."""

    def __init__(
        self,
        config: AudioToolsConfig,
        overwrite: bool = False,
        min_file_size: int = 5000,
        min_duration: float = 0.3,
        max_retries: int = 3,
        limit_voices: int | None = None,
    ):
        self.config = config
        self.base_output_path = self.config.base_audio_dir / self.config.language_code
        self.overwrite = overwrite
        self.min_file_size = min_file_size
        self.min_duration = min_duration
        self.max_retries = max_retries
        self.limit_voices = limit_voices
        ensure_directory_exists(self.base_output_path)

    def process_word_recording(
        self, word_text: str, transliteration: str, voice_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a single word recording and save to tree structure."""
        # Add appropriate punctuation for natural speech based on language
        if self.config.language_code == "bn-IN":
            text_with_stop = f"{word_text}।"  # Bengali full stop
        elif self.config.language_code == "es-US":
            text_with_stop = f"{word_text}."  # Spanish period
        else:
            text_with_stop = f"{word_text}."  # Default period

        # Create word-specific directory
        word_dir = self.base_output_path / transliteration
        ensure_directory_exists(word_dir)

        # Get minimal voice name for filename
        minimal_voice_name = get_minimal_voice_name(voice_config["voice_name"])

        # Create filename in tree structure: word/word_voicename.mp3 (changed from .wav)
        output_filename = f"{transliteration}_{minimal_voice_name}.mp3"
        output_file = word_dir / output_filename

        # Check if file already exists and skip if not overwriting
        if output_file.exists() and not self.overwrite:
            validation = validate_audio_file(output_file, self.min_file_size, self.min_duration)
            if validation["valid"]:
                return {
                    "status": "skipped",
                    "file_size": validation["file_size"],
                    "duration": validation["duration"],
                    "path": output_file,
                }
            else:
                return {
                    "status": "regenerate",
                    "file_size": validation["file_size"],
                    "reason": validation["reason"],
                    "path": output_file,
                }

        # Retry logic for failed recordings
        for attempt in range(self.max_retries):
            try:
                # Generate audio with timeout
                sample_rate, samples = synthesize_raw_audio(
                    text_with_stop,
                    volume_gain_db=voice_config["volume_gain_db"],
                    effects_profile_id=voice_config["effects_profile"],
                    voice_name=voice_config["voice_name"],
                    language_code=self.config.language_code,
                    timeout=30,  # 30 second timeout to prevent hanging
                )

                # Trim silence from beginning and end only (preserve all speech content)
                trimmed_samples = librosa.effects.trim(samples, top_db=30)[0]

                # Convert to 16-bit integers for pydub
                if trimmed_samples.dtype != np.int16:
                    # Normalize and convert to int16
                    if trimmed_samples.dtype == np.float32 or trimmed_samples.dtype == np.float64:
                        # Samples are in [-1, 1] range
                        trimmed_samples = (trimmed_samples * 32767).astype(np.int16)
                    else:
                        # Samples might already be int16 or need different conversion
                        trimmed_samples = trimmed_samples.astype(np.int16)

                # Create AudioSegment and export as MP3
                audio_segment = AudioSegment(
                    trimmed_samples.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1,  # mono
                )

                # Export as MP3 with high quality
                audio_segment.export(str(output_file), format="mp3", bitrate="192k")

                # Validate the generated audio file
                validation = validate_audio_file(output_file, self.min_file_size, self.min_duration)

                if validation["valid"]:
                    return {
                        "status": "success",
                        "file_size": validation["file_size"],
                        "duration": validation["duration"],
                        "path": output_file,
                        "voice": voice_config["voice_name"],
                    }
                else:
                    # File failed validation, delete it and retry
                    if output_file.exists():
                        output_file.unlink()

                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return {
                            "status": "failed",
                            "reason": f"Audio validation failed: {validation['reason']}",
                            "file_size": validation.get("file_size", 0),
                            "duration": validation.get("duration", 0),
                        }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    continue
                else:
                    return {"status": "failed", "reason": str(e)}

        return {"status": "failed", "reason": f"Failed after {self.max_retries} attempts"}

    def discover_missing_files(self, unique_words, voices):
        """Discover which word/voice combinations are missing audio files."""
        missing = []
        existing = []

        for bengali_text, transliteration in unique_words:
            word_dir = self.base_output_path / transliteration

            for voice_name in voices:
                minimal_voice_name = get_minimal_voice_name(voice_name)
                output_filename = f"{transliteration}_{minimal_voice_name}.mp3"
                output_file = word_dir / output_filename

                if not output_file.exists() or self.overwrite:
                    missing.append((bengali_text, transliteration, voice_name))
                else:
                    # Validate existing file
                    validation = validate_audio_file(
                        output_file, self.min_file_size, self.min_duration
                    )
                    if not validation["valid"]:
                        missing.append((bengali_text, transliteration, voice_name))
                    else:
                        existing.append((bengali_text, transliteration, voice_name))

        return missing, existing

    def generate_all_audio(
        self,
        pairs_data_path: Path | None = None,
        volume_gain_db: float = 0.0,
        effects_profile: str = "headphone-class-device",
        voice_type: str = "all",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Generate audio for all words in the minimal pairs database."""
        # Load data
        pairs_data = load_minimal_pairs_data(
            pairs_data_path if pairs_data_path else self.config.pairs_file_path
        )
        unique_words = get_unique_words(pairs_data, self.config.language_code)

        # Get voices based on type
        all_voices = get_all_voice_names(self.config.language_code)
        if voice_type == "chirp":
            voices = all_voices.get("chirp3_hd", []) + all_voices.get("chirp_hd", [])
        elif voice_type == "wavenet":
            voices = all_voices.get("wavenet", [])
        else:
            # Combine all voice types for the language
            voices = []
            for voice_list in all_voices.values():
                voices.extend(voice_list)

        # Limit voices if requested
        if self.limit_voices:
            voices = voices[: self.limit_voices]

        # Discover missing files
        console.print("[dim]Discovering missing audio files...[/dim]")
        missing_files, existing_files = self.discover_missing_files(unique_words, voices)

        # Calculate unique words that need generation
        words_needing_generation = set()
        for _, transliteration, _ in missing_files:
            words_needing_generation.add(transliteration)

        # Statistics
        total_words = len(unique_words)
        words_to_generate = len(words_needing_generation)
        total_to_generate = len(missing_files)
        total_existing = len(existing_files)

        console.print("[bold blue]Audio File Status:[/bold blue]")
        console.print(f"  • Total words in database: {total_words}")
        console.print(f"  • Words needing audio: [yellow]{words_to_generate}[/yellow]")
        console.print(f"  • Voices per word: {len(voices)}")
        console.print(f"  • Existing audio files: [green]{total_existing}[/green]")
        console.print(
            f"  • Files to generate: [yellow]{total_to_generate}[/yellow] ({words_to_generate} words × up to {len(voices)} voices)"
        )

        if dry_run:
            console.print("\n[bold yellow]DRY RUN MODE - No files will be generated[/bold yellow]")
            if missing_files:
                console.print("\n[dim]Files that would be generated:[/dim]")
                for i, (bengali, transliteration, voice) in enumerate(missing_files[:10]):
                    minimal_voice = get_minimal_voice_name(voice)
                    console.print(f"  {i + 1}. {bengali} ({transliteration}) - {minimal_voice}")
                if len(missing_files) > 10:
                    console.print(f"  ... and {len(missing_files) - 10} more")
            return {"dry_run": True, "missing": total_to_generate, "existing": total_existing}

        if total_to_generate == 0:
            console.print("\n[bold green]✓ All audio files already exist![/bold green]")
            return {
                "successful": 0,
                "failed": 0,
                "skipped": total_existing,
                "regenerated": 0,
                "words": {},
            }

        results = {
            "successful": 0,
            "failed": 0,
            "skipped": total_existing,  # Count existing files as skipped
            "regenerated": 0,
            "words": {},
        }

        # Process only missing files
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,  # Reduce refresh rate to minimize visual duplication
            transient=True,  # Clear progress bar when done
        ) as progress:
            main_task = progress.add_task(
                "Generating missing audio files...", total=total_to_generate
            )

            for idx, (bengali_text, transliteration, voice_name) in enumerate(missing_files):
                minimal_voice = get_minimal_voice_name(voice_name)

                # Update progress description before processing
                progress.update(
                    main_task,
                    description=f"[{idx + 1}/{total_to_generate}] {transliteration} - {minimal_voice}",
                )

                voice_config = {
                    "voice_name": voice_name,
                    "volume_gain_db": volume_gain_db,
                    "effects_profile": effects_profile,
                }

                try:
                    result = self.process_word_recording(
                        bengali_text, transliteration, voice_config
                    )
                except TimeoutError as e:
                    console.print(
                        f"[red]⏰ Timeout processing {transliteration} with {minimal_voice}: {e}[/red]"
                    )
                    result = {"status": "failed", "reason": f"Timeout: {str(e)}"}
                except Exception as e:
                    console.print(
                        f"[red]❌ Error processing {transliteration} with {minimal_voice}: {e}[/red]"
                    )
                    result = {"status": "failed", "reason": f"Unexpected error: {str(e)}"}

                # Update statistics based on actual result (not assumed skipped)
                if result["status"] == "success":
                    results["successful"] += 1
                elif result["status"] == "failed":
                    results["failed"] += 1
                elif result["status"] == "regenerate":
                    results["regenerated"] += 1

                # Store result for this word/voice combo
                if transliteration not in results["words"]:
                    results["words"][transliteration] = {"bengali": bengali_text, "results": []}
                results["words"][transliteration]["results"].append(result)

                # Advance the progress bar after processing
                progress.update(main_task, advance=1)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print generation summary."""
        console.print("\n[bold green]Audio Generation Complete![/bold green]")

        summary_table = Table(title="Generation Summary", box=box.ROUNDED)
        summary_table.add_column("Status", style="cyan", no_wrap=True)
        summary_table.add_column("Count", justify="right", style="magenta")

        summary_table.add_row("Successful", str(results["successful"]))
        summary_table.add_row("Failed", str(results["failed"]))
        summary_table.add_row("Skipped", str(results["skipped"]))
        summary_table.add_row("Regenerated", str(results["regenerated"]))
        summary_table.add_row(
            "Total Attempts",
            str(
                results["successful"]
                + results["failed"]
                + results["skipped"]
                + results["regenerated"]
            ),
        )

        console.print(summary_table)

        # Show problem words if any
        problem_words = []
        for word, word_data in results["words"].items():
            failures = [r for r in word_data["results"] if r["status"] == "failed"]
            if failures:
                problem_words.append((word, word_data["bengali"], len(failures)))

        if problem_words:
            console.print("\n[bold red]Words with failures:[/bold red]")
            for word, bengali, failure_count in problem_words[:10]:
                console.print(f"  • {bengali} ({word}): {failure_count} failures")
            if len(problem_words) > 10:
                console.print(f"  ... and {len(problem_words) - 10} more")
