"""Async audio generation functionality for minimal pairs with parallel processing."""

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from google.cloud import texttospeech_v1
from pydub import AudioSegment
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from scipy.io import wavfile

from .config import AudioToolsConfig
from .generator import (
    get_all_voice_names,
    get_minimal_voice_name,
    validate_audio_file,
)
from .utils import ensure_directory_exists, get_unique_words, load_minimal_pairs_data

console = Console()


async def synthesize_raw_audio_async(
    text: str,
    volume_gain_db: float = 0.0,
    effects_profile_id: str = "headphone-class-device",
    voice_name: str = "bn-IN-Chirp3-HD-Aoede",
    language_code: str = "bn-IN",
    timeout: int = 30,
) -> tuple[int, Any]:
    """Async version of synthesize_raw_audio using TextToSpeechAsyncClient."""
    try:
        # Create async client
        client = texttospeech_v1.TextToSpeechAsyncClient()

        input_text = texttospeech_v1.SynthesisInput(text=text)

        voice = texttospeech_v1.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )

        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,  # Keep WAV for processing
            sample_rate_hertz=44100,  # Standard sample rate for better browser compatibility
            volume_gain_db=volume_gain_db,
            effects_profile_id=[effects_profile_id] if effects_profile_id else [],
        )

        # Make async TTS request with timeout
        response = await asyncio.wait_for(
            client.synthesize_speech(
                request={
                    "input": input_text,
                    "voice": voice,
                    "audio_config": audio_config,
                }
            ),
            timeout=timeout,
        )

        # Convert response to audio samples for processing
        sample_rate, samples = wavfile.read(io.BytesIO(response.audio_content))
        return sample_rate, samples

    except asyncio.TimeoutError:
        raise TimeoutError(f"TTS synthesis timed out after {timeout} seconds") from None
    except Exception as e:
        raise Exception(f"TTS synthesis failed: {str(e)}") from e


class AsyncAudioGenerator:
    """Async audio generation with parallel processing for minimal pairs."""

    def __init__(
        self,
        config: AudioToolsConfig,
        overwrite: bool = False,
        limit_voices: int | None = None,
        min_file_size: int = 5000,
        min_duration: float = 0.3,
        max_concurrent: int = 10,  # Limit concurrent TTS requests
        max_concurrent_io: int = 20,  # Limit concurrent I/O operations
    ):
        self.config = config
        self.base_output_path = self.config.base_audio_dir / self.config.language_code
        self.overwrite = overwrite
        self.limit_voices = limit_voices
        self.min_file_size = min_file_size
        self.min_duration = min_duration
        self.max_concurrent = max_concurrent
        self.max_concurrent_io = max_concurrent_io

        # Create semaphores to limit concurrency
        self.tts_semaphore = asyncio.Semaphore(max_concurrent)
        self.io_semaphore = asyncio.Semaphore(max_concurrent_io)

        # Thread pool for CPU-intensive audio processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)

    async def process_audio_in_thread(
        self, samples: np.ndarray, sample_rate: int, output_file: Path
    ) -> dict[str, Any]:
        """Process audio (trimming and MP3 conversion) in a separate thread."""
        loop = asyncio.get_event_loop()

        def process_audio():
            try:
                # Trim silence from beginning and end only (preserve all speech content)
                trimmed_samples = librosa.effects.trim(samples, top_db=30)[0]

                # Convert to AudioSegment and export as MP3
                audio_segment = AudioSegment(
                    trimmed_samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
                )

                # Export as MP3 with high quality
                audio_segment.export(str(output_file), format="mp3", bitrate="192k")

                # Validate the generated file
                validation = validate_audio_file(output_file, self.min_file_size, self.min_duration)
                return {
                    "status": "success" if validation["valid"] else "failed",
                    "reason": validation["reason"],
                    "file_size": validation["file_size"],
                    "duration": validation["duration"],
                    "path": output_file,
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "reason": f"Audio processing error: {str(e)}",
                    "file_size": 0,
                    "duration": 0,
                    "path": output_file,
                }

        return await loop.run_in_executor(self.thread_pool, process_audio)

    async def process_word_recording_async(
        self, word_text: str, transliteration: str, voice_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Async version of process_word_recording with semaphore-controlled concurrency."""
        # Add appropriate punctuation for natural speech based on language
        if self.config.language_code == "bn-IN":
            text_with_stop = f"{word_text}।"  # Bengali full stop
        elif self.config.language_code == "es-US":
            text_with_stop = f"{word_text}."  # Spanish period
        else:
            text_with_stop = f"{word_text}."  # Default period

        # Create word-specific directory
        word_dir = self.base_output_path / transliteration

        async with self.io_semaphore:
            ensure_directory_exists(word_dir)

        # Get minimal voice name for filename
        minimal_voice_name = get_minimal_voice_name(voice_config["voice_name"])

        # Create filename in tree structure: word/word_voicename.mp3
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
                # File exists but is invalid, regenerate
                pass

        # Use semaphore to limit concurrent TTS requests
        async with self.tts_semaphore:
            try:
                console.print(
                    f"[dim]Synthesizing: {word_text} ({voice_config['voice_name']})[/dim]"
                )
                # Synthesize audio
                sample_rate, samples = await synthesize_raw_audio_async(
                    text_with_stop,
                    volume_gain_db=voice_config["volume_gain_db"],
                    effects_profile_id=voice_config["effects_profile"],
                    voice_name=voice_config["voice_name"],
                    language_code=self.config.language_code,
                    timeout=30,
                )

                console.print(f"[dim]Processing audio for: {output_file}[/dim]")
                # Process audio in thread pool (CPU-intensive work)
                result = await self.process_audio_in_thread(samples, sample_rate, output_file)
                console.print(f"[dim]Result for {output_file}: {result['status']}[/dim]")
                return result

            except TimeoutError as e:
                return {
                    "status": "failed",
                    "reason": f"Timeout: {str(e)}",
                    "file_size": 0,
                    "duration": 0,
                    "path": output_file,
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "reason": f"TTS error: {str(e)}",
                    "file_size": 0,
                    "duration": 0,
                    "path": output_file,
                }

    async def generate_all_audio_async(
        self,
        pairs_data_path: Path | None = None,
        volume_gain_db: float = 0.0,
        effects_profile: str = "headphone-class-device",
        voice_type: str = "all",
        batch_size: int = 50,  # Process in batches to manage memory
    ) -> dict[str, Any]:
        """Async version of generate_all_audio with parallel processing."""
        start_time = time.time()

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

        # Statistics
        total_words = len(unique_words)
        total_attempts = total_words * len(voices)

        console.print(
            f"[bold blue]🚀 Async Processing {total_words} unique words with {len(voices)} voices[/bold blue]"
        )
        console.print(f"[dim]Total recordings to generate: {total_attempts}[/dim]")
        console.print(f"[dim]Max concurrent TTS requests: {self.max_concurrent}[/dim]")
        console.print(f"[dim]Max concurrent I/O operations: {self.max_concurrent_io}[/dim]")

        results = {
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "regenerated": 0,
            "words": {},
            "total_time": 0,
            "avg_time_per_recording": 0,
        }

        # Create all tasks
        all_tasks = []
        task_metadata = []  # To track which task belongs to which word/voice

        for word_idx, (bengali_text, transliteration) in enumerate(unique_words):
            for voice_idx, voice_name in enumerate(voices):
                voice_config = {
                    "voice_name": voice_name,
                    "volume_gain_db": volume_gain_db,
                    "effects_profile": effects_profile,
                }

                task = self.process_word_recording_async(
                    bengali_text, transliteration, voice_config
                )
                all_tasks.append(task)
                task_metadata.append(
                    {
                        "word_idx": word_idx,
                        "voice_idx": voice_idx,
                        "transliteration": transliteration,
                        "bengali_text": bengali_text,
                        "voice_name": voice_name,
                    }
                )

        # Process tasks in batches with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("🔄 Processing audio recordings...", total=total_attempts)

            completed = 0
            for i in range(0, len(all_tasks), batch_size):
                batch_tasks = all_tasks[i : i + batch_size]
                batch_metadata = task_metadata[i : i + batch_size]

                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Process results
                for result, metadata in zip(batch_results, batch_metadata, strict=False):
                    completed += 1
                    transliteration = metadata["transliteration"]

                    # Initialize word results if not exists
                    if transliteration not in results["words"]:
                        results["words"][transliteration] = {
                            "bengali": metadata["bengali_text"],
                            "results": [],
                        }

                    # Handle exceptions
                    if isinstance(result, Exception):
                        result = {
                            "status": "failed",
                            "reason": f"Unexpected error: {str(result)}",
                            "file_size": 0,
                            "duration": 0,
                        }

                    results["words"][transliteration]["results"].append(result)

                    # Update statistics
                    if result["status"] == "success":
                        results["successful"] += 1
                    elif result["status"] == "failed":
                        results["failed"] += 1
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                    elif result["status"] == "regenerate":
                        results["regenerated"] += 1

                    # Update progress
                    progress.update(
                        main_task,
                        completed=completed,
                        description=f"🔄 Processed {completed}/{total_attempts} recordings",
                    )

        # Calculate timing statistics
        end_time = time.time()
        results["total_time"] = end_time - start_time
        if total_attempts > 0:
            results["avg_time_per_recording"] = results["total_time"] / total_attempts

        # Print summary
        self._print_async_summary(results)

        return results

    def _print_async_summary(self, results: dict[str, Any]) -> None:
        """Print async generation summary with performance metrics."""
        console.print("\n[bold green]🎉 Async Audio Generation Complete![/bold green]")

        # Generation summary table
        summary_table = Table(title="Generation Summary", box=box.ROUNDED)
        summary_table.add_column("Status", style="cyan", no_wrap=True)
        summary_table.add_column("Count", justify="right", style="magenta")

        summary_table.add_row("Successful", str(results["successful"]))
        summary_table.add_row("Failed", str(results["failed"]))
        summary_table.add_row("Skipped", str(results["skipped"]))
        summary_table.add_row("Regenerated", str(results["regenerated"]))

        total_attempts = (
            results["successful"] + results["failed"] + results["skipped"] + results["regenerated"]
        )
        summary_table.add_row("Total Attempts", str(total_attempts))

        console.print(summary_table)

        # Performance metrics table
        perf_table = Table(title="Performance Metrics", box=box.ROUNDED)
        perf_table.add_column("Metric", style="yellow", no_wrap=True)
        perf_table.add_column("Value", justify="right", style="green")

        perf_table.add_row("Total Time", f"{results['total_time']:.1f}s")
        perf_table.add_row("Avg Time/Recording", f"{results['avg_time_per_recording']:.2f}s")

        if total_attempts > 0:
            recordings_per_minute = (total_attempts / results["total_time"]) * 60
            perf_table.add_row("Recordings/Minute", f"{recordings_per_minute:.1f}")

        console.print(perf_table)

        # Print words with failures
        if results["failed"] > 0:
            failed_words = set()
            for word, word_data in results["words"].items():
                for result in word_data["results"]:
                    if result["status"] == "failed":
                        failed_words.add(word)
                        break

            if failed_words:
                console.print("\n[red]Words with failures:[/red]")
                for word in sorted(failed_words):
                    # Count failures for this word
                    failure_count = sum(
                        1
                        for result in results["words"][word]["results"]
                        if result["status"] == "failed"
                    )
                    bengali = results["words"][word]["bengali"]
                    console.print(f"  • {bengali} ({word}): {failure_count} failures")
