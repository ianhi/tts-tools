"""Audio verification functionality for minimal pairs."""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

from .models import BengaliTextNormalizer, SpanishTextNormalizer, TextNormalizer, VerificationModel, GcpStandardModel, Chirp2Model
from .utils import load_minimal_pairs_data, load_audio_manifest
from .config import AudioToolsConfig


console = Console()


@dataclass
class VerificationResult:
    """Results from STT verification of an audio file."""
    file_path: str
    expected_text: str
    stt_transcription: str
    stt_confidence: float
    exact_match: bool
    normalized_match: bool
    edit_distance: int
    category: str
    voice_name: str
    model_name: str


class PronunciationVerifier:
    """Verify pronunciation accuracy using speech-to-text models."""

    def __init__(self, model: VerificationModel, config: AudioToolsConfig):
        self.model = model
        self.config = config
        self.normalizer = self._get_normalizer(config.language_code)
    
    def _get_normalizer(self, language_code: str) -> TextNormalizer:
        """Get the appropriate text normalizer for the language."""
        if language_code.startswith("bn"):
            return BengaliTextNormalizer()
        elif language_code.startswith("es"):
            return SpanishTextNormalizer()
        else:
            # Default to Bengali for backward compatibility
            return BengaliTextNormalizer()

    def verify_audio_file(
        self, 
        audio_file_path: str, 
        expected_text: str,
        category: str, 
        voice_name: str
    ) -> VerificationResult:
        """Verify a single audio file against expected text."""
        stt_transcription, stt_confidence = self.model.transcribe(audio_file_path)

        exact_match = expected_text == stt_transcription
        normalized_expected = self.normalizer.normalize(expected_text)
        normalized_stt = self.normalizer.normalize(stt_transcription)
        normalized_match = normalized_expected == normalized_stt
        edit_dist = self.normalizer.calculate_edit_distance(expected_text, stt_transcription)

        return VerificationResult(
            file_path=audio_file_path,
            expected_text=expected_text,
            stt_transcription=stt_transcription,
            stt_confidence=stt_confidence,
            exact_match=exact_match,
            normalized_match=normalized_match,
            edit_distance=edit_dist,
            category=category,
            voice_name=voice_name,
            model_name=self.model.__class__.__name__
        )

    def verify_all_audio(
        self,
        pairs_data_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        words: Optional[List[str]] = None,
        category: Optional[str] = None,
        max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """Verify all audio files in the manifest."""
        # Load data
        pairs_data = load_minimal_pairs_data(pairs_data_path if pairs_data_path else self.config.pairs_file_path)
        audio_manifest = load_audio_manifest(manifest_path, language_code=self.config.language_code)

        # Collect verification tasks
        verification_tasks = self._collect_verification_tasks(
            pairs_data, audio_manifest, words, category, max_files
        )

        console.print(f"Found {len(verification_tasks)} audio files to verify")

        # Verify each file
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Verifying pronunciations...", total=len(verification_tasks))
            
            for task_info in verification_tasks:
                progress.update(
                    task, 
                    description=f"Verifying {task_info['transliteration']} ({task_info['voice_name']})..."
                )
                
                result = self.verify_audio_file(
                    task_info["audio_file"],
                    task_info["expected_text"],
                    task_info["category"],
                    task_info["voice_name"]
                )
                results.append(result)
                progress.advance(task)
                time.sleep(0.1)  # Small delay to avoid rate limiting

        # Generate summary
        summary = self._generate_summary(results)
        
        # Print results
        self._print_results(results, summary)
        
        return {
            "summary": summary,
            "results": [asdict(r) for r in results]
        }

    def _collect_verification_tasks(
        self,
        pairs_data: Dict[str, Any],
        audio_manifest: Dict[str, Any],
        words: Optional[List[str]],
        category: Optional[str],
        max_files: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Collect all verification tasks based on filters."""
        tasks = []
        lang_data = pairs_data.get(self.config.language_code, {})
        types = lang_data.get("types", {})

        for category_name, category_data in types.items():
            if category and category_name != category:
                continue
            
            pairs = category_data.get("pairs", [])
            for pair in pairs:
                for word_data in pair:
                    bengali_word, transliteration = word_data
                    
                    if words and transliteration not in words:
                        continue
                    
                    if transliteration in audio_manifest.get("words", {}):
                        word_info = audio_manifest["words"][transliteration]
                        voices = word_info.get("voices", [])
                        extension = word_info.get("extension", "wav")
                        
                        for voice in voices:
                            audio_file = f"{self.config.base_audio_dir}/{self.config.language_code}/{transliteration}/{transliteration}_{voice}.{extension}"
                            if Path(audio_file).exists():
                                tasks.append({
                                    "audio_file": audio_file,
                                    "expected_text": bengali_word,
                                    "category": category_name,
                                    "voice_name": voice,
                                    "transliteration": transliteration
                                })
                                
                                if max_files and len(tasks) >= max_files:
                                    return tasks
        
        return tasks

    def _generate_summary(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate summary statistics from verification results."""
        total_files = len(results)
        exact_matches = sum(1 for r in results if r.exact_match)
        normalized_matches = sum(1 for r in results if r.normalized_match)
        high_confidence = sum(1 for r in results if r.stt_confidence > 0.8)
        
        return {
            "total_files": total_files,
            "exact_matches": exact_matches,
            "normalized_matches": normalized_matches,
            "high_confidence": high_confidence,
            "exact_match_rate": exact_matches / total_files if total_files > 0 else 0,
            "normalized_match_rate": normalized_matches / total_files if total_files > 0 else 0,
            "high_confidence_rate": high_confidence / total_files if total_files > 0 else 0
        }

    def _print_results(self, results: List[VerificationResult], summary: Dict[str, Any]) -> None:
        """Print verification results to console."""
        # Summary table
        summary_table = Table(title="Verification Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", justify="right", style="green")
        summary_table.add_column("Percentage", justify="right", style="yellow")
        
        total = summary["total_files"]
        summary_table.add_row("Total Files", str(total), "100.0%")
        
        if total > 0:
            summary_table.add_row(
                "Exact Matches", 
                str(summary["exact_matches"]), 
                f"{summary['exact_match_rate']*100:.1f}%"
            )
            summary_table.add_row(
                "Normalized Matches", 
                str(summary["normalized_matches"]), 
                f"{summary['normalized_match_rate']*100:.1f}%"
            )
            summary_table.add_row(
                "High Confidence (>0.8)", 
                str(summary["high_confidence"]), 
                f"{summary['high_confidence_rate']*100:.1f}%"
            )
        
        console.print(summary_table)
        
        # Problem files
        problematic = [r for r in results if not r.normalized_match or r.stt_confidence < 0.7]
        if problematic:
            console.print(f"\nFound {len(problematic)} files that need attention:")
            
            problem_table = Table(title="Files Needing Review", box=box.ROUNDED)
            problem_table.add_column("File", style="red", width=25, no_wrap=True)
            problem_table.add_column("Expected", style="cyan", width=8)
            problem_table.add_column("STT Result", style="yellow", width=8)
            problem_table.add_column("Conf", justify="right", style="green", width=5)
            problem_table.add_column("Category", style="blue", width=20, no_wrap=True)
            
            for result in problematic[:20]:
                file_name = Path(result.file_path).name
                if len(file_name) > 24:
                    file_name = file_name[:21] + "..."
                
                category = result.category
                if len(category) > 19:
                    category = category[:16] + "..."
                
                problem_table.add_row(
                    file_name,
                    result.expected_text,
                    result.stt_transcription or "[empty]",
                    f"{result.stt_confidence:.2f}",
                    category
                )
            
            console.print(problem_table)
            
            if len(problematic) > 20:
                console.print(f"... and {len(problematic) - 20} more files")


def save_verification_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save verification results to JSON file."""
    results_with_timestamp = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_timestamp, f, ensure_ascii=False, indent=2)
    
    console.print(f"\nDetailed results saved to: {output_path}")