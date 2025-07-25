"""Command-line interface for minimal pairs audio tools."""

import click
from pathlib import Path
from rich.console import Console

from .generator import AudioGenerator, get_all_voice_names, get_minimal_voice_name
from .verifier import PronunciationVerifier, save_verification_results
from .manifest import ManifestGenerator
from .models import GcpStandardModel, Chirp2Model
from .utils import load_minimal_pairs_data, get_unique_words


console = Console()


@click.group()
def main():
    """Minimal Pairs Audio Tools - Generate and verify audio for language learning."""
    pass


@main.command()
@click.option('--output-dir', '-o', default='public/audio/bn-IN', 
              help='Output directory for audio files')
@click.option('--overwrite', is_flag=True, 
              help='Overwrite existing audio files')
@click.option('--limit-voices', '-l', type=int, 
              help='Limit number of voices to use (for testing)')
@click.option('--voice-type', type=click.Choice(['all', 'chirp', 'wavenet']), 
              default='all', help='Type of voices to use')
@click.option('--volume-gain', type=float, default=0.0, 
              help='Volume gain in dB')
@click.option('--pairs-file', '-p', type=click.Path(exists=True), 
              help='Path to minimal pairs JSON file')
def generate(output_dir, overwrite, limit_voices, voice_type, volume_gain, pairs_file):
    """Generate audio files for all minimal pairs."""
    console.print("[bold blue]🎵 Generating audio files...[/bold blue]")
    
    generator = AudioGenerator(
        base_output_path=output_dir,
        overwrite=overwrite,
        limit_voices=limit_voices
    )
    
    pairs_path = Path(pairs_file) if pairs_file else None
    
    results = generator.generate_all_audio(
        pairs_data_path=pairs_path,
        volume_gain_db=volume_gain,
        voice_type=voice_type
    )
    
    # Generate manifest after audio generation
    console.print("\n[bold blue]📋 Generating audio manifest...[/bold blue]")
    manifest_gen = ManifestGenerator(output_dir)
    manifest_gen.generate_manifest()


@main.command()
@click.option('--model', type=click.Choice(['standard', 'chirp2']), 
              default='chirp2', help='STT model to use')
@click.option('--words', '-w', multiple=True, 
              help='Specific words to verify (can be used multiple times)')
@click.option('--category', '-c', 
              help='Specific category to verify')
@click.option('--max-files', '-m', type=int, 
              help='Maximum number of files to verify')
@click.option('--output', '-o', default='verification_results.json', 
              help='Output file for results')
@click.option('--pairs-file', '-p', type=click.Path(exists=True), 
              help='Path to minimal pairs JSON file')
@click.option('--manifest-file', type=click.Path(exists=True), 
              help='Path to audio manifest JSON file')
def verify(model, words, category, max_files, output, pairs_file, manifest_file):
    """Verify pronunciation accuracy using speech-to-text."""
    console.print(f"[bold blue]🎙️ Verifying pronunciations with {model} model...[/bold blue]")
    
    # Create appropriate model
    if model == 'chirp2':
        stt_model = Chirp2Model()
    else:
        stt_model = GcpStandardModel()
    
    verifier = PronunciationVerifier(stt_model)
    
    # Convert paths
    pairs_path = Path(pairs_file) if pairs_file else None
    manifest_path = Path(manifest_file) if manifest_file else None
    
    # Run verification
    results = verifier.verify_all_audio(
        pairs_data_path=pairs_path,
        manifest_path=manifest_path,
        words=list(words) if words else None,
        category=category,
        max_files=max_files
    )
    
    # Save results
    save_verification_results(results, Path(output))


@main.command()
@click.option('--audio-dir', '-d', default='public/audio/bn-IN', 
              help='Audio directory to scan')
@click.option('--output', '-o', 
              help='Output file for manifest (default: audio_dir/../audio_manifest.json)')
@click.option('--verify-files', is_flag=True, 
              help='Verify that all files in manifest exist')
@click.option('--fix-missing', is_flag=True, 
              help='Remove missing files from manifest')
def manifest(audio_dir, output, verify_files, fix_missing):
    """Generate or verify audio manifest."""
    manifest_gen = ManifestGenerator(audio_dir)
    
    if verify_files:
        console.print("[bold blue]🔍 Verifying manifest...[/bold blue]")
        manifest_path = Path(output) if output else None
        manifest_gen.verify_manifest(manifest_path, fix_missing=fix_missing)
    else:
        console.print("[bold blue]📋 Generating manifest...[/bold blue]")
        output_path = Path(output) if output else None
        manifest_gen.generate_manifest(output_path)


@main.command()
@click.option('--output-dir', '-o', default='public/audio/bn-IN', 
              help='Output directory for audio files')
@click.option('--overwrite', is_flag=True, 
              help='Overwrite existing audio files')
@click.option('--limit-voices', '-l', type=int, 
              help='Limit number of voices to use (for testing)')
@click.option('--voice-type', type=click.Choice(['all', 'chirp', 'wavenet']), 
              default='chirp', help='Type of voices to use')
@click.option('--volume-gain', type=float, default=0.0, 
              help='Volume gain in dB')
@click.option('--pairs-file', '-p', type=click.Path(exists=True), 
              help='Path to minimal pairs JSON file')
@click.option('--model', type=click.Choice(['standard', 'chirp2']), 
              default='chirp2', help='STT model to use for verification')
@click.option('--max-verify-files', type=int, default=50,
              help='Maximum number of files to verify')
@click.option('--skip-verification', is_flag=True,
              help='Skip the verification step')
def full_pipeline(output_dir, overwrite, limit_voices, voice_type, volume_gain, 
                 pairs_file, model, max_verify_files, skip_verification):
    """Complete pipeline: generate audio, create manifest, and verify quality."""
    console.print("[bold green]🚀 Running Full Audio Pipeline[/bold green]")
    console.print("[dim]This will: 1) Generate audio, 2) Create manifest, 3) Verify quality[/dim]\n")
    
    # Step 1: Generate Audio
    console.print("[bold blue]Step 1/3: 🎵 Generating audio files...[/bold blue]")
    generator = AudioGenerator(
        base_output_path=output_dir,
        overwrite=overwrite,
        limit_voices=limit_voices
    )
    
    pairs_path = Path(pairs_file) if pairs_file else None
    
    generation_results = generator.generate_all_audio(
        pairs_data_path=pairs_path,
        volume_gain_db=volume_gain,
        voice_type=voice_type
    )
    
    # Step 2: Generate Manifest
    console.print("\n[bold blue]Step 2/3: 📋 Generating audio manifest...[/bold blue]")
    manifest_gen = ManifestGenerator(output_dir)
    manifest_gen.generate_manifest()
    
    # Step 3: Verify Audio Quality (optional)
    if not skip_verification:
        console.print(f"\n[bold blue]Step 3/3: 🎙️ Verifying audio quality with {model} model...[/bold blue]")
        
        # Create appropriate model
        if model == 'chirp2':
            stt_model = Chirp2Model()
        else:
            stt_model = GcpStandardModel()
        
        verifier = PronunciationVerifier(stt_model)
        
        # Convert paths
        manifest_path = Path(output_dir).parent / "audio_manifest.json"
        
        # Run verification
        verification_results = verifier.verify_all_audio(
            pairs_data_path=pairs_path,
            manifest_path=manifest_path if manifest_path.exists() else None,
            max_files=max_verify_files
        )
        
        # Save verification results
        output_file = "verification_results.json"
        save_verification_results(verification_results, Path(output_file))
        
        console.print(f"\n[bold green]✅ Pipeline Complete![/bold green]")
        console.print(f"📊 Generated {generation_results['successful']} audio files")
        console.print(f"🎙️ Verified {verification_results['summary']['total_files']} files")
        console.print(f"📋 Manifest and verification results saved")
    else:
        console.print(f"\n[bold green]✅ Pipeline Complete![/bold green]")
        console.print(f"📊 Generated {generation_results['successful']} audio files")
        console.print(f"📋 Manifest created (verification skipped)")


@main.command()
@click.option('--word', '-w', required=True, 
              help='Word to regenerate (transliteration)')
@click.option('--voice', '-v', 
              help='Specific voice to regenerate (if not provided, regenerates all voices for the word)')
@click.option('--output-dir', '-o', default='public/audio/bn-IN', 
              help='Output directory for audio files')
@click.option('--volume-gain', type=float, default=0.0, 
              help='Volume gain in dB')
@click.option('--pairs-file', '-p', type=click.Path(exists=True), 
              help='Path to minimal pairs JSON file')
def regenerate(word, voice, output_dir, volume_gain, pairs_file):
    """Regenerate audio for a specific word and voice combination."""
    console.print(f"[bold blue]🔄 Regenerating audio for '{word}'...[/bold blue]")
    
    # Load pairs data to get the Bengali text for this word
    pairs_data = load_minimal_pairs_data(Path(pairs_file) if pairs_file else None)
    unique_words = get_unique_words(pairs_data)
    
    # Find the Bengali text for this transliteration
    bengali_text = None
    for bengali, transliteration in unique_words:
        if transliteration == word:
            bengali_text = bengali
            break
    
    if not bengali_text:
        console.print(f"[red]Error: Word '{word}' not found in minimal pairs data[/red]")
        return
    
    # Get voices to regenerate
    all_voices = get_all_voice_names()
    if voice:
        # Validate the voice exists
        all_voice_list = all_voices["chirp3_hd"] + all_voices["wavenet"]
        if voice not in [get_minimal_voice_name(v) for v in all_voice_list]:
            console.print(f"[red]Error: Voice '{voice}' not found[/red]")
            return
        
        # Convert back to full voice name
        voices_to_regenerate = []
        for full_voice in all_voice_list:
            if get_minimal_voice_name(full_voice) == voice:
                voices_to_regenerate = [full_voice]
                break
    else:
        # Regenerate all voices for this word
        voices_to_regenerate = all_voices["chirp3_hd"] + all_voices["wavenet"]
    
    console.print(f"[dim]Bengali text: {bengali_text}[/dim]")
    console.print(f"[dim]Regenerating {len(voices_to_regenerate)} voice(s)[/dim]")
    
    # Create generator and regenerate
    generator = AudioGenerator(
        base_output_path=output_dir,
        overwrite=True  # Always overwrite when regenerating
    )
    
    results = {
        "successful": 0,
        "failed": 0,
        "word": word,
        "bengali_text": bengali_text,
        "voices": []
    }
    
    for voice_name in voices_to_regenerate:
        voice_config = {
            "voice_name": voice_name,
            "volume_gain_db": volume_gain,
            "effects_profile": "headphone-class-device"
        }
        
        result = generator.process_word_recording(
            bengali_text, 
            word, 
            voice_config
        )
        
        minimal_voice = get_minimal_voice_name(voice_name)
        results["voices"].append({
            "voice": minimal_voice,
            "status": result["status"],
            "reason": result.get("reason", "")
        })
        
        if result["status"] == "success":
            results["successful"] += 1
            console.print(f"[green]✓[/green] {minimal_voice}")
        else:
            results["failed"] += 1
            console.print(f"[red]✗[/red] {minimal_voice}: {result.get('reason', 'Unknown error')}")
    
    # Print summary
    console.print(f"\n[bold]Results:[/bold] {results['successful']} successful, {results['failed']} failed")
    
    # Update manifest after regeneration
    if results["successful"] > 0:
        console.print("[dim]Updating audio manifest...[/dim]")
        manifest_gen = ManifestGenerator(output_dir)
        manifest_gen.generate_manifest(include_stats=False)
    
    # Return results as JSON for API calls
    import json
    console.print(f"\n[dim]{json.dumps(results, indent=2)}[/dim]")


@main.command()
@click.option('--min-size', type=int, default=5000, 
              help='Minimum file size in bytes')
@click.option('--audio-dir', '-d', default='public/audio/bn-IN', 
              help='Audio directory to clean')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be deleted without actually deleting')
def clean(min_size, audio_dir, dry_run):
    """Clean up small/invalid audio files."""
    from .generator import validate_audio_file
    
    console.print(f"[bold blue]🧹 Cleaning audio files smaller than {min_size} bytes...[/bold blue]")
    
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        console.print(f"[red]Audio directory not found: {audio_path}[/red]")
        return
    
    deleted_count = 0
    checked_count = 0
    
    # Scan all audio files
    for word_dir in audio_path.iterdir():
        if not word_dir.is_dir():
            continue
        
        for audio_file in word_dir.iterdir():
            if audio_file.suffix in ['.wav', '.mp3']:
                checked_count += 1
                validation = validate_audio_file(audio_file)
                
                if not validation["valid"] or validation["file_size"] < min_size:
                    if dry_run:
                        console.print(
                            f"[yellow]Would delete:[/yellow] {audio_file} "
                            f"({validation['file_size']} bytes) - {validation['reason']}"
                        )
                    else:
                        audio_file.unlink()
                        console.print(
                            f"[red]Deleted:[/red] {audio_file} "
                            f"({validation['file_size']} bytes) - {validation['reason']}"
                        )
                    deleted_count += 1
    
    console.print(f"\n[bold]Checked {checked_count} files[/bold]")
    if dry_run:
        console.print(f"[yellow]Would delete {deleted_count} files[/yellow]")
    else:
        console.print(f"[green]Deleted {deleted_count} files[/green]")


if __name__ == '__main__':
    main()