"""Command-line interface for minimal pairs audio tools."""

import asyncio
import click
from pathlib import Path
from typing import Optional
from rich.console import Console

from .generator import AudioGenerator, get_all_voice_names, get_minimal_voice_name
from .async_generator import AsyncAudioGenerator
from .verifier import PronunciationVerifier, save_verification_results
from .manifest import ManifestGenerator
from .models import GcpStandardModel, Chirp2Model
from .utils import load_minimal_pairs_data, get_unique_words
from .config import AudioToolsConfig # Import the new config class


console = Console()


def create_language_config(base_config: AudioToolsConfig, language_code: str) -> AudioToolsConfig:
    """Create a language-specific config from the base config."""
    return AudioToolsConfig(
        language_code=language_code,
        base_audio_dir=base_config.base_audio_dir,
        pairs_file_path=base_config.pairs_file_path
    )


def create_audio_generator(config: AudioToolsConfig, overwrite: bool = False, 
                         limit_voices: Optional[int] = None) -> AudioGenerator:
    """Create an AudioGenerator instance with common settings."""
    return AudioGenerator(
        config=config,
        overwrite=overwrite,
        limit_voices=limit_voices
    )


def create_verification_model(language_code: str, model_type: str):
    """Create a verification model based on type."""
    if model_type == "chirp2":
        return Chirp2Model(language_code=language_code)
    else:
        return GcpStandardModel(language_code=language_code)


def get_output_directory(output_dir: Optional[str], base_dir: Path) -> Path:
    """Get the output directory, using base_dir if not specified."""
    return Path(output_dir) if output_dir else base_dir


pass_config = click.make_pass_decorator(AudioToolsConfig, ensure=True)

@click.group()
@click.option('--language', '-lang', type=click.Choice(['bn-IN', 'es-US']), 
              multiple=True, help='Language(s) to operate on')
@pass_config
def main(config: AudioToolsConfig, language):
    """Minimal Pairs Audio Tools - Generate and verify audio for language learning."""
    if not language: # If no language is provided via CLI
        # Dynamically discover language directories
        base_audio_path = config.base_audio_dir
        if base_audio_path.exists():
            discovered_languages = [d.name for d in base_audio_path.iterdir() if d.is_dir() and '-' in d.name]
            if discovered_languages:
                config.languages = discovered_languages
            else:
                config.languages = ["bn-IN"] # Fallback to default if none found
        else:
            config.languages = ["bn-IN"] # Fallback to default if base audio path doesn't exist
    else:
        config.languages = list(language) # Store all provided languages
    
    # Set language_code for backward compatibility with single-language operations
    config.language_code = config.languages[0] if config.languages else "bn-IN"
    pass


@main.command()
@click.option('--output-dir', '-o', 
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
@click.option('--dry-run', is_flag=True,
              help='Show what would be generated without actually generating files')
@pass_config
def generate(config: AudioToolsConfig, output_dir, overwrite, limit_voices, voice_type, volume_gain, pairs_file, dry_run):
    """Generate audio files for all minimal pairs."""
    for lang_code in config.languages:
        console.print(f"[bold blue]🎵 Generating audio files for {lang_code}...[/bold blue]")
        
        # Create language-specific config and generator
        lang_config = create_language_config(config, lang_code)
        generator = create_audio_generator(lang_config, overwrite, limit_voices)
        
        # Get paths
        output_dir_for_lang = get_output_directory(output_dir, config.base_audio_dir)
        pairs_path = Path(pairs_file) if pairs_file else config.pairs_file_path
        
        # Generate audio
        results = generator.generate_all_audio(
            pairs_data_path=pairs_path,
            volume_gain_db=volume_gain,
            voice_type=voice_type,
            dry_run=dry_run
        )



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
@pass_config
def verify(config: AudioToolsConfig, model, words, category, max_files, output, pairs_file, manifest_file):
    """Verify pronunciation accuracy using speech-to-text."""
    for lang_code in config.languages:
        console.print(f"[bold blue]🎙️ Verifying pronunciations for {lang_code} with {model} model...[/bold blue]")
        
        # Create language-specific config and model
        lang_config = create_language_config(config, lang_code)
        stt_model = create_verification_model(lang_code, model)
        verifier = PronunciationVerifier(stt_model, config=lang_config)
        
        # Get paths
        pairs_path = Path(pairs_file) if pairs_file else lang_config.pairs_file_path
        
        if manifest_file is None:
            manifest_path = lang_config.base_audio_dir / f"audio_manifest_{lang_code}.json"
        else:
            manifest_path = Path(manifest_file)

        # Run verification
        results = verifier.verify_all_audio(
            pairs_data_path=pairs_path,
            manifest_path=manifest_path if manifest_path.exists() else None,
            words=list(words) if words else None,
            category=category,
            max_files=max_files
        )
        
        # Save results
        save_verification_results(results, Path(output).parent / f"verification_results_{lang_code}.json")


@main.command()
@click.option('--audio-dir', '-d', 
              help='Base audio directory to scan')
@click.option('--output', '-o', 
              help='Output file for manifest (default: audio_dir/audio_manifest_{language}.json)')
@click.option('--verify-files', is_flag=True, 
              help='Verify that all files in manifest exist')
@click.option('--fix-missing', is_flag=True, 
              help='Remove missing files from manifest')
@pass_config
def manifest(config: AudioToolsConfig, audio_dir, output, verify_files, fix_missing):
    """Generate or verify audio manifest."""
    for lang_code in config.languages:
        console.print(f"[bold blue]📋 Processing manifest for {lang_code}...[/bold blue]")

        # Create language-specific config
        lang_config = create_language_config(config, lang_code)
        manifest_gen = ManifestGenerator(config=lang_config)
        
        # Determine manifest path
        if output is None:
            manifest_path = lang_config.base_audio_dir / f"audio_manifest_{lang_code}.json"
        else:
            manifest_path = Path(output)
        
        if verify_files:
            console.print("[bold blue]🔍 Verifying manifest...[/bold blue]")
            manifest_gen.verify_manifest(manifest_path, fix_missing=fix_missing)
        else:
            console.print("[bold blue]📋 Generating manifest...[/bold blue]")
            manifest_gen.generate_manifest(manifest_path)


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
@click.option('--language', '-lang', type=click.Choice(['bn-IN', 'es-US']), 
              default='bn-IN', help='Language to generate audio for')
@click.option('--dry-run', is_flag=True,
              help='Show what would be generated without actually generating files')
def full_pipeline(output_dir, overwrite, limit_voices, voice_type, volume_gain, 
                 pairs_file, model, max_verify_files, skip_verification, language, dry_run):
    """Complete pipeline: generate audio, create manifest, and verify quality."""
    console.print("[bold green]🚀 Running Full Audio Pipeline[/bold green]")
    console.print("[dim]This will: 1) Generate audio, 2) Create manifest, 3) Verify quality[/dim]\n")
    
    # Create config for the specified language
    base_config = AudioToolsConfig()
    config = create_language_config(base_config, language)
    
    if output_dir is None:
        output_dir = str(config.base_audio_dir)

    # Step 1: Generate Audio
    console.print("[bold blue]Step 1/3: 🎵 Generating audio files...[/bold blue]")
    generator = AudioGenerator(
        config=config,
        overwrite=overwrite,
        limit_voices=limit_voices
    )
    
    pairs_path = Path(pairs_file) if pairs_file else config.pairs_file_path
    
    generation_results = generator.generate_all_audio(
        pairs_data_path=pairs_path,
        volume_gain_db=volume_gain,
        voice_type=voice_type,
        dry_run=dry_run
    )
    
    # If dry run, skip manifest and verification steps
    if dry_run:
        return generation_results
    
    # Step 2: Generate Manifest
    console.print("\n[bold blue]Step 2/3: 📋 Generating audio manifest...[/bold blue]")
    manifest_gen = ManifestGenerator(config=config)
    manifest_gen.generate_manifest()
    
    # Step 3: Verify Audio Quality (optional)
    if not skip_verification:
        console.print(f"\n[bold blue]Step 3/3: 🎙️ Verifying audio quality with {model} model...[/bold blue]")
        
        # Create appropriate model
        if model == 'chirp2':
            stt_model = Chirp2Model(language_code=config.language_code)
        else:
            stt_model = GcpStandardModel(language_code=config.language_code)
        
        verifier = PronunciationVerifier(stt_model, config=config)
        
        # Convert paths
        manifest_path = config.base_audio_dir / f"audio_manifest_{config.language_code}.json"
        
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
@click.option('--output-dir', '-o', 
              help='Output directory for audio files (default: public/audio)')
@click.option('--volume-gain', type=float, default=0.0, 
              help='Volume gain in dB')
@click.option('--pairs-file', '-p', type=click.Path(exists=True), 
              help='Path to minimal pairs JSON file')
@pass_config
def regenerate(config: AudioToolsConfig, word, voice, output_dir, volume_gain, pairs_file):
    """Regenerate audio for a specific word and voice combination."""
    console.print(f"[bold blue]🔄 Regenerating audio for '{word}' in {config.language_code}...[/bold blue]")
    
    # Get paths
    output_dir = get_output_directory(output_dir, config.base_audio_dir)
    pairs_path = Path(pairs_file) if pairs_file else config.pairs_file_path
    
    # Load pairs data to get the native text for this word
    pairs_data = load_minimal_pairs_data(pairs_path)
    unique_words = get_unique_words(pairs_data, config.language_code)
    
    # Find the Bengali text for this transliteration
    bengali_text = None
    for bengali, transliteration in unique_words:
        if transliteration == word:
            bengali_text = bengali
            break
    
    if not bengali_text:
        console.print(f"[red]Error: Word '{word}' not found in minimal pairs data for {config.language_code}[/red]")
        return
    
    # Get voices to regenerate
    all_voices = get_all_voice_names(config.language_code)
    if voice:
        # Validate the voice exists
        all_voice_list = []
        for voice_type_list in all_voices.values():
            all_voice_list.extend(voice_type_list)

        if voice not in [get_minimal_voice_name(v) for v in all_voice_list]:
            console.print(f"[red]Error: Voice '{voice}' not found for {config.language_code}[/red]")
            return
        
        # Convert back to full voice name
        voices_to_regenerate = []
        for full_voice in all_voice_list:
            if get_minimal_voice_name(full_voice) == voice:
                voices_to_regenerate = [full_voice]
                break
    else:
        # Regenerate all voices for this word
        voices_to_regenerate = []
        for voice_type_list in all_voices.values():
            voices_to_regenerate.extend(voice_type_list)
    
    console.print(f"[dim]Text: {bengali_text}[/dim]")
    console.print(f"[dim]Regenerating {len(voices_to_regenerate)} voice(s)[/dim]")
    
    # Create generator and regenerate
    generator = AudioGenerator(
        config=config,
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
        manifest_gen = ManifestGenerator(config=config)
        manifest_gen.generate_manifest()
    
    # Return results as JSON for API calls
    import json
    console.print(f"\n[dim]{json.dumps(results, indent=2)}[/dim]")



@main.command()
@click.option('--min-size', type=int, default=5000, 
              help='Minimum file size in bytes')
@click.option('--audio-dir', '-d', 
              help='Base audio directory to clean (default: public/audio)')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be deleted without actually deleting')
@pass_config
def clean(config: AudioToolsConfig, min_size, audio_dir, dry_run):
    """Clean up small/invalid audio files."""
    from .generator import validate_audio_file
    
    if audio_dir is None:
        audio_dir = config.base_audio_dir

    audio_path = config.base_audio_dir / config.language_code
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


@main.command()
@click.option('--output-dir', '-o', 
              help='Output directory for audio files (default: public/audio)')
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
@click.option('--max-concurrent', type=int, default=10,
              help='Maximum concurrent TTS requests')
@click.option('--max-concurrent-io', type=int, default=20,
              help='Maximum concurrent I/O operations')
@click.option('--batch-size', type=int, default=50,
              help='Batch size for processing')
@click.option('--language', '-lang', type=click.Choice(['bn-IN', 'es-US']), 
              default='bn-IN', help='Language to generate audio for')
def generate_async(output_dir, overwrite, limit_voices, voice_type, volume_gain, 
                  pairs_file, max_concurrent, max_concurrent_io, batch_size, language):
    """Generate audio files asynchronously with parallel processing (faster!)."""
    console.print("[bold blue]🚀 Generating audio files with async processing...[/bold blue]")
    
    if output_dir is None:
        output_dir = str(config.base_audio_dir)

    async def run_async_generation():
        generator = AsyncAudioGenerator(
            base_output_path=output_dir,
            overwrite=overwrite,
            limit_voices=limit_voices,
            max_concurrent=max_concurrent,
            max_concurrent_io=max_concurrent_io,
            language_code=language
        )
        
        pairs_path = Path(pairs_file) if pairs_file else None
        
        results = await generator.generate_all_audio_async(
            pairs_data_path=pairs_path,
            volume_gain_db=volume_gain,
            effects_profile="headphone-class-device",
            voice_type=voice_type,
            batch_size=batch_size
        )
        
        return results
    
    # Run the async generation
    try:
        results = asyncio.run(run_async_generation())
        
        if results["failed"] > 0:
            console.print(f"\n[yellow]⚠️  {results['failed']} recordings failed to generate[/yellow]")
            exit(1)
        else:
            console.print(f"\n[green]✅ All recordings generated successfully![/green]")
            
    except KeyboardInterrupt:
        console.print("\n[red]❌ Generation cancelled by user[/red]")
        exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Generation failed: {str(e)}[/red]")
        exit(1)


@main.command()
@click.option('--output-dir', '-o', 
              help='Output directory for audio files (default: public/audio)')
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
@click.option('--max-concurrent', type=int, default=10,
              help='Maximum concurrent TTS requests (async only)')
@click.option('--use-async', is_flag=True,
              help='Use async processing for faster generation')
@pass_config
def full_pipeline_async(config: AudioToolsConfig, output_dir, overwrite, limit_voices, voice_type, volume_gain, 
                       pairs_file, max_concurrent, use_async):
    """Run the full pipeline (generate + manifest + verify) with optional async processing."""
    console.print("[bold blue]🚀 Running Full Audio Pipeline[/bold blue]")
    console.print("This will: 1) Generate audio, 2) Create manifest, 3) Verify quality")
    
    if output_dir is None:
        output_dir = config.base_audio_dir

    # Step 1: Generate audio
    console.print(f"\n[bold]Step 1/3: 🎵 Generating audio files...[/bold]")
    
    if use_async:
        # Use async generation
        async def run_async_pipeline():
            generator = AsyncAudioGenerator(
                config=config,
                overwrite=overwrite,
                limit_voices=limit_voices,
                max_concurrent=max_concurrent
            )
            
            pairs_path = Path(pairs_file) if pairs_file else config.pairs_file_path
            
            return await generator.generate_all_audio_async(
                pairs_data_path=pairs_path,
                volume_gain_db=volume_gain,
                effects_profile="headphone-class-device",
                voice_type=voice_type
            )
        
        try:
            results = asyncio.run(run_async_pipeline())
        except KeyboardInterrupt:
            console.print("\n[red]❌ Pipeline cancelled by user[/red]")
            exit(1)
        except Exception as e:
            console.print(f"\n[red]❌ Audio generation failed: {str(e)}[/red]")
            exit(1)
    else:
        # Use sync generation
        # Use sync generation
        generator = AudioGenerator(
            config=config,
            overwrite=overwrite,
            limit_voices=limit_voices
        )
        
        pairs_path = Path(pairs_file) if pairs_file else config.pairs_file_path
        
        results = generator.generate_all_audio(
            pairs_data_path=pairs_path,
            volume_gain_db=volume_gain,
            effects_profile="headphone-class-device",
            voice_type=voice_type,
            dry_run=False
        )
    
    if results["failed"] > 0:
        console.print(f"[yellow]⚠️  {results['failed']} audio files failed to generate[/yellow]")
    
    # Step 2: Generate manifest
    console.print(f"\n[bold]Step 2/3: 📋 Generating audio manifest...[/bold]")
    manifest_gen = ManifestGenerator(config=config)
    manifest_gen.generate_manifest()
    
    # Step 3: Verify audio quality (optional, for now just mention it)
    console.print(f"\n[bold]Step 3/3: ✅ Audio pipeline complete![/bold]")
    console.print("[dim]Run 'minimal-pairs-audio verify' separately to verify audio quality[/dim]")
    
    if results["failed"] == 0:
        console.print(f"\n[green]🎉 Full pipeline completed successfully![/green]")
    else:
        console.print(f"\n[yellow]⚠️  Pipeline completed with {results['failed']} failures[/yellow]")




if __name__ == '__main__':
    main()