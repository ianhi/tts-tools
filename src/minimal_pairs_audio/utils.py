"""Utility functions for minimal pairs audio processing."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_minimal_pairs_data(json_path: Path = None) -> Dict[str, Any]:
    """Load minimal pairs data from JSON file.
    
    Args:
        json_path: Path to JSON file. If None, uses default location.
        
    Returns:
        Dictionary containing minimal pairs data.
    """
    if json_path is None:
        # Import here to avoid circular dependency
        from .config import AudioToolsConfig
        default_config = AudioToolsConfig()
        json_path = default_config.pairs_file_path
    
    if not json_path.exists():
        raise FileNotFoundError(f"Minimal pairs data not found at {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_audio_manifest(manifest_path: Path = None, language_code: str = "bn-IN") -> Dict[str, Any]:
    """Load audio manifest to get available audio files.
    
    Args:
        manifest_path: Path to manifest file. If None, uses default location.
        language_code: Language code for the manifest file.
        
    Returns:
        Dictionary containing audio manifest data.
    """
    if manifest_path is None:
        # Import here to avoid circular dependency
        from .config import AudioToolsConfig
        default_config = AudioToolsConfig()
        manifest_path = default_config.base_audio_dir / f"audio_manifest_{language_code}.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Audio manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_unique_words(pairs_data: Dict[str, Any], language_code: str = "bn-IN") -> List[Tuple[str, str]]:
    """Extract unique words from minimal pairs data for a specific language.
    
    Args:
        pairs_data: Minimal pairs data dictionary.
        language_code: Language code to extract words for (e.g., "bn-IN", "es-ES").
        
    Returns:
        For languages with transliteration (like Bengali): List of (native_word, transliteration) tuples.
        For languages without transliteration (like Spanish): List of (word, word) tuples.
    """
    unique_words = {}
    lang_data = pairs_data.get(language_code, {})
    types = lang_data.get("types", {})
    
    for category_data in types.values():
        pairs = category_data.get("pairs", [])
        for pair in pairs:
            for word_data in pair:
                if len(word_data) == 2:
                    # Language with transliteration (like Bengali)
                    native_word, transliteration = word_data
                    if transliteration not in unique_words:
                        unique_words[transliteration] = native_word
                elif len(word_data) == 1:
                    # Language without transliteration (like Spanish)
                    word = word_data[0]
                    if word not in unique_words:
                        unique_words[word] = word
    
    return [(native, key) for key, native in unique_words.items()]


def get_google_cloud_project_id() -> str:
    """Get Google Cloud project ID from environment or gcloud config."""
    import os
    import subprocess
    
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        try:
            project_id = subprocess.check_output(
                ['gcloud', 'config', 'get-value', 'project']
            ).strip().decode('utf-8')
        except (FileNotFoundError, subprocess.CalledProcessError):
            project_id = None
    
    if not project_id:
        raise ValueError(
            "Google Cloud project ID not found. Please set the GOOGLE_CLOUD_PROJECT "
            "environment variable or configure it using 'gcloud config set project YOUR_PROJECT_ID'."
        )
    
    return project_id


def ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)