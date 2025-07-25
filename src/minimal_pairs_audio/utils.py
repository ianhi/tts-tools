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
        json_path = Path("public/minimal_pairs_db.json")
    
    if not json_path.exists():
        raise FileNotFoundError(f"Minimal pairs data not found at {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_audio_manifest(manifest_path: Path = None) -> Dict[str, Any]:
    """Load audio manifest to get available audio files.
    
    Args:
        manifest_path: Path to manifest file. If None, uses default location.
        
    Returns:
        Dictionary containing audio manifest data.
    """
    if manifest_path is None:
        manifest_path = Path("public/audio/audio_manifest.json")
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Audio manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_unique_words(pairs_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract unique words from minimal pairs data.
    
    Args:
        pairs_data: Minimal pairs data dictionary.
        
    Returns:
        List of (bengali_word, transliteration) tuples.
    """
    unique_words = {}
    bn_data = pairs_data.get("bn-IN", {})
    types = bn_data.get("types", {})
    
    for category_data in types.values():
        pairs = category_data.get("pairs", [])
        for pair in pairs:
            for word_data in pair:
                bengali_word, transliteration = word_data
                # Only add if we haven't seen this transliteration before
                if transliteration not in unique_words:
                    unique_words[transliteration] = bengali_word
    
    return [(bengali, translit) for translit, bengali in unique_words.items()]


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