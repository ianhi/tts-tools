"""Minimal Pairs Audio Tools - Generate and verify audio for language learning."""

from .generator import AudioGenerator, get_all_voice_names, validate_audio_file
from .verifier import PronunciationVerifier, VerificationResult, save_verification_results
from .manifest import ManifestGenerator
from .models import (
    BengaliTextNormalizer,
    VerificationModel,
    GcpStandardModel,
    Chirp2Model
)
from .utils import (
    load_minimal_pairs_data,
    load_audio_manifest,
    get_unique_words,
    get_google_cloud_project_id,
    ensure_directory_exists
)

__version__ = "0.1.0"

__all__ = [
    # Generator
    "AudioGenerator",
    "get_all_voice_names",
    "validate_audio_file",
    # Verifier
    "PronunciationVerifier",
    "VerificationResult",
    "save_verification_results",
    # Manifest
    "ManifestGenerator",
    # Models
    "BengaliTextNormalizer",
    "VerificationModel",
    "GcpStandardModel",
    "Chirp2Model",
    # Utils
    "load_minimal_pairs_data",
    "load_audio_manifest",
    "get_unique_words",
    "get_google_cloud_project_id",
    "ensure_directory_exists",
]
