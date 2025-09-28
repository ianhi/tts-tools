from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioToolsConfig:
    """Configuration for the audio tools."""

    language_code: str = "bn-IN"
    languages: list[str] = None
    base_audio_dir: Path = Path("public/audio")
    pairs_file_path: Path = Path("public/minimal_pairs_db.json")

    def __post_init__(self):
        """Set default values that need to be mutable."""
        if self.languages is None:
            self.languages = ["bn-IN"]
