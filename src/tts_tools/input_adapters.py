"""Input adapters for reading text from various sources."""

import csv
import json
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import get_unique_words, load_minimal_pairs_data


@dataclass
class TextItem:
    """Represents a text item to generate audio for."""

    text: str  # The actual text to synthesize
    identifier: str  # Unique identifier for file naming (transliteration, slug, etc.)
    language_code: str  # Language code (e.g., "bn-IN", "es-US")
    metadata: dict[str, Any] | None = None  # Additional data (source, note_id, field_name, etc.)

    def __post_init__(self):
        """Validate and normalize the text item."""
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not self.identifier or not self.identifier.strip():
            raise ValueError("Identifier cannot be empty")
        if not self.language_code:
            raise ValueError("Language code cannot be empty")

        # Normalize text and identifier
        self.text = self.text.strip()
        self.identifier = self.identifier.strip()

        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}


class InputAdapter(ABC):
    """Abstract base class for reading text from various input sources."""

    @abstractmethod
    def get_text_items(self) -> list[TextItem]:
        """Extract text items from the input source.

        Returns:
            List of TextItem objects containing text to generate audio for.
        """
        pass

    @abstractmethod
    def get_language_code(self) -> str:
        """Get the primary language code for this input source.

        Returns:
            Language code string (e.g., "bn-IN", "es-US").
        """
        pass

    def get_total_items(self) -> int:
        """Get total number of text items.

        Returns:
            Number of text items that will be generated.
        """
        return len(self.get_text_items())


class MinimalPairsAdapter(InputAdapter):
    """Adapter for the existing minimal pairs JSON format."""

    def __init__(self, pairs_file_path: Path, language_code: str = "bn-IN"):
        """Initialize the minimal pairs adapter.

        Args:
            pairs_file_path: Path to minimal pairs JSON file.
            language_code: Language code to extract words for.
        """
        self.pairs_file_path = pairs_file_path
        self.language_code = language_code
        self._text_items = None

    def get_language_code(self) -> str:
        """Get the language code."""
        return self.language_code

    def get_text_items(self) -> list[TextItem]:
        """Extract text items from minimal pairs data."""
        if self._text_items is None:
            pairs_data = load_minimal_pairs_data(self.pairs_file_path)
            unique_words = get_unique_words(pairs_data, self.language_code)

            self._text_items = []
            for native_text, transliteration in unique_words:
                item = TextItem(
                    text=native_text,
                    identifier=transliteration,
                    language_code=self.language_code,
                    metadata={
                        "source": "minimal_pairs",
                        "source_file": str(self.pairs_file_path),
                        "native_text": native_text,
                        "transliteration": transliteration,
                    },
                )
                self._text_items.append(item)

        return self._text_items


class TextListAdapter(InputAdapter):
    """Adapter for reading text from various list formats (TXT, CSV, JSON)."""

    def __init__(
        self,
        file_path: Path,
        language_code: str,
        format_type: str | None = None,
        text_column: str = "text",
        identifier_column: str = "identifier",
        encoding: str = "utf-8",
    ):
        """Initialize the text list adapter.

        Args:
            file_path: Path to the text file.
            language_code: Language code for the text.
            format_type: Format type ("txt", "csv", "json"). Auto-detected if None.
            text_column: Column name containing text (for CSV/JSON).
            identifier_column: Column name containing identifier (for CSV/JSON).
            encoding: File encoding.
        """
        self.file_path = file_path
        self.language_code = language_code
        self.format_type = format_type or self._detect_format()
        self.text_column = text_column
        self.identifier_column = identifier_column
        self.encoding = encoding
        self._text_items = None

    def _detect_format(self) -> str:
        """Auto-detect file format from extension."""
        suffix = self.file_path.suffix.lower()
        if suffix == ".txt":
            return "txt"
        elif suffix == ".csv":
            return "csv"
        elif suffix == ".json":
            return "json"
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def get_language_code(self) -> str:
        """Get the language code."""
        return self.language_code

    def get_text_items(self) -> list[TextItem]:
        """Extract text items from the file."""
        if self._text_items is None:
            if self.format_type == "txt":
                self._text_items = self._read_txt()
            elif self.format_type == "csv":
                self._text_items = self._read_csv()
            elif self.format_type == "json":
                self._text_items = self._read_json()
            else:
                raise ValueError(f"Unsupported format: {self.format_type}")

        return self._text_items

    def _read_txt(self) -> list[TextItem]:
        """Read from plain text file (one item per line)."""
        items = []
        with open(self.file_path, encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                text = line.strip()
                if text:  # Skip empty lines
                    # Generate identifier from text (simple slug)
                    identifier = self._generate_identifier(text, line_num)
                    item = TextItem(
                        text=text,
                        identifier=identifier,
                        language_code=self.language_code,
                        metadata={
                            "source": "text_file",
                            "source_file": str(self.file_path),
                            "line_number": line_num,
                        },
                    )
                    items.append(item)
        return items

    def _read_csv(self) -> list[TextItem]:
        """Read from CSV file."""
        items = []
        with open(self.file_path, encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                text = row.get(self.text_column, "").strip()
                identifier = row.get(self.identifier_column, "").strip()

                if text:  # Skip rows without text
                    if not identifier:
                        identifier = self._generate_identifier(text, row_num)

                    # Include all other columns as metadata
                    metadata = {
                        "source": "csv_file",
                        "source_file": str(self.file_path),
                        "row_number": row_num,
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in [self.text_column, self.identifier_column]
                        },
                    }

                    item = TextItem(
                        text=text,
                        identifier=identifier,
                        language_code=self.language_code,
                        metadata=metadata,
                    )
                    items.append(item)
        return items

    def _read_json(self) -> list[TextItem]:
        """Read from JSON file."""
        with open(self.file_path, encoding=self.encoding) as f:
            data = json.load(f)

        items = []
        if isinstance(data, list):
            # Array of objects
            for item_num, item_data in enumerate(data, 1):
                if isinstance(item_data, dict):
                    text = item_data.get(self.text_column, "").strip()
                    identifier = item_data.get(self.identifier_column, "").strip()

                    if text:
                        if not identifier:
                            identifier = self._generate_identifier(text, item_num)

                        metadata = {
                            "source": "json_file",
                            "source_file": str(self.file_path),
                            "item_number": item_num,
                            **{
                                k: v
                                for k, v in item_data.items()
                                if k not in [self.text_column, self.identifier_column]
                            },
                        }

                        item = TextItem(
                            text=text,
                            identifier=identifier,
                            language_code=self.language_code,
                            metadata=metadata,
                        )
                        items.append(item)
                elif isinstance(item_data, str):
                    # Simple array of strings
                    text = item_data.strip()
                    if text:
                        identifier = self._generate_identifier(text, item_num)
                        item = TextItem(
                            text=text,
                            identifier=identifier,
                            language_code=self.language_code,
                            metadata={
                                "source": "json_file",
                                "source_file": str(self.file_path),
                                "item_number": item_num,
                            },
                        )
                        items.append(item)

        elif isinstance(data, dict):
            # Single object or nested structure
            text = data.get(self.text_column, "").strip()
            identifier = data.get(self.identifier_column, "").strip()

            if text:
                if not identifier:
                    identifier = self._generate_identifier(text, 1)

                metadata = {
                    "source": "json_file",
                    "source_file": str(self.file_path),
                    **{
                        k: v
                        for k, v in data.items()
                        if k not in [self.text_column, self.identifier_column]
                    },
                }

                item = TextItem(
                    text=text,
                    identifier=identifier,
                    language_code=self.language_code,
                    metadata=metadata,
                )
                items.append(item)

        return items

    def _generate_identifier(self, text: str, fallback_num: int) -> str:
        """Generate a safe identifier from text.

        Args:
            text: The text to create identifier from.
            fallback_num: Fallback number if text can't be used.

        Returns:
            Safe identifier string.
        """
        import re

        # Try to create identifier from text (first 50 chars, alphanumeric only)
        identifier = re.sub(r"[^\w\s-]", "", text[:50].lower())
        identifier = re.sub(r"[-\s]+", "_", identifier).strip("_")

        if identifier and len(identifier) >= 3:
            return identifier
        else:
            # Fallback to numbered identifier
            return f"item_{fallback_num:04d}"


class AnkiDeckAdapter(InputAdapter):
    """Adapter for reading text from Anki deck files (.apkg/.anki2)."""

    def __init__(
        self,
        deck_path: Path,
        language_code: str,
        text_fields: list[str],
        identifier_field: str | None = None,
        deck_name_filter: str | None = None,
    ):
        """Initialize the Anki deck adapter.

        Args:
            deck_path: Path to .apkg or .anki2 file.
            language_code: Language code for the text.
            text_fields: List of field names containing text to generate audio for.
            identifier_field: Field name to use as identifier (defaults to first text field).
            deck_name_filter: Optional deck name to filter cards by.
        """
        self.deck_path = deck_path
        self.language_code = language_code
        self.text_fields = text_fields
        self.identifier_field = identifier_field or text_fields[0]
        self.deck_name_filter = deck_name_filter
        self._text_items = None

        # Check if deck file exists
        if not self.deck_path.exists():
            raise FileNotFoundError(f"Anki deck file not found: {deck_path}")

    def get_language_code(self) -> str:
        """Get the language code."""
        return self.language_code

    def get_text_items(self) -> list[TextItem]:
        """Extract text items from Anki deck."""
        if self._text_items is None:
            # Handle .apkg files by extracting to temp directory
            if self.deck_path.suffix.lower() == ".apkg":
                self._text_items = self._read_apkg()
            else:
                # Assume .anki2 file
                self._text_items = self._read_anki2()

        return self._text_items

    def _read_apkg(self) -> list[TextItem]:
        """Read from .apkg file (zip archive)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract .apkg file
            with zipfile.ZipFile(self.deck_path, "r") as zip_file:
                zip_file.extractall(temp_path)

            # Find collection.anki2 file
            collection_file = temp_path / "collection.anki2"
            if not collection_file.exists():
                raise ValueError("Invalid .apkg file: missing collection.anki2")

            return self._read_anki2_file(collection_file)

    def _read_anki2(self) -> list[TextItem]:
        """Read from .anki2 file directly."""
        return self._read_anki2_file(self.deck_path)

    def _read_anki2_file(self, anki2_path: Path) -> list[TextItem]:
        """Read from .anki2 collection file using direct SQLite access."""
        import json
        import re
        import sqlite3

        items = []

        try:
            conn = sqlite3.connect(str(anki2_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get note types (models) to understand field structure
            cursor.execute("SELECT models FROM col")
            models_json = cursor.fetchone()["models"]
            models = json.loads(models_json)

            # Get deck information if filtering by deck name
            deck_id = None
            if self.deck_name_filter:
                cursor.execute("SELECT decks FROM col")
                decks_json = cursor.fetchone()["decks"]
                decks = json.loads(decks_json)

                for deck_data in decks.values():
                    if deck_data["name"] == self.deck_name_filter:
                        deck_id = deck_data["id"]
                        break

                if deck_id is None:
                    return []  # Deck not found

            # Query notes, optionally filtered by deck
            if deck_id:
                query = """
                SELECT DISTINCT n.id, n.mid, n.flds, n.tags
                FROM notes n
                JOIN cards c ON n.id = c.nid
                WHERE c.did = ?
                """
                cursor.execute(query, (deck_id,))
            else:
                query = "SELECT id, mid, flds, tags FROM notes"
                cursor.execute(query)

            notes = cursor.fetchall()

            for note in notes:
                note_id = note["id"]
                model_id = str(note["mid"])
                fields_data = note["flds"]

                # Get field names from model
                if model_id in models:
                    field_names = [field["name"] for field in models[model_id]["flds"]]
                    field_values = fields_data.split("\x1f")  # Anki field separator

                    # Create field mapping
                    note_fields = {}
                    for i, field_name in enumerate(field_names):
                        if i < len(field_values):
                            note_fields[field_name] = field_values[i]

                    # Extract text from specified fields
                    for field_name in self.text_fields:
                        if field_name in note_fields:
                            text = note_fields[field_name]

                            # Clean HTML tags and whitespace
                            if text:
                                text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
                                text = text.strip()

                            if text:  # Only process non-empty text
                                # Generate identifier
                                identifier_text = note_fields.get(self.identifier_field, text)
                                identifier_text = re.sub(r"<[^>]+>", "", identifier_text).strip()
                                identifier = self._generate_identifier(
                                    identifier_text, note_id, field_name
                                )

                                item = TextItem(
                                    text=text,
                                    identifier=identifier,
                                    language_code=self.language_code,
                                    metadata={
                                        "source": "anki_deck",
                                        "source_file": str(self.deck_path),
                                        "note_id": note_id,
                                        "field_name": field_name,
                                        "deck_name": self.deck_name_filter,
                                        "model_id": model_id,
                                        "all_fields": note_fields,
                                    },
                                )
                                items.append(item)

        except sqlite3.Error as e:
            raise ValueError(f"Error reading Anki database: {e}") from e
        finally:
            if "conn" in locals():
                conn.close()

        return items

    def _generate_identifier(self, text: str, note_id: int, field_name: str) -> str:
        """Generate identifier for Anki note field."""
        import re

        # Try to create identifier from text
        identifier = re.sub(r"[^\w\s-]", "", text[:50].lower())
        identifier = re.sub(r"[-\s]+", "_", identifier).strip("_")

        if identifier and len(identifier) >= 3:
            return f"{identifier}_{note_id}_{field_name}"
        else:
            return f"note_{note_id}_{field_name}"

    def get_missing_audio_items(self, existing_audio_dir: Path) -> list[TextItem]:
        """Get text items that don't have existing audio files.

        Args:
            existing_audio_dir: Directory containing existing audio files.

        Returns:
            List of TextItem objects for text without audio.
        """
        all_items = self.get_text_items()
        missing_items = []

        for item in all_items:
            # Check if audio file exists for this item
            audio_dir = existing_audio_dir / item.identifier
            if not audio_dir.exists() or not any(audio_dir.glob("*.mp3")):
                missing_items.append(item)

        return missing_items


def create_adapter(
    source_type: str, source_path: str | Path, language_code: str, **kwargs
) -> InputAdapter:
    """Factory function to create appropriate input adapter.

    Args:
        source_type: Type of input source ("minimal_pairs", "text_list", "anki_deck").
        source_path: Path to the input source file.
        language_code: Language code for the text.
        **kwargs: Additional arguments specific to adapter type.

    Returns:
        Configured InputAdapter instance.
    """
    source_path = Path(source_path)

    if source_type == "minimal_pairs":
        return MinimalPairsAdapter(source_path, language_code)

    elif source_type == "text_list":
        return TextListAdapter(source_path, language_code, **kwargs)

    elif source_type == "anki_deck":
        text_fields = kwargs.get("text_fields", ["Front"])
        return AnkiDeckAdapter(source_path, language_code, text_fields, **kwargs)

    else:
        raise ValueError(f"Unsupported source type: {source_type}")
