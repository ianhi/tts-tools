"""Models for speech recognition and text normalization."""

import unicodedata
import re
from typing import Tuple
from abc import ABC, abstractmethod

from google.cloud import speech
from google.cloud.speech_v2 import SpeechClient as SpeechV2Client
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import edit_distance

from .utils import get_google_cloud_project_id


class TextNormalizer(ABC):
    """Abstract base class for text normalization."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text for accurate comparison."""
        pass

    @abstractmethod
    def calculate_edit_distance(self, expected: str, actual: str) -> int:
        """Calculate edit distance between two texts."""
        pass


class BengaliTextNormalizer(TextNormalizer):
    """Normalize Bengali text for comparison between expected and STT results."""

    def normalize(self, text: str) -> str:
        """Normalize Bengali text for accurate comparison."""
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        # Remove Bengali punctuation and digits
        text = re.sub(r'[।,;:?!।০-৯]', '', text)
        text = ' '.join(text.split())
        text = text.lower()
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        return text.strip()

    def calculate_edit_distance(self, expected: str, actual: str) -> int:
        """Calculate edit distance between two Bengali texts."""
        norm_expected = self.normalize(expected)
        norm_actual = self.normalize(actual)
        return edit_distance.edit_distance(norm_expected, norm_actual)


class SpanishTextNormalizer(TextNormalizer):
    """Normalize Spanish text for comparison between expected and STT results."""

    def normalize(self, text: str) -> str:
        """Normalize Spanish text for accurate comparison."""
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        # Remove common punctuation
        text = re.sub(r'[.,;:?!¡¿]', '', text)
        text = ' '.join(text.split())
        text = text.lower()
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        return text.strip()

    def calculate_edit_distance(self, expected: str, actual: str) -> int:
        """Calculate edit distance between two Spanish texts."""
        norm_expected = self.normalize(expected)
        norm_actual = self.normalize(actual)
        return edit_distance.edit_distance(norm_expected, norm_actual)


class VerificationModel(ABC):
    """Abstract base class for speech verification models."""
    
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> Tuple[str, float]:
        """Transcribe audio file and return (transcript, confidence)."""
        pass


class GcpStandardModel(VerificationModel):
    """Google Cloud Speech-to-Text standard model."""
    
    def __init__(self, language_code: str):
        self.language_code = language_code
        self.client = speech.SpeechClient()
        self.recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=self.language_code,
            model="default",
            use_enhanced=False,
            enable_word_confidence=True,
            enable_automatic_punctuation=True,
        )

    def transcribe(self, audio_file_path: str) -> Tuple[str, float]:
        """Transcribe audio file using GCP standard model."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            response = self.client.recognize(config=self.recognition_config, audio=audio)

            if not response.results:
                logger.warning(f"No transcription results for {audio_file_path}")
                return "", 0.0

            result = response.results[0]
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence
            return transcript, confidence
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing {audio_file_path} with GCP Standard", exc_info=True)
            raise RuntimeError(f"Failed to transcribe {audio_file_path}: {str(e)}") from e


class Chirp2Model(VerificationModel):
    """Google Cloud Speech-to-Text Chirp2 model."""
    
    def __init__(self, language_code: str, region: str = "us-central1"):
        self.language_code = language_code
        self.region = region
        self.project_id = get_google_cloud_project_id()
        
        api_endpoint = f"{self.region}-speech.googleapis.com"
        self.client = SpeechV2Client(client_options=ClientOptions(api_endpoint=api_endpoint))
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language_code],
            model="chirp_2",
            features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
        )

    def transcribe(self, audio_file_path: str) -> Tuple[str, float]:
        """Transcribe audio file using Chirp2 model."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            with open(audio_file_path, "rb") as f:
                content = f.read()

            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{self.project_id}/locations/{self.region}/recognizers/_",
                config=self.config,
                content=content,
            )
            response = self.client.recognize(request=request)

            if not response.results:
                logger.warning(f"No transcription results for {audio_file_path}")
                return "", 0.0

            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            return transcript, confidence
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing {audio_file_path} with Chirp2", exc_info=True)
            raise RuntimeError(f"Failed to transcribe {audio_file_path}: {str(e)}") from e