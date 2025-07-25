"""Models for speech recognition and text normalization."""

import os
import unicodedata
import re
import subprocess
from typing import Tuple
from abc import ABC, abstractmethod

from google.cloud import speech
from google.cloud.speech_v2 import SpeechClient as SpeechV2Client
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import edit_distance


class BengaliTextNormalizer:
    """Normalize Bengali text for comparison between expected and STT results."""

    @staticmethod
    def normalize(text: str) -> str:
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

    @staticmethod
    def calculate_edit_distance(expected: str, actual: str) -> int:
        """Calculate edit distance between two Bengali texts."""
        norm_expected = BengaliTextNormalizer.normalize(expected)
        norm_actual = BengaliTextNormalizer.normalize(actual)
        return edit_distance.edit_distance(norm_expected, norm_actual)


class VerificationModel(ABC):
    """Abstract base class for speech verification models."""
    
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> Tuple[str, float]:
        """Transcribe audio file and return (transcript, confidence)."""
        pass


class GcpStandardModel(VerificationModel):
    """Google Cloud Speech-to-Text standard model."""
    
    def __init__(self, language_code: str = "bn-IN"):
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
        try:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            response = self.client.recognize(config=self.recognition_config, audio=audio)

            if not response.results:
                return "", 0.0

            result = response.results[0]
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence
            return transcript, confidence
        except Exception as e:
            print(f"Error transcribing {audio_file_path} with GCP Standard: {e}")
            return "", 0.0


class Chirp2Model(VerificationModel):
    """Google Cloud Speech-to-Text Chirp2 model."""
    
    def __init__(self, language_code: str = "bn-IN", region: str = "us-central1"):
        self.language_code = language_code
        self.region = region
        project_id = self._get_project_id()
        if not project_id:
            raise ValueError(
                "Google Cloud project ID not found. Please set the GOOGLE_CLOUD_PROJECT "
                "environment variable or configure it using 'gcloud config set project YOUR_PROJECT_ID'."
            )
        self.project_id = project_id
        
        api_endpoint = f"{self.region}-speech.googleapis.com"
        self.client = SpeechV2Client(client_options=ClientOptions(api_endpoint=api_endpoint))
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language_code],
            model="chirp_2",
            features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
        )

    def _get_project_id(self) -> str:
        """Get Google Cloud project ID from environment or gcloud config."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            try:
                project_id = subprocess.check_output(
                    ['gcloud', 'config', 'get-value', 'project']
                ).strip().decode('utf-8')
            except (FileNotFoundError, subprocess.CalledProcessError):
                project_id = None
        return project_id

    def transcribe(self, audio_file_path: str) -> Tuple[str, float]:
        """Transcribe audio file using Chirp2 model."""
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
                return "", 0.0

            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            return transcript, confidence
        except Exception as e:
            print(f"Error transcribing {audio_file_path} with Chirp2: {e}")
            return "", 0.0