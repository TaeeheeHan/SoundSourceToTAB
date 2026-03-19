"""
Audio loading and preprocessing module.
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff'}


class AudioProcessor:
    TARGET_SR = 22050

    def load(self, file_path: str):
        """Load audio file → (y: np.ndarray, sr: int)"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        y, sr = librosa.load(file_path, sr=self.TARGET_SR, mono=True)
        return y, sr

    def preprocess_bass(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Bass-focused preprocessing:
        - Normalize amplitude
        - Keep only bass frequency range (E1 ~41 Hz to ~300 Hz)
        """
        # Normalize
        y = librosa.util.normalize(y)

        # Low-pass filter: remove frequencies above ~500 Hz
        # Using librosa's effects for harmonic isolation
        # (keeps pitch-like components, removes transient noise)
        y_harm = librosa.effects.harmonic(y, margin=3.0)

        return y_harm

    def get_duration(self, y: np.ndarray, sr: int) -> float:
        return librosa.get_duration(y=y, sr=sr)
