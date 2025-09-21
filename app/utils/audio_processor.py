import librosa
import numpy as np
import noisereduce as nr
from scipy import signal
import soundfile as sf  # Add this import

def load_and_preprocess_audio(file_path, target_sr=22050):
    """Load and preprocess audio file with better error handling"""
    try:
        # Try to load with soundfile first (more reliable)
        try:
            y, sr = sf.read(file_path)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
        except Exception as e:
            print(f"SoundFile failed: {e}, falling back to librosa")
            # Fall back to librosa if soundfile fails
            y, sr = librosa.load(file_path, sr=target_sr)

        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        # Noise reduction
        y_clean = nr.reduce_noise(y=y, sr=sr)

        # Normalize audio
        y_normalized = librosa.util.normalize(y_clean)

        return y_normalized, sr
    except Exception as e:
        raise Exception(f"Audio loading failed: {str(e)}")

# Rest of the functions remain the same...