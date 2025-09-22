import librosa
import numpy as np
from scipy import stats

def extract_acoustic_features(y, sr):
    """Extract comprehensive acoustic features from audio"""
    features = {}

    # Time-domain features
    features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
    features['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # Spectral features
    S = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    features['spectral_centroid'] = float(np.mean(spectral_centroid))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))

    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    features['spectral_rolloff'] = float(np.mean(spectral_rolloff))

    # Harmonic features
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(y_harmonic**2) / (np.mean(y_harmonic**2) + np.mean(y_percussive**2))
    features['harmonic_ratio'] = float(harmonic_ratio)

    # Pitch content
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) > 0:
        features['average_pitch'] = float(np.mean(pitches))
        features['pitch_range'] = float(np.max(pitches) - np.min(pitches))
        # Trumpet range detection (typically 165Hz to 1000Hz)
        trumpet_pitches = pitches[(pitches >= 165) & (pitches <= 1000)]
        trumpet_pitch_ratio = len(trumpet_pitches) / len(pitches) if len(pitches) > 0 else 0
        features['trumpet_pitch_ratio'] = float(trumpet_pitch_ratio)
    else:
        features['average_pitch'] = 0.0
        features['pitch_range'] = 0.0
        features['trumpet_pitch_ratio'] = 0.0

    # MFCCs - convert to list of floats
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(min(5, mfccs.shape[0]))]
    features['mfcc_coefficients'] = mfcc_means

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast'] = float(np.mean(spectral_contrast))

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_energy'] = float(np.mean(chroma))

    # Derived features for decision making
    features['energy_sufficient'] = features['rms_energy'] > 0.01
    features['centroid_in_range'] = 500 < features['spectral_centroid'] < 2500  # Trumpet range
    features['low_zcr'] = features['zcr'] < 0.1  # Trumpets have lower ZCR
    features['rolloff_sufficient'] = features['spectral_rolloff'] > 1500
    features['harmonic_sufficient'] = features['harmonic_ratio'] > 0.3
    features['has_pitch_content'] = features['trumpet_pitch_ratio'] > 0.1
    features['pitch_in_trumpet_range'] = features['trumpet_pitch_ratio'] > 0.5
    features['has_tonal_content'] = features['chroma_energy'] > 0.1

    return features

def generate_recommendations(features, confidence):
    """Generate recommendations based on feature analysis"""
    recommendations = []

    if not features['energy_sufficient']:
        recommendations.extend([
            "Play louder and with more confidence",
            "Play with more volume - the signal is too quiet"
        ])

    if not features['centroid_in_range']:
        recommendations.append("Check the pitch range of the trumpet")

    if not features['harmonic_sufficient']:
        recommendations.append("Work on tone quality - more harmonic content needed")

    if not features['has_pitch_content']:
        recommendations.append("Ensure you're playing definite pitches")

    if not features['pitch_in_trumpet_range']:
        recommendations.append("Check if you're playing in the trumpet's range")

    if confidence < 0.7:
        recommendations.extend([
            "Ensure proper microphone placement",
            "Check for background noise interference"
        ])

    return list(set(recommendations))  # Remove duplicates

def generate_warning(features, confidence):
    """Generate warning message based on feature analysis"""
    if confidence < 0.6:
        return "Weak trumpet signal detected. Audio quality may affect analysis accuracy."

    if not features['energy_sufficient']:
        return "Low energy signal. Trumpet may be too quiet or too far from microphone."

    if not features['has_pitch_content']:
        return "Limited pitch content detected. This may not be a pitched instrument."

    return None