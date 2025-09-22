import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

def extract_features_from_audio(file_path, target_sr=22050):
    """Extract features from a single audio file"""
    try:
        # Load and preprocess audio
        y, sr = librosa.load(file_path, sr=target_sr)

        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        # Extract features
        features = {}

        # Basic features
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # Spectral features
        S = np.abs(librosa.stft(y))
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)

        # Harmonic features
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(y_harmonic**2) / (np.mean(y_harmonic**2) + np.mean(y_percussive**2))
        features['harmonic_ratio'] = harmonic_ratio

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(min(5, mfccs.shape[0])):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast'] = np.mean(spectral_contrast)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_energy'] = np.mean(chroma)

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def create_dataset(trumpet_dir, non_trumpet_dir, output_file):
    """Create a dataset from trumpet and non-trumpet directories"""
    data = []
    labels = []

    # Process trumpet files
    print("Processing trumpet files...")
    for filename in tqdm(os.listdir(trumpet_dir)):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(trumpet_dir, filename)
            features = extract_features_from_audio(file_path)
            if features:
                data.append(features)
                labels.append(1)  # 1 for trumpet

    # Process non-trumpet files
    print("Processing non-trumpet files...")
    for filename in tqdm(os.listdir(non_trumpet_dir)):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(non_trumpet_dir, filename)
            features = extract_features_from_audio(file_path)
            if features:
                data.append(features)
                labels.append(0)  # 0 for non-trumpet

    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset created with {len(df)} samples and saved to {output_file}")

    return df

if __name__ == "__main__":
    trumpet_dir = "data/raw/trumpet"
    non_trumpet_dir = "data/raw/non_trumpet"
    output_file = "data/processed/dataset.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    create_dataset(trumpet_dir, non_trumpet_dir, output_file)