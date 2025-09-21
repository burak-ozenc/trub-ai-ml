import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from app.utils.audio_processor import load_and_preprocess_audio
from app.utils.feature_extractor import extract_acoustic_features
from app.models.trumpet_detector import TrumpetDetector

def load_dataset(data_dir):
    """Load labeled dataset from directory"""
    # This is a placeholder function
    # In a real implementation, you would:
    # 1. Iterate through subdirectories (trumpet/non-trumpet)
    # 2. Load each audio file
    # 3. Extract features
    # 4. Create labeled dataset

    print("Loading dataset...")
    # Placeholder for actual implementation
    return None, None

def train_and_save_model():
    """Train the model and save it"""
    # Load dataset
    X, y = load_dataset("app/data/raw")

    if X is None or y is None:
        print("No dataset found. Using rule-based detection only.")
        return

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    detector = TrumpetDetector()
    detector.model = model
    detector.feature_columns = X.columns.tolist() if hasattr(X, 'columns') else None
    detector.save_model("app/data/trained_models/trumpet_detector.pkl")

    print("Model trained and saved successfully")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("app/data/trained_models", exist_ok=True)

    # Train model
    train_and_save_model()