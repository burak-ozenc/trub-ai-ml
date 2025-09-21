import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

from app.utils.feature_extractor import extract_acoustic_features

class TrumpetDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = None
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with a simple model (to be trained)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model_loaded = False

    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            loaded_data = joblib.load(model_path)
            self.model = loaded_data['model']
            self.feature_columns = loaded_data['feature_columns']
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model_loaded = False

    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is not None and self.feature_columns is not None:
            data_to_save = {
                'model': self.model,
                'feature_columns': self.feature_columns
            }
            joblib.dump(data_to_save, model_path)
            print(f"Model saved successfully to {model_path}")
        else:
            print("No model to save")

    def prepare_features(self, features_dict):
        """Convert features dictionary to model input format"""
        if self.feature_columns is None:
            # If no feature columns defined, use all available
            feature_values = []
            feature_names = []
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    feature_values.append(value)
                    feature_names.append(key)
            return np.array(feature_values).reshape(1, -1), feature_names
        else:
            # Use the predefined feature columns
            feature_values = [features_dict[col] for col in self.feature_columns]
            return np.array(feature_values).reshape(1, -1), self.feature_columns

    def predict(self, features_dict):
        """Make prediction using the loaded model"""
        if not self.model_loaded:
            # Fallback to rule-based detection if no model loaded
            return self.rule_based_detection(features_dict)

        try:
            features_array, feature_names = self.prepare_features(features_dict)
            prediction = self.model.predict(features_array)
            probability = self.model.predict_proba(features_array)

            is_trumpet = bool(prediction[0])
            confidence = float(probability[0][1] if is_trumpet else probability[0][0])

            return is_trumpet, confidence
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            return self.rule_based_detection(features_dict)

    def rule_based_detection(self, features):
        """Fallback rule-based detection if model is not available"""
        # Simple rule-based detection
        score = 0

        if features.get('energy_sufficient', False):
            score += 0.1

        if features.get('centroid_in_range', False):
            score += 0.2

        if features.get('low_zcr', False):
            score += 0.1

        if features.get('harmonic_sufficient', False):
            score += 0.2

        if features.get('pitch_in_trumpet_range', False):
            score += 0.2

        if features.get('has_tonal_content', False):
            score += 0.2

        is_trumpet = score > 0.5
        confidence = min(0.95, max(0.05, score))

        return is_trumpet, confidence

    def train_model(self, data_dir, test_size=0.2):
        """Train the model on labeled data"""
        # This would be implemented to load labeled data and train the model
        # For now, we'll just create a placeholder
        print("Training model... This would load data from", data_dir)

        # In a real implementation, you would:
        # 1. Load audio files and labels
        # 2. Extract features for each file
        # 3. Split into train/test sets
        # 4. Train the model
        # 5. Evaluate performance

        # Placeholder for actual implementation
        self.model_loaded = True
        print("Model training complete (placeholder)")