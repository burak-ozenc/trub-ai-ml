import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset for training"""
    df = pd.read_csv(csv_path)

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y

def train_model(X, y, test_size=0.2, random_state=42):
    """Train a Random Forest classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced'  # Handle class imbalance
    )

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train on the full training set
    model.fit(X_train_scaled, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Trumpet', 'Trumpet'],
                yticklabels=['Non-Trumpet', 'Trumpet'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('data/processed/confusion_matrix.png')
    plt.close()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png')
    plt.close()

    return model, scaler, feature_importance, X.columns

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_

def save_model(model, scaler, feature_columns, model_path, scaler_path):
    """Save the trained model and scaler"""
    model_data = {
        'model': model,
        'feature_columns': feature_columns
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load and prepare data
    csv_path = "data/processed/dataset.csv"
    X, y = load_and_prepare_data(csv_path)

    # Train the model
    print("Training model...")
    model, scaler, feature_importance, feature_columns = train_model(X, y)

    # Display feature importance
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    # Save the model
    model_path = "app/data/trained_models/trumpet_detector.pkl"
    scaler_path = "app/data/trained_models/scaler.pkl"
    save_model(model, scaler, feature_columns, model_path, scaler_path)

    print("Model training complete!")