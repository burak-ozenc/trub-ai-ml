from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import os
from datetime import datetime

from app.models.trumpet_detector import TrumpetDetector
from app.utils.audio_processor import load_and_preprocess_audio
from app.utils.feature_extractor import extract_acoustic_features, generate_recommendations, generate_warning
from app.schemas.response import TrumpetDetectionResult

app = FastAPI(
    title="Trumpet Detection API",
    description="API for detecting trumpet sounds in audio files",
    version="1.0.0"
)

# Initialize the detector with model and scaler paths
model_path = "app/data/trained_models/trumpet_detector.pkl"
scaler_path = "app/data/trained_models/scaler.pkl"
detector = TrumpetDetector(model_path, scaler_path)

@app.post("/detect-trumpet", response_model=TrumpetDetectionResult)
async def detect_trumpet(audio_file: UploadFile = File(..., description="WAV audio file to analyze")):
    # Validate file type
    if not audio_file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Save uploaded file temporarily
    temp_path = f"temp_{datetime.now().timestamp()}.wav"
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)

        # Load and preprocess audio
        y, sr = load_and_preprocess_audio(temp_path)

        # Extract features
        features = extract_acoustic_features(y, sr)
        
        print('features: ', features)

        # Make prediction
        is_trumpet, confidence = detector.predict(features)

        # Generate recommendations and warnings
        recommendations = generate_recommendations(features, confidence)
        warning_message = generate_warning(features, confidence)

        # Prepare response
        result = TrumpetDetectionResult(
            is_trumpet=is_trumpet,
            confidence_score=confidence,
            detection_features=features,
            warning_message=warning_message,
            recommendations=recommendations
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Rest of the file remains the same...