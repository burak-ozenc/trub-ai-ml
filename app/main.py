from fastapi import FastAPI, File, UploadFile, HTTPException
import aiofiles
import os
from datetime import datetime

from app.models.trumpet_detector import TrumpetDetector
from app.utils.audio_processor import load_and_preprocess_audio
from app.utils.feature_extractor import extract_acoustic_features
from app.schemas.response import TrumpetDetectionResult

app = FastAPI(
    title="Trumpet Detection API",
    description="API for detecting trumpet sounds in audio files",
    version="1.0.0"
)

# Initialize the detector
detector = TrumpetDetector("app/data/trained_models/trumpet_detector.pkl")

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

        # Make prediction
        is_trumpet, confidence = detector.predict(features)

        # Generate recommendations and warnings
        # recommendations = generate_recommendations(features, confidence)
        # warning_message = generate_warning(features, confidence)

        # Prepare response
        result = TrumpetDetectionResult(
            is_trumpet=is_trumpet,
            confidence_score=confidence,
            detection_features=features,
            warning_message="warning_message",
            recommendations=[]
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.model_loaded,
        "service": "Trumpet Detection API"
    }

@app.get("/")
async def root():
    return {
        "message": "Trumpet Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect-trumpet": "POST /detect-trumpet",
            "health": "GET /health"
        }
    }