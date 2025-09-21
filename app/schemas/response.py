from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class TrumpetDetectionResult(BaseModel):
    is_trumpet: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    detection_features: Dict[str, Any]
    warning_message: Optional[str] = None
    recommendations: List[str] = []