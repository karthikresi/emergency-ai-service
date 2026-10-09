from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmergencyRequest(BaseModel):
    emergency_text: str = Field(..., min_length=1)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location: Optional[str] = None
    timestamp: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class SkillPrediction(BaseModel):
    skill: str
    confidence: float


class CategoryPrediction(BaseModel):
    prediction: str
    confidence: float


class SeverityPrediction(BaseModel):
    prediction: str
    confidence: float
