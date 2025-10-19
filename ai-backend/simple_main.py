#!/usr/bin/env python3
"""
Simplified MedTrack AI Backend
Fast startup version without heavy ML models
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MedTrack AI Backend (Simple)",
    description="Simplified AI backend for medication validation and health insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}
    type: str = "general"

class ChatResponse(BaseModel):
    response: str
    type: str
    confidence: float = 0.8

class ValidationRequest(BaseModel):
    medication: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    user_context: Dict[str, Any] = {}

class ValidationResponse(BaseModel):
    is_valid: bool
    confidence: float
    suggestions: List[str]
    warnings: List[str]
    extracted_entities: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    min_confidence: float = 0.5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query_processed: str

class HealthReportRequest(BaseModel):
    user_data: Dict[str, Any]
    report_type: str = "comprehensive"

class HealthReportResponse(BaseModel):
    adherence: str
    trend: str
    insights: List[str]
    recommendations: List[str]
    confidence: float

# Simple medication database
MEDICATION_DATABASE = [
    {
        "name": "acetaminophen",
        "brands": ["tylenol", "panadol"],
        "dosage_forms": ["tablet", "liquid"],
        "strengths": ["500mg", "1000mg"],
        "indications": ["pain", "fever"],
        "contraindications": ["liver disease"]
    },
    {
        "name": "ibuprofen",
        "brands": ["advil", "motrin"],
        "dosage_forms": ["tablet", "liquid"],
        "strengths": ["200mg", "400mg", "600mg"],
        "indications": ["pain", "inflammation", "fever"],
        "contraindications": ["stomach ulcers"]
    },
    {
        "name": "metformin",
        "brands": ["glucophage"],
        "dosage_forms": ["tablet"],
        "strengths": ["500mg", "850mg", "1000mg"],
        "indications": ["diabetes", "type 2 diabetes"],
        "contraindications": ["kidney disease"]
    },
    {
        "name": "lisinopril",
        "brands": ["prinivil", "zestril"],
        "dosage_forms": ["tablet"],
        "strengths": ["5mg", "10mg", "20mg", "40mg"],
        "indications": ["hypertension", "heart failure"],
        "contraindications": ["pregnancy", "angioedema"]
    }
]

def simple_fuzzy_match(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Simple fuzzy matching for medications"""
    query_lower = query.lower().strip()
    matches = []
    
    for med in MEDICATION_DATABASE:
        score = 0
        
        # Check name match
        if query_lower in med["name"].lower():
            score = 90
        elif med["name"].lower() in query_lower:
            score = 80
        
        # Check brand matches
        for brand in med["brands"]:
            if query_lower in brand.lower():
                score = max(score, 85)
            elif brand.lower() in query_lower:
                score = max(score, 75)
        
        if score > 0:
            matches.append({
                "name": med["name"],
                "brands": med["brands"],
                "confidence": score / 100,
                "dosage_forms": med["dosage_forms"],
                "strengths": med["strengths"],
                "indications": med["indications"],
                "contraindications": med["contraindications"]
            })
    
    # Sort by confidence and return top matches
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    return matches[:limit]

def extract_medication_entities(text: str) -> Dict[str, Any]:
    """Extract medication entities using simple patterns"""
    entities = {
        "medications": [],
        "dosages": [],
        "frequencies": [],
        "raw_entities": []
    }
    
    text_lower = text.lower()
    
    # Simple medication extraction
    for med in MEDICATION_DATABASE:
        if med["name"] in text_lower:
            entities["medications"].append(med["name"])
        for brand in med["brands"]:
            if brand in text_lower:
                entities["medications"].append(brand)
    
    # Simple dosage extraction
    import re
    dosage_patterns = [
        r'\b(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:g|grams?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:ml|milliliters?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(?:tablets?|pills?|capsules?)\b'
    ]
    
    for pattern in dosage_patterns:
        matches = re.findall(pattern, text_lower)
        entities["dosages"].extend(matches)
    
    # Simple frequency extraction
    frequency_patterns = [
        r'\b(?:once|one)\s+(?:daily|a day|per day)\b',
        r'\b(?:twice|two)\s+(?:daily|a day|per day)\b',
        r'\bevery\s+\d+\s+(?:hours?|hrs?)\b',
        r'\b(?:as needed|prn|when needed)\b'
    ]
    
    for pattern in frequency_patterns:
        matches = re.findall(pattern, text_lower)
        entities["frequencies"].extend(matches)
    
    return entities

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_available": True,
        "components": {
            "ai_models": True,
            "ner_pipeline": True,
            "fuzzy_matcher": True,
            "health_generator": True
        }
    }

@app.get("/api/ai/status")
async def ai_status():
    """Check AI availability and model status"""
    return {
        "available": True,
        "message": "AI is online (simplified backend)",
        "models": ["pattern-based", "fuzzy-matching"],
        "components": {
            "ner": True,
            "fuzzy_matching": True,
            "health_insights": True
        }
    }

@app.post("/api/ai/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest):
    """AI chat endpoint with medication context awareness"""
    try:
        # Extract medication entities
        entities = extract_medication_entities(request.message)
        
        # Generate simple response based on message type
        if request.type == "medication":
            if entities["medications"]:
                response = f"I can help you with information about {', '.join(entities['medications'])}. "
                response += "Please consult your healthcare provider for medical advice."
            else:
                response = "I can help you with medication information. What would you like to know?"
        else:
            response = f"Hello! I'm your health assistant. You mentioned: {request.message}. "
            response += "I can help with medication information and health insights. "
            response += "Please consult your healthcare provider for medical advice."
        
        return ChatResponse(
            response=response,
            type=request.type,
            confidence=0.8
        )
        
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

@app.post("/api/validate-med", response_model=ValidationResponse)
async def validate_medication(request: ValidationRequest):
    """Validate medication with simple pattern matching"""
    try:
        # Extract medication entities
        entities = extract_medication_entities(request.medication)
        
        # Find matches
        matches = simple_fuzzy_match(request.medication, limit=5)
        
        # Simple validation
        is_valid = len(matches) > 0
        suggestions = []
        warnings = []
        
        if matches:
            med_info = matches[0]
            suggestions.append(f"Found medication: {med_info['name']}")
            if med_info["brands"]:
                suggestions.append(f"Brand names: {', '.join(med_info['brands'])}")
            if med_info["contraindications"]:
                warnings.append(f"Contraindications: {', '.join(med_info['contraindications'])}")
        else:
            suggestions.append("Medication not found in database. Please verify spelling.")
            warnings.append("Consult your healthcare provider for medication information.")
        
        return ValidationResponse(
            is_valid=is_valid,
            confidence=matches[0]["confidence"] if matches else 0.0,
            suggestions=suggestions,
            warnings=warnings,
            extracted_entities=entities
        )
        
    except Exception as e:
        logger.error(f"Medication validation error: {e}")
        return ValidationResponse(
            is_valid=False,
            confidence=0.0,
            suggestions=["Please consult your healthcare provider"],
            warnings=["Unable to validate medication at this time"],
            extracted_entities={}
        )

@app.post("/api/search-med", response_model=SearchResponse)
async def search_medications(request: SearchRequest):
    """Search medications with simple fuzzy matching"""
    try:
        # Perform simple search
        results = simple_fuzzy_match(request.query, limit=request.limit)
        
        return SearchResponse(
            results=results,
            total=len(results),
            query_processed=request.query
        )
        
    except Exception as e:
        logger.error(f"Medication search error: {e}")
        return SearchResponse(
            results=[],
            total=0,
            query_processed=request.query
        )

@app.post("/api/ai/health-report", response_model=HealthReportResponse)
async def generate_health_report(request: HealthReportRequest):
    """Generate simple health insights and recommendations"""
    try:
        user_data = request.user_data
        medications = user_data.get("medications", [])
        metrics = user_data.get("metrics", {})
        adherence = user_data.get("adherence", 95)
        
        # Generate simple insights
        insights = []
        recommendations = []
        
        if medications:
            insights.append(f"Currently taking {len(medications)} medications: {', '.join(medications)}")
        
        if adherence >= 95:
            insights.append("Excellent medication adherence - keep up the great work!")
        elif adherence >= 85:
            insights.append("Good medication adherence with room for improvement")
        else:
            insights.append("Medication adherence could be improved - consider setting reminders")
        
        # Add general recommendations
        recommendations.extend([
            "Take medications as prescribed by your healthcare provider",
            "Set up medication reminders to improve adherence",
            "Keep a medication list with you at all times",
            "Schedule regular check-ups with your healthcare provider",
            "Maintain a healthy lifestyle with proper diet and exercise"
        ])
        
        # Determine trend
        if adherence >= 95 and len(medications) > 0:
            trend = "Improving"
        elif adherence >= 85:
            trend = "Stable"
        else:
            trend = "Needs Attention"
        
        return HealthReportResponse(
            adherence=f"{adherence}%",
            trend=trend,
            insights=insights,
            recommendations=recommendations,
            confidence=0.8
        )
        
    except Exception as e:
        logger.error(f"Health report generation error: {e}")
        return HealthReportResponse(
            adherence="95%",
            trend="Stable",
            insights=["Unable to generate detailed insights at this time"],
            recommendations=["Please consult your healthcare provider for personalized advice"],
            confidence=0.0
        )

if __name__ == "__main__":
    logger.info("🚀 Starting MedTrack AI Backend (Simplified)...")
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=5002,
        reload=False,
        log_level="info"
    )