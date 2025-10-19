#!/usr/bin/env python3
"""
MedTrack AI Backend
Local, privacy-compliant AI stack for medication validation and health insights
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# AI Components
from ai_models import AIModelManager
from ner_pipeline import MedicationNERPipeline
from fuzzy_matcher import MedicationFuzzyMatcher
from health_insights import HealthInsightsGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AI components
ai_models = None
ner_pipeline = None
fuzzy_matcher = None
health_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AI models on startup"""
    global ai_models, ner_pipeline, fuzzy_matcher, health_generator
    
    logger.info("🚀 Initializing MedTrack AI Backend...")
    
    try:
        # Initialize AI model manager
        ai_models = AIModelManager()
        await ai_models.load_models()
        logger.info("✅ AI Models loaded successfully")
        
        # Initialize NER pipeline
        ner_pipeline = MedicationNERPipeline()
        await ner_pipeline.initialize()
        logger.info("✅ NER Pipeline initialized")
        
        # Initialize fuzzy matcher
        fuzzy_matcher = MedicationFuzzyMatcher()
        await fuzzy_matcher.load_medication_database()
        logger.info("✅ Fuzzy Matcher initialized")
        
        # Initialize health insights generator
        health_generator = HealthInsightsGenerator(ai_models)
        logger.info("✅ Health Insights Generator initialized")
        
        logger.info("🎉 MedTrack AI Backend ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI backend: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down AI backend...")

# Create FastAPI app
app = FastAPI(
    title="MedTrack AI Backend",
    description="Local, privacy-compliant AI for medication validation and health insights",
    version="1.0.0",
    lifespan=lifespan
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
    confidence: float = 0.0

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

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_available": ai_models is not None and ai_models.is_ready(),
        "components": {
            "ai_models": ai_models is not None,
            "ner_pipeline": ner_pipeline is not None,
            "fuzzy_matcher": fuzzy_matcher is not None,
            "health_generator": health_generator is not None
        }
    }

@app.get("/api/ai/status")
async def ai_status():
    """Check AI availability and model status"""
    if not ai_models or not ai_models.is_ready():
        return {
            "available": False,
            "message": "AI models not loaded",
            "models": []
        }
    
    return {
        "available": True,
        "message": "AI is online and ready",
        "models": ai_models.get_loaded_models(),
        "components": {
            "ner": ner_pipeline is not None,
            "fuzzy_matching": fuzzy_matcher is not None,
            "health_insights": health_generator is not None
        }
    }

@app.post("/api/ai/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest):
    """AI chat endpoint with medication context awareness"""
    try:
        if not ai_models or not ai_models.is_ready():
            raise HTTPException(status_code=503, detail="AI models not available")
        
        # Extract medication entities from user message
        entities = {}
        if ner_pipeline:
            entities = await ner_pipeline.extract_medications(request.message)
        
        # Generate AI response with medication context
        response = await ai_models.generate_response(
            message=request.message,
            context=request.context,
            entities=entities,
            response_type=request.type
        )
        
        return ChatResponse(
            response=response["text"],
            type=request.type,
            confidence=response.get("confidence", 0.8)
        )
        
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

@app.post("/api/validate-med", response_model=ValidationResponse)
async def validate_medication(request: ValidationRequest):
    """Validate medication with NER and fuzzy matching"""
    try:
        if not ner_pipeline or not fuzzy_matcher:
            raise HTTPException(status_code=503, detail="Validation services not available")
        
        # Extract medication entities
        entities = await ner_pipeline.extract_medications(request.medication)
        
        # Fuzzy match against medication database
        matches = await fuzzy_matcher.find_matches(
            query=request.medication,
            limit=5,
            min_confidence=0.6
        )
        
        # Validate dosage and frequency if provided
        dosage_valid = True
        frequency_valid = True
        warnings = []
        suggestions = []
        
        if request.dosage and not await fuzzy_matcher.validate_dosage(request.dosage, entities):
            dosage_valid = False
            warnings.append("Dosage format may be incorrect")
            suggestions.append("Please check dosage format (e.g., '500mg', '2 tablets')")
        
        if request.frequency and not await fuzzy_matcher.validate_frequency(request.frequency):
            frequency_valid = False
            warnings.append("Frequency format may be incorrect")
            suggestions.append("Please check frequency format (e.g., 'twice daily', 'every 8 hours')")
        
        # Check for drug interactions if multiple medications in context
        if len(entities.get("medications", [])) > 1:
            interactions = await fuzzy_matcher.check_interactions(entities["medications"])
            if interactions:
                warnings.extend(interactions)
        
        is_valid = len(matches) > 0 and dosage_valid and frequency_valid
        
        return ValidationResponse(
            is_valid=is_valid,
            confidence=matches[0]["confidence"] if matches else 0.0,
            suggestions=suggestions,
            warnings=warnings,
            extracted_entities=entities
        )
        
    except Exception as e:
        logger.error(f"Medication validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/search-med", response_model=SearchResponse)
async def search_medications(request: SearchRequest):
    """Search medications with fuzzy matching"""
    try:
        if not fuzzy_matcher:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        # Extract entities for better search
        entities = {}
        if ner_pipeline:
            entities = await ner_pipeline.extract_medications(request.query)
        
        # Perform fuzzy search
        results = await fuzzy_matcher.find_matches(
            query=request.query,
            limit=request.limit,
            min_confidence=request.min_confidence
        )
        
        return SearchResponse(
            results=results,
            total=len(results),
            query_processed=request.query
        )
        
    except Exception as e:
        logger.error(f"Medication search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/ai/health-report", response_model=HealthReportResponse)
async def generate_health_report(request: HealthReportRequest):
    """Generate AI-powered health insights and recommendations"""
    try:
        if not health_generator:
            raise HTTPException(status_code=503, detail="Health insights not available")
        
        # Generate comprehensive health report
        report = await health_generator.generate_report(
            user_data=request.user_data,
            report_type=request.report_type
        )
        
        return HealthReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Health report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Health report failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )