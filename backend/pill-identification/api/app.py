"""
FastAPI microservice for pill identification
Runs on localhost:8005
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
import traceback
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..inference import PillIdentifier

# Initialize FastAPI app
app = FastAPI(
    title="Pill Identification API",
    description="ePillID-based pill identification service",
    version="1.0.0"
)

# Global identifier instance
identifier: Optional[PillIdentifier] = None


class IdentifyRequest(BaseModel):
    """Request model for /identify endpoint"""
    image: str  # Base64-encoded image
    k: Optional[int] = 5  # Number of results
    min_confidence: Optional[float] = 0.0  # Minimum confidence threshold


class EmbedRequest(BaseModel):
    """Request model for /embed endpoint"""
    image: str  # Base64-encoded image


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


def initialize_identifier():
    """Initialize the pill identifier with model and index (auto-detects if not provided)"""
    global identifier
    
    if identifier is not None:
        return identifier
    
    try:
        # Get paths from environment variables or use defaults
        model_path = os.getenv('PILL_MODEL_PATH', None)
        index_path = os.getenv('PILL_INDEX_PATH', None)
        metadata_path = os.getenv('PILL_METADATA_PATH', None)
        network = os.getenv('PILL_NETWORK', 'resnet18')
        embedding_dim = int(os.getenv('PILL_EMBEDDING_DIM', '2048'))
        device = os.getenv('PILL_DEVICE', None)
        metric = os.getenv('PILL_METRIC', 'cosine')
        auto_detect = os.getenv('PILL_AUTO_DETECT', 'true').lower() == 'true'
        
        # Resolve relative paths if provided
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if index_path and not os.path.isabs(index_path):
            index_path = os.path.join(base_dir, index_path)
        if metadata_path and not os.path.isabs(metadata_path):
            metadata_path = os.path.join(base_dir, metadata_path)
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)
        
        identifier = PillIdentifier(
            model_path=model_path,
            index_path=index_path,
            metadata_path=metadata_path,
            network=network,
            embedding_dim=embedding_dim,
            device=device,
            metric=metric,
            auto_detect=auto_detect
        )
        
        return identifier
    except Exception as e:
        print(f"Error initializing identifier: {e}")
        traceback.print_exc()
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        initialize_identifier()
    except Exception as e:
        print(f"Warning: Could not initialize identifier on startup: {e}")
        print("The service will attempt to initialize on first request.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Pill Identification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "identify": "POST /identify",
            "embed": "POST /embed",
            "health": "GET /health",
            "info": "GET /info"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        initialize_identifier()
        if identifier is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Identifier not initialized"}
            )
        
        info = identifier.get_model_info()
        return {
            "status": "healthy",
            "model_loaded": True,
            "index_loaded": info.get('has_index', False),
            "device": info.get('device', 'unknown')
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/info")
async def info():
    """Get model and index information"""
    try:
        initialize_identifier()
        if identifier is None:
            raise HTTPException(status_code=503, detail="Identifier not initialized")
        
        return identifier.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify")
async def identify(request: IdentifyRequest):
    """
    Identify a pill from a base64-encoded image.
    
    Returns top matches with confidence scores and metadata.
    """
    try:
        initialize_identifier()
        if identifier is None:
            raise HTTPException(
                status_code=503,
                detail="Pill identifier not initialized. Check model and index paths."
            )
        
        # Validate image
        if not request.image:
            raise HTTPException(status_code=400, detail="Image is required")
        
        # Perform identification
        result = identifier.identify(
            image_input=request.image,
            k=request.k or 5,
            min_confidence=request.min_confidence or 0.0
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/embed")
async def embed(request: EmbedRequest):
    """
    Generate embedding vector for an image.
    
    Returns the embedding vector as a list.
    """
    try:
        initialize_identifier()
        if identifier is None:
            raise HTTPException(
                status_code=503,
                detail="Pill identifier not initialized. Check model path."
            )
        
        # Validate image
        if not request.image:
            raise HTTPException(status_code=400, detail="Image is required")
        
        # Generate embedding
        embedding = identifier.embed(request.image)
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/identify/file")
async def identify_file(file: UploadFile = File(...), k: int = Form(5), min_confidence: float = Form(0.0)):
    """
    Identify a pill from an uploaded image file.
    """
    try:
        initialize_identifier()
        if identifier is None:
            raise HTTPException(
                status_code=503,
                detail="Pill identifier not initialized. Check model and index paths."
            )
        
        # Read file
        image_bytes = await file.read()
        
        # Perform identification
        result = identifier.identify(
            image_input=image_bytes,
            k=k,
            min_confidence=min_confidence
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/embed/file")
async def embed_file(file: UploadFile = File(...)):
    """
    Generate embedding for an uploaded image file.
    """
    try:
        initialize_identifier()
        if identifier is None:
            raise HTTPException(
                status_code=503,
                detail="Pill identifier not initialized. Check model path."
            )
        
        # Read file
        image_bytes = await file.read()
        
        # Generate embedding
        embedding = identifier.embed(image_bytes)
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005, reload=True)

