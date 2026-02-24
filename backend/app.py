from fastapi import FastAPI, HTTPException, Cookie, Request
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
from jose import JWTError, jwt

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd
import io

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import AsyncOpenAI

from google.generativeai import GenerativeModel
import time
from graph import enrich_cell_with_graph, LLMProvider, OpenAIProvider, GeminiProvider
from utils import count_enrichable_items, check_api_key, TableData
import requests

# Import refactored streaming function
# Use relative import since we're in the backend directory
try:
    from .streaming import enrich_table_with_research_streaming
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from streaming import enrich_table_with_research_streaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
APP_URL = os.getenv("VITE_APP_URL")

JWT_SECRET = os.getenv('JWT_SECRET')

app = FastAPI(
    title="Data Enrichment API",
    description="API for enriching spreadsheet data using Tavily and AI models",
    version="1.0.0"
)

print(f"APP_URL: {APP_URL}")    

# Build allowed origins list
# Add your production domain here when deploying
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
if APP_URL:
    allowed_origins.append(APP_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients at module level (similar to enrich.py)
tavily_client = None
openai_provider = None
gemini_provider = None

def init_clients(tavily_api_key: str):
    """Initialize all clients that have valid API keys"""
    global tavily_client, openai_provider, gemini_provider
    
    # Only initialize once
    if tavily_client is None:
        # Tavily client (required for all operations)
        if not tavily_api_key:
            raise ValueError("Tavily API key is required")
        tavily_client = TavilyClient(api_key=tavily_api_key)
        # OpenAI provider (optional)
        if openai_api_key:
            openai_client = AsyncOpenAI(api_key=openai_api_key)
            openai_provider = OpenAIProvider(openai_client, model="gpt-4o")
        
        # Gemini provider (optional)
        if gemini_api_key:
            gemini_model = GenerativeModel(model_name="gemini-1.5-flash")
            gemini_provider = GeminiProvider(gemini_model)

def get_llm_provider(provider_name: str, tavily_api_key: str) -> LLMProvider:
    """Get the requested LLM provider or raise an error if not available"""
    init_clients(tavily_api_key)
    
    if provider_name == "openai" and openai_provider:
        return openai_provider
    elif provider_name == "gemini" and gemini_provider:
        return gemini_provider
    else:
        available_providers = []
        if openai_provider: available_providers.append("openai") 
        if gemini_provider: available_providers.append("gemini")
        
        if not available_providers:
            raise ValueError("No LLM providers are available. Please configure at least one provider.")
        elif provider_name not in ["openai", "gemini"]:
            raise ValueError(f"Invalid provider name: {provider_name}. Available options: openai, gemini")
        else:
            raise ValueError(f"Provider {provider_name} is missing an API key. Available providers with API keys: {', '.join(available_providers)}")

class SearchResult(BaseModel):
    title: str
    url: str
    favicon: Optional[str] = None

class EnrichTableResponse(BaseModel):
    enriched_values: Dict[str, List[str]]
    sources: Dict[str, List[List[SearchResult]]] = {}
    status: str
    error: Optional[str] = None

class TableWideEnrichmentRequest(BaseModel):
    """Request to enrich entire table with a single research query"""
    headers: List[str]  # Column headers
    rows: List[List[str]]  # Table rows (each row is a list of cell values)
    context_column: Optional[str] = None  # Optional: column to use as context for enrichment
    target_column: Optional[str] = None  # Optional: if specified, only enrich this column (ignores other empty columns)


@app.get("/api/verify-jwt")
async def verify_jwt(jwt_token: str = Cookie(None)):  # Renamed to avoid conflicts
    if not jwt_token:
        raise HTTPException(status_code=401, detail="Unauthorized: No token provided")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server error: JWT_SECRET not configured")
    try:
        decoded = jwt.decode(jwt_token, JWT_SECRET, algorithms=["HS256"])
        return JSONResponse(content={"success": True, "data": decoded['apiKey']})
    except JWTError:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {str(e)}")


@app.post("/api/enrich/table-wide/stream")
async def enrich_table_wide_stream(
    request: TableWideEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "openai"
):
    """
    Stream research progress and results for table-wide enrichment.
    Returns Server-Sent Events (SSE) with real-time updates.
    """
    try:
        api_key = fastapi_request.headers.get("Authorization")
        init_clients(api_key)
        llm_provider = get_llm_provider(provider, api_key)

        logger.info(f"Starting streaming table-wide enrichment for {len(request.rows)} rows")
        
        return StreamingResponse(
            enrich_table_with_research_streaming(
                headers=request.headers,
                rows=request.rows,
                context_column=request.context_column,
                target_column=request.target_column,
                tavily_client=tavily_client,
                llm_provider=llm_provider,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )

    except Exception as e:
        logger.error(f"Error starting streaming enrichment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        async def error_stream():
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

