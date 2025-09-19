from fastapi import FastAPI, HTTPException, Cookie, Request
from fastapi.responses import StreamingResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
from jose import JWTError, jwt
import json

import asyncio
import logging
import sys
import time
import uuid
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from backend.graph import (
    enrich_cell_with_graph, 
    enrich_medical_field,
    LLMProvider, 
    VertexAIProvider,
    MedicalEnrichmentContext
)
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
APP_URL = os.getenv("VITE_APP_URL")
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")

JWT_SECRET = os.getenv('JWT_SECRET')

app = FastAPI(
    title="Data Enrichment API",
    description="API for enriching spreadsheet data using Tavily and AI models",
    version="1.0.0"
)

# Configure CORS
# Allow multiple development ports for flexibility
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174", 
    "http://localhost:3000",  # common React dev port
]

# Add APP_URL if it exists and is different
if APP_URL and APP_URL not in allowed_origins:
    allowed_origins.append(APP_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients at module level
tavily_client = None
vertex_provider = None

def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking"""
    return str(uuid.uuid4())

def parse_authorization_header(auth_header: Optional[str]) -> Optional[str]:
    """
    Parse Authorization header and extract API key.
    Supports both 'Bearer <key>' and raw key formats.
    """
    if not auth_header:
        return None
    
    auth_header = auth_header.strip()
    
    # Handle 'Bearer <key>' format
    if auth_header.lower().startswith('bearer '):
        return auth_header[7:].strip()
    
    # Handle raw key
    return auth_header

def init_clients(tavily_api_key: str):
    """Initialize all clients that have valid API keys"""
    global tavily_client, vertex_provider
    
    if not tavily_api_key:
        raise ValueError("Tavily API key is required")
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=tavily_api_key)
        
    # Vertex AI provider (uses service account credentials)
    try:
        if VERTEX_AI_PROJECT_ID:
            vertex_provider = VertexAIProvider(project_id=VERTEX_AI_PROJECT_ID)
            logger.info("Successfully initialized Vertex AI provider")
        else:
            logger.error("VERTEX_AI_PROJECT_ID not found")
            vertex_provider = None
    except Exception as e:
        logger.error(f"Failed to initialize vertex provider: {e}")
        vertex_provider = None

def get_llm_provider(provider_name: str, tavily_api_key: str) -> LLMProvider:
    """Get the requested LLM provider or raise an error if not available"""
    init_clients(tavily_api_key)
    
    if provider_name == "vertex" and vertex_provider:
        return vertex_provider
    else:
        available_providers = []
        if vertex_provider: 
            available_providers.append("vertex")
        
        if not available_providers:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No LLM providers available",
                    "message": "VERTEX_AI_PROJECT_ID is required for Vertex AI provider. Please configure your environment variables.",
                    "required_env": "VERTEX_AI_PROJECT_ID"
                }
            )
        elif provider_name not in ["vertex"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Invalid provider: {provider_name}",
                    "message": f"Available providers: {', '.join(available_providers)}",
                    "available_providers": available_providers
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"Provider {provider_name} not configured",
                    "message": f"Provider {provider_name} requires additional configuration. Available providers: {', '.join(available_providers)}",
                    "available_providers": available_providers
                }
            )

class WorkerPool:
    """Bounded worker pool for processing tasks with rate limiting"""
    
    def __init__(self, max_workers: int = 2):  # Reduced from 8 to 2
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch(self, tasks, worker_func):
        """Process a batch of tasks with controlled concurrency"""
        results = [None] * len(tasks)
        
        async def worker(task_idx, task_data):
            async with self.semaphore:
                try:
                    result = await worker_func(task_data)
                    results[task_idx] = result
                except Exception as e:
                    logger.error(f"Worker error for task {task_idx}: {str(e)}")
                    results[task_idx] = {"error": str(e)}
        
        # Create worker tasks
        worker_tasks = [
            worker(idx, task_data) 
            for idx, task_data in enumerate(tasks)
        ]
        
        # Wait for all workers to complete
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        return results

# Global worker pool
worker_pool = WorkerPool(max_workers=2)  # Reduced to protect Tavily credits

class SearchResult(BaseModel):
    title: str
    url: str

class MedicalEnrichmentRequest(BaseModel):
    """Request model for medical professional data enrichment"""
    name: str
    target_field: str  # Field to enrich: email, specialty, credentials, etc.
    hospital_name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None

class MedicalBatchEnrichmentRequest(BaseModel):
    """Request model for batch medical professional enrichment"""
    surgeons: List[Dict[str, str]]  # List of surgeon data
    target_fields: List[str]  # Fields to enrich for each surgeon

class MedicalEnrichmentResponse(BaseModel):
    """Response model for medical enrichment"""
    name: str
    enriched_data: Dict[str, str]  # field_name -> enriched_value
    sources: List[SearchResult] = []
    enrichment_status: List[str] = []
    credits_used: Optional[int] = None  # Track Tavily credits used
    search_strategy: Optional[str] = None  # Track search strategy used
    status: str
    error: Optional[str] = None
    trace_id: Optional[str] = None

class EnrichmentRequest(BaseModel):
    column_name: str
    target_value: str
    context_values: Dict[str, str]
    answer: str = None
    search_result: str = None

class BatchEnrichmentRequest(BaseModel):
    column_name: str
    rows: List[str]  # List of target values to enrich
    context_values: Dict[str, str]
    answer: str = None
    search_result: str = None

class TableData(BaseModel):
    rows: List[str]  # List of target values to enrich
    context_values: Dict[str, str]
    answer: str = None
    search_result: str = None

class TableEnrichmentRequest(BaseModel):
    data: Dict[str, TableData]

class EnrichmentResponse(BaseModel):
    enriched_value: str
    status: str
    error: Optional[str] = None
    sources: List[SearchResult] = []
    trace_id: Optional[str] = None

class BatchEnrichmentResponse(BaseModel):
    enriched_values: List[str]
    status: str
    error: Optional[str] = None
    sources: List[List[SearchResult]] = []
    trace_id: Optional[str] = None

class TableEnrichmentColumnResponse(BaseModel):
    enriched_values: List[str]
    sources: List[List[SearchResult]] = []

class EnrichTableResponse(BaseModel):
    enriched_values: Dict[str, List[str]]
    sources: Dict[str, List[List[SearchResult]]] = {}
    status: str
    error: Optional[str] = None
    trace_id: Optional[str] = None


@app.post("/api/enrich-medical", response_model=MedicalEnrichmentResponse)
async def enrich_medical_data(
    request: MedicalEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "vertex"  # Default to vertex for medical data
):
    """Enrich a single field for a medical professional."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            logger.warning(f"[{trace_id}] Missing Authorization header")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization header",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )
        
        # Early exit for blank input
        if not request.name or not request.name.strip():
            return MedicalEnrichmentResponse(
                name=request.name or "",
                enriched_data={request.target_field: ""},
                status="success",
                trace_id=trace_id
            )
        
        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"[{trace_id}] Processing medical enrichment for {request.name}, field: {request.target_field}")

        enrich_start_time = time.time()
        result = await enrich_medical_field(
            name=request.name,
            target_field=request.target_field,
            hospital_name=request.hospital_name,
            address=request.address,
            phone=request.phone,
            tavily_client=tavily_client,
            llm_provider=llm_provider
        )
        enrich_time = time.time() - enrich_start_time
        
        total_time = time.time() - start_time
        logger.info(f"[{trace_id}] Medical enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s)")
        
        # Extract sources from the result
        sources = []
        if result.sources:
            for source in result.sources:
                if isinstance(source, dict):
                    sources.append(SearchResult(
                        title=source.get("title", "Unknown"),
                        url=source.get("url", "")
                    ))
                else:
                    sources.append(SearchResult(title="Source", url=str(source)))
        
        return MedicalEnrichmentResponse(
            name=request.name,
            enriched_data={request.target_field: result.answer or "Information not found"},
            sources=sources,
            enrichment_status=result.enrichment_status,
            credits_used=result.credits_used,
            search_strategy=result.search_strategy,
            status="success",
            trace_id=trace_id
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
        
    except ValueError as e:
        logger.error(f"[{trace_id}] Provider configuration error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Provider configuration error",
                "trace_id": trace_id,
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error in medical enrichment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )

@app.post("/api/enrich-medical/batch")
async def enrich_medical_batch(
    request: MedicalBatchEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "vertex"
):
    """Enrich multiple fields for multiple medical professionals using controlled concurrency."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization header",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )
        
        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"[{trace_id}] Starting batch medical enrichment for {len(request.surgeons)} surgeons")

        # Create enrichment tasks
        tasks = []
        task_metadata = []
        
        for surgeon_idx, surgeon_data in enumerate(request.surgeons):
            name = surgeon_data.get("name", "")
            if not name.strip():
                continue
                
            for field in request.target_fields:
                task_data = {
                    "name": name,
                    "target_field": field,
                    "hospital_name": surgeon_data.get("hospital_name"),
                    "address": surgeon_data.get("address"),
                    "phone": surgeon_data.get("phone"),
                    "tavily_client": tavily_client,
                    "llm_provider": llm_provider
                }
                tasks.append(task_data)
                task_metadata.append((surgeon_idx, name, field))

        # Worker function for processing individual enrichments
        async def enrich_worker(task_data):
            return await enrich_medical_field(**task_data)

        # Execute enrichments with controlled concurrency
        enrich_start_time = time.time()
        results = await worker_pool.process_batch(tasks, enrich_worker)
        enrich_time = time.time() - enrich_start_time
        
        # Organize results by surgeon
        surgeon_results = {}
        total_credits_used = 0
        
        for (surgeon_idx, name, field), result in zip(task_metadata, results):
            if name not in surgeon_results:
                surgeon_results[name] = {
                    "name": name,
                    "enriched_data": {},
                    "sources": [],
                    "enrichment_status": [],
                    "credits_used": 0,
                    "status": "success"
                }
            
            if isinstance(result, dict) and "error" in result:
                surgeon_results[name]["enriched_data"][field] = "Error during enrichment"
                surgeon_results[name]["status"] = "partial_error"
                surgeon_results[name]["enrichment_status"].append(f"error:{field}:{result['error']}")
            else:
                surgeon_results[name]["enriched_data"][field] = result.answer or "Information not found"
                surgeon_results[name]["enrichment_status"].extend(result.enrichment_status)
                
                # Track credits used
                field_credits = result.credits_used or 0
                surgeon_results[name]["credits_used"] += field_credits
                total_credits_used += field_credits
                
                # Add sources from this enrichment
                if result.sources:
                    for source in result.sources:
                        if isinstance(source, dict):
                            surgeon_results[name]["sources"].append(SearchResult(
                                title=source.get("title", "Unknown"),
                                url=source.get("url", "")
                            ))

        total_time = time.time() - start_time
        avg_time_per_task = enrich_time / len(tasks) if tasks else 0
        logger.info(f"[{trace_id}] Batch medical enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s)")

        return {
            "results": list(surgeon_results.values()),
            "summary": {
                "total_surgeons": len(request.surgeons),
                "total_enrichments": len(tasks),
                "success_count": sum(1 for r in surgeon_results.values() if r["status"] == "success"),
                "partial_error_count": sum(1 for r in surgeon_results.values() if r["status"] == "partial_error"),
                "total_time": total_time,
                "avg_time_per_enrichment": avg_time_per_task,
                "total_credits_used": total_credits_used,
                "avg_credits_per_enrichment": total_credits_used / len(tasks) if tasks else 0
            },
            "status": "success",
            "trace_id": trace_id
        }
        
    except HTTPException:
        raise
        
    except ValueError as e:
        logger.error(f"[{trace_id}] Provider configuration error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Provider configuration error",
                "trace_id": trace_id,
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error in batch medical enrichment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during batch enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )

class StreamRequest(BaseModel):
    surgeons: List[Dict]
    target_fields: List[str]
    provider: str = "vertex"

@app.post("/api/enrich-medical/stream")
async def stream_medical_enrichment(
    request_data: StreamRequest,
    fastapi_request: Request
):
    """Stream medical enrichment results in real-time using Server-Sent Events."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization from header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )
        
        # Get data from request body
        surgeons_list = request_data.surgeons
        fields_list = request_data.target_fields
        provider = request_data.provider
        
        llm_provider = get_llm_provider(provider, api_key)
        
        async def generate_stream():
            """Generate SSE stream of enrichment results."""
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'trace_id': trace_id, 'timestamp': time.time()})}\n\n"
            
            total_enrichments = len(surgeons_list) * len(fields_list)
            completed = 0
            total_credits = 0
            
            # Process each surgeon
            for surgeon_idx, surgeon in enumerate(surgeons_list):
                surgeon_name = surgeon.get('name', '')
                if not surgeon_name.strip():
                    continue
                    
                # Send surgeon start event
                yield f"data: {json.dumps({'type': 'surgeon_start', 'surgeon_idx': surgeon_idx, 'name': surgeon_name, 'timestamp': time.time()})}\n\n"
                
                surgeon_results = {
                    "name": surgeon_name,
                    "enriched_data": {},
                    "sources": [],
                    "enrichment_status": [],
                    "credits_used": 0
                }
                
                # Process each field for this surgeon
                for field_idx, field in enumerate(fields_list):
                    try:
                        # Send field start event
                        yield f"data: {json.dumps({'type': 'field_start', 'surgeon_idx': surgeon_idx, 'field': field, 'timestamp': time.time()})}\n\n"
                        
                        result = await enrich_medical_field(
                            name=surgeon_name,
                            target_field=field,
                            hospital_name=surgeon.get('hospital_name'),
                            address=surgeon.get('address'),
                            phone=surgeon.get('phone'),
                            tavily_client=tavily_client,
                            llm_provider=llm_provider
                        )
                        
                        # Update results
                        surgeon_results["enriched_data"][field] = result.answer or "Information not found"
                        surgeon_results["credits_used"] += result.credits_used or 0
                        surgeon_results["enrichment_status"].extend(result.enrichment_status)
                        
                        # Add sources
                        if result.sources:
                            for source in result.sources:
                                if isinstance(source, dict):
                                    surgeon_results["sources"].append({
                                        "title": source.get("title", "Unknown"),
                                        "url": source.get("url", "")
                                    })
                        
                        total_credits += result.credits_used or 0
                        completed += 1
                        
                        # Send field complete event with result
                        field_result = {
                            "type": "field_complete",
                            "surgeon_idx": surgeon_idx,
                            "field": field,
                            "value": result.answer or "Information not found",
                            "credits_used": result.credits_used or 0,
                            "search_strategy": result.search_strategy,
                            "sources": surgeon_results["sources"][-len(result.sources):] if result.sources else [],
                            "progress": completed / total_enrichments,
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(field_result)}\n\n"
                        
                    except Exception as e:
                        logger.error(f"Error enriching {field} for {surgeon_name}: {str(e)}")
                        
                        # Send error event
                        error_event = {
                            "type": "field_error",
                            "surgeon_idx": surgeon_idx,
                            "field": field,
                            "error": str(e),
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(error_event)}\n\n"
                
                # Send surgeon complete event
                surgeon_complete = {
                    "type": "surgeon_complete",
                    "surgeon_idx": surgeon_idx,
                    "name": surgeon_name,
                    "enriched_data": surgeon_results["enriched_data"],
                    "credits_used": surgeon_results["credits_used"],
                    "timestamp": time.time()
                }
                yield f"data: {json.dumps(surgeon_complete)}\n\n"
            
            # Send final completion event
            completion_event = {
                "type": "complete",
                "total_surgeons": len(surgeons_list),
                "total_enrichments": completed,
                "total_credits_used": total_credits,
                "total_time": time.time() - start_time,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(completion_event)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error in streaming medical enrichment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during streaming enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )

async def verify_jwt(jwt_token: str = Cookie(None)):  # Renamed to avoid conflicts
    try:
        decoded = jwt.decode(jwt_token, JWT_SECRET, algorithms=["HS256"])
        return JSONResponse(content={"success": True, "data": decoded['apiKey']})
    except JWTError:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {str(e)}")
    
@app.post("/api/enrich", response_model=EnrichmentResponse)
async def enrich_data(
    request: EnrichmentRequest,
    fastapi_request: Request,
    provider: str = "vertex"  # Changed default to vertex for better medical data handling
):
    """Enrich a single cell's data."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization header",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )
        
        # Early exit for blank input
        if not request.target_value or not request.target_value.strip():
            return EnrichmentResponse(
                enriched_value="",
                status="success",
                sources=[],
                trace_id=trace_id
            )
        
        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"[{trace_id}] Processing single enrichment request for column: {request.column_name}")

        enrich_start_time = time.time()
        enriched_value = await enrich_cell_with_graph(
            column_name=request.column_name,
            target_value=request.target_value,
            context_values=request.context_values,
            tavily_client=tavily_client,
            llm_provider=llm_provider
        )
        enrich_time = time.time() - enrich_start_time
        
        total_time = time.time() - start_time
        logger.info(f"[{trace_id}] Enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s)")
        
        # Extract sources from the result
        sources = []
        if isinstance(enriched_value, dict) and 'search_result' in enriched_value:
            search_results = enriched_value['search_result'].get('results', [])
            for result in search_results:
                sources.append(SearchResult(
                    title=result.get('title', 'Unknown'),
                    url=result.get('url', '')
                ))
        
        final_value = enriched_value.get('answer', enriched_value) if isinstance(enriched_value, dict) else str(enriched_value)
        
        return EnrichmentResponse(
            enriched_value=final_value,
            status="success",
            sources=sources,
            trace_id=trace_id
        )
        
    except HTTPException:
        raise
        
    except ValueError as e:
        logger.error(f"[{trace_id}] Provider configuration error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Provider configuration error",
                "trace_id": trace_id,
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error in single enrichment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )

@app.post("/api/enrich/batch", response_model=BatchEnrichmentResponse)
async def enrich_batch(
    request: BatchEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "vertex"  # Changed default to vertex
):
    """Enrich multiple rows with controlled concurrency."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization header",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )

        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"[{trace_id}] Starting batch enrichment for column: {request.column_name}")
        logger.info(f"[{trace_id}] Number of rows to process: {len(request.rows)}")

        # Create tasks only for non-empty rows
        tasks = []
        task_indices = []
        
        for idx, row in enumerate(request.rows):
            if row and row.strip():
                task_data = {
                    "column_name": request.column_name,
                    "target_value": row,
                    "context_values": request.context_values,
                    "tavily_client": tavily_client,
                    "llm_provider": llm_provider
                }
                tasks.append(task_data)
                task_indices.append(idx)

        # Worker function for processing individual enrichments
        async def enrich_worker(task_data):
            return await enrich_cell_with_graph(**task_data)

        # Execute enrichments with controlled concurrency
        enrich_start_time = time.time()
        enriched_results = await worker_pool.process_batch(tasks, enrich_worker)
        enrich_time = time.time() - enrich_start_time
        
        # Process results and maintain original order
        final_values = [""] * len(request.rows)  # Initialize with empty strings
        all_sources = [[] for _ in request.rows]  # Initialize with empty lists
        
        # Fill in results for processed rows
        for task_idx, row_idx in enumerate(task_indices):
            result = enriched_results[task_idx]
            sources = []
            
            if isinstance(result, dict) and "error" in result:
                final_values[row_idx] = "Error during enrichment"
            elif isinstance(result, dict):
                # Extract value
                final_values[row_idx] = result.get('answer', str(result))
                
                # Extract sources
                if result.get('search_result', {}).get('results'):
                    for search_result in result['search_result']['results']:
                        sources.append(SearchResult(
                            title=search_result.get('title', 'Unknown'),
                            url=search_result.get('url', '')
                        ))
            else:
                final_values[row_idx] = str(result)
            
            all_sources[row_idx] = sources
        
        total_time = time.time() - start_time
        avg_time_per_row = enrich_time / len(tasks) if tasks else 0
        logger.info(f"[{trace_id}] Batch enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s)")
        
        return BatchEnrichmentResponse(
            enriched_values=final_values,
            status="success",
            sources=all_sources,
            trace_id=trace_id
        )
    
    except HTTPException:
        raise
        
    except ValueError as e:
        logger.error(f"[{trace_id}] Provider configuration error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Provider configuration error",
                "trace_id": trace_id,
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error in batch enrichment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during batch enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )

@app.post("/api/enrich-table", response_model=EnrichTableResponse)
async def enrich_table(
    request: TableEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "vertex"  # Changed default to vertex
):
    """Enrich the entire table (all columns) with controlled concurrency."""
    trace_id = generate_trace_id()
    start_time = time.time()
    
    try:
        # Parse Authorization header
        auth_header = fastapi_request.headers.get("Authorization")
        api_key = parse_authorization_header(auth_header)
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Missing Authorization header",
                    "trace_id": trace_id,
                    "message": "Please provide Tavily API key in Authorization header"
                }
            )
        
        llm_provider = get_llm_provider(provider, api_key)

        logger.info(f"[{trace_id}] Starting table enrichment for {len(request.data)} columns")

        # Prepare enrichment tasks for all non-empty cells
        tasks = []
        task_metadata = []  # Track which column and row index each task belongs to
        
        for column_name, table_data in request.data.items():
            for row_idx, row_value in enumerate(table_data.rows):
                if row_value and row_value.strip():
                    task_data = {
                        "column_name": column_name,
                        "target_value": row_value,
                        "context_values": table_data.context_values,
                        "tavily_client": tavily_client,
                        "llm_provider": llm_provider
                    }
                    tasks.append(task_data)
                    task_metadata.append((column_name, row_idx))

        # Worker function for processing individual enrichments
        async def enrich_worker(task_data):
            return await enrich_cell_with_graph(**task_data)

        # Execute enrichments with controlled concurrency
        enrich_start_time = time.time()
        enriched_results = await worker_pool.process_batch(tasks, enrich_worker)
        enrich_time = time.time() - enrich_start_time

        # Initialize response containers
        enriched_table = {col: [""] * len(data.rows) for col, data in request.data.items()}
        sources_table = {col: [[] for _ in data.rows] for col, data in request.data.items()}

        # Fill in results
        for task_idx, (column_name, row_idx) in enumerate(task_metadata):
            result = enriched_results[task_idx]
            
            if isinstance(result, dict) and "error" in result:
                enriched_table[column_name][row_idx] = "Error during enrichment"
                sources_table[column_name][row_idx] = []
            elif isinstance(result, dict):
                # Extract value
                enriched_table[column_name][row_idx] = result.get('answer', str(result))
                
                # Extract sources
                sources = []
                if result.get('search_result', {}).get('results'):
                    for search_result in result['search_result']['results']:
                        sources.append(SearchResult(
                            title=search_result.get('title', 'Unknown'),
                            url=search_result.get('url', '')
                        ))
                sources_table[column_name][row_idx] = sources
            else:
                enriched_table[column_name][row_idx] = str(result)
                sources_table[column_name][row_idx] = []

        total_time = time.time() - start_time
        avg_time_per_cell = enrich_time / len(tasks) if tasks else 0
        logger.info(f"[{trace_id}] Table enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s)")

        return EnrichTableResponse(
            enriched_values=enriched_table,
            sources=sources_table,
            status="success",
            trace_id=trace_id
        )

    except HTTPException:
        raise

    except ValueError as e:
        logger.error(f"[{trace_id}] Provider configuration error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Provider configuration error",
                "trace_id": trace_id,
                "message": str(e)
            }
        )

    except Exception as e:
        logger.error(f"[{trace_id}] Error enriching table: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during table enrichment",
                "trace_id": trace_id,
                "message": "An unexpected error occurred"
            }
        )


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
