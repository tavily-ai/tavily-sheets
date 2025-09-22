from fastapi import FastAPI, HTTPException, Cookie, Request
from fastapi.responses import StreamingResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
import re
from jose import JWTError, jwt
import json
from contextlib import asynccontextmanager

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

# Initialize API keys from environment
APP_URL = os.getenv("VITE_APP_URL")
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")

JWT_SECRET = os.getenv('JWT_SECRET')

# Global client storage with caching
class AppState:
    def __init__(self):
        self.vertex_provider: Optional[VertexAIProvider] = None
        self.query_llm_provider: Optional[VertexAIProvider] = None
        self.tavily_clients: Dict[str, TavilyClient] = {}  # Cache clients by API key hash
        # Track active streaming jobs for potential cancellation/cleanup
        self.active_streams: Dict[str, list] = {}
        
    def get_tavily_client(self, api_key: str) -> TavilyClient:
        """Get cached Tavily client or create new one"""
        if not api_key:
            raise ValueError("Tavily API key is required")
        
        # Use first 8 chars of API key as cache key (for security)
        cache_key = api_key[:8] if len(api_key) >= 8 else api_key
        
        if cache_key not in self.tavily_clients:
            self.tavily_clients[cache_key] = TavilyClient(api_key=api_key)
            logger.info(f"Created new Tavily client for key ending in {cache_key}")
        
        return self.tavily_clients[cache_key]

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients at startup and cleanup at shutdown"""
    logger.info("Starting up application...")
    
    # Initialize both Vertex AI providers at startup
    try:
        if VERTEX_AI_PROJECT_ID:
            # Create main provider
            app_state.vertex_provider = VertexAIProvider(project_id=VERTEX_AI_PROJECT_ID)
            logger.info("Successfully initialized main Vertex AI provider at startup")
            
            # Create lightweight query provider that reuses the client infrastructure
            app_state.query_llm_provider = app_state.vertex_provider.create_lightweight_version("gemini-2.5-flash-lite")
            logger.info("Successfully initialized lightweight query LLM provider at startup")
        else:
            logger.warning("VERTEX_AI_PROJECT_ID not found - Vertex AI providers not available")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI providers at startup: {e}")
    
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="Data Enrichment API",
    description="API for enriching spreadsheet data using Tavily and AI models",
    version="1.0.0",
    lifespan=lifespan
)

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

# ------------------------
# Field canonicalization
# ------------------------
_FIELD_SYNONYMS = {
    "email": {"email", "e_mail", "mail", "emailaddress"},
    "linkedin_url": {"linkedin", "linkedin_url", "linked_in", "linkedinprofile", "linkedinprofileurl"},
    "phone": {"phone", "phone_number", "telephone", "tel"},
    "hospital_name": {"hospital", "hospital_name", "institution", "clinic"},
}

def canonicalize_field(field_name: str) -> str:
    """Normalize incoming field names to canonical snake_case keys."""
    if not isinstance(field_name, str):
        return field_name
    normalized = re.sub(r"[^a-z0-9]+", "_", field_name.strip().lower()).strip("_")
    for canonical, variants in _FIELD_SYNONYMS.items():
        if normalized in variants:
            return canonical
    return normalized

def canonicalize_record_keys(record: Dict) -> Dict:
    """Return a copy of record with canonicalized keys (values preserved)."""
    if not isinstance(record, dict):
        return record
    return {canonicalize_field(k): v for k, v in record.items()}

def init_clients(tavily_api_key: str):
    """Initialize all clients that have valid API keys"""
    # This function is now deprecated - clients are initialized at startup
    # Keeping for backward compatibility but no longer modifying global state
    if not tavily_api_key:
        raise ValueError("Tavily API key is required")

def get_tavily_client(tavily_api_key: str) -> TavilyClient:
    """Get cached Tavily client with the provided API key"""
    return app_state.get_tavily_client(tavily_api_key)

def get_llm_provider(provider_name: str) -> LLMProvider:
    """Get the requested LLM provider or raise an error if not available"""
    if provider_name == "vertex" and app_state.vertex_provider:
        return app_state.vertex_provider
    else:
        available_providers = []
        if app_state.vertex_provider: 
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

def get_query_llm_provider() -> LLMProvider:
    """Get the query LLM provider or raise an error if not available"""
    if app_state.query_llm_provider:
        return app_state.query_llm_provider
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query LLM provider not available",
                "message": "Query LLM provider was not properly initialized at startup"
            }
        )

class SequentialBatchProcessor:
    """
    Sequential batch processor that respects rate limits by processing small batches sequentially.
    
    This approach processes 3-4 cells at a time, then waits appropriately to respect Tavily's
    100 req/min rate limit (with buffer at 80 req/min).
    """
    
    def __init__(self, batch_size: int = 3, requests_per_minute: int = 80):
        """
        Initialize sequential batch processor
        
        Args:
            batch_size: Number of tasks to process concurrently in each batch (3-4 recommended)
            requests_per_minute: Rate limit for API calls (80 req/min with buffer from 100)
        """
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.min_delay_between_batches = 60.0 / requests_per_minute * batch_size  # Time to wait between batches
        
        logger.info(f"Initialized SequentialBatchProcessor: {self.batch_size} tasks per batch, "
                   f"{self.requests_per_minute} req/min limit, {self.min_delay_between_batches:.1f}s between batches")
    
    async def process_tasks_sequentially(self, tasks, worker_func, progress_callback=None):
        """
        Process tasks in sequential batches with rate limiting
        
        Args:
            tasks: List of task data to process
            worker_func: Async function to process each task
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results corresponding to input tasks
        """
        results = [None] * len(tasks)
        total_tasks = len(tasks)
        completed_tasks = 0
        
        logger.info(f"Processing {total_tasks} tasks in batches of {self.batch_size}")
        
        # Process tasks in sequential batches
        for batch_start in range(0, total_tasks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]
            batch_size = len(batch_tasks)
            
            batch_start_time = time.time()
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{(total_tasks + self.batch_size - 1)//self.batch_size} "
                       f"(tasks {batch_start + 1}-{batch_end})")
            
            # Process current batch concurrently (but limited to batch_size)
            batch_results = await self._process_single_batch(batch_tasks, worker_func, batch_start)
            
            # Store results
            for i, result in enumerate(batch_results):
                results[batch_start + i] = result
            
            completed_tasks += batch_size
            batch_duration = time.time() - batch_start_time
            
            # Progress callback
            if progress_callback:
                try:
                    await progress_callback(completed_tasks, total_tasks, batch_duration)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
            
            # Rate limiting: wait before next batch if needed
            if batch_end < total_tasks:  # Not the last batch
                elapsed_time = batch_duration
                required_delay = self.min_delay_between_batches
                
                if elapsed_time < required_delay:
                    wait_time = required_delay - elapsed_time
                    logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next batch")
                    await asyncio.sleep(wait_time)
                else:
                    logger.info(f"Batch took {elapsed_time:.1f}s, no additional delay needed")
        
        logger.info(f"Completed processing {total_tasks} tasks")
        return results
    
    async def _process_single_batch(self, batch_tasks, worker_func, batch_start_idx):
        """Process a single batch of tasks concurrently"""
        batch_results = [None] * len(batch_tasks)
        
        async def batch_worker(local_idx, task_data):
            global_idx = batch_start_idx + local_idx
            try:
                result = await worker_func(task_data)
                batch_results[local_idx] = result
                logger.debug(f"Task {global_idx + 1} completed successfully")
            except Exception as e:
                logger.error(f"Task {global_idx + 1} failed: {str(e)}")
                batch_results[local_idx] = {"error": str(e), "status": "failed"}
        
        # Execute batch tasks concurrently
        batch_workers = [
            batch_worker(idx, task_data) 
            for idx, task_data in enumerate(batch_tasks)
        ]
        
        await asyncio.gather(*batch_workers, return_exceptions=True)
        return batch_results

# Global sequential batch processor - optimized for rate-limited processing
sequential_processor = SequentialBatchProcessor(batch_size=3, requests_per_minute=80)

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
        
        llm_provider = get_llm_provider(provider)
        query_llm_provider = get_query_llm_provider()
        tavily_client = get_tavily_client(api_key)
        
        canonical_field = canonicalize_field(request.target_field)
        logger.info(f"[{trace_id}] Processing medical enrichment for {request.name}, field: {canonical_field}")

        enrich_start_time = time.time()
        result = await enrich_medical_field(
            name=request.name,
            target_field=canonical_field,
            hospital_name=request.hospital_name,
            address=request.address,
            phone=request.phone,
            tavily_client=tavily_client,
            llm_provider=llm_provider,
            query_llm_provider=query_llm_provider
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
            enriched_data={canonical_field: result.answer or "Information not found"},
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
        
        llm_provider = get_llm_provider(provider)
        query_llm_provider = get_query_llm_provider()
        tavily_client = get_tavily_client(api_key)
        
        # Canonicalize fields and surgeon records
        canonical_fields = [canonicalize_field(f) for f in request.target_fields]
        surgeons_canonical = [canonicalize_record_keys(s) for s in request.surgeons]

        logger.info(f"[{trace_id}] Starting sequential batch medical enrichment for {len(surgeons_canonical)} surgeons with {len(canonical_fields)} fields each")

        # Filter to only process empty cells and create enrichment tasks 
        cell_filter = CellEnrichmentFilter()
        tasks = []
        task_metadata = []
        
        for surgeon_idx, surgeon_data in enumerate(surgeons_canonical):
            name = surgeon_data.get("name", "")
            if not name.strip():
                continue
            
            # Get only fields that need enrichment (empty cells)
            fields_to_enrich = cell_filter.get_fields_to_enrich(surgeon_data, canonical_fields)
            
            if not fields_to_enrich:
                logger.info(f"[{trace_id}] Skipping surgeon {name} - all target fields already populated")
                continue
                
            for field in fields_to_enrich:
                task_data = {
                    "name": name,
                    "target_field": field,
                    "hospital_name": surgeon_data.get("hospital_name"),
                    "address": surgeon_data.get("address"),
                    "phone": surgeon_data.get("phone"),
                    "tavily_client": tavily_client,
                    "llm_provider": llm_provider,
                    "query_llm_provider": query_llm_provider
                }
                tasks.append(task_data)
                task_metadata.append((surgeon_idx, name, field))

        # Worker function for processing individual enrichments
        async def enrich_worker(task_data):
            return await enrich_medical_field(**task_data)

        # Execute all enrichments using sequential batch processing with rate limiting
        logger.info(f"[{trace_id}] Processing {len(tasks)} enrichment tasks using sequential batches of {sequential_processor.batch_size}")
        enrich_start_time = time.time()
        results = await sequential_processor.process_tasks_sequentially(tasks, enrich_worker)
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
        total_tasks = len(results)
        logger.info(f"[{trace_id}] Sequential batch medical enrichment completed in {enrich_time:.2f}s (total: {total_time:.2f}s) for {total_tasks} enrichments")

        return {
            "results": list(surgeon_results.values()),
            "summary": {
                "total_surgeons": len(request.surgeons),
                "total_enrichments": total_tasks,
                "success_count": sum(1 for r in surgeon_results.values() if r["status"] == "success"),
                "partial_error_count": sum(1 for r in surgeon_results.values() if r["status"] == "partial_error"),
                "total_time": total_time,
                "enrichment_time": enrich_time,
                "avg_time_per_enrichment": enrich_time / total_tasks if total_tasks else 0,
                "total_credits_used": total_credits_used,
                "avg_credits_per_enrichment": total_credits_used / total_tasks if total_tasks else 0
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

class CellEnrichmentFilter:
    """Utility class to determine which cells need enrichment"""
    
    @staticmethod
    def is_cell_empty(value) -> bool:
        """Check if a cell value is empty and needs enrichment"""
        if value is None:
            return True
        if isinstance(value, str):
            cleaned = value.strip().lower()
            # Consider these as empty
            empty_indicators = {
                '', 'n/a', 'na', 'null', 'none', 'unknown', 'not found', 
                'information not found', 'not available', '-', '--', '---'
            }
            return cleaned in empty_indicators
        return False
    
    @staticmethod
    def should_enrich_field(surgeon_data: Dict, field: str) -> bool:
        """Determine if a specific field should be enriched for a surgeon"""
        current_value = surgeon_data.get(field)
        return CellEnrichmentFilter.is_cell_empty(current_value)

class StreamingMedicalEnricher:
    """
    High-performance streaming medical enrichment processor that respects rate limits
    and provides real-time field-by-field updates via Server-Sent Events.
    """
    
    def __init__(self, batch_size: int = 3, requests_per_minute: int = 80):
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.min_batch_delay = 60.0 / requests_per_minute * batch_size
        
    async def stream_enrichments(
        self,
        surgeons: List[Dict],
        fields: List[str],
        tavily_client,
        llm_provider,
        query_llm_provider,
        trace_id: str
    ):
        """
        Stream enrichment results with real-time field updates and proper rate limiting.
        
        Yields SSE-formatted events for each field completion as they happen.
        """
        start_time = time.time()
        
        # Send connection event
        yield self._format_sse("connected", {
            "trace_id": trace_id,
            "total_surgeons": len(surgeons),
            "total_fields": len(fields),
            "timestamp": time.time()
        })
        
        # Initialize surgeon tracking
        surgeon_results = {}
        all_tasks = []
        
        # Create task queue with metadata
        for surgeon_idx, surgeon in enumerate(surgeons):
            surgeon_name = surgeon.get('name', '').strip()
            if not surgeon_name:
                continue
                
            surgeon_results[surgeon_idx] = {
                "name": surgeon_name,
                "enriched_data": {},
                "sources": [],
                "credits_used": 0,
                "fields_completed": 0,
                "total_fields": len(fields)
            }
            
            # Send surgeon initialization
            yield self._format_sse("surgeon_init", {
                "surgeon_idx": surgeon_idx,
                "name": surgeon_name,
                "fields_to_enrich": fields,
                "timestamp": time.time()
            })
            
            # Create task queue with metadata - only for empty fields
            for field in fields:
                if CellEnrichmentFilter.should_enrich_field(surgeon, field):
                    task_data = {
                        "surgeon_idx": surgeon_idx,
                        "surgeon_name": surgeon_name,
                        "field": field,
                        "enrich_params": {
                            "name": surgeon_name,
                            "target_field": field,
                            "hospital_name": surgeon.get('hospital_name'),
                            "address": surgeon.get('address'),
                            "phone": surgeon.get('phone'),
                            "all_context": surgeon,  # Pass all surgeon data as context
                            "tavily_client": tavily_client,
                            "llm_provider": llm_provider,
                            "query_llm_provider": query_llm_provider
                        }
                    }
                    all_tasks.append(task_data)
                else:
                    # Field already has data, skip enrichment
                    logger.info(f"Skipping {field} for {surgeon_name} - already has value: {surgeon.get(field)}")
            
            # Update total fields count to only include fields that need enrichment
            surgeon_results[surgeon_idx]["total_fields"] = len([f for f in fields if CellEnrichmentFilter.should_enrich_field(surgeon, f)])
        
        total_tasks = len(all_tasks)
        completed_tasks = 0
        total_credits = 0
        
        # Process tasks in rate-limited sequential batches
        for batch_start in range(0, total_tasks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_tasks)
            batch_tasks = all_tasks[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (total_tasks + self.batch_size - 1) // self.batch_size
            
            batch_start_time = time.time()
            logger.info(f"[{trace_id}] Processing batch {batch_num}/{total_batches} with {len(batch_tasks)} tasks")
            
            # Send batch start event
            yield self._format_sse("batch_start", {
                "batch_num": batch_num,
                "total_batches": total_batches,
                "batch_size": len(batch_tasks),
                "timestamp": time.time()
            })
            
            # Process batch tasks concurrently
            batch_coros = []
            for task in batch_tasks:
                batch_coros.append(self._process_single_enrichment(task, surgeon_results))
            
            # Execute batch and stream results as they complete
            batch_results = []
            for coro in asyncio.as_completed(batch_coros):
                try:
                    result = await coro
                    batch_results.append(result)
                    
                    # Stream field completion event immediately
                    task_data = result["task_data"]
                    surgeon_idx = task_data["surgeon_idx"]
                    field = task_data["field"]
                    enrichment_result = result["enrichment_result"]
                    
                    completed_tasks += 1
                    total_credits += enrichment_result.get("credits_used", 0)
                    
                    # Update surgeon results
                    surgeon_data = surgeon_results[surgeon_idx]
                    surgeon_data["enriched_data"][field] = enrichment_result.get("answer", "Information not found")
                    surgeon_data["credits_used"] += enrichment_result.get("credits_used", 0)
                    surgeon_data["fields_completed"] += 1
                    
                    # Add sources if available
                    if enrichment_result.get("sources"):
                        surgeon_data["sources"].extend([
                            {"title": src.get("title", "Unknown"), "url": src.get("url", "")}
                            for src in enrichment_result["sources"]
                            if isinstance(src, dict)
                        ])
                    
                    # Send field completion event
                    yield self._format_sse("field_complete", {
                        "surgeon_idx": surgeon_idx,
                        "surgeon_name": task_data["surgeon_name"],
                        "field": field,
                        "value": enrichment_result.get("answer", "Information not found"),
                        "credits_used": enrichment_result.get("credits_used", 0),
                        "search_strategy": enrichment_result.get("search_strategy"),
                        "sources": surgeon_data["sources"][-len(enrichment_result.get("sources", [])):] if enrichment_result.get("sources") else [],
                        "progress": completed_tasks / total_tasks,
                        "timestamp": time.time()
                    })
                    
                    # Check if surgeon is complete
                    if surgeon_data["fields_completed"] >= surgeon_data["total_fields"]:
                        yield self._format_sse("surgeon_complete", {
                            "surgeon_idx": surgeon_idx,
                            "name": task_data["surgeon_name"],
                            "enriched_data": surgeon_data["enriched_data"],
                            "total_sources": len(surgeon_data["sources"]),
                            "total_credits": surgeon_data["credits_used"],
                            "timestamp": time.time()
                        })
                        
                except Exception as e:
                    logger.error(f"[{trace_id}] Batch task error: {str(e)}")
                    # Find the failed task and send error event
                    yield self._format_sse("field_error", {
                        "error": str(e),
                        "timestamp": time.time()
                    })
            
            batch_duration = time.time() - batch_start_time
            
            # Send batch completion event
            yield self._format_sse("batch_complete", {
                "batch_num": batch_num,
                "duration": batch_duration,
                "completed_tasks": len(batch_results),
                "timestamp": time.time()
            })
            
            # Rate limiting between batches
            if batch_end < total_tasks:
                wait_time = max(0, self.min_batch_delay - batch_duration)
                if wait_time > 0:
                    logger.info(f"[{trace_id}] Rate limiting: waiting {wait_time:.1f}s before next batch")
                    yield self._format_sse("rate_limit_wait", {
                        "wait_time": wait_time,
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(wait_time)
        
        # Send final completion event
        total_time = time.time() - start_time
        yield self._format_sse("enrichment_complete", {
            "total_surgeons": len(surgeon_results),
            "total_enrichments": completed_tasks,
            "total_credits_used": total_credits,
            "total_time": total_time,
            "avg_time_per_enrichment": total_time / max(completed_tasks, 1),
            "timestamp": time.time()
        })
    
    async def _process_single_enrichment(self, task_data: Dict, surgeon_results: Dict) -> Dict:
        """Process a single enrichment and return structured result."""
        surgeon_idx = task_data["surgeon_idx"]
        field = task_data["field"]
        surgeon_name = task_data["surgeon_name"]
        
        try:
            # Call the enrichment function
            enrichment_result = await enrich_medical_field(**task_data["enrich_params"])
            
            # Convert to dict format
            result_dict = {
                "answer": enrichment_result.answer,
                "credits_used": enrichment_result.credits_used,
                "sources": enrichment_result.sources,
                "search_strategy": enrichment_result.search_strategy,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Enrichment failed for {surgeon_name} - {field}: {str(e)}")
            result_dict = {
                "answer": "Information not found",
                "credits_used": 0,
                "sources": [],
                "search_strategy": None,
                "status": "error",
                "error": str(e)
            }
        
        return {
            "task_data": task_data,
            "enrichment_result": result_dict
        }
    
    def _format_sse(self, event_type: str, data: Dict) -> str:
        """Format data as Server-Sent Event."""
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

@app.post("/api/enrich-medical/stream")
async def stream_medical_enrichment(
    request_data: StreamRequest,
    fastapi_request: Request
):
    """Stream medical enrichment results in real-time using Server-Sent Events."""
    trace_id = generate_trace_id()
    
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
        
        # Initialize providers
        llm_provider = get_llm_provider(request_data.provider)
        query_llm_provider = get_query_llm_provider()
        tavily_client = get_tavily_client(api_key)
        
        # Canonicalize fields and surgeon objects
        canonical_fields = [canonicalize_field(f) for f in request_data.target_fields]
        surgeons_canonical = [canonicalize_record_keys(s) for s in request_data.surgeons]

        # Create streaming enricher
        enricher = StreamingMedicalEnricher(batch_size=3, requests_per_minute=80)
        
        # Create the streaming generator
        async def generate_stream():
            try:
                async for event in enricher.stream_enrichments(
                    surgeons=surgeons_canonical,
                    fields=canonical_fields,
                    tavily_client=tavily_client,
                    llm_provider=llm_provider,
                    query_llm_provider=query_llm_provider,
                    trace_id=trace_id
                ):
                    yield event
            except Exception as e:
                logger.error(f"[{trace_id}] Streaming error: {str(e)}")
                yield f"data: {json.dumps({'type': 'stream_error', 'error': str(e), 'timestamp': time.time()})}\n\n"
        
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
                "message": str(e)
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
        
        llm_provider = get_llm_provider(provider)
        tavily_client = get_tavily_client(api_key)
        
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

        llm_provider = get_llm_provider(provider)
        tavily_client = get_tavily_client(api_key)
        
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

        # Execute enrichments using sequential batch processing with rate limiting
        enrich_start_time = time.time()
        enriched_results = await sequential_processor.process_tasks_sequentially(tasks, enrich_worker)
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
        
        llm_provider = get_llm_provider(provider)
        tavily_client = get_tavily_client(api_key)

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

        # Execute enrichments using sequential batch processing with rate limiting
        enrich_start_time = time.time()
        enriched_results = await sequential_processor.process_tasks_sequentially(tasks, enrich_worker)
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
