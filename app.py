from fastapi import FastAPI, HTTPException, Cookie, Request
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
from jose import JWTError, jwt

import asyncio
import logging
import sys
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import AsyncOpenAI
from google.generativeai import GenerativeModel
import time
from backend.graph import enrich_cell_with_graph, LLMProvider, OpenAIProvider, GeminiProvider
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
            openai_provider = OpenAIProvider(openai_client, model="gpt-4.1")
        
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

class EnrichmentRequest(BaseModel):
    column_name: str
    target_value: str
    context_values: Dict[str, str]
    answer: Optional[str] = None
    search_result: Optional[str] = None

class BatchEnrichmentRequest(BaseModel):
    column_name: str
    rows: List[str]  # List of target values to enrich
    context_values: Dict[str, str]
    input_source_type: Optional[str] = None
    input_datas: Optional[List[str]] = None
    custom_prompt: Optional[str] = None
    answer: Optional[str] = None
    search_result: Optional[str] = None

class TableData(BaseModel):
    rows: List[str]  # List of target values to enrich
    context_values: Dict[str, str]
    input_source_type: Optional[str] = None
    input_datas: Optional[List[str]] = None
    custom_prompt: Optional[str] = None
    answer: Optional[str] = None
    search_result: Optional[str] = None

class TableEnrichmentRequest(BaseModel):
    data: Dict[str, TableData]

class EnrichmentResponse(BaseModel):
    enriched_value: str
    status: str
    error: Optional[str] = None
    sources: List[SearchResult] = []

class BatchEnrichmentResponse(BaseModel):
    enriched_values: List[str]
    status: str
    error: Optional[str] = None
    sources: List[List[SearchResult]] = []

class TableEnrichmentColumnResponse(BaseModel):
    enriched_values: List[str]
    sources: List[List[SearchResult]] = []

class EnrichTableResponse(BaseModel):
    enriched_values: Dict[str, List[str]]
    sources: Dict[str, List[List[SearchResult]]] = {}
    status: str
    error: Optional[str] = None

@app.get("/api/verify-jwt")
async def verify_jwt(jwt_token: str = Cookie(None)):
    return JSONResponse(content={"success": True, "data": "FAKE_API_KEY_FOR_DEV"})

@app.post("/api/enrich", response_model=EnrichmentResponse)
async def enrich_data(
    request: EnrichmentRequest,
    fastapi_request: Request,
    provider: str = "openai" 
):
    """Enrich a single cell's data."""
    start_time = time.time()
    try:
        api_key = fastapi_request.headers.get("Authorization")
        # BYPASS for local development: use a default API key if not provided
        if not api_key:
            api_key = "FAKE_API_KEY_FOR_DEV"
        # Get the appropriate provider
        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"Processing single enrichment request for column: {request.column_name}")
        logger.info(f"Target value: {request.target_value}")
        logger.info(f"Using provider: {provider}")

        # Measure the time for the enrichment operation
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
        logger.info(f"Enrichment completed in {enrich_time:.2f}s (total request: {total_time:.2f}s)")
        
        # Extract sources from the result
        sources = []
        if isinstance(enriched_value, dict) and 'search_result' in enriched_value:
            for result in enriched_value['search_result']['results']:
                sources.append(SearchResult(
                    title=result['title'],
                    url=result['url']
                ))
        
        return EnrichmentResponse(
            enriched_value=enriched_value.get('answer', enriched_value),
            status="success",
            sources=sources
        )
        
    except ValueError as e:
        logger.error(f"Invalid provider configuration: {str(e)}")
        total_time = time.time() - start_time
        logger.info(f"Request failed in {total_time:.2f}s")
        return EnrichmentResponse(
            enriched_value="Provider configuration error",
            status="error",
            error=str(e),
            sources=[]
        )
    except Exception as e:
        logger.error(f"Error in single enrichment: {str(e)}")
        total_time = time.time() - start_time
        logger.info(f"Request failed in {total_time:.2f}s")
        return EnrichmentResponse(
            enriched_value="Error during enrichment",
            status="error",
            error=str(e),
            sources=[]
        )

@app.post("/api/enrich/batch", response_model=BatchEnrichmentResponse)
async def enrich_batch(
    request: BatchEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "openai" 
):
    """Enrich multiple rows in parallel."""
    start_time = time.time()
    try:
        api_key = fastapi_request.headers.get("Authorization")
        # BYPASS for local development: use a default API key if not provided
        if not api_key:
            api_key = "FAKE_API_KEY_FOR_DEV"
        # Get the appropriate provider
        llm_provider = get_llm_provider(provider, api_key)
        
        logger.info(f"Starting batch enrichment for column: {request.column_name}")
        logger.info(f"Number of rows to process: {len(request.rows)}")
        logger.info(f"Using provider: {provider}")

        # Process each row
        tasks = []
        non_empty_indices = []  # Track indices of non-empty rows
        for row_idx, row in enumerate(request.rows):
            if row.strip():
                input_data = request.input_datas[row_idx] if request.input_datas and len(request.input_datas) > row_idx else None
                task = enrich_cell_with_graph(
                    column_name=request.column_name,
                    target_value=row,
                    context_values=request.context_values,
                    tavily_client=tavily_client,
                    llm_provider=llm_provider,
                    input_source_type=request.input_source_type,
                    input_data=input_data,
                    custom_prompt=request.custom_prompt
                )
                tasks.append(task)
                non_empty_indices.append(row_idx)

        # Measure the time for the enrichment operations
        enrich_start_time = time.time()
        enriched_values = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        enrich_time = time.time() - enrich_start_time
        
        # Process results and fill empty rows
        final_values = []
        all_sources = []
        processed_idx = 0
        
        for row_idx, row in enumerate(request.rows):
            if not row.strip():
                final_values.append("")
                all_sources.append([])
            else:
                value = enriched_values[processed_idx]
                sources = []
                
                if isinstance(value, dict) and 'search_result' in value:
                    for result in value['search_result']['results']:
                        sources.append(SearchResult(
                            title=result['title'],
                            url=result['url']
                        ))
                    final_values.append(value.get('answer', str(value)))
                elif isinstance(value, Exception):
                    final_values.append("Error during enrichment")
                else:
                    final_values.append(str(value))
                
                all_sources.append(sources)
                processed_idx += 1
        
        total_time = time.time() - start_time
        avg_time_per_row = enrich_time / len(tasks) if tasks else 0
        logger.info(f"Batch enrichment completed in {enrich_time:.2f}s (total request: {total_time:.2f}s)")
        logger.info(f"Average time per row: {avg_time_per_row:.2f}s")
        
        return BatchEnrichmentResponse(
            enriched_values=final_values,
            status="success",
            sources=all_sources
        )
    
    except ValueError as e:
        logger.error(f"Invalid provider configuration: {str(e)}")
        total_time = time.time() - start_time
        logger.info(f"Request failed in {total_time:.2f}s")
        return BatchEnrichmentResponse(
            enriched_values=["Provider configuration error"] * len(request.rows),
            status="error",
            error=str(e)
        )    
    except Exception as e:
        logger.error(f"Error in batch enrichment: {str(e)}")
        total_time = time.time() - start_time
        logger.info(f"Request failed in {total_time:.2f}s")
        return BatchEnrichmentResponse(
            enriched_values=["Error during enrichment"] * len(request.rows),
            status="error",
            error=str(e),
            sources=[[] for _ in request.rows]
        )

@app.post("/api/enrich-table", response_model=EnrichTableResponse)
async def enrich_table(
    request: TableEnrichmentRequest,
    fastapi_request: Request,
    provider: str = "openai"
):
    """Enrich the entire table (all columns) in parallel."""
    start_time = time.time()
    try:
        api_key = fastapi_request.headers.get("Authorization")
        # BYPASS for local development: use a default API key if not provided
        if not api_key:
            api_key = "FAKE_API_KEY_FOR_DEV"
        llm_provider = get_llm_provider(provider, api_key)

        # Prepare enrichment tasks for all non-empty cells
        tasks = []
        index_map = []  # Track which column and row index each task belongs to
        for column_name, table_data in request.data.items():
            for row_idx, row_value in enumerate(table_data.rows):
                if row_value.strip():
                    input_data = table_data.input_datas[row_idx] if table_data.input_datas and len(table_data.input_datas) > row_idx else None
                    task = enrich_cell_with_graph(
                        column_name=column_name,
                        target_value=row_value,
                        context_values=table_data.context_values,
                        tavily_client=tavily_client,
                        llm_provider=llm_provider,
                        input_source_type=table_data.input_source_type,
                        input_data=input_data,
                        custom_prompt=table_data.custom_prompt
                    )
                    tasks.append(task)
                    index_map.append((column_name, row_idx))

        # Run all enrichment tasks
        enrich_start_time = time.time()
        enriched_results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        enrich_time = time.time() - enrich_start_time

        # Prepare response containers
        enriched_table = {col: [""] * len(data.rows) for col, data in request.data.items()}
        sources_table = {col: [[] for _ in data.rows] for col, data in request.data.items()}

        for (column_name, row_idx), result in zip(index_map, enriched_results):
            if isinstance(result, dict) and 'search_result' in result:
                enriched_table[column_name][row_idx] = result.get('answer', str(result))
                sources_table[column_name][row_idx] = [
                    SearchResult(title=r['title'], url=r['url'])
                    for r in result['search_result']['results']
                ]
            elif isinstance(result, Exception):
                enriched_table[column_name][row_idx] = "Error during enrichment"
                sources_table[column_name][row_idx] = []
            else:
                enriched_table[column_name][row_idx] = str(result)
                sources_table[column_name][row_idx] = []

        total_time = time.time() - start_time
        avg_time_per_cell = enrich_time / len(tasks) if tasks else 0
        logger.info(f"Table enrichment done in {enrich_time:.2f}s (total: {total_time:.2f}s)")
        logger.info(f"Avg time per cell: {avg_time_per_cell:.2f}s")

        return EnrichTableResponse(
            enriched_values=enriched_table,
            sources=sources_table,
            status="success"
        )

    except ValueError as e:
        logger.error(f"Invalid provider configuration: {str(e)}")
        return EnrichTableResponse(
            enriched_values={col: ["Provider configuration error"] * len(data.rows) for col, data in request.data.items()},
            sources={col: [[] for _ in data.rows] for col, data in request.data.items()},
            status="error",
            error=str(e)
        )

    except Exception as e:
        logger.error(f"Error enriching table: {str(e)}")
        return EnrichTableResponse(
            enriched_values={col: ["Error during enrichment"] * len(data.rows) for col, data in request.data.items()},
            sources={col: [[] for _ in data.rows] for col, data in request.data.items()},
            status="error",
            error=str(e)
        )


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
