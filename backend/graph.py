import asyncio
import logging
import os
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union
from functools import wraps

from dotenv import load_dotenv
from google import genai
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from aiolimiter import AsyncLimiter
from cachetools import TTLCache
import threading

# Import our simple extraction system
from .smart_extractor import SmartFieldExtractor

# Initialize extractor
smart_extractor = SmartFieldExtractor()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Global rate limiter, cache, and circuit breaker
_tavily_rate_limiter = AsyncLimiter(80, 60)  # 80 requests per minute (leaving buffer from 100)
_tavily_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests to match batch size
_enrichment_cache = TTLCache(maxsize=1000, ttl=900)  # 15-minute TTL cache
_cache_lock = threading.RLock()

# Circuit breaker state
class CircuitBreakerState:
    def __init__(self):
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = 5
        self.recovery_timeout = 60  # seconds
        self.success_threshold = 2  # successes needed to close circuit
        self.consecutive_successes = 0

_circuit_breaker = CircuitBreakerState()

# Note: Removed file persistence to make backend stateless between application runs.
# The in-memory TTLCache (_enrichment_cache) is kept for de-duplication within a single session.

# Global trace context
_trace_context = {}

def get_trace_id() -> str:
    """Generate or retrieve trace ID for current request"""
    return str(uuid.uuid4())

async def _tavily_wrapper(tavily_client, operation: str, **kwargs):
    """
    Enhanced Tavily API wrapper with circuit breaker, rate limiting, and retry logic
    """
    trace_id = get_trace_id()
    start_time = time.time()
    
    # Circuit breaker check
    current_time = time.time()
    if _circuit_breaker.state == "OPEN":
        if current_time - _circuit_breaker.last_failure_time > _circuit_breaker.recovery_timeout:
            _circuit_breaker.state = "HALF_OPEN"
            _circuit_breaker.consecutive_successes = 0
            logger.info(f"[{trace_id}] Circuit breaker moving to HALF_OPEN state")
        else:
            logger.warning(f"[{trace_id}] Circuit breaker is OPEN - failing fast")
            raise Exception("Circuit breaker is OPEN - Tavily API temporarily unavailable")
    
    # Rate limiting
    async with _tavily_rate_limiter:
        async with _tavily_semaphore:
            retry_count = 0
            max_retries = 3 if _circuit_breaker.state != "HALF_OPEN" else 1
            
            while retry_count < max_retries:
                try:
                    logger.info(f"[{trace_id}] Executing Tavily {operation} (attempt {retry_count + 1}) - Circuit: {_circuit_breaker.state}")
                    
                    if operation == "search":
                        result = await asyncio.to_thread(
                            lambda: tavily_client.search(**kwargs)
                        )
                    else:
                        raise ValueError(f"Unknown Tavily operation: {operation}")
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[{trace_id}] Tavily {operation} completed in {elapsed:.2f}s")
                    
                    # Success - update circuit breaker state
                    if _circuit_breaker.state == "HALF_OPEN":
                        _circuit_breaker.consecutive_successes += 1
                        if _circuit_breaker.consecutive_successes >= _circuit_breaker.success_threshold:
                            _circuit_breaker.state = "CLOSED"
                            _circuit_breaker.failure_count = 0
                            logger.info(f"[{trace_id}] Circuit breaker moved to CLOSED state")
                    elif _circuit_breaker.state == "CLOSED":
                        _circuit_breaker.failure_count = max(0, _circuit_breaker.failure_count - 1)
                    
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    elapsed = time.time() - start_time
                    
                    # Update circuit breaker on failure
                    _circuit_breaker.failure_count += 1
                    _circuit_breaker.last_failure_time = time.time()
                    
                    # Open circuit if threshold exceeded
                    if (_circuit_breaker.state == "CLOSED" and 
                        _circuit_breaker.failure_count >= _circuit_breaker.failure_threshold):
                        _circuit_breaker.state = "OPEN"
                        logger.error(f"[{trace_id}] Circuit breaker opened due to {_circuit_breaker.failure_count} failures")
                    elif _circuit_breaker.state == "HALF_OPEN":
                        _circuit_breaker.state = "OPEN"
                        logger.error(f"[{trace_id}] Circuit breaker reopened after failure in HALF_OPEN state")
                    
                    # Don't retry on auth errors (4xx)
                    if "401" in str(e) or "403" in str(e) or "api_key" in str(e).lower():
                        logger.error(f"[{trace_id}] Tavily auth error: {str(e)}")
                        raise
                    
                    if retry_count >= max_retries:
                        logger.error(f"[{trace_id}] Tavily {operation} failed after {max_retries} attempts: {str(e)}")
                        raise
                    
                    # Exponential backoff with jitter
                    base_delay = (2 ** retry_count) * 5  # 5s, 10s, 20s base delays
                    jitter = min(base_delay * 0.1, 2)  # Up to 10% jitter, max 2s
                    delay = base_delay + (jitter * (0.5 - asyncio.get_event_loop().time() % 1))
                    
                    logger.warning(f"[{trace_id}] Tavily {operation} failed (attempt {retry_count}), retrying in {delay:.1f}s: {str(e)}")
                    await asyncio.sleep(delay)

# Medical specialties for validation
MEDICAL_SPECIALTIES = [
    "General Surgery", "Orthopedic Surgery", "Neurosurgery", "Cardiac Surgery", 
    "Plastic Surgery", "Trauma Surgery", "Vascular Surgery", "Pediatric Surgery",
    "Oncological Surgery", "Thoracic Surgery", "Urological Surgery", "ENT Surgery",
    "Ophthalmology", "Dermatology", "Anesthesiology", "Emergency Medicine",
    "Internal Medicine", "Family Medicine", "Pediatrics", "Radiology"
]

# Model selection strategy for cost optimization
MODEL_COMPLEXITY = {
    "simple": "gemini-2.5-flash-lite",    # Basic extractions, credentials
    "medium": "gemini-2.5-flash",         # Professional summaries 
    "complex": "gemini-2.5-flash"           # Strategic analysis, influence assessment
}


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, complexity: str = "medium") -> str:
        """
        Generate content with complexity-based optimization.
        
        Args:
            prompt: The input prompt
            complexity: "simple", "medium", or "complex" for model selection
            
        Returns:
            Generated text response
        """
        pass

class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider using service account credentials."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", project_id: Optional[str] = None):
        # Use Vertex AI with service account credentials
        self.project_id = project_id or os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID environment variable is required")
            
        # Initialize with Vertex AI configuration
        try:
            # Configure for Vertex AI with project and location
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location
            )
            self.model_name = model_name
            
            logger.info(f"Initialized Vertex AI provider with project {self.project_id} in {self.location}")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI provider: {e}")
        
    async def generate(self, prompt: str, complexity: str = "medium") -> str:
        model_name = MODEL_COMPLEXITY.get(complexity, MODEL_COMPLEXITY["medium"])
        if model_name != self.model_name:
            self.model_name = model_name
            
        max_retries = 3
        base_delay = 2.0  # Start with 2 second delay
        
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                )
                return response.text
                
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limiting (429 or RESOURCE_EXHAUSTED)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        # Exponential backoff with jitter for rate limiting
                        delay = base_delay * (2 ** attempt) + (0.5 * attempt)
                        logger.warning(f"Vertex AI rate limited (attempt {attempt + 1}), retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Vertex AI rate limiting persisted after {max_retries} retries")
                        raise Exception("Rate limiting error - please try again later")
                else:
                    # For other errors, fail immediately
                    logger.error(f"Vertex AI generation failed: {e}")
                    raise
    
    def create_lightweight_version(self, model_name: str = "gemini-2.5-flash-lite") -> 'VertexAIProvider':
        """Create a lightweight version that reuses the same client infrastructure"""
        lightweight = VertexAIProvider.__new__(VertexAIProvider)  # Create without calling __init__
        lightweight.client = self.client  # Reuse existing client
        lightweight.project_id = self.project_id
        lightweight.location = self.location
        lightweight.model_name = model_name
        logger.info(f"Created lightweight Vertex AI provider with model {model_name} (reusing client)")
        return lightweight


@dataclass
class MedicalEnrichmentContext:
    """State for medical professional data enrichment"""
    # Input data (may be partially filled)
    name: str
    hospital_name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    
    # Full context from CSV row - ALL available data
    all_context: Optional[Dict[str, str]] = field(default_factory=dict)
    
    # Fields to enrich
    target_field: str = None  # Which field we're enriching
    
    # Enrichment results
    email: Optional[str] = None
    specialty: Optional[str] = None
    subspecialty: Optional[str] = None
    credentials: Optional[str] = None
    linkedin_url: Optional[str] = None
    influence_summary: Optional[str] = None
    strategic_summary: Optional[str] = None
    additional_contacts: Optional[str] = None
    
    # System fields
    search_result: Optional[Dict] = None
    sources: List[str] = field(default_factory=list)
    enrichment_status: List[str] = field(default_factory=list)
    credits_used: int = 0  # Track Tavily credits used
    search_strategy: Optional[str] = None  # Track which search strategy was used
    last_updated: Optional[datetime] = None
    answer: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "name": self.name,
            "hospital_name": self.hospital_name,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "specialty": self.specialty,
            "subspecialty": self.subspecialty,
            "credentials": self.credentials,
            "linkedin_url": self.linkedin_url,
            "influence_summary": self.influence_summary,
            "strategic_summary": self.strategic_summary,
            "additional_contacts": self.additional_contacts,
            "sources": self.sources,
            "enrichment_status": self.enrichment_status,
            "credits_used": self.credits_used,
            "search_strategy": self.search_strategy,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
class MedicalEnrichmentPipeline:
    """Specialized pipeline for medical professional data enrichment"""
    
    def __init__(self, tavily_client, llm_provider: LLMProvider, query_llm_provider: LLMProvider):
        self.tavily = tavily_client
        self.llm = llm_provider
        self.query_llm = query_llm_provider

    async def search_medical_data(self, state: MedicalEnrichmentContext) -> MedicalEnrichmentContext:
        """Search for medical professional data with fallback logic for emails"""
        try:
            # Build cache key that includes request-specific context to prevent cross-contamination
            cache_key = f"{state.name.strip()}:{state.target_field}:{state.hospital_name or 'no_hospital'}"
            
            with _cache_lock:
                cached_result = _enrichment_cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for {state.name} - {state.target_field}")
                    # IMPORTANT: Create fresh copies to prevent source contamination
                    state.search_result = cached_result["search_result"].copy() if cached_result["search_result"] else None
                    state.sources = [source.copy() for source in cached_result.get("sources", [])]  # Fresh copy
                    state.credits_used += cached_result["credits_used"]
                    state.search_strategy = f"cached - {cached_result['search_strategy']}"
                    return state
            
            # Special handling for email field with fallback search
            if state.target_field == "email":
                return await self._search_with_email_fallback(state, cache_key)
            
            # Standard single search for other fields
            return await self._perform_single_search(state, cache_key)
            
        except Exception as e:
            logger.error(f"Error in medical search for {state.name}: {str(e)}")
            state.answer = "Search error occurred"
            state.enrichment_status.append(f"search_error:{datetime.now().isoformat()}:{str(e)}")
            return state

    async def _search_with_email_fallback(self, state: MedicalEnrichmentContext, cache_key: str) -> MedicalEnrichmentContext:
        """Perform primary search for email with fallback if needed"""
        # Resolve hospital/practice domain when possible to scope queries
        resolved_domain = await self._resolve_hospital_domain(state.hospital_name) if state.hospital_name else None

        # Primary search - focused on direct contact
        primary_query = await self._build_primary_email_query(state, resolved_domain)
        search_config = self._get_search_config("email")
        if resolved_domain:
            # Scope to the official domain for better precision
            search_config = {**search_config, "include_domains": [resolved_domain]}
        
        logger.info(f"Primary email search for {state.name}: {primary_query}")
        
        result = await self._execute_search(primary_query, search_config)
        credits_used = search_config['credits']
        strategy_parts = [f"primary-{search_config['search_depth']}"]
        
        # Check if primary search found useful email content
        has_email_content = self._has_email_content(result)
        
        # Fallback search if primary didn't find email content
        if not has_email_content:
            fallback_query = await self._build_fallback_email_query(state, resolved_domain)
            logger.info(f"Fallback email search for {state.name}: {fallback_query}")
            
            # For fallback, remove strict domain scoping if primary failed
            fallback_search_config = dict(search_config)
            if resolved_domain:
                fallback_search_config.pop("include_domains", None)
            fallback_result = await self._execute_search(fallback_query, fallback_search_config)
            credits_used += search_config['credits']
            strategy_parts.append(f"fallback-{search_config['search_depth']}")
            
            # Combine results, prioritizing fallback if it has better content
            if self._has_email_content(fallback_result):
                # Merge results with fallback taking priority
                result = self._merge_search_results(fallback_result, result)
            else:
                # Keep primary results even if no email found
                result = self._merge_search_results(result, fallback_result)
        
        # Update state
        state.search_result = result
        state.credits_used += credits_used
        state.search_strategy = "+".join(strategy_parts)
        
        # Cache the combined result
        await self._cache_result(cache_key, result, {"credits": credits_used}, state.search_strategy)
        
        logger.info(f"Email search completed for {state.name} (credits used: {credits_used})")
        return state

    async def _perform_single_search(self, state: MedicalEnrichmentContext, cache_key: str) -> MedicalEnrichmentContext:
        """Perform standard single search for non-email fields"""
        query = await self._build_simple_search_query(state)
        search_config = self._get_search_config(state.target_field)
        
        logger.info(f"Searching for {state.target_field} with query: {query}")
        
        result = await self._execute_search(query, search_config)
        
        logger.info(f"Tavily search completed for {state.name} (credits used: {search_config['credits']})")
        
        # Update state with results
        state.search_result = result
        state.credits_used += search_config['credits']
        state.search_strategy = f"{search_config['search_depth']} ({search_config['reason']})"
        
        # Cache the result
        await self._cache_result(cache_key, result, search_config, state.search_strategy)
        
        return state

    def _has_email_content(self, search_result: dict) -> bool:
        """Check if search result contains email-related content"""
        if not search_result or not search_result.get("results"):
            return False
        
        # Check Tavily's answer for email patterns
        answer = search_result.get("answer", "")
        if "@" in answer and any(domain in answer.lower() for domain in [".com", ".org", ".edu", ".gov"]):
            return True
        
        # Check results content for email patterns
        email_indicators = ["@", "email", "contact", "reach", "mail"]
        for result in search_result.get("results", []):
            content = (result.get("content", "") + result.get("raw_content", "")).lower()
            if "@" in content and any(indicator in content for indicator in email_indicators):
                return True
        
        return False

    def _merge_search_results(self, primary: dict, secondary: dict) -> dict:
        """Merge two search results, prioritizing primary"""
        if not secondary:
            return primary
        if not primary:
            return secondary
        
        # Combine results while avoiding duplicates
        merged_results = list(primary.get("results", []))
        
        for sec_result in secondary.get("results", []):
            sec_url = sec_result.get("url", "")
            if not any(res.get("url") == sec_url for res in merged_results):
                merged_results.append(sec_result)
        
        # Prefer primary answer, fallback to secondary
        answer = primary.get("answer") or secondary.get("answer", "")
        
        return {
            "results": merged_results[:15],  # Keep reasonable limit
            "answer": answer,
            **{k: v for k, v in primary.items() if k not in ["results", "answer"]}
        }

    async def _build_primary_email_query(self, state: MedicalEnrichmentContext, domain: Optional[str] = None) -> str:
        """Build focused query for direct email contact"""
        name = state.name
        hospital = state.hospital_name or ""
        
        if domain:
            # Prefer scoping to official domain when available
            query_parts = [f'site:{domain}', f'"{name}"', 'email', 'contact']
        else:
            query_parts = [f'"{name}" email address contact']
        if hospital:
            query_parts.append(f'"{hospital}"')
        query_parts.extend(["doctor", "surgeon", "professional"])
        
        return " ".join(query_parts)

    async def _build_fallback_email_query(self, state: MedicalEnrichmentContext, domain: Optional[str] = None) -> str:
        """Build broader query for email with department/hospital contacts"""
        name = state.name
        hospital = state.hospital_name or ""
        
        # Include department/specialty info if available
        specialty_info = ""
        if state.all_context:
            for key, value in state.all_context.items():
                if key.lower() in ['specialty', 'department'] and value:
                    specialty_info = str(value)
                    break
        
        if domain:
            query_parts = [f'site:{domain}', f'"{name}"']
        else:
            query_parts = [f'"{name}"']
        if hospital:
            query_parts.append(f'"{hospital}"')
        if specialty_info:
            query_parts.append(specialty_info)
        query_parts.extend(["contact", "email", "directory", "department", "office"])
        
        return " ".join(query_parts)

    async def _build_linkedin_query(self, state: MedicalEnrichmentContext) -> str:
        """Build optimized LinkedIn search query - clean and targeted"""
        name = state.name
        
        # Get specialty for better targeting, but keep it simple
        specialty_word = ""
        if state.all_context:
            for key, value in state.all_context.items():
                if key.lower() in ['specialty', 'department'] and value:
                    # Use only the first meaningful word
                    specialty_word = str(value).split()[0].lower()
                    break
        
        # LinkedIn works best with: site restriction + name + medical identifiers
        query_parts = [f'site:linkedin.com "{name}"']
        query_parts.extend(["doctor", "MD"])
        
        # Add specialty if available and recognized
        if specialty_word and specialty_word in ["orthopedic", "cardiac", "neuro", "plastic", "general"]:
            query_parts.append(specialty_word)
        
        return " ".join(query_parts)

    async def _build_simple_search_query(self, state: MedicalEnrichmentContext) -> str:
        """Route to optimized queries for specific fields, LLM for others"""
        # Use optimized queries for fields with known search patterns
        if state.target_field == "linkedin_url":
            return await self._build_linkedin_query(state)
        
        # Use LLM for complex fields that benefit from intelligent query generation
        return await self._generate_intelligent_query_with_llm(state)
    
    async def _generate_intelligent_query_with_llm(self, state: MedicalEnrichmentContext) -> str:
        """Generate intelligent search queries using LLM with ALL available context"""
        
        # Build comprehensive context summary
        context_parts = [f"Name: {state.name}"]
        
        # Add structured context
        if state.hospital_name:
            context_parts.append(f"Hospital: {state.hospital_name}")
        if state.address:
            context_parts.append(f"Location: {state.address}")
        if state.phone:
            context_parts.append(f"Phone: {state.phone}")
        
        # Add ALL other context from CSV
        additional_context = []
        if state.all_context:
            for key, value in state.all_context.items():
                if key.lower() not in ['name', 'hospital', 'hospital_name', 'address', 'phone'] and value and str(value).strip():
                    additional_context.append(f"{key}: {value}")
        
        if additional_context:
            context_parts.append("Additional Info: " + "; ".join(additional_context))
            
        full_context = "\n".join(context_parts)
        
        query_prompt = f"""You are an expert at creating search queries to find specific information about medical professionals. 
Create ONE intelligent search query that will help find {state.target_field} for this person.

PERSON CONTEXT:
{full_context}

SEARCH TARGET: {state.target_field}

{self._get_field_specific_guidance(state.target_field)}

QUERY CREATION STRATEGY:
1. Write a natural, conversational query - not just keywords
2. Be specific about the person (use quotes around full name)
3. Include hospital/institution for precision
4. Ask a direct question that Tavily's LLM can answer
5. For emails: "What is Dr [Name]'s email address at [Hospital]?" or "Contact email for [Name] at [Institution]"
6. For complex fields: Frame as analytical questions that require understanding

EXAMPLES:
- Email: "What is Dr Sarah Johnson's email address at Toronto General Hospital?"
- Phone: "What is the office phone number for Dr Michael Chen at Vancouver General Hospital?"  
- Influence: "What are Dr Lisa Wang's research contributions and academic influence in cardiology?"
- Strategic: "What leadership positions does Dr Robert Kim hold at Mount Sinai Hospital?"

Create ONE precise query that asks directly for the information needed:"""

        try:
            # Use lightweight model for query generation

            response = await self.query_llm.generate(query_prompt, complexity="simple")
            generated_query = response.strip()
            
            # Clean up the response
            if "Query:" in generated_query:
                generated_query = generated_query.split("Query:")[-1].strip()
            if generated_query.startswith('"') and generated_query.endswith('"'):
                generated_query = generated_query[1:-1]
            
            # Ensure query doesn't exceed Tavily's 400 character limit
            if len(generated_query) > 400:
                logger.warning(f"Query too long ({len(generated_query)} chars), truncating for {state.name}")
                # Smart truncation - keep name and essential parts
                name_part = f'"{state.name}"'
                field_part = state.target_field
                remaining_chars = 400 - len(name_part) - len(field_part) - 20  # Buffer for connectors
                
                if state.hospital_name and len(state.hospital_name) < remaining_chars:
                    generated_query = f'{name_part} {field_part} {state.hospital_name[:remaining_chars]}'
                else:
                    generated_query = f'{name_part} {field_part} research publications'
            
            logger.info(f"LLM generated query for {state.name} ({state.target_field}): {generated_query}")
            return generated_query
            
        except Exception as e:
            logger.error(f"Error generating LLM query: {e}")
            
            # Field-specific intelligent fallback
            return self._build_intelligent_fallback_query(state)
    
    def _build_intelligent_fallback_query(self, state: MedicalEnrichmentContext) -> str:
        """Build field-specific intelligent fallback queries"""
        name = state.name
        field = state.target_field
        
        # Base context
        context_keywords = []
        if state.hospital_name:
            context_keywords.append(f'"{state.hospital_name}"')
        if state.address:
            location = state.address.split(',')[0]  # Get city/state
            context_keywords.append(location)
        
        # Add specialty/department info from context
        specialty_keywords = []
        if state.all_context:
            for key, value in state.all_context.items():
                if key.lower() in ['specialty', 'department', 'division', 'subspecialty'] and value:
                    specialty_keywords.append(str(value))
        
        context_str = " ".join(context_keywords + specialty_keywords[:2])  # Limit to avoid too long queries
        
        # Field-specific fallback queries
        field_specific_queries = {
            "email": f'"{name}" email contact directory {context_str} surgeon doctor medical',
            "phone": f'"{name}" phone office contact {context_str} surgeon doctor medical',
            "credentials": f'"{name}" MD PhD credentials education {context_str} surgeon doctor medical',
            "specialty": f'"{name}" specialty department {context_str} surgeon doctor medical',
            "linkedin_url": f'site:linkedin.com "{name}" {context_str} surgeon doctor medical',
            "influence_summary": f'"{name}" publications citations research h-index awards {context_str} surgeon doctor',
            "strategic_summary": f'"{name}" leadership director chair board committee {context_str} surgeon doctor',
            "additional_contacts": f'"{name}" assistant coordinator office staff {context_str} medical'
        }
        
        fallback_query = field_specific_queries.get(
            field, 
            f'"{name}" {field} {context_str} surgeon doctor medical professional'
        )
        
        logger.info(f"Using field-specific fallback query: {fallback_query}")
        return fallback_query
    
    def _get_field_specific_guidance(self, field: str) -> str:
        """Enhanced field-specific search guidance for intelligent query generation"""
        guidance = {
            "email": """
TARGET: Any email address that can be used to reach this doctor professionally

PRIORITY APPROACH:
1st PREFERENCE: Doctor's personal professional email 
2nd PREFERENCE: Department/specialty email if relevant to their practice
3rd PREFERENCE: Hospital/clinic contact emails, personal emails if verifiably linked to the doctor

SEARCH STRATEGY:
- Primary search: Direct professional contact information
- If no results: Broader search including department, personal, and hospital emails
- Maximum 2 search attempts for email field only

RELEVANCE CRITERIA:
✓ ACCEPT: Any email that can reasonably be used to contact this specific doctor
✓ ACCEPT: Department emails if they work in that department  
✓ ACCEPT: Hospital emails if they're the doctor's official contact
✓ ACCEPT: Personal emails if clearly belonging to the doctor
✗ REJECT: Generic emails with no clear connection to the doctor


QUERY CONSTRUCTION:
- Use natural, direct questions about contact information
- Be specific with full name and hospital context
- Ask for any way to reach the doctor professionally""",
            
            "phone": """
TARGET: Professional phone number - office, clinic, department, or hospital
SEARCH STRATEGY: Find ANY phone number for contacting this person professionally
KEYWORDS: phone, telephone, office, clinic, contact, appointment, department
QUERY EXAMPLES:
- "Dr [Name] phone number [Hospital]"
- "[Name] office phone [Location]"
- "[Hospital] [Department] contact number"
FALLBACK: Department or hospital main numbers are acceptable""",
            
            "specialty": """
TARGET: Primary medical specialty
SEARCH STRATEGY: Find the main medical specialty they practice
KEYWORDS: specialty, specializes in, department of, division of, board certified, practice area
QUERY EXAMPLES:
- "Dr [Name] medical specialty [Hospital]"
- "[Name] what kind of doctor [Location]"
MEDICAL SPECIALTIES: Surgery, Internal Medicine, Pediatrics, Cardiology, Neurology, etc.""",
            
            "credentials": """
TARGET: Medical degrees, certifications, board certifications
SEARCH STRATEGY: Find educational background and professional certifications
KEYWORDS: MD, DO, PhD, FACS, board certified, education, medical school, residency, fellowship
QUERY EXAMPLES:
- "Dr [Name] credentials education [Hospital]"
- "[Name] medical degree board certification"
FORMATS: MD, DO, FACS, Board Certified in [Specialty]""",
            
            "linkedin_url": """
TARGET: Professional LinkedIn profile URL
SEARCH STRATEGY: Find their LinkedIn profile with medical context
KEYWORDS: site:linkedin.com, LinkedIn profile, professional network
QUERY EXAMPLES:
- "[Name] LinkedIn profile surgeon doctor [Hospital]"
- "site:linkedin.com [Name] [Specialty] [Location]"
VALIDATION: Ensure profile matches the medical professional""",
            
            "influence_summary": """
TARGET: Research impact, academic influence, professional recognition
SEARCH STRATEGY: Quantify their influence through publications, citations, leadership
KEYWORDS: publications, citations, h-index, research, awards, editorial board, department chair
METRICS TO FIND:
- Publication count ("100+ publications", "published X papers")
- Citation metrics ("cited X times", "h-index of Y")
- Leadership roles ("department chair", "program director", "editorial board")
- Awards and recognition ("best doctor", "top surgeon", "research awards")
QUERY EXAMPLES:
- "Dr [Name] publications citations research impact [Specialty]"
- "[Name] awards recognition leadership [Hospital] [Specialty]"
OUTPUT FORMAT: [Number] publications with [impact metrics]. [Leadership roles]. [Awards/recognition].""",
            
            "strategic_summary": """
TARGET: Strategic value, decision-making authority, institutional influence
SEARCH STRATEGY: Find leadership positions, budget authority, strategic roles
KEYWORDS: chief, director, president, board member, committee chair, advisory, consultant
POSITIONS TO FIND:
- Executive roles ("Chief of Surgery", "Medical Director", "Department Head")
- Board positions ("Hospital Board", "Medical Society Leadership")
- Committee leadership ("Chair", "Committee Member", "Advisory Board")
- Institutional influence ("Decision maker", "Budget authority")
QUERY EXAMPLES:
- "Dr [Name] leadership position director chief [Hospital]"
- "[Name] board member committee chair [Institution]"
OUTPUT FORMAT: [Title/Position] with authority over [scope]. [Key responsibilities]. [Strategic influence].""",
            
            "additional_contacts": """
TARGET: Alternative contact methods - assistants, department contacts, scheduling
SEARCH STRATEGY: Find supporting contacts and alternative ways to reach them
KEYWORDS: assistant, coordinator, scheduler, department contact, office manager, secretary
QUERY EXAMPLES:
- "Dr [Name] assistant scheduler [Hospital]"
- "[Hospital] [Department] contact staff directory"
ACCEPTABLE: Office staff, department contacts, scheduling lines"""
        }
        
        return guidance.get(field, f"""
TARGET: {field} information for medical professional
SEARCH STRATEGY: Find professional information related to {field}
KEYWORDS: {field}, professional, medical, doctor, surgeon
QUERY EXAMPLE: "Dr [Name] {field} [Hospital] [Location]"
""")
    
    
    async def _execute_search(self, query: str, search_config: dict):
        """Execute a single search with the given parameters"""
        search_params = {
            "query": query,
            "search_depth": search_config["search_depth"],
            "max_results": search_config["max_results"],
            "include_raw_content": search_config["include_raw_content"],
            "include_answer": search_config["include_answer"],
            "auto_parameters": search_config["auto_parameters"]
        }
        
        # Add optional parameters if present
        if "chunks_per_source" in search_config:
            search_params["chunks_per_source"] = search_config["chunks_per_source"]
        if "include_domains" in search_config:
            search_params["include_domains"] = search_config["include_domains"]
        
        return await _tavily_wrapper(self.tavily, "search", **search_params)

    async def _cache_result(self, cache_key: str, result: dict, search_config: dict, search_strategy: str):
        """Cache the search result"""
        with _cache_lock:
            cache_entry = {
                "search_result": result.copy() if result else None,
                "sources": [],  # Sources will be populated during extraction
                "credits_used": search_config['credits'],
                "search_strategy": search_strategy,
                "timestamp": datetime.now()
            }
            _enrichment_cache[cache_key] = cache_entry

    def _get_search_config(self, field: str) -> Dict:
        """
        Optimized search configuration based on Tavily API documentation.
        
        Key insights from Tavily docs:
        - max_results can go up to 20 within same credit cost (1 or 2 credits based on search_depth)
        - search_depth determines credit cost: basic=1, advanced=2
        - include_answer="advanced" provides better LLM-generated responses 
        - include_raw_content="markdown" gives structured content for extraction
        - auto_parameters=False prevents unexpected credit usage
        """
        
        # Advanced search (2 credits) - for fields requiring deep analysis or comprehensive discovery
        if field in ["email", "influence_summary", "strategic_summary"]:
            return {
                "search_depth": "advanced",
                "max_results": 15,  # Maximize results within 2-credit cost
                "include_raw_content": "markdown",  # Structured content extraction
                "include_answer": "advanced",  # Detailed LLM-generated answers
                "chunks_per_source": 3,  # Maximum content chunks
                "auto_parameters": False,  # Prevent unexpected upgrades
                "credits": 2,
                "reason": f"Advanced search for comprehensive {field} discovery"
            }
        
        # Basic search (1 credit) - for straightforward factual information
        else:
            config = {
                "search_depth": "basic", 
                "max_results": 12,
                "include_raw_content": "markdown",
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": f"Basic search for {field} information"
            }
            # Scope LinkedIn to linkedin.com
            if field == "linkedin_url":
                config["include_domains"] = ["linkedin.com"]
                config["reason"] = "Domain-scoped LinkedIn lookup"
            return config

    async def _resolve_hospital_domain(self, hospital_name: Optional[str]) -> Optional[str]:
        """Resolve an institution's primary domain via a quick search."""
        try:
            if not hospital_name or not hospital_name.strip():
                return None
            query = f"{hospital_name} official website"
            search_params = {
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_raw_content": "markdown",
                "include_answer": "basic",
                "auto_parameters": False,
            }
            result = await _tavily_wrapper(self.tavily, "search", **search_params)
            if not result or not result.get("results"):
                return None
            # Pick the first sensible domain
            skip_domains = {"linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "wikipedia.org"}
            for item in result["results"]:
                url = item.get("url") or ""
                if not url:
                    continue
                try:
                    from urllib.parse import urlparse
                    netloc = urlparse(url).netloc.lower()
                    # Strip www.
                    if netloc.startswith("www."):
                        netloc = netloc[4:]
                    root = netloc
                    if any(skip in root for skip in skip_domains):
                        continue
                    return root
                except Exception:
                    continue
            return None
        except Exception:
            return None

    async def extract_field_data(self, state: MedicalEnrichmentContext) -> MedicalEnrichmentContext:
        """Extract specific field data using appropriate complexity model and leveraging Tavily's answers"""
        if not state.search_result or not state.search_result.get("results"):
            state.answer = "Information not found"
            state.enrichment_status.append(f"extract_no_results:{datetime.now().isoformat()}")
            return state
        
        try:
            # Get search configuration to understand what type of search was used
            search_config = self._get_search_config(state.target_field)
            
            # Use Tavily's answer if available and relevant (especially for basic searches)
            tavily_answer = state.search_result.get("answer", "")
            
            # For basic searches with good Tavily answers, use them directly with minimal LLM processing
            if search_config["search_depth"] == "basic" and tavily_answer and len(tavily_answer.strip()) > 10:
                # Simple fields can often use Tavily's answer directly with light validation
                if state.target_field in ["email", "phone", "credentials", "specialty", "linkedin_url"]:
                    # Create context for smart extraction
                    validation_context = {
                        'doctor_name': state.name,
                        'tavily_answer': tavily_answer,
                        'search_content': state.search_result.get('raw_content', ''),
                        'search_results': state.search_result.get('results', []),
                        'source_quality': state.search_result.get('source_quality', {}),
                        'direct_answer': True
                    }
                    validated_answer = self._validate_and_extract_field_data(state.target_field, tavily_answer, validation_context)
                    if validated_answer != "Information not found":
                        logger.info(f"Using Tavily answer directly for {state.target_field}: {validated_answer}")
                        state.answer = validated_answer
                        self._update_sources(state)
                        state.last_updated = datetime.now()
                        
                        return state
            
            # For advanced searches or when Tavily answer needs processing, use LLM
            content = self._extract_content(state.search_result)
            if not content.strip():
                state.answer = "Information not found"
                state.enrichment_status.append(f"extract_no_content:{datetime.now().isoformat()}")
                return state
            
            # Deterministic first for LinkedIn/email to reduce variance
            extraction_context = {
                'doctor_name': state.name,
                'tavily_answer': tavily_answer,
                'search_content': content,
                'search_results': state.search_result.get('results', []) if state.search_result else [],
                'source_quality': self._assess_source_quality(state.search_result.get('results', [])) if state.search_result else {},
                'num_sources': len(state.search_result.get('results', [])) if state.search_result else 0,
                'has_linkedin_mentions': any('linkedin' in str(r).lower() for r in (state.search_result.get('results', []) if state.search_result else [])),
                'has_email_mentions': any('@' in str(r) for r in (state.search_result.get('results', []) if state.search_result else [])),
                'hospital_name': state.hospital_name or "",
            }

            if state.target_field in ["linkedin_url", "email"]:
                deterministic_answer = self._validate_and_extract_field_data(state.target_field, tavily_answer or content, extraction_context)
                if deterministic_answer and deterministic_answer != "Information not found":
                    state.answer = deterministic_answer
                    self._update_sources(state)
                    state.last_updated = datetime.now()
                    state.enrichment_status.append(f"extract_success:{datetime.now().isoformat()}:deterministic")
                    return state

            # Determine complexity level for LLM model selection
            complexity = self._get_complexity_level(state.target_field)

            # Build field-specific prompt
            prompt = self._build_extraction_prompt(state, tavily_answer, content)
            
            logger.info(f"Extracting {state.target_field} for {state.name} using {complexity} complexity")
            
            # Add debug logging for LinkedIn URL extraction
            if state.target_field == "linkedin_url":
                logger.info(f"LinkedIn extraction debug - Tavily Answer: '{tavily_answer[:200]}...'")
                logger.info(f"LinkedIn extraction debug - Content length: {len(content)} chars")

            answer = await self.llm.generate(prompt, complexity=complexity)
            
            # Add debug logging for raw GenAI response
            if state.target_field == "linkedin_url":
                logger.info(f"LinkedIn extraction debug - Raw GenAI response: '{answer}'")

            # Enhanced context for smart extraction
            # Reuse the richer context for smart extractor validation
                
            validated_answer = self._validate_and_extract_field_data(
                state.target_field, answer, extraction_context
            )
            
            logger.info(f"Extracted {state.target_field}: {validated_answer}")
            
            state.answer = validated_answer
            self._update_sources(state)
            state.last_updated = datetime.now()
            state.enrichment_status.append(f"extract_success:{datetime.now().isoformat()}:{complexity}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error extracting {state.target_field} for {state.name}: {str(e)}")
            state.answer = "Error during enrichment"
            state.enrichment_status.append(f"extract_error:{datetime.now().isoformat()}:{str(e)}")
            return state

    def _update_sources(self, state: MedicalEnrichmentContext):
        """Update sources from search results with proper isolation per request"""
        if state.search_result and state.search_result.get("results"):
            # Create completely fresh sources array for this specific request
            fresh_sources = []
            for result in state.search_result["results"]:
                # Create fresh source object to prevent any reference sharing
                source = {
                    "title": str(result.get("title", "Unknown Title")),  # Force string copy
                    "url": str(result.get("url", "")),  # Force string copy
                    "snippet": (str(result.get("content", result.get("raw_content", "")))[:200] + "...") if result.get("content", result.get("raw_content", "")) else ""
                }
                fresh_sources.append(source)
            
            # Assign fresh sources list to prevent any cross-contamination
            state.sources = fresh_sources
            
            # Log for debugging source isolation
            logger.debug(f"Updated sources for {state.name} - {state.target_field}: {len(fresh_sources)} sources")
        else:
            state.sources = []  # Ensure empty list if no results

    def _extract_content(self, search_result: Dict) -> str:
        """Extract and concatenate content from search results"""
        content_parts = []
        
        for result in search_result.get("results", []):
            if result.get("raw_content"):
                content_parts.append(result["raw_content"])
            elif result.get("content"):
                content_parts.append(result["content"])
        
        return "\n\n---\n\n".join(content_parts)

    def _get_complexity_level(self, field: str) -> str:
        """Determine model complexity based on field type"""
        complexity_mapping = {
            "email": "simple",
            "phone": "simple", 
            "credentials": "simple",
            "linkedin_url": "simple",
            "specialty": "medium",
            "subspecialty": "medium",
            "additional_contacts": "medium",
            "influence_summary": "complex",
            "strategic_summary": "complex"
        }
        
        return complexity_mapping.get(field, "medium")

    def _build_extraction_prompt(self, state: MedicalEnrichmentContext, tavily_answer: str, content: str) -> str:
        """Build precise, field-optimized extraction prompts that prioritize Tavily's answer"""
        
        base_context = f"Surgeon: {state.name}"
        if state.hospital_name:
            base_context += f" at {state.hospital_name}"
        if state.address:
            base_context += f", {state.address}"
        
        # Clean and limit content to most relevant sections
        content_focused = content[:2500] if len(content) > 2500 else content
        
        prompts = {
            "email": f"""TASK: Extract professional email address for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION INSTRUCTIONS:
You are an expert at finding professional contact information. Find any valid email address for this medical professional.

WHAT TO LOOK FOR:
1. Professional emails: firstname.lastname@hospital.com, doctor@clinic.org
2. Institutional emails: name@university.edu, contact@medicalcenter.org  
3. Practice emails: info@surgicalpractice.com (if associated with this doctor)
4. Personal emails: name@gmail.com (if clearly belonging to this doctor)

ACCEPTANCE CRITERIA:
✓ Direct professional emails (name@hospital.com)
✓ Department emails if they work there (surgery@hospital.com)
✓ Practice/clinic contact emails  
✓ Personal emails if clearly linked to the doctor
✓ Any valid email that can reach this specific doctor

REJECTION CRITERIA:
✗ Automated emails (noreply@, no-reply@, system@)
✗ Generic emails not connected to this person
✗ Broken or incomplete email addresses

RESPONSE GUIDELINES:
- If valid email found: Return ONLY the email address
- If no email available: Say "No email address found"
- If uncertain: Say "Email address not available"

Email address:""",

            "specialty": f"""TASK: Extract primary medical specialty for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

VALID SPECIALTIES ONLY: {', '.join(MEDICAL_SPECIALTIES[:15])}... (and other standard medical specialties)

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, identify specialty mentioned in Tavily Answer
2. Match to closest valid medical specialty from the standard list
3. Look for keywords like "Department of", "specializes in", "board certified in"
4. Return ONLY the specialty name - no explanations
5. If unclear between multiple, choose the most prominent one

RETURN EXACTLY:""",

            "subspecialty": f"""TASK: Extract subspecialty/focus area for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, look for subspecialty in Tavily Answer (e.g., "pediatric surgery", "breast cancer", "spine surgery")
2. Look for fellowship training, specialized procedures, or focus areas
3. Return the most specific clinical focus area
4. Return ONLY the subspecialty - no explanations
5. If multiple, return the most prominent one

RETURN EXACTLY:""",

            "credentials": f"""TASK: Extract medical credentials for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION INSTRUCTIONS:
You are an expert at identifying medical credentials. Find the degrees, certifications, and qualifications for this medical professional.

WHAT TO LOOK FOR:
1. Medical degrees: MD, DO, PhD, DDS, DVM, PharmD
2. Board certifications: FACS, FRCSC, FRCS, FACC, FACP, FACEP, FAAOS
3. Speciality certifications: Board Certified in [Specialty]
4. Fellowship designations: Fellow of [Organization]
5. Academic credentials: Professor, Associate Professor, etc.

EXTRACTION PATTERNS:
- Look for: "Dr. Name, MD, FACS" or "John Smith, MD, FACS, Board Certified"
- Include degrees AND certifications: "MD, FACS" or "DO, Board Certified in Surgery"
- Separate multiple credentials with commas

RESPONSE GUIDELINES:
- If credentials found: Return in format "MD, FACS" or "MD, FACS, Board Certified"
- Include only verified credentials from reliable sources
- If no clear credentials: Say "Credentials not available"
- DO NOT make up or assume credentials

Credentials:""",

            "linkedin_url": f"""TASK: Extract LinkedIn profile URL for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION INSTRUCTIONS:
You are an expert at extracting LinkedIn profile URLs. Your task is to find the correct LinkedIn profile for this medical professional.

WHAT TO LOOK FOR:
1. Complete LinkedIn URLs: https://linkedin.com/in/firstname-lastname-md, https://www.linkedin.com/pub/name
2. Profile URLs matching the surgeon's name and medical background
3. URLs from reliable sources in search results

RESPONSE GUIDELINES:
- If you find a valid LinkedIn URL: Return ONLY the complete URL
- If clearly no profile exists: Say "No LinkedIn profile found"  
- If uncertain or information is unclear: Say "LinkedIn profile not available"
- DO NOT make up URLs or return partial/broken links
- DO NOT return generic responses like "Information not found"

VALIDATION:
- Verify the profile name reasonably matches the surgeon
- Prefer complete https:// URLs over partial ones
- Choose the most professional/complete profile if multiple found

LinkedIn URL:""",

            "influence_summary": f"""TASK: Create influence summary for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, use information from Tavily Answer to assess influence
2. Focus on: publications count, citations, h-index, research leadership, awards
3. Look for quantifiable metrics: "100+ publications", "h-index of 25", "cited 1000+ times"
4. Include leadership positions: department chair, society president, editorial board
5. Write 2-3 concise sentences focusing on measurable impact

FORMAT: [Number] publications with [impact metrics]. [Leadership roles]. [Recognition/awards].""",

            "strategic_summary": f"""TASK: Create strategic positioning summary for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, use Tavily Answer to identify strategic position
2. Focus on: department leadership, decision-making authority, budget responsibility
3. Look for titles: Chief, Director, Head, Chair, Vice President
4. Include scope: department size, patient volume, procedure specialization
5. Write 2-3 concise sentences on organizational influence

FORMAT: [Title/Position] with authority over [scope]. [Key responsibilities]. [Strategic influence].""",

            "additional_contacts": f"""TASK: Extract additional contact information for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, look for contact info in Tavily Answer
2. Find: office manager, assistant, appointment line, department contact
3. Include names and roles: "Mary Smith, Office Manager"
4. Return ONLY the contact information - no explanations
5. Format: "Name, Role" or "Department: Phone"

RETURN EXACTLY:""",

            "phone": f"""TASK: Extract phone number for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, extract phone from Tavily Answer if present
2. Look for patterns: (xxx) xxx-xxxx, xxx-xxx-xxxx, +1-xxx-xxx-xxxx
3. Prioritize office/practice numbers over personal
4. Return ONLY the phone number - no explanations
5. If multiple, return the most professional/official one

RETURN EXACTLY:"""
        }
        
        return prompts.get(state.target_field, f"""TASK: Extract {state.target_field} for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"
BACKUP CONTENT: {content_focused}

Extract the requested information, prioritizing the Tavily Answer.
RETURN EXACTLY:""")

    def _assess_source_quality(self, search_results: List[dict]) -> dict:
        """Simple source quality assessment."""
        return {
            'has_results': len(search_results) > 0,
            'result_count': len(search_results)
        }

    def _validate_and_extract_field_data(self, field: str, raw_response: str, context: dict = None) -> str:
        """
        Clean, working field data extraction using the smart extractor.
        This replaces the old complex validation with a simple, effective approach.
        """
        try:
            # Use the new smart field extractor
            context_dict = context or {}
            
            result = smart_extractor.extract_field_data(field, raw_response, context_dict)
            
            # Log the result
            if result != "Information not found":
                logger.info(f"{field.capitalize()} extraction successful: {result}")
            else:
                logger.info(f"{field.capitalize()} extraction failed - Raw response: '{raw_response}'")
            
            return result
                
        except Exception as e:
            logger.error(f"Error in field validation for {field}: {str(e)}")
            return "Information not found"

    def build_graph(self):
        """Build and compile the medical enrichment graph with a lightweight planner."""
        async def planner(state: MedicalEnrichmentContext) -> str:
            # Route deterministically for fields with known strategies
            if state.target_field in ["linkedin_url", "email", "phone"]:
                return "search"  # use specialized search configs
            return "search"

        graph = StateGraph(MedicalEnrichmentContext)
        graph.add_node("plan", planner)
        graph.add_node("search", self.search_medical_data)
        graph.add_node("extract", self.extract_field_data)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "extract")
        graph.add_edge("extract", END)
        return graph.compile()


async def enrich_medical_field(
    name: str,
    target_field: str,
    hospital_name: Optional[str] = None,
    address: Optional[str] = None,
    phone: Optional[str] = None,
    all_context: Optional[Dict[str, str]] = None,
    tavily_client=None,
    llm_provider: LLMProvider = None,
    query_llm_provider: LLMProvider = None
) -> MedicalEnrichmentContext:
    """
    Enrich a specific field for a medical professional.
    
    Args:
        name: Surgeon's full name
        target_field: Field to enrich (email, specialty, credentials, etc.)
        hospital_name: Primary hospital affiliation
        address: Physical address
        phone: Contact phone
        tavily_client: Tavily search client
        llm_provider: LLM provider instance for content extraction
        query_llm_provider: Lightweight LLM provider instance for query generation
        
    Returns:
        MedicalEnrichmentContext with enriched data and metadata
    """
    try:
        # Early exit for blank names
        if not name or not name.strip():
            return MedicalEnrichmentContext(
                name=name or "",
                target_field=target_field,
                answer="",
                enrichment_status=["blank_input:skipped"],
                last_updated=datetime.now()
            )
        
        logger.info(f"Starting medical enrichment for {name}, field: {target_field}")
        
        pipeline = MedicalEnrichmentPipeline(tavily_client, llm_provider, query_llm_provider)
        
        initial_context = MedicalEnrichmentContext(
            name=name,
            hospital_name=hospital_name,
            address=address,
            phone=phone,
            all_context=all_context or {},
            target_field=target_field,
            last_updated=datetime.now()
        )
        
        graph = pipeline.build_graph()
        result = await graph.ainvoke(initial_context)
        
        # Handle langgraph returning dict instead of MedicalEnrichmentContext object
        if isinstance(result, dict):
            # Extract the final state from the dict - langgraph returns the state dict
            # The result dict contains the final state values
            logger.info(f"Converting dict result to MedicalEnrichmentContext for {name}")
            return MedicalEnrichmentContext(
                name=result.get("name", name),
                hospital_name=result.get("hospital_name", hospital_name),
                address=result.get("address", address),
                phone=result.get("phone", phone),
                target_field=result.get("target_field", target_field),
                email=result.get("email"),
                specialty=result.get("specialty"),
                subspecialty=result.get("subspecialty"),
                credentials=result.get("credentials"),
                linkedin_url=result.get("linkedin_url"),
                influence_summary=result.get("influence_summary"),
                strategic_summary=result.get("strategic_summary"),
                additional_contacts=result.get("additional_contacts"),
                search_result=result.get("search_result"),
                sources=result.get("sources", []),
                enrichment_status=result.get("enrichment_status", []),
                credits_used=result.get("credits_used", 0),
                search_strategy=result.get("search_strategy"),
                last_updated=result.get("last_updated") or datetime.now(),
                answer=result.get("answer", "Information not found")
            )
        
        # Ensure result is a MedicalEnrichmentContext object (this should not be needed now)
        if not isinstance(result, MedicalEnrichmentContext):
            logger.error(f"Graph returned unexpected type {type(result)} for {name}")
            return MedicalEnrichmentContext(
                name=name,
                hospital_name=hospital_name,
                target_field=target_field,
                answer="Error during enrichment",
                enrichment_status=[f"type_error:{datetime.now().isoformat()}"],
                last_updated=datetime.now()
            )
        
        # Update enrichment status with credit information
        credits_used = getattr(result, 'credits_used', 0)
        strategy = getattr(result, 'search_strategy', 'unknown')
        credits_info = f"credits:{credits_used}"
        strategy_info = f"strategy:{strategy}"
        result.enrichment_status.append(f"{target_field}:{datetime.now().isoformat()}:{credits_info}:{strategy_info}")
        
        logger.info(f"Completed medical enrichment for {name} - Credits used: {credits_used}")
        return result
        
    except Exception as e:
        logger.error(f"Error in medical enrichment for {name}: {str(e)}")
        # Always return a proper MedicalEnrichmentContext
        error_context = MedicalEnrichmentContext(
            name=name,
            hospital_name=hospital_name,
            target_field=target_field,
            answer="Error during enrichment",
            enrichment_status=[f"error:{datetime.now().isoformat()}:{str(e)}"],
            last_updated=datetime.now(),
            credits_used=0  # Ensure credits_used is set
        )
        return error_context


async def enrich_cell_with_graph(
    column_name: str,
    target_value: str,
    context_values: Dict[str, str],
    tavily_client,
    llm_provider: LLMProvider
) -> Dict:
    """
    Legacy function for backward compatibility.
    Maps to medical enrichment pipeline.
    """
    try:
        # Early exit for blank values
        if not target_value or not target_value.strip():
            return {
                "answer": "",
                "search_result": {},
                "sources": []
            }
        
        # Extract medical context from the general context
        hospital_name = context_values.get("hospital_name") or context_values.get("Hospital")
        address = context_values.get("address") or context_values.get("Address") 
        phone = context_values.get("phone") or context_values.get("Phone")
        
        # Keep ALL context for intelligent query generation
        all_context = {k: v for k, v in context_values.items() if v and str(v).strip()}
        
        # Map column_name to medical field
        field_mapping = {
            "email": "email",
            "Email": "email", 
            "specialty": "specialty",
            "Specialty": "specialty",
            "credentials": "credentials",
            "Credentials": "credentials",
            "linkedin": "linkedin_url",
            "LinkedIn": "linkedin_url",
            "phone": "phone",
            "Phone": "phone"
        }
        
        target_field = field_mapping.get(column_name, column_name.lower())
        
        result = await enrich_medical_field(
            name=target_value,
            target_field=target_field,
            hospital_name=hospital_name,
            address=address,
            phone=phone,
            all_context=all_context,
            tavily_client=tavily_client,
            llm_provider=llm_provider
        )
        
        # Convert sources to legacy format
        legacy_sources = []
        for source in result.sources:
            if isinstance(source, dict):
                legacy_sources.append({
                    "title": source.get("title", ""),
                    "url": source.get("url", "")
                })
            else:
                # Handle old string format
                legacy_sources.append({
                    "title": "Source",
                    "url": str(source)
                })
        
        # Return in legacy format for compatibility
        return {
            "answer": result.answer or "Information not found",
            "search_result": result.search_result or {"results": []},
            "sources": legacy_sources
        }
        
    except Exception as e:
        logger.error(f"Error in legacy enrichment function: {str(e)}")
        return {
            "answer": "Error during enrichment",
            "search_result": {"results": []},
            "sources": []
        }
