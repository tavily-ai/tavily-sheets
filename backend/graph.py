import asyncio
import logging
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union
import os
import asyncio
import logging
import time
import uuid
from functools import wraps

from dotenv import load_dotenv
from google import genai
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from aiolimiter import AsyncLimiter
from cachetools import TTLCache
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Global rate limiter and cache - VERY conservative to protect credits
_tavily_rate_limiter = AsyncLimiter(10, 60)  # Only 10 requests per minute (was 100)
_tavily_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests (was 8)
_enrichment_cache = TTLCache(maxsize=1000, ttl=900)  # 15-minute TTL cache
_cache_lock = threading.RLock()

# Data persistence settings
PERSISTENCE_DIR = "data"
RESULTS_FILE = os.path.join(PERSISTENCE_DIR, "enrichment_results.json")

def ensure_persistence_dir():
    """Ensure the persistence directory exists"""
    if not os.path.exists(PERSISTENCE_DIR):
        os.makedirs(PERSISTENCE_DIR)

def save_enrichment_result(name: str, field: str, result: dict):
    """Save enrichment result to persistent storage"""
    try:
        ensure_persistence_dir()
        
        # Load existing results
        all_results = {}
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        
        # Create key for this person
        person_key = name.strip()
        if person_key not in all_results:
            all_results[person_key] = {}
        
        # Save the result with timestamp
        all_results[person_key][field] = {
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'timestamp': datetime.now().isoformat(),
            'search_strategy': result.get('search_strategy', ''),
            'credits_used': result.get('credits_used', 0)
        }
        
        # Save back to file
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved enrichment result for {name} - {field}")
        
    except Exception as e:
        logger.error(f"Error saving enrichment result: {e}")

def load_enrichment_result(name: str, field: str) -> Optional[dict]:
    """Load enrichment result from persistent storage"""
    try:
        if not os.path.exists(RESULTS_FILE):
            return None
            
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
            
        person_key = name.strip()
        if person_key in all_results and field in all_results[person_key]:
            result = all_results[person_key][field]
            logger.info(f"Loaded cached result for {name} - {field}")
            return result
            
    except Exception as e:
        logger.error(f"Error loading enrichment result: {e}")
    
    return None

# Global trace context
_trace_context = {}

def get_trace_id() -> str:
    """Generate or retrieve trace ID for current request"""
    return str(uuid.uuid4())

async def _tavily_wrapper(tavily_client, operation: str, **kwargs):
    """
    Centralized wrapper for all Tavily API calls with rate limiting, 
    retry logic, and telemetry.
    """
    trace_id = get_trace_id()
    start_time = time.time()
    
    # Rate limiting
    async with _tavily_rate_limiter:
        async with _tavily_semaphore:
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    logger.info(f"[{trace_id}] Executing Tavily {operation} (attempt {retry_count + 1})")
                    
                    if operation == "search":
                        result = await asyncio.to_thread(
                            lambda: tavily_client.search(**kwargs)
                        )
                    else:
                        raise ValueError(f"Unknown Tavily operation: {operation}")
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[{trace_id}] Tavily {operation} completed in {elapsed:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    elapsed = time.time() - start_time
                    
                    # Don't retry on auth errors (4xx)
                    if "401" in str(e) or "403" in str(e) or "api_key" in str(e).lower():
                        logger.error(f"[{trace_id}] Tavily auth error: {str(e)}")
                        raise
                    
                    if retry_count >= max_retries:
                        logger.error(f"[{trace_id}] Tavily {operation} failed after {max_retries} attempts: {str(e)}")
                        raise
                    
                    # Exponential backoff with much longer delays to protect credits
                    delay = (2 ** retry_count) * 5  # 5s, 10s, 20s delays
                    logger.warning(f"[{trace_id}] Tavily {operation} failed (attempt {retry_count}), retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)

# Medical specialties for validation
MEDICAL_SPECIALTIES = [
    "Anesthesiology", "Cardiology", "Dermatology", "Emergency Medicine", 
    "Family Medicine", "Gastroenterology", "General Surgery", "Internal Medicine",
    "Neurology", "Neurosurgery", "Obstetrics and Gynecology", "Oncology",
    "Ophthalmology", "Orthopedic Surgery", "Otolaryngology", "Pathology",
    "Pediatrics", "Plastic Surgery", "Psychiatry", "Pulmonology",
    "Radiology", "Rheumatology", "Urology", "Vascular Surgery",
    "Cardiac Surgery", "Thoracic Surgery", "Transplant Surgery",
    "Trauma Surgery", "Pediatric Surgery", "Interventional Cardiology"
]

"""
TAVILY CREDIT OPTIMIZATION STRATEGY:

Credit Usage:
- Basic Search: 1 credit - for straightforward factual data (email, phone, credentials, specialty, linkedin)
- Advanced Search: 2 credits - for complex analysis requiring quality content (influence_summary, strategic_summary, subspecialty)

Field Classification:
- BASIC (1 credit): email, phone, credentials, specialty, linkedin_url, additional_contacts
  - Reason: These are typically found in standard bio/contact sections
  - Strategy: Use basic search + Tavily's basic answer, minimal LLM processing
  
- ADVANCED (2 credits): influence_summary, strategic_summary, subspecialty
  - Reason: Require deep content analysis, nuanced understanding
  - Strategy: Advanced search + advanced answer + full LLM processing with appropriate complexity

This approach can reduce credit usage by ~40% while maintaining quality for complex fields.
"""

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
    "complex": "gemini-2.5-flash-pro"     # Strategic analysis, influence assessment
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
            
        try:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            raise


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
    
    def __init__(self, tavily_client, llm_provider: LLMProvider):
        self.tavily = tavily_client
        self.llm = llm_provider
        
        # Create lightweight LLM for query generation (cheaper, faster)
        if hasattr(llm_provider, 'project_id'):
            self.query_llm = VertexAIProvider("gemini-2.5-flash-lite", llm_provider.project_id)
        else:
            self.query_llm = llm_provider  # Fallback to main LLM

    async def search_medical_data(self, state: MedicalEnrichmentContext) -> MedicalEnrichmentContext:
        """Search for medical professional data with simple, effective approach"""
        try:
            # Check for persistent cached result first
            cached_result = load_enrichment_result(state.name, state.target_field)
            if cached_result:
                logger.info(f"Using persistent cached result for {state.name} - {state.target_field}")
                state.answer = cached_result.get('answer', 'Information not found')
                state.sources = cached_result.get('sources', [])
                state.search_strategy = f"cached - {cached_result.get('search_strategy', 'unknown')}"
                state.credits_used = 0  # No credits used for cached results
                return state
            
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
            
            # Build intelligent, context-aware query
            query = await self._build_simple_search_query(state)
            search_config = self._get_search_config(state.target_field)
            
            logger.info(f"Searching for {state.target_field} with query: {query}")
            
            # Perform single search
            result = await self._execute_search(query, search_config)
            
            logger.info(f"Tavily search completed for {state.name} (credits used: {search_config['credits']})")
            
            # Update state with results
            state.search_result = result
            state.credits_used += search_config['credits']
            state.search_strategy = f"{search_config['search_depth']} ({search_config['reason']})"
            
            # Cache the result
            await self._cache_result(cache_key, result, search_config, state.search_strategy)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in medical search for {state.name}: {str(e)}")
            state.answer = "Search error occurred"
            state.enrichment_status.append(f"search_error:{datetime.now().isoformat()}:{str(e)}")
            return state

    async def _build_simple_search_query(self, state: MedicalEnrichmentContext) -> str:
        """Use LLM to generate intelligent, context-aware search queries"""
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
        
        query_prompt = f"""Generate ONE precise web search query to find {state.target_field} for this medical professional.

PERSON DETAILS:
{full_context}

SEARCH GOAL: Find their {state.target_field}

{self._get_field_specific_guidance(state.target_field)}

RULES:
1. Use the person's full name in quotes for precision
2. Include hospital/institution if provided - critical for accuracy
3. Use ALL relevant context details that help identify this specific person
4. Add medical keywords to avoid non-medical results
5. Make it specific enough to find the RIGHT person
6. Keep it concise but comprehensive

Return only the search query:"""

        try:
            # Use lightweight model for query generation
            response = await self.query_llm.generate(query_prompt, complexity="simple")
            generated_query = response.strip()
            
            # Clean up the response
            if "Query:" in generated_query:
                generated_query = generated_query.split("Query:")[-1].strip()
            if generated_query.startswith('"') and generated_query.endswith('"'):
                generated_query = generated_query[1:-1]
            
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
        """Get field-specific search guidance for more targeted queries"""
        guidance = {
            "email": """
TARGET: Professional email address (personal, clinic, hospital, or department)
SEARCH FOR: Contact pages, staff directories, hospital listings, practice websites, "email" or "contact"
KEYWORDS: email, contact, directory, staff, faculty""",
            
            "phone": """
TARGET: Professional phone number (office, clinic, or hospital)
SEARCH FOR: Practice listings, hospital directories, office contact information
KEYWORDS: phone, telephone, office, clinic, contact, appointment""",
            
            "specialty": """
TARGET: Medical specialty and subspecialty
SEARCH FOR: Department listings, medical credentials, hospital biographies, practice descriptions
KEYWORDS: specialty, specializes in, department of, division of, board certified""",
            
            "credentials": """
TARGET: Medical degrees, certifications, board certifications
SEARCH FOR: Education background, medical qualifications, certifications, CV/resume
KEYWORDS: MD, DO, PhD, FACS, board certified, education, medical school, residency""",
            
            "linkedin_url": """
TARGET: Professional LinkedIn profile
SEARCH FOR: LinkedIn profile with medical/professional context
KEYWORDS: site:linkedin.com, LinkedIn profile, professional network""",
            
            "influence_summary": """
TARGET: Research impact, publications, citations, leadership roles, recognition
SEARCH FOR: Publication counts, h-index, citation metrics, research leadership, awards, editorial roles
KEYWORDS: publications, citations, h-index, research, awards, editorial board, department chair, fellowship director
SPECIFIC METRICS: Look for "100+ publications", "h-index", "cited X times", "impact factor", "research grants"
ACADEMIC ROLES: Department head, program director, editorial board, society leadership""",
            
            "strategic_summary": """
TARGET: Strategic value, network connections, decision-making influence, institutional roles
SEARCH FOR: Leadership positions, committee memberships, advisory roles, industry connections
KEYWORDS: chief, director, president, board member, committee chair, advisory, consultant, key opinion leader
STRATEGIC ROLES: Hospital board, medical society leadership, advisory committees, consultant roles""",
            
            "additional_contacts": """
TARGET: Alternative contact methods, assistant contacts, department contacts
SEARCH FOR: Office staff, administrative contacts, department phone/email, scheduling contacts
KEYWORDS: assistant, coordinator, scheduler, department contact, office manager"""
        }
        
        return guidance.get(field, f"""
TARGET: {field} information
SEARCH FOR: Professional information related to {field}
KEYWORDS: {field}, professional, medical""")
    
    
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
        Determine optimal search configuration based on field requirements.
        
        Credit optimization strategy (based on Tavily documentation):
        - Basic (1 credit): Simple factual data, max_results ≤ 5 optimal, include_raw_content for extraction
        - Advanced (2 credits): Complex analysis, chunks_per_source 1-3, include_raw_content + advanced answers
        
        Key optimizations:
        - Explicitly set search_depth to prevent auto_parameters upgrading to advanced
        - Use include_domains for targeted searches when beneficial
        - Minimize max_results while maintaining quality
        - Use include_answer strategically to reduce LLM processing
        """
        
        # Fields that need ADVANCED search (2 credits) - require deep analysis and quality content
        advanced_fields = {
            "influence_summary": {
                "search_depth": "advanced",
                "max_results": 3,  # Reduced from 5 - advanced gets better quality per result
                "include_raw_content": "markdown",  # More structured than True
                "include_answer": "advanced",  # Detailed answer reduces LLM work
                "chunks_per_source": 2,  # Reduced from 3 - still quality but less tokens
                "auto_parameters": False,  # Prevent automatic upgrades
                "credits": 2,
                "reason": "Complex analysis of publications, citations, leadership roles"
            },
            "strategic_summary": {
                "search_depth": "advanced", 
                "max_results": 3,  # Reduced from 5
                "include_raw_content": "markdown",
                "include_answer": "advanced",
                "chunks_per_source": 2,  # Focused chunks
                "auto_parameters": False,
                "credits": 2,
                "reason": "Organizational impact analysis requires quality content extraction"
            },
            "subspecialty": {
                "search_depth": "advanced",
                "max_results": 3,  # Reduced from 4
                "include_raw_content": "markdown",
                "include_answer": "basic",  # Basic answer sufficient for subspecialty
                "chunks_per_source": 2,
                "auto_parameters": False,
                "credits": 2,
                "reason": "Subspecialties require nuanced understanding from quality sources"
            }
        }
        
        # Fields that work well with BASIC search (1 credit) - optimized for efficiency
        basic_fields = {
            "email": {
                "search_depth": "basic",
                "max_results": 10,  # Focused results for better quality
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": "Context-driven email search"
            },
            "phone": {
                "search_depth": "basic",
                "max_results": 8,
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": "Context-driven phone search"
            },
            "credentials": {
                "search_depth": "basic",
                "max_results": 3,  # Credentials may be in multiple places
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": "Medical credentials in bio sections"
            },
            "specialty": {
                "search_depth": "basic",
                "max_results": 5,  # Usually prominently displayed
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": "Primary specialty prominently listed"
            },
            "linkedin_url": {
                "search_depth": "basic",
                "max_results": 3,  # LinkedIn should be top result
                "include_raw_content": False,  # URLs don't need content extraction
                "include_answer": "basic",
                "auto_parameters": False,
                "credits": 1,
                "reason": "Context-driven LinkedIn search"
            },
            "additional_contacts": {
                "search_depth": "basic",
                "max_results": 5,  # Increased for broader search
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,
                # Removed domain restrictions
                "credits": 1,
                "reason": "Comprehensive contact search across all sources"
            }
        }
        
        # Return appropriate configuration
        if field in advanced_fields:
            return advanced_fields[field]
        elif field in basic_fields:
            return basic_fields[field]
        else:
            # Default to basic for unknown fields to minimize costs
            return {
                "search_depth": "basic",
                "max_results": 2,
                "include_raw_content": "text",
                "include_answer": "basic",
                "auto_parameters": False,  # Critical: prevent auto-upgrade to advanced
                "credits": 1,
                "reason": "Unknown field - cost-efficient basic search"
            }

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
                    validated_answer = self._validate_field_data(state.target_field, tavily_answer)
                    if validated_answer != "Information not found":
                        logger.info(f"Using Tavily answer directly for {state.target_field}: {validated_answer}")
                        state.answer = validated_answer
                        self._update_sources(state)
                        state.last_updated = datetime.now()
                        
                        # Save result to persistent storage
                        result_data = {
                            'answer': state.answer,
                            'sources': state.sources,
                            'search_strategy': state.search_strategy,
                            'credits_used': state.credits_used
                        }
                        save_enrichment_result(state.name, state.target_field, result_data)
                        
                        return state
            
            # For advanced searches or when Tavily answer needs processing, use LLM
            content = self._extract_content(state.search_result)
            if not content.strip():
                state.answer = "Information not found"
                state.enrichment_status.append(f"extract_no_content:{datetime.now().isoformat()}")
                return state
            
            # Determine complexity level for LLM model selection
            complexity = self._get_complexity_level(state.target_field)
            
            # Build field-specific prompt
            prompt = self._build_extraction_prompt(state, tavily_answer, content)
            
            logger.info(f"Extracting {state.target_field} for {state.name} using {complexity} complexity")
            
            answer = await self.llm.generate(prompt, complexity=complexity)
            
            # Validate and clean the answer
            validated_answer = self._validate_field_data(state.target_field, answer)
            
            logger.info(f"Extracted {state.target_field}: {validated_answer}")
            
            state.answer = validated_answer
            self._update_sources(state)
            state.last_updated = datetime.now()
            state.enrichment_status.append(f"extract_success:{datetime.now().isoformat()}:{complexity}")
            
            # Save result to persistent storage
            result_data = {
                'answer': state.answer,
                'sources': state.sources,
                'search_strategy': state.search_strategy,
                'credits_used': state.credits_used
            }
            save_enrichment_result(state.name, state.target_field, result_data)
            
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
            "email": f"""Find the email address for {base_context}

Search this content for email addresses:

Tavily Answer: {tavily_answer}

Additional Content: {content_focused}

Instructions:
- Look for any email address (personal, department, clinic, office)
- If no direct email found, look for clinic/department/hospital emails
- Extract just the email address, nothing else
- If truly no email exists anywhere, return "Information not found"

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

EXTRACTION RULES:
1. First, extract credentials from Tavily Answer (MD, DO, FACS, board certifications, etc.)
2. Look for patterns: "Dr. Name, MD", "John Smith, MD, FACS", board certifications
3. Include degrees and certifications, separate with commas
4. Return ONLY credentials - no explanations
5. Standard format: "MD, FACS" or "MD, FACS, Board Certified"

RETURN EXACTLY:""",

            "linkedin_url": f"""TASK: Extract LinkedIn profile URL for {base_context}

PRIORITY SOURCE - Tavily Answer: "{tavily_answer}"

BACKUP CONTENT (only if Tavily Answer is insufficient):
{content_focused}

EXTRACTION RULES:
1. First, look for LinkedIn URL in Tavily Answer
2. Search for patterns like "linkedin.com/in/name" or "linkedin.com/pub/name"
3. Verify the profile matches the surgeon's name
4. Return ONLY the complete LinkedIn URL - no explanations
5. If multiple profiles, return the most complete/professional one

RETURN EXACTLY:""",

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

    def _validate_field_data(self, field: str, data: str) -> str:
        """Validate and clean extracted field data with flexible acceptance criteria"""
        if not data or data.strip().lower() in ["information not found", "not found", "n/a", "", "none", "null"]:
            return "Information not found"
        
        cleaned = data.strip()
        
        # Field-specific validation with enhanced flexibility
        if field == "email":
            return self._validate_email_data(cleaned)
                
        elif field == "phone":
            # Basic phone number validation
            import re
            phone_pattern = r'[\(\)\s\-\.\+\d]+'
            if re.search(r'\d', cleaned):  # Must contain at least one digit
                return cleaned
            return "Information not found"
            
        elif field == "specialty":
            # Find closest match in MEDICAL_SPECIALTIES
            for specialty in MEDICAL_SPECIALTIES:
                if specialty.lower() in cleaned.lower() or cleaned.lower() in specialty.lower():
                    return specialty
                    
        elif field == "linkedin_url":
            if "linkedin.com" in cleaned.lower():
                # Extract URL if it's in longer text
                import re
                url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
                urls = re.findall(url_pattern, cleaned)
                linkedin_urls = [url for url in urls if "linkedin.com" in url.lower()]
                if linkedin_urls:
                    return linkedin_urls[0]
                return cleaned
            return "Information not found"
        
        return cleaned

    def _validate_email_data(self, data: str) -> str:
        """Simple email validation that extracts clean email addresses"""
        import re
        
        # First try to extract standard email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, data)
        if emails:
            return emails[0]  # Return first valid email found
        
        # If no email found, return not found
        return "Information not found"

    def build_graph(self):
        """Build and compile the medical enrichment graph"""
        graph = StateGraph(MedicalEnrichmentContext)
        graph.add_node("search", self.search_medical_data)
        graph.add_node("extract", self.extract_field_data)
        graph.add_edge(START, "search")
        graph.add_edge("search", "extract")
        graph.add_edge("extract", END)
        compiled_graph = graph.compile()
        return compiled_graph


async def enrich_medical_field(
    name: str,
    target_field: str,
    hospital_name: Optional[str] = None,
    address: Optional[str] = None,
    phone: Optional[str] = None,
    all_context: Optional[Dict[str, str]] = None,
    tavily_client=None,
    llm_provider: LLMProvider = None
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
        llm_provider: LLM provider instance
        
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
        
        pipeline = MedicalEnrichmentPipeline(tavily_client, llm_provider)
        
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
