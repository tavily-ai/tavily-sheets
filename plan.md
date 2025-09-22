# Sylke Medical Enrichment System: Intelligent Agent Architecture Plan

## Executive Summary

The current system suffers from rigid, hardcoded logic that prevents intelligent adaptation and contextual reasoning. This plan outlines a transformation to create an intelligent agent that understands business context (Sylke silk fibroin wound closure technology) and can reason about its data collection objectives with flexibility and sophistication.

**Product Context:** Sylke is a silk fibroin-based wound closure technology targeting surgeons in:
- **Plastic Surgery:** Reconstructive, Aesthetic, Cosmetic procedures
- **Orthopedics:** Joint, Spine, Sports Medicine, Extremity procedures  
- **Cardio:** Cardiothoracic, Cardiovascular procedures
- **OB/GYN:** C-sections, Robotic surgery

## Core Issues Identified

### 1. **Rigid Email Finding Logic**
**Current Problem:** Hardcoded fallback patterns without contextual understanding of why contact information is needed.

**Current Flow:**
```
Primary Search → Fallback Search → Extract → Done
```

**Intelligent Solution:** Context-aware reasoning with business objective understanding:
```
Objective Understanding → Adaptive Strategy → Contextual Search → Validation → Re-strategize (if needed)
```

### 2. **Hardcoded Search Configurations**
**Current Problem:** Fixed search depths and strategies in `_get_search_config()` without contextual adaptation.

**Solution:** Dynamic configuration through prompt-driven strategy selection using standard Tavily tool with context-specific parameters.

### 3. **Linear Pipeline Architecture**
**Current Problem:** Rigid START → plan → search → extract → END flow with no learning or adaptation.

**Solution:** Intelligent routing with self-correction capabilities using gemini-2.5-flash for efficient reasoning.

### 4. **Poor Retrieval Performance**
**Current Problem:** Generic queries that don't understand the business objective or target audience context.

### 5. **Lack of Business Context Intelligence**
**Current Problem:** No understanding that this is for Sylke wound closure technology outreach to surgical decision-makers.

## Intelligent Agent Architecture Plan

### Phase 1: Context-Aware Intelligence

#### 1.1 Business Context Understanding
**Enhancement to:** `backend/graph.py` - Modify existing `MedicalEnrichmentContext`

Add business context awareness without rigid hierarchies:

```python
@dataclass
class MedicalEnrichmentContext:
    # ... existing fields ...
    business_objective: str = "Find professional contact information for introducing Sylke silk fibroin wound closure technology"
    target_specialties: List[str] = field(default_factory=lambda: [
        "plastic surgery", "orthopedics", "cardiothoracic", "ob/gyn"
    ])
    contact_reasoning: Optional[str] = None  # Why this contact method was chosen
    
    def get_specialty_relevance(self) -> str:
        """Determine relevance to Sylke's target specialties"""
        if self.all_context:
            specialty = self.all_context.get('specialty', '').lower()
            for target_spec in self.target_specialties:
                if target_spec in specialty:
                    return f"High relevance - {specialty} procedures use wound closure"
        return "General surgical relevance - all procedures require wound closure"
```

#### 1.2 Prompt Manager for Dynamic Instructions
**File:** `prompts/field_instructions.py` (NEW - lightweight prompts only)

```python
FIELD_INSTRUCTIONS = {
    "email": {
        "objective": "Find any professional email that allows us to introduce Sylke wound closure technology to this surgeon or their decision-making team",
        "reasoning": "We need a contact method to share how Sylke can improve their surgical outcomes",
        "acceptable_contacts": [
            "Surgeon's direct professional email (ideal for personalized introduction)",
            "Department email (surgical department can evaluate new technologies)", 
            "Clinic email (practice-level decisions)",
            "Administrative email (if connected to this specific surgeon)",
            "Any professional contact that reaches surgical decision-makers"
        ],
        "search_guidance": "Ask natural questions about how to reach this surgeon professionally for introducing innovative wound closure technology"
    },
    
    "specialty": {
        "objective": "Determine surgical specialty to understand relevance to Sylke's wound closure applications",
        "reasoning": "Different specialties have different wound closure needs and decision-making processes",
        "relevance_mapping": {
            "plastic_surgery": "High - extensive suturing, aesthetic outcomes critical",
            "orthopedics": "High - joint surgeries, sports medicine applications",
            "cardiothoracic": "High - precision closure for cardiac procedures",
            "ob_gyn": "High - cesarean sections, robotic surgery applications"
        }
    },
    
    "credentials": {
        "objective": "Understand surgeon's qualifications to assess their influence in wound closure technology adoption",
        "reasoning": "Board-certified surgeons with fellowships are key opinion leaders"
    }
}
```

### Phase 2: Agentic Pipeline Architecture

#### 2.1 Replace Linear Graph with Cyclical Reasoning
**File:** `backend/agentic_pipeline.py` (NEW)

```python
class AgenticEnrichmentPipeline:
    def __init__(self, tavily_client, llm_provider, business_context):
        self.context_manager = business_context
        self.max_iterations = 5
        self.success_threshold = 0.8
        
    async def intelligent_planner(self, state: MedicalEnrichmentContext) -> str:
        """LLM-powered planning that considers context and previous attempts"""
        
        planning_prompt = f"""
        You are an intelligent sales research agent for a silk fibroin wound closure technology company.
        
        TARGET SURGEON: {state.name}
        HOSPITAL: {state.hospital_name}
        FIELD TO FIND: {state.target_field}
        PREVIOUS ATTEMPTS: {state.enrichment_status}
        BUSINESS OBJECTIVE: Find contact information for medical device sales outreach
        
        Available strategies for {state.target_field}:
        {self._get_available_strategies(state.target_field)}
        
        Analyze the surgeon profile and choose the BEST strategy:
        1. Consider their seniority level (decision-making authority)
        2. Consider department type (surgery dept more relevant)
        3. Consider hospital type (teaching hospitals have different structures)
        4. Consider previous failure patterns
        
        Choose ONE strategy and explain your reasoning:
        """
        
        strategy = await self.llm.generate(planning_prompt, complexity="medium")
        return self._parse_strategy_choice(strategy)
    
    async def adaptive_searcher(self, state: MedicalEnrichmentContext) -> MedicalEnrichmentContext:
        """Execute search with dynamic parameter adjustment"""
        
        strategy = state.current_strategy
        
        # Adjust search parameters based on strategy and context
        search_config = await self._generate_search_config(state, strategy)
        
        # Generate contextual query
        query = await self._generate_contextual_query(state, strategy)
        
        # Execute search with monitoring
        result = await self._execute_monitored_search(query, search_config)
        
        state.search_result = result
        state.search_confidence = self._assess_search_quality(result, state.target_field)
        
        return state
    
    async def validator_and_router(self, state: MedicalEnrichmentContext) -> str:
        """Intelligent validation and routing decisions"""
        
        if state.search_confidence < self.success_threshold and state.iteration_count < self.max_iterations:
            return "replanning"  # Try different strategy
        elif state.search_confidence >= self.success_threshold:
            return "extraction"  # Proceed to extraction
        else:
            return "fallback_extraction"  # Best effort extraction
    
    def build_agentic_graph(self):
        """Build cyclical, self-correcting graph"""
        graph = StateGraph(MedicalEnrichmentContext)
        
        # Nodes
        graph.add_node("planning", self.intelligent_planner)
        graph.add_node("search", self.adaptive_searcher) 
        graph.add_node("validation", self.validator_and_router)
        graph.add_node("extraction", self.smart_extractor)
        graph.add_node("replanning", self.strategy_adjuster)
        graph.add_node("fallback_extraction", self.fallback_extractor)
        
        # Edges - creating cycles for self-correction
        graph.add_edge(START, "planning")
        graph.add_edge("planning", "search")
        graph.add_edge("search", "validation")
        
        # Conditional routing based on validation
        graph.add_conditional_edges(
            "validation",
            lambda state: state.next_action,
            {
                "extraction": "extraction",
                "replanning": "replanning", 
                "fallback_extraction": "fallback_extraction"
            }
        )
        
        # Self-correction loop
        graph.add_edge("replanning", "planning")
        graph.add_edge("extraction", END)
        graph.add_edge("fallback_extraction", END)
        
        return graph.compile()
```

#### 2.2 Dynamic Query Generation with Business Context
**Enhancement to:** `backend/graph.py`

Replace the hardcoded query builders with LLM-powered generation:

```python
async def _generate_business_aware_query(self, state: MedicalEnrichmentContext, strategy: str) -> str:
    """Generate queries that understand business context and sales objectives"""
    
    context_prompt = f"""
    You are an expert at finding contact information for medical device sales outreach.
    
    BUSINESS CONTEXT:
    - Product: Silk fibroin-based wound closure technology
    - Target: Surgeons who make OR purchasing decisions
    - Goal: Find ANY professional contact method for sales outreach
    
    SURGEON PROFILE:
    - Name: {state.name}
    - Hospital: {state.hospital_name}
    - Specialty: {state.specialty or 'Unknown'}
    - All Context: {state.all_context}
    
    CONTACT STRATEGY: {strategy}
    TARGET FIELD: {state.target_field}
    
    For EMAIL searches, prioritize in this order:
    1. Direct surgeon email (best for personalized outreach)
    2. Department email (surgery department, wound care clinic)
    3. OR coordinator email (they influence purchasing)
    4. Hospital procurement email (for institutional sales)
    5. ANY contact email connected to this surgeon
    
    The search query should be:
    - Natural and conversational (Tavily works better with questions)
    - Specific enough to find this person
    - Broad enough to find alternative contacts if direct contact fails
    - Focused on the business need (professional communication for medical device introduction)
    
    Create ONE precise search query:
    """
    
    return await self.llm.generate(context_prompt, complexity="simple")
```

### Phase 3: Tool-Calling Architecture

#### 3.1 Implement Tool-Based Search System
**File:** `backend/search_tools.py` (NEW)

```python
class SearchToolkit:
    """Collection of specialized search tools that can be called dynamically"""
    
    def __init__(self, tavily_client, llm_provider):
        self.tavily = tavily_client
        self.llm = llm_provider
        
    async def domain_resolver_tool(self, hospital_name: str) -> Optional[str]:
        """Tool to resolve hospital domain for targeted searching"""
        pass
    
    async def linkedin_hunter_tool(self, name: str, context: Dict) -> Dict:
        """Specialized LinkedIn profile hunting with medical context"""
        pass
    
    async def email_discovery_tool(self, name: str, domain: str, strategy: str) -> Dict:
        """Multi-strategy email discovery tool"""
        pass
    
    async def department_finder_tool(self, hospital: str, specialty: str) -> Dict:
        """Find department contact information"""
        pass
    
    async def hierarchy_mapper_tool(self, hospital: str) -> Dict:
        """Map hospital organizational hierarchy for contact strategies"""
        pass

class AgenticSearchOrchestrator:
    """Orchestrates tool calls based on planning decisions"""
    
    async def execute_tool_sequence(self, plan: List[str], context: MedicalEnrichmentContext) -> Dict:
        """Execute a sequence of tools based on LLM planning"""
        
        results = {}
        for tool_name in plan:
            tool_result = await self._call_tool(tool_name, context)
            results[tool_name] = tool_result
            
            # Update context with new information
            context = self._update_context_with_tool_result(context, tool_result)
            
        return results
```

#### 3.2 Replace Smart Extractor with Tool-Calling Extractor
**Enhancement to:** `backend/smart_extractor.py`

```python
class AgenticFieldExtractor:
    """Tool-calling extractor that reasons about extraction strategies"""
    
    async def extract_with_tools(self, field: str, search_results: Dict, context: Dict) -> str:
        """Use appropriate extraction tools based on field and content analysis"""
        
        # Analyze content to choose extraction strategy
        analysis_prompt = f"""
        Analyze the search results for extracting {field}.
        
        Search Results Quality:
        - Number of results: {len(search_results.get('results', []))}
        - Has Tavily answer: {bool(search_results.get('answer'))}
        - Content types found: {self._analyze_content_types(search_results)}
        
        Business Context: Finding {field} for surgical device outreach
        
        Choose the best extraction approach:
        1. direct_extraction - Clear answer in Tavily response
        2. pattern_matching - Use regex patterns on content
        3. llm_reasoning - Need LLM to interpret and reason
        4. tool_chaining - Combine multiple extraction tools
        
        Choose ONE approach and explain why:
        """
        
        strategy = await self.llm.generate(analysis_prompt, complexity="simple")
        
        return await self._execute_extraction_strategy(strategy, field, search_results, context)
```

### Phase 4: Configuration-Driven Flexibility

#### 4.1 External Field Configuration System
**File:** `config/fields/email.yaml`
```yaml
field_name: email
business_priority: high
strategies:
  - name: direct_professional
    search_depth: advanced
    max_results: 15
    context_aware: true
    business_value: "Direct contact for personalized medical device introduction"
    
  - name: department_contact
    search_depth: basic
    max_results: 10
    context_aware: true
    business_value: "Department contacts can influence purchasing decisions"
    
success_criteria:
  - contains_at_symbol: true
  - domain_relevance: high
  - contact_type_appropriate: true
  
failure_patterns:
  - noreply_emails
  - broken_addresses
  - unrelated_contacts
```

#### 4.2 Dynamic Configuration Loader
**File:** `backend/config_manager.py` (NEW)
```python
class FieldConfigManager:
    """Loads and manages field-specific configurations"""
    
    def __init__(self, config_dir: str = "config/fields"):
        self.config_dir = config_dir
        self.field_configs = {}
        self._load_configurations()
    
    def get_field_strategies(self, field: str) -> List[Dict]:
        """Get available strategies for a field"""
        return self.field_configs.get(field, {}).get('strategies', [])
    
    def get_business_context(self, field: str) -> Dict:
        """Get business context for field enrichment"""
        return self.field_configs.get(field, {}).get('business_context', {})
```

### Phase 5: Performance and Intelligence Improvements

#### 5.1 Caching with Context Awareness
**Enhancement to:** `backend/graph.py`

Replace the simple cache with context-aware caching:

```python
class ContextAwareCache:
    """Cache that considers business context and strategy effectiveness"""
    
    def __init__(self):
        self.cache = TTLCache(maxsize=2000, ttl=1800)  # 30-minute TTL
        self.strategy_performance = {}  # Track strategy success rates
        
    def get_cache_key(self, name: str, field: str, strategy: str, business_context: Dict) -> str:
        """Generate context-aware cache key"""
        context_hash = hashlib.md5(
            f"{name}:{field}:{strategy}:{business_context.get('priority_level')}".encode()
        ).hexdigest()[:8]
        return f"{name}:{field}:{strategy}:{context_hash}"
    
    def update_strategy_performance(self, strategy: str, success: bool):
        """Track strategy effectiveness for future planning"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {"attempts": 0, "successes": 0}
        
        self.strategy_performance[strategy]["attempts"] += 1
        if success:
            self.strategy_performance[strategy]["successes"] += 1
```

#### 5.2 Retrieval Quality Assessment
**File:** `backend/quality_assessor.py` (NEW)
```python
class RetrievalQualityAssessor:
    """Assess search result quality and suggest improvements"""
    
    async def assess_search_quality(self, results: Dict, target_field: str, context: Dict) -> float:
        """Return confidence score 0-1 for search results"""
        
        assessment_prompt = f"""
        Assess the quality of these search results for finding {target_field}.
        
        Results Summary:
        - Number of results: {len(results.get('results', []))}
        - Tavily answer present: {bool(results.get('answer'))}
        - Result domains: {self._extract_domains(results)}
        
        Business Context: Medical device sales outreach
        Target: {context.get('doctor_name')} at {context.get('hospital_name')}
        
        Rate the search quality 0-100 considering:
        1. Relevance to the specific doctor
        2. Usefulness for business outreach
        3. Presence of contact information
        4. Authority of sources
        
        Return only the numeric score:
        """
        
        score_str = await self.llm.generate(assessment_prompt, complexity="simple")
        try:
            return float(score_str.strip()) / 100.0
        except:
            return 0.5  # Default moderate confidence
```

## Implementation Priority

### Phase 1 (Immediate - Week 1)
1. Implement `BusinessContextManager` with silk fibroin context
2. Replace hardcoded email strategies with intelligent hierarchy
3. Add business context to all query generation

### Phase 2 (Week 2)
1. Implement cyclical graph architecture
2. Add intelligent planning with LLM
3. Implement validation and re-planning logic

### Phase 3 (Week 3-4)
1. Build tool-calling search system
2. Implement agentic field extraction
3. Add configuration management system

### Phase 4 (Week 5-6)
1. Implement context-aware caching
2. Add retrieval quality assessment
3. Performance optimization and testing

## Expected Improvements

### Email Finding Effectiveness
- **Current:** Rigid fallback pattern
- **New:** Intelligent hierarchy with business context
- **Expected Improvement:** 40-60% better contact discovery

### Retrieval Performance
- **Current:** "Barely able to find anything"
- **New:** Business-aware queries with self-correction
- **Expected Improvement:** 50-70% better relevant results

### System Intelligence
- **Current:** Hardcoded rules
- **New:** LLM-powered planning and adaptation
- **Expected Improvement:** Truly agentic behavior with continuous improvement

This architecture transforms your system from a rigid rule-based pipeline into an intelligent, adaptive agent that understands business context and can reason about its tasks.
