import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv
from google.generativeai import GenerativeModel
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI
from tavily import TavilyClient

MAX_TOKENS = 20000

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, client, model="gpt-4.1-2025-04-14"):
        self.client = client
        self.model = model

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    def __init__(self, model):
        self.model = model

    async def generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(lambda: self.model.generate_content(prompt))
        return response.text.strip()


@dataclass
class EnrichmentContext:
    column_name: str
    target_value: str
    context_values: Dict[str, str]
    search_result: Dict = None
    answer: str = None
class EnrichmentPipeline:
    def __init__(self, tavily_client, llm_provider: LLMProvider):
        self.tavily = tavily_client
        self.llm = llm_provider

    async def search_tavily(self, state: EnrichmentContext):
        """Run Tavily research in a separate thread"""
        try:
            query = f"{state.column_name} of {state.target_value}?"
            logger.info(f"Researching Tavily with query: {query}")
            
            # Use Tavily Research API (async - requires polling)
            # Research API uses 'input' parameter instead of 'query'
            def start_research():
                # Create generic schema for structured output
                # Tavily only accepts 'properties' and 'required' keys, not full JSON Schema
                output_schema = {
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": f"Comprehensive information about {state.column_name} of {state.target_value}. Format as a clean report without markdown links or citations."
                        }
                    },
                    "required": ["answer"]
                }
                return self.tavily.research(
                    input=query,
                    model="mini",
                    output_schema=output_schema
                )
            
            # Start the research task
            initial_response = await asyncio.to_thread(start_research)
            logger.info(f"Research task started: {initial_response}")
            
            # Check if we got a request_id (async task)
            request_id = initial_response.get('request_id')
            if request_id:
                # Poll for completion
                max_attempts = 120   # Maximum 10 minutes (120 * 5 seconds)
                attempt = 0
                
                while attempt < max_attempts:
                    await asyncio.sleep(5)  # Wait 5 seconds between polls
                    attempt += 1
                    
                    def get_research_status():
                        return self.tavily.get_research(request_id)
                    
                    status_response = await asyncio.to_thread(get_research_status)
                    status = status_response.get('status', 'pending')
                    
                    logger.info(f"Research status check {attempt}: {status}")
                    
                    if status == 'completed':
                        logger.info(f"Research completed after {attempt * 5} seconds")
                        result = status_response
                        break
                    elif status == 'failed':
                        logger.error(f"Research task failed: {status_response}")
                        raise Exception(f"Research task failed: {status_response.get('error', 'Unknown error')}")
                    # Continue polling if status is 'pending' or other
                else:
                    # Timeout after max attempts
                    raise Exception(f"Research task timed out after {max_attempts * 5} seconds")
            else:
                # If no request_id, assume it's already completed (synchronous response)
                result = initial_response
            
            logger.info(f"Tavily research result: {result}")
            return {"search_result": result}
        except Exception as e:
            logger.error(f"Error in search_tavily: {str(e)}")
            raise

    async def extract_minimal_answer(
        self, state: EnrichmentContext
    ) -> Dict:
        """Use LLM to extract a minimal answer from Tavily's research results."""
        search_result = state.search_result
        
        # Log for debugging
        logger.info(f"Extract minimal answer for {state.target_value}")
        logger.info(f"Search result type: {type(search_result)}, keys: {list(search_result.keys()) if isinstance(search_result, dict) else 'Not a dict'}")
        
        # Research API response format (always research mode now)
        if isinstance(search_result, dict):
            # Check if it's structured output (from output_schema)
            if "answer" in search_result:
                # Structured output from output_schema - already clean and formatted
                answer = search_result["answer"]
                logger.info(f"Using structured research answer for {state.target_value} (length: {len(answer) if answer else 0})")
                # Still process through LLM to ensure it's clean and remove any remaining links
                try:
                    truncated_answer = answer[:MAX_TOKENS] if len(answer) > MAX_TOKENS else answer
                    prompt = f"""
                        Process this research answer and ensure it's clean:

                        {truncated_answer}

                        Rules:
                            1. Keep the content as-is but ensure no markdown links remain
                            2. Remove any citation references like [1], [2], etc.
                            3. Ensure clean formatting
                            4. Return the cleaned answer
                            
                        Clean Answer:
                        """
                    logger.info(f"Final cleaning of structured research answer for {state.target_value}")
                    cleaned_answer = await self.llm.generate(prompt)
                    return {"answer": cleaned_answer}
                except Exception as e:
                    logger.error(f"Error cleaning structured answer: {str(e)}")
                    # Fallback to structured answer if LLM fails
                    return {"answer": answer}
            elif search_result.get("content"):
                # Original markdown format (fallback if output_schema not used)
                research_content = search_result["content"]
                logger.info(f"Processing Research API content through LLM for {state.target_value} (length: {len(research_content) if research_content else 0})")
                
                # Truncate content if too long (respect MAX_TOKENS)
                truncated_content = research_content[:MAX_TOKENS] if len(research_content) > MAX_TOKENS else research_content
                
                try:
                    prompt = f"""
                        Process this research report and provide a clean, formatted answer:

                        {truncated_content}

                        Rules:
                        1. Provide a well-formatted report answer based on the research content
                        2. Remove all markdown links (e.g., [text](url) or [1], [2] citations)
                        3. Remove source citations and reference numbers
                        4. Keep the structure and formatting but make it clean
                        5. Focus on answering: {state.column_name} of {state.target_value}
                        6. Be comprehensive but concise
                        
                        Clean Report Answer:
                    """
                    logger.info(f"Processing research content through LLM for {state.target_value}")
                    answer = await self.llm.generate(prompt)
                    logger.info(f"LLM processed research answer for {state.target_value}")
                    return {"answer": answer}
                except Exception as e:
                    logger.error(f"Error processing research content through LLM: {str(e)}")
                    # Fallback to original content if LLM fails
                    return {"answer": research_content}
        
        # Fallback: if no structured answer or content, return empty
        logger.warning(f"No answer or content found in research result for {state.target_value}")
        return {"answer": "Information not found"}

    def build_graph(self):
        """build and compile the graph"""
        graph = StateGraph(EnrichmentContext)
        
        graph.add_node("search", self.search_tavily)
        graph.add_node("extract", self.extract_minimal_answer)
        #graph.add_node("enrich", self.enrich)
        graph.add_edge(START, "search")
        graph.add_edge("search", "extract")
        graph.add_edge("extract", END)
        compiled_graph = graph.compile()
        return compiled_graph


async def enrich_cell_with_graph(
    column_name: str,
    target_value: str,
    context_values: Dict[str, str],
    tavily_client,
    llm_provider: LLMProvider,
    tavily_mode: str = "research"  # Keep for backward compatibility but always use research
) -> Dict:
    """Helper function to enrich a single cell using langgraph."""
    try:
        logger.info(f"Starting enrich_cell_with_graph for {target_value}")
        pipeline = EnrichmentPipeline(tavily_client, llm_provider)
        initial_context = EnrichmentContext(
            column_name=column_name,
            target_value=target_value,
            context_values=context_values,
            search_result=None,
            answer=None
        )
        graph = pipeline.build_graph()
        result = await graph.ainvoke(initial_context)
        #print(f"Result: {result}")
        logger.info(f"Completed enrich_cell_with_graph for {target_value}")
        return result #, result['urls']
    except Exception as e:
        logger.error(f"Error in enrich_cell_with_graph: {str(e)}")
        return "Error during enrichment"


# Example usage:
if __name__ == "__main__":
    context = EnrichmentContext(
        column_name="CEO",
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994",
            "Location": "Seattle, WA",
        },
    )

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    gemini_model = GenerativeModel(model_name="gemini-1.5-flash")

    #Example with OpenAI
    openai_provider = OpenAIProvider(openai_client)
    pipeline_openai = EnrichmentPipeline(tavily_client, openai_provider)

    # Example with Gemini
    gemini_provider = GeminiProvider(gemini_model)
    pipeline_gemini = EnrichmentPipeline(tavily_client, gemini_provider)

    # Using the graph
    graph = pipeline_gemini.build_graph()
    initial_context = EnrichmentContext(
        column_name="CEO", 
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994", 
            "Location": "Seattle, WA",
        },
        search_result=None,
        answer=None
    )
    result = asyncio.run(graph.ainvoke(initial_context))
    #print(f"Enrichment result: {result['answer']}")

    # Or using the helper function
    result_helper = asyncio.run(enrich_cell_with_graph(
        column_name="CEO",
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994",
            "Location": "Seattle, WA",
        },
        tavily_client=tavily_client,
        llm_provider=gemini_provider
    ))
    #print(f"Helper function result: {result_helper}")
