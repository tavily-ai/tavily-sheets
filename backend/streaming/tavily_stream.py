"""Async streaming from Tavily Research API using httpx."""

import json
import httpx
import logging
from typing import AsyncGenerator, Dict, Any, Optional

logger = logging.getLogger(__name__)

TAVILY_API_BASE = "https://api.tavily.com"


async def stream_tavily_research(
    api_key: str,
    query: str,
    output_schema: Dict[str, Any],
    model: str = "mini"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream research results from Tavily API using async HTTP.
    
    Args:
        api_key: Tavily API key
        query: Research query/input
        output_schema: Output schema for structured data
        model: Model to use (default: "mini")
        
    Yields:
        Event data dictionaries from the stream
    """
    url = f"{TAVILY_API_BASE}/research"
    
    # Ensure API key is not None or empty
    if not api_key:
        logger.error("Tavily API key is missing or empty")
        yield {'type': 'error', 'message': 'Tavily API key is required'}
        return
    
    payload = {
        "input": query,
        "model": model,
        "output_schema": output_schema,
        "stream": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": api_key  # Tavily API expects API key directly in Authorization header
    }
    
    logger.debug(f"Making Tavily API request to {url} with model={model}")
    
    buffer = ""
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                # Check status code manually (can't use raise_for_status on streaming response)
                if response.status_code != 200:
                    # For non-200 responses, try to read error message safely
                    error_text = ""
                    try:
                        # Read a limited amount of error response
                        chunk_count = 0
                        async for chunk in response.aiter_bytes():
                            error_text += chunk.decode('utf-8', errors='ignore')
                            chunk_count += 1
                            if chunk_count > 10 or len(error_text) > 1000:  # Limit reading
                                break
                    except Exception as read_error:
                        logger.debug(f"Could not read error response: {read_error}")
                    
                    error_msg = f"Tavily API HTTP error: {response.status_code}"
                    if error_text:
                        error_msg += f" - {error_text[:200]}"
                    logger.error(error_msg)
                    yield {'type': 'error', 'message': f'Tavily API error: {response.status_code}. Check API key and request format.'}
                    return
                
                async for chunk in response.aiter_bytes():
                    # Decode bytes to string
                    chunk_str = chunk.decode('utf-8', errors='ignore')
                    buffer += chunk_str
                    
                    # Process complete lines (SSE format)
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        # Handle SSE format: "data: {...}" or "event: done"
                        if line == 'event: done':
                            # Process any remaining buffer before exiting
                            if buffer.strip():
                                for remaining_line in buffer.split('\n'):
                                    remaining_line = remaining_line.strip()
                                    if remaining_line and remaining_line.startswith('data: '):
                                        try:
                                            event_data = json.loads(remaining_line[6:])
                                            yield event_data
                                        except (json.JSONDecodeError, Exception):
                                            pass
                            return
                        
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                yield event_data
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                logger.warning(f"Failed to parse SSE data: {line[:100]}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {str(e)}")
                                continue
                
                # Process any remaining buffer after stream ends
                if buffer.strip():
                    for line in buffer.split('\n'):
                        line = line.strip()
                        if line and line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])
                                yield event_data
                            except (json.JSONDecodeError, Exception):
                                pass
                                
    except httpx.HTTPStatusError as e:
        # For streaming responses, we can't access .text directly
        error_msg = f"Tavily API HTTP error: {e.response.status_code}"
        try:
            # Try to get error details if response is not streaming
            if hasattr(e.response, 'text') and not hasattr(e.response, 'aiter_bytes'):
                error_msg += f" - {e.response.text[:200]}"
        except:
            pass
        logger.error(error_msg)
        yield {'type': 'error', 'message': f'Tavily API error: {e.response.status_code}'}
    except httpx.RequestError as e:
        logger.error(f"Tavily API request error: {str(e)}")
        yield {'type': 'error', 'message': f'Tavily API request failed: {str(e)}'}
    except Exception as e:
        logger.error(f"Unexpected error in Tavily streaming: {str(e)}")
        yield {'type': 'error', 'message': f'Streaming error: {str(e)}'}

