"""Functions for handling different event types from Tavily Research stream."""

import json
import logging
from typing import Dict, List, Any, Generator

logger = logging.getLogger(__name__)


def handle_tool_call_event(event_data: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Handle tool call events from the stream.
    
    Args:
        event_data: Event data containing tool call information
        
    Yields:
        SSE formatted event strings
    """
    if 'choices' in event_data and len(event_data['choices']) > 0:
        delta = event_data['choices'][0].get('delta', {})
        
        # Tool call events
        if 'tool_calls' in delta:
            tool_calls = delta['tool_calls']
            if tool_calls.get('type') == 'tool_call':
                for tool_call in tool_calls.get('tool_call', []):
                    tool_name = tool_call.get('name', 'Unknown')
                    tool_id = tool_call.get('id', '')
                    arguments = tool_call.get('arguments', '')
                    queries = tool_call.get('queries', [])
                    
                    event = {
                        'type': 'tool_call',
                        'tool': tool_name,
                        'id': tool_id,
                        'arguments': arguments
                    }
                    if queries:
                        event['queries'] = queries
                    
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # Send user-friendly message
                    if tool_name == 'Planning':
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Planning research strategy...'})}\n\n"
                    elif tool_name == 'WebSearch':
                        query_count = len(queries) if queries else 0
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'Searching the web ({query_count} queries)...'})}\n\n"
                    elif tool_name == 'Generating':
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Generating research report...'})}\n\n"
            elif tool_calls.get('type') == 'tool_response':
                # Tool response events (with sources)
                for tool_response in tool_calls.get('tool_response', []):
                    tool_name = tool_response.get('name', 'Unknown')
                    sources = tool_response.get('sources', [])
                    
                    if sources:
                        # Stream sources with favicons as they're found
                        sources_data = [
                            {
                                'title': s.get('title', ''),
                                'url': s.get('url', ''),
                                'favicon': s.get('favicon')
                            }
                            for s in sources
                        ]
                        yield f"data: {json.dumps({'type': 'sources_found', 'count': len(sources), 'tool': tool_name, 'sources': sources_data})}\n\n"


def handle_content_event(
    event_data: Dict[str, Any],
    accumulated_content: Dict[str, str]
) -> Generator[str, None, None]:
    """
    Handle content events from the stream.

    Args:
        event_data: Event data containing content
        accumulated_content: Dictionary to accumulate structured content

    Yields:
        SSE formatted event strings
    """
    if 'choices' in event_data and len(event_data['choices']) > 0:
        delta = event_data['choices'][0].get('delta', {})

        # Content events (research report chunks)
        if 'content' in delta:
            content = delta['content']

            if isinstance(content, str):
                # Text content chunk - also accumulate it for debugging
                if '_raw_text' not in accumulated_content:
                    accumulated_content['_raw_text'] = ""
                accumulated_content['_raw_text'] += content
                yield f"data: {json.dumps({'type': 'content_chunk', 'content': content})}\n\n"
            elif isinstance(content, dict):
                # Structured output (from output_schema)
                for key, value in content.items():
                    if key not in accumulated_content:
                        accumulated_content[key] = ""
                    if isinstance(value, str):
                        accumulated_content[key] += value
                    else:
                        accumulated_content[key] = str(value)


def handle_sources_event(event_data: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Handle sources event from the stream.
    
    Args:
        event_data: Event data containing sources
        
    Yields:
        SSE formatted event strings
    """
    if 'choices' in event_data and len(event_data['choices']) > 0:
        delta = event_data['choices'][0].get('delta', {})
        
        # Sources event (final list)
        if 'sources' in delta:
            final_sources = delta['sources']
            # Stream final sources with favicons
            sources_data = [
                {
                    'title': s.get('title', ''),
                    'url': s.get('url', ''),
                    'favicon': s.get('favicon')
                }
                for s in final_sources
            ]
            yield f"data: {json.dumps({'type': 'sources_complete', 'count': len(final_sources), 'sources': sources_data})}\n\n"


def handle_error_event(event_data: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Handle error events from the stream.
    
    Args:
        event_data: Event data containing error information
        
    Yields:
        SSE formatted error event string
    """
    if event_data.get('object') == 'error':
        error_msg = event_data.get('error', 'Unknown error')
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


def process_stream_event(
    event_data: Dict[str, Any],
    accumulated_content: Dict[str, str],
    all_sources: List[Dict[str, Any]]
) -> Generator[str, None, None]:
    """
    Process a single event from the stream and yield appropriate SSE events.
    
    Args:
        event_data: Event data from stream
        accumulated_content: Dictionary to accumulate structured content
        all_sources: List to accumulate all sources
        
    Yields:
        SSE formatted event strings
    """
    # Handle tool call events
    for event in handle_tool_call_event(event_data):
        yield event
    
    # Handle content events
    for event in handle_content_event(event_data, accumulated_content):
        yield event
    
    # Handle sources events
    if 'choices' in event_data and len(event_data['choices']) > 0:
        delta = event_data['choices'][0].get('delta', {})
        if 'sources' in delta:
            final_sources = delta['sources']
            all_sources.extend(final_sources)
            for event in handle_sources_event(event_data):
                yield event
    
    # Handle error events
    for event in handle_error_event(event_data):
        yield event

