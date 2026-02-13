"""Functions for processing streaming chunks from Tavily Research API."""

import asyncio
import json
import queue
import logging
from typing import Generator, Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def get_placeholder_values() -> List[str]:
    """Get list of placeholder values to filter out."""
    return [
        "not disclosed", "not publically available", "not publicly available",
        "n/a", "na", "unknown", "none", "", "null", "not available",
        "not found", "unavailable", "data not available"
    ]


def collect_chunks(stream, chunk_queue: queue.Queue, error_flag: Dict[str, bool]):
    """
    Collect chunks from stream in background thread.
    
    Args:
        stream: Tavily research stream
        chunk_queue: Queue to put chunks into
        error_flag: Dictionary with 'error_occurred' key to signal errors
    """
    try:
        for chunk in stream:
            if error_flag.get('error_occurred', False):
                break
            try:
                chunk_queue.put(chunk, timeout=1.0)
            except queue.Full:
                logger.warning("Chunk queue full, dropping chunk")
                continue
        chunk_queue.put(None)  # Sentinel to signal completion
    except Exception as e:
        logger.error(f"Error collecting chunks: {str(e)}")
        try:
            chunk_queue.put(('error', str(e)), timeout=1.0)
        except queue.Full:
            pass


def get_chunk(chunk_queue: queue.Queue, timeout: float = 0.05) -> Optional[Any]:
    """
    Get chunk from queue with timeout.
    This is a synchronous function that will be run in a thread.
    
    Args:
        chunk_queue: Queue to get chunks from
        timeout: Timeout in seconds
        
    Returns:
        Chunk from queue or None if timeout
    """
    try:
        return chunk_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def process_buffer_content(buffer: str) -> Generator[Dict[str, Any], None, None]:
    """
    Process buffer content and yield event data dictionaries.
    
    Args:
        buffer: String buffer containing SSE-formatted lines
        
    Yields:
        Event data dictionaries parsed from buffer
    """
    if buffer.strip():
        for line in buffer.split('\n'):
            line = line.strip()
            if line and line.startswith('data: '):
                try:
                    event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                    yield event_data
                except (json.JSONDecodeError, Exception):
                    # Skip invalid JSON lines
                    pass


async def process_stream_chunks(
    chunk_queue: queue.Queue,
    collection_task: asyncio.Task,
    error_flag: Dict[str, bool]
) -> Generator[str, None, None]:
    """
    Process chunks from the stream and yield event data dictionaries.
    
    Args:
        chunk_queue: Queue containing stream chunks
        collection_task: Background task collecting chunks
        error_flag: Dictionary with 'error_occurred' key
        
    Yields:
        Event data dictionaries (not SSE formatted)
    """
    buffer = ""
    
    while True:
        try:
            # Get chunk with small timeout
            chunk = await asyncio.wait_for(
                asyncio.to_thread(get_chunk, chunk_queue, 0.05),
                timeout=0.1
            )
            
            if chunk is None:  # Timeout or sentinel
                # Check if collection task is done
                if collection_task.done():
                    # Try to get any remaining chunks without timeout
                    try:
                        while True:
                            chunk = chunk_queue.get_nowait()
                            if chunk is None:  # Sentinel - done
                                # Process any remaining buffer content
                                for event_data in process_buffer_content(buffer):
                                    yield event_data
                                # Stream is complete, exit
                                return
                            if isinstance(chunk, tuple) and chunk[0] == 'error':
                                yield {'type': 'error', 'message': chunk[1]}
                                error_flag['error_occurred'] = True
                                return
                            
                            # Process remaining chunk
                            if isinstance(chunk, bytes):
                                chunk_str = chunk.decode('utf-8')
                            else:
                                chunk_str = str(chunk)
                            
                            buffer += chunk_str
                            
                            # Process any remaining lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                if line and line.startswith('data: '):
                                    try:
                                        event_data = json.loads(line[6:])
                                        yield event_data
                                    except:
                                        pass
                    except queue.Empty:
                        # No more chunks, process remaining buffer and exit
                        for event_data in process_buffer_content(buffer):
                            yield event_data
                        return
                # If timeout and task still running, continue waiting
                await asyncio.sleep(0.01)
                continue
            
            if isinstance(chunk, tuple) and chunk[0] == 'error':
                yield {'type': 'error', 'message': chunk[1]}
                error_flag['error_occurred'] = True
                return
            
            # Process chunk and yield events immediately
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode('utf-8')
            else:
                chunk_str = str(chunk)
            
            buffer += chunk_str
            
            # Process complete lines and yield events immediately
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if not line:
                    continue
                
                # Parse SSE format: "data: {...}" or "event: done"
                if line == 'event: done':
                    # Process any remaining buffer before exiting
                    for event_data in process_buffer_content(buffer):
                        yield event_data
                    return
                
                if line.startswith('data: '):
                    try:
                        event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                        yield event_data
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {str(e)}")
                        continue
        
        except asyncio.TimeoutError:
            # This should not happen now since we handle timeout in get_chunk
            # But keep it as a safety net
            if collection_task.done():
                # Process any remaining buffer before exiting
                for event_data in process_buffer_content(buffer):
                    yield event_data
                return
            await asyncio.sleep(0.01)
            continue
        except Exception as e:
            logger.error(f"Unexpected error in stream processing: {str(e)}")
            yield {'type': 'error', 'message': f'Stream processing error: {str(e)}'}
            error_flag['error_occurred'] = True
            return

