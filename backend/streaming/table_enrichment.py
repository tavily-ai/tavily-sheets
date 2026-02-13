"""Main orchestrator for streaming table-wide enrichment."""

import json
import asyncio
import queue
import logging
import time
from typing import List, Optional, Generator, Dict, Any

from .table_preparation import find_empty_columns, collect_context_values
from .stream_processor import collect_chunks, process_stream_chunks, get_placeholder_values
from .event_handler import process_stream_event
from .data_extractor import extract_data_from_research
from graph import LLMProvider, OpenAIProvider

logger = logging.getLogger(__name__)


def filter_empty_rows(rows: List[List[str]]) -> tuple[List[List[str]], List[int]]:
    """
    Filter out rows where all cells are empty.

    Args:
        rows: List of rows, where each row is a list of cell values

    Returns:
        Tuple of (filtered_rows, original_indices) where original_indices maps
        each filtered row back to its original index
    """
    filtered_rows = []
    original_indices = []

    for idx, row in enumerate(rows):
        # Check if row has any non-empty cells
        has_content = any(cell.strip() for cell in row if cell)
        if has_content:
            filtered_rows.append(row)
            original_indices.append(idx)

    return filtered_rows, original_indices


async def enrich_table_with_research_streaming(
    headers: List[str],
    rows: List[List[str]],
    context_column: Optional[str],
    tavily_client,
    llm_provider: Optional[LLMProvider],
    target_column: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream research progress and results for table-wide enrichment.
    
    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of cell values
        context_column: Optional column name to use as context
        tavily_client: Tavily client instance
        llm_provider: LLM provider instance
        target_column: Optional column name to enrich (if specified, only this column will be enriched)
        
    Yields:
        SSE formatted event strings
    """
    
    # Filter out empty rows before processing
    filtered_rows, original_row_indices = filter_empty_rows(rows)
    total_rows = len(rows)

    logger.info(f"Filtered {total_rows - len(filtered_rows)} empty rows, processing {len(filtered_rows)} rows with content")

    if not filtered_rows:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No rows with content to enrich'})}\n\n"
        return

    # If target_column is specified, only enrich that column
    if target_column:
        target_col_idx = None
        for idx, header in enumerate(headers):
            if header == target_column:
                target_col_idx = idx
                break
        
        if target_col_idx is None:
            error_msg = f'Target column "{target_column}" not found in headers'
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return
        
        # Only enrich the target column (always include it, even if it has values)
        empty_columns = [(target_col_idx, target_column)]
        logger.info(f"Target column specified: {target_column}, will only enrich this column")
    else:
        # Find empty columns that need enrichment (use filtered_rows)
        empty_columns = find_empty_columns(headers, filtered_rows)
    
    if not empty_columns:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No columns to enrich'})}\n\n"
        return
    
    column_names = [header for _, header in empty_columns]

    # Collect context values (use filtered_rows)
    context_values = collect_context_values(headers, filtered_rows, context_column)
    
    if not context_values:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No context values found'})}\n\n"
        return
    
    # Split entities into batches of 5
    BATCH_SIZE = 1
    entity_batches = []
    for i in range(0, len(context_values), BATCH_SIZE):
        batch = context_values[i:i + BATCH_SIZE]
        entity_batches.append(batch)
    
    logger.info(f"Split {len(context_values)} entities into {len(entity_batches)} batches of up to {BATCH_SIZE} entities each")
    
    # Send initial event
    yield f"data: {json.dumps({'type': 'start', 'message': f'Starting research for {len(context_values)} entities', 'total_entities': len(context_values)})}\n\n"
    
    # Create output schema helper function
    def create_multi_property_schema(column_names: List[str], entities: List[str] = None) -> dict:
        """Create output_schema with multiple column properties for Research API"""
        properties = {}
        entities_str = f" for the following entities: {', '.join(entities)}" if entities else ""
        
        for col_name in column_names:
            # Create a more detailed description based on column name
            col_lower = col_name.lower()

            # Contact / Personal Information columns (check these FIRST before other conditions)
            # if 'email' in col_lower:
            #     detail = f"Provide the {col_name} (email address)."
            # elif any(keyword in col_lower for keyword in ['phone', 'telephone', 'mobile', 'cell']):
            #     detail = f"Provide the {col_name} (phone number with country/area code if available)."
            # elif any(keyword in col_lower for keyword in ['linkedin', 'twitter', 'social']):
            #     detail = f"Provide the {col_name} (social media profile URL or handle)."
            # elif any(keyword in col_lower for keyword in ['address', 'location', 'headquarters', 'hq']):
            #     detail = f"Provide the {col_name} (physical address or location, e.g., 'San Francisco, CA' or full street address)."
            # # Drug Trials / Biomedical Research columns
            # elif any(keyword in col_lower for keyword in ['trial name', 'trial']):
            #     detail = f"Provide the {col_name} (clinical trial identifier or name, e.g., 'KEYNOTE-671', 'CARTITUDE-1')."
            # elif any(keyword in col_lower for keyword in ['drug', 'intervention', 'therapeutic']):
            #     detail = f"Provide the {col_name} (drug name, intervention, or therapeutic agent, e.g., 'Pembrolizumab', 'Cilta-cel', 'Liraglutide')."
            # elif any(keyword in col_lower for keyword in ['indication']):
            #     detail = f"Provide the {col_name} (medical condition or disease the trial is targeting, e.g., 'Non-small cell lung cancer', 'Multiple Myeloma', 'Type 2 Diabetes')."
            # elif any(keyword in col_lower for keyword in ['phase']):
            #     detail = f"Provide the {col_name} (clinical trial phase: Phase I, Phase II, Phase III, or Phase IV)."
            # elif any(keyword in col_lower for keyword in ['sponsor', 'organization']):
            #     detail = f"Provide the {col_name} (trial sponsor or organization conducting the trial, e.g., 'Merck', 'Janssen', 'AstraZeneca')."
            # elif any(keyword in col_lower for keyword in ['mechanism of action', 'moa']):
            #     detail = f"Provide the {col_name} (how the drug works at the molecular or cellular level, e.g., 'PD-1 inhibitor', 'CAR-T cell therapy', 'GLP-1 receptor agonist')."
            # elif any(keyword in col_lower for keyword in ['primary endpoint', 'endpoints']):
            #     detail = f"Provide the {col_name} (main outcome measures of the trial, e.g., 'Overall Survival', 'Progression-Free Survival', 'Response Rate')."
            # elif any(keyword in col_lower for keyword in ['trial status', 'status']):
            #     detail = f"Provide the {col_name} (current trial status, e.g., 'Recruiting', 'Active, not recruiting', 'Completed', 'Terminated')."
            # elif any(keyword in col_lower for keyword in ['key results', 'results summary']):
            #     detail = f"Provide the {col_name} (summary of key trial results and outcomes, including statistical significance if available)."
            # elif any(keyword in col_lower for keyword in ['safety signal', 'safety', 'adverse']):
            #     detail = f"Provide the {col_name} (safety concerns, adverse events, or safety signals identified in the trial)."
            # elif any(keyword in col_lower for keyword in ['regulatory', 'fda', 'ema', 'approval']):
            #     detail = f"Provide the {col_name} (regulatory status, approvals, or regulatory notes from FDA, EMA, or other agencies)."
            # elif any(keyword in col_lower for keyword in ['biomarker', 'stratification criteria', 'stratification']):
            #     detail = f"Provide the {col_name} (specific molecular gatekeeper or patient selection criteria required for treatment effectiveness, highlighting precision medicine requirements and companion diagnostics, e.g., 'PD-L1 ≥ 1% (FDA label) with maximum benefit in PD-L1 ≥ 50%', 'High-risk patients; Ki-67 ≥ 20%', 'Residual pathologic disease after neoadjuvant chemoradiotherapy')."
            # # Biomedical / Scientific Landscape columns
            # elif any(keyword in col_lower for keyword in ['entity name', 'gene', 'protein', 'pathway']):
            #     detail = f"Provide the {col_name} (gene name, protein name, or biological pathway identifier, e.g., 'KRAS G12C', 'PD-1', 'BRCA1')."
            # elif any(keyword in col_lower for keyword in ['associated condition', 'condition', 'disease']):
            #     detail = f"Provide the {col_name} (medical conditions or diseases associated with the entity, comma-separated if multiple)."
            # elif any(keyword in col_lower for keyword in ['therapeutic area']):
            #     detail = f"Provide the {col_name} (therapeutic area or medical specialty, e.g., 'Oncology', 'Cardiology', 'Neurology', 'Immunology')."
            # elif any(keyword in col_lower for keyword in ['key finding', 'findings']):
            #     detail = f"Provide the {col_name} (key scientific findings or research discoveries related to the entity)."
            # elif any(keyword in col_lower for keyword in ['evidence strength', 'evidence']):
            #     detail = f"Provide the {col_name} (strength of evidence, e.g., 'Strong', 'Moderate', 'Weak', 'Preliminary', include study types if available)."
            # elif any(keyword in col_lower for keyword in ['publication', 'publications']):
            #     detail = f"Provide the {col_name} (recent or key publications, include journal names and years if available, comma-separated if multiple)."
            # elif any(keyword in col_lower for keyword in ['active trial', 'trials']):
            #     detail = f"Provide the {col_name} (number of active clinical trials or list of trial identifiers related to the entity)."
            # elif any(keyword in col_lower for keyword in ['open question', 'questions']):
            #     detail = f"Provide the {col_name} (unanswered research questions or areas requiring further investigation)."
            # elif any(keyword in col_lower for keyword in ['research momentum', 'momentum']):
            #     detail = f"Provide the {col_name} (trend in research activity, e.g., 'Increasing', 'Stable', 'Declining', include recent developments)."
            # # Financial / Market Analysis columns
            # elif any(keyword in col_lower for keyword in ['asset', 'company', 'ticker', 'symbol']):
            #     detail = f"Provide the {col_name} (asset name, company name, or ticker symbol, e.g., 'NVIDIA (NVDA)', 'Bitcoin (BTC)', 'EUR/USD')."
            # elif any(keyword in col_lower for keyword in ['sector', 'market']):
            #     detail = f"Provide the {col_name} (sector or market classification, e.g., 'Semiconductors', 'Cryptocurrency', 'Foreign Exchange', 'Energy')."
            # elif any(keyword in col_lower for keyword in ['key driver', 'drivers']):
            #     detail = f"Provide the {col_name} (main factors driving the asset's performance or market dynamics)."
            # elif any(keyword in col_lower for keyword in ['recent event', 'events']):
            #     detail = f"Provide the {col_name} (recent market events, news, or developments affecting the asset, include dates if significant)."
            # elif any(keyword in col_lower for keyword in ['bull case', 'bull']):
            #     detail = f"Provide the {col_name} (optimistic scenario or positive factors supporting upward price movement)."
            # elif any(keyword in col_lower for keyword in ['bear case', 'bear']):
            #     detail = f"Provide the {col_name} (pessimistic scenario or negative factors supporting downward price movement)."
            # elif any(keyword in col_lower for keyword in ['risk factor', 'risks']):
            #     detail = f"Provide the {col_name} (key risk factors or potential downside risks, comma-separated if multiple)."
            # elif any(keyword in col_lower for keyword in ['correlation profile', 'correlation']):
            #     detail = f"Provide the {col_name} (correlation relationship with other assets or markets, e.g., 'High Positive Correlation to Nasdaq-100', 'Inversely Correlated to US Dollar', include correlation strength and direction if available)."
            # elif any(keyword in col_lower for keyword in ['volatility regime', 'volatility']):
            #     detail = f"Provide the {col_name} (current volatility characteristics and market regime, e.g., 'Low Vol / Mean-Reverting', 'High Vol / Breakout Mode', include volatility level and market behavior pattern)."
            # elif any(keyword in col_lower for keyword in ['forward-looking', 'indicator', 'indicators', 'outlook']):
            #     detail = f"Provide the {col_name} (forward-looking indicators, forecasts, or outlook for future performance)."
            # # Source Citations (common to all)
            # elif any(keyword in col_lower for keyword in ['source citation', 'citation', 'sources']):
            #     detail = f"Provide the {col_name} (list of source citations, URLs, or references, comma-separated if multiple)."
            # # General fallbacks
            # elif any(keyword in col_lower for keyword in ['ceo', 'founder', 'president', 'director', 'manager', 'leader']):
            #     detail = f"Provide the {col_name} (person's full name)."
            # elif any(keyword in col_lower for keyword in ['founded', 'established', 'created', 'started']):
            #     detail = f"Provide the {col_name} (year when the entity was founded or established, format as YYYY)."
            # elif any(keyword in col_lower for keyword in ['revenue', 'income', 'sales', 'profit']):
            #     detail = f"Provide the {col_name} (financial figure with currency symbol and units, e.g., '$391.035B' or '$391 billion')."
            # elif any(keyword in col_lower for keyword in ['market cap', 'market capitalization', 'valuation']):
            #     detail = f"Provide the {col_name} (market capitalization with currency symbol and units, e.g., '$3,553.5B' or '$3.5 trillion')."
            # else:
            #     detail = f"Provide comprehensive information about {col_name}."

            detail = f"Provide comprehensive information about {col_name}."
            properties[col_name] = {
                "type": "string",
                "description": f"{detail} Return the data as a JSON object string mapping each entity name exactly as provided to its corresponding {col_name} value. CRITICAL: You MUST include data for ALL entities listed: {', '.join(entities) if entities else 'all entities'}. Format the JSON string as: {{\"EntityName1\": \"value1\", \"EntityName2\": \"value2\", ...}}. Use the exact entity names as provided. IMPORTANT: Only return actual, usable values - NOT masked/redacted values (like 'n******@domain.com', '***-***-1234', etc.). If you find a masked value, infer the actual value based on patterns or context. For emails, use the company's email format (e.g., first.last@company.com). Always provide your best answer - do not leave values empty."
            }
        
        return {
            "properties": properties,
            "required": column_names if column_names else []
        }
    
    # Process all batches in parallel
    all_sources = []
    all_accumulated_content = {}
    
    async def process_single_batch(
        batch_idx: int,
        batch_entities: List[str],
        event_queue: asyncio.Queue,
        content_dict: Dict[str, Dict[str, str]],
        sources_list: List[Dict[str, Any]]
    ):
        """Process a single batch and put events into the shared queue"""
        batch_num = batch_idx + 1
        logger.info(f"Processing batch {batch_num}/{len(entity_batches)} with {len(batch_entities)} entities: {batch_entities}")
        
        try:
            # Build query for this batch
            query = f"For each of these entities: {', '.join(batch_entities)}, provide information about: {', '.join(column_names)}. Return structured data for each entity separately."
            output_schema = create_multi_property_schema(column_names, batch_entities)

            # Start streaming research for this batch
            def start_streaming_research():
                return tavily_client.research(
                    input=query,
                    model="mini",
                    output_schema=output_schema,
                    stream=True
                )
            
            logger.info(f"Starting streaming research for batch {batch_num}: {query}")
            
            # Get the stream
            stream = await asyncio.to_thread(start_streaming_research)
            
            # Simple queue for chunks with size limit to prevent memory issues
            chunk_queue = queue.Queue(maxsize=100)
            error_flag = {'error_occurred': False}
            
            # Start collecting chunks in background
            collection_task = asyncio.create_task(
                asyncio.to_thread(collect_chunks, stream, chunk_queue, error_flag)
            )
            
            # Process chunks as they arrive for this batch
            batch_accumulated_content = {}
            batch_sources = []
            
            try:
                # Process stream events
                async for event_data in process_stream_chunks(chunk_queue, collection_task, error_flag):
                    if isinstance(event_data, dict):
                        # Process event and put SSE strings into shared queue
                        for sse_event in process_stream_event(event_data, batch_accumulated_content, batch_sources):
                            await event_queue.put(sse_event)
                        
                        # Check for error
                        if event_data.get('type') == 'error':
                            error_flag['error_occurred'] = True
                            break
            
            finally:
                # Cleanup: stop collection if still running
                error_flag['error_occurred'] = True
                if not collection_task.done():
                    collection_task.cancel()
                    try:
                        await collection_task
                    except asyncio.CancelledError:
                        pass
            
            # Store batch results for later merging
            content_dict[f"batch_{batch_idx}"] = batch_accumulated_content
            sources_list.extend(batch_sources)

            logger.info(f"Completed batch {batch_num}/{len(entity_batches)}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            await event_queue.put(f"data: {json.dumps({'type': 'error', 'message': f'Error in batch {batch_num}: {str(e)}'})}\n\n")
    
    try:
        # Create shared queue for events from all batches
        event_queue = asyncio.Queue()
        batch_content_dict = {}  # Store each batch's content separately
        batch_sources_list = []  # Store sources from all batches
        
        # Start all batches in parallel
        batch_tasks = []
        for batch_idx, batch_entities in enumerate(entity_batches):
            task = asyncio.create_task(
                process_single_batch(
                    batch_idx,
                    batch_entities,
                    event_queue,
                    batch_content_dict,
                    batch_sources_list
                )
            )
            batch_tasks.append(task)
        
        # Merge events from all batches as they arrive
        active_tasks = set(range(len(batch_tasks)))
        
        # Track which progress messages we've already sent (to avoid duplicates from parallel batches)
        planning_sent = False
        generating_sent = False
        
        while active_tasks:
            try:
                # Get event from any batch with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                
                # Filter duplicate "Planning" and "Generating" messages
                if event.startswith("data: "):
                    try:
                        event_data = json.loads(event[6:])  # Remove 'data: ' prefix
                        
                        # Check if it's a progress message we want to deduplicate
                        if event_data.get('type') == 'progress':
                            message = event_data.get('message', '')
                            
                            # Only send "Planning" once
                            if 'Planning research strategy' in message:
                                if not planning_sent:
                                    planning_sent = True
                                    yield event
                                continue
                            
                            # Only send "Generating" once
                            if 'Generating research report' in message:
                                if not generating_sent:
                                    generating_sent = True
                                    yield event
                                continue
                        
                        # For all other events (including search queries), yield them
                        yield event
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # If parsing fails, yield the event as-is
                        yield event
                else:
                    # Not an SSE event, yield as-is
                    yield event
                    
            except asyncio.TimeoutError:
                # Check if any batch has completed
                completed_indices = []
                for i in active_tasks:
                    if batch_tasks[i].done():
                        completed_indices.append(i)
                
                # Remove completed tasks
                for i in completed_indices:
                    active_tasks.remove(i)
                
                # If all batches are done, drain remaining events
                if not active_tasks:
                    # Drain remaining events (but still filter duplicates)
                    while not event_queue.empty():
                        try:
                            event = event_queue.get_nowait()
                            
                            # Apply same filtering logic
                            if event.startswith("data: "):
                                try:
                                    event_data = json.loads(event[6:])
                                    if event_data.get('type') == 'progress':
                                        message = event_data.get('message', '')
                                        if 'Planning research strategy' in message and planning_sent:
                                            continue
                                        if 'Generating research report' in message and generating_sent:
                                            continue
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    pass
                            
                            yield event
                        except asyncio.QueueEmpty:
                            break
                    break
        
        # Wait for all tasks to complete (in case of any errors)
        await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Merge all batch results into overall accumulated content
        for batch_key, batch_content in batch_content_dict.items():
            for key, value in batch_content.items():
                if key in all_accumulated_content:
                    # Try to merge JSON strings if both are JSON
                    if isinstance(all_accumulated_content[key], str) and isinstance(value, str):
                        try:
                            import json as json_module
                            existing_json = json_module.loads(all_accumulated_content[key])
                            new_json = json_module.loads(value)
                            # Merge the JSON objects
                            if isinstance(existing_json, dict) and isinstance(new_json, dict):
                                merged_json = {**existing_json, **new_json}
                                all_accumulated_content[key] = json_module.dumps(merged_json)
                            else:
                                # If not dicts, just use the new value
                                all_accumulated_content[key] = value
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, concatenate or use new value
                            all_accumulated_content[key] = value
                    else:
                        all_accumulated_content[key] = value
                else:
                    all_accumulated_content[key] = value
        
        # Merge sources
        all_sources.extend(batch_sources_list)
        
        # After all batches complete, process the final combined result
        if all_accumulated_content:
            final_result = all_accumulated_content
        else:
            # Fallback: use empty dict if no content
            final_result = {}
        
        # Extract sources (using dict format for now, will be converted to SearchResult in app.py)
        sources = []
        if all_sources:
            sources = [
                {
                    'title': s.get('title', ''),
                    'url': s.get('url', ''),
                    'favicon': s.get('favicon')
                }
                for s in all_sources
            ]
        
        # Initialize result structures for filtered rows only
        sources_table_filtered = {header: [sources for _ in filtered_rows] for _, header in empty_columns}

        # Extract data from research results (using filtered_rows)
        yield f"data: {json.dumps({'type': 'progress', 'message': 'Processing research results...'})}\n\n"

        enriched_table_filtered = await extract_data_from_research(
            final_result=final_result,
            context_values=context_values,
            column_names=column_names,
            empty_columns=empty_columns,
            rows=filtered_rows,
            llm_provider=llm_provider
        )

        # Map filtered results back to original row indices
        # Empty rows will have empty string values (not "Not Found")
        enriched_table = {}
        sources_table = {}
        for _, header in empty_columns:
            # Initialize with empty strings for all original rows
            enriched_table[header] = [""] * total_rows
            sources_table[header] = [[] for _ in range(total_rows)]

            # Map filtered results back to original positions
            for filtered_idx, original_idx in enumerate(original_row_indices):
                if filtered_idx < len(enriched_table_filtered.get(header, [])):
                    enriched_table[header][original_idx] = enriched_table_filtered[header][filtered_idx]
                if filtered_idx < len(sources_table_filtered.get(header, [])):
                    sources_table[header][original_idx] = sources_table_filtered[header][filtered_idx]

        # Send final result
        sources_dict = {}
        for k, v in sources_table.items():
            sources_dict[k] = [
                [{'title': s.get('title', ''), 'url': s.get('url', ''), 'favicon': s.get('favicon')} for s in row_sources]
                for row_sources in v
            ]
        yield f"data: {json.dumps({'type': 'complete', 'enriched_values': enriched_table, 'sources': sources_dict, 'status': 'success'})}\n\n"
        

    except Exception as e:
        logger.error(f"Error in streaming research: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

