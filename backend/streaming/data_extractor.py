"""Functions for extracting and matching data from research results."""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any

from graph import LLMProvider, OpenAIProvider

logger = logging.getLogger(__name__)


def get_placeholder_values() -> List[str]:
    """Get list of placeholder values to filter out."""
    return [
        "not disclosed", "not publically available", "not publicly available",
        "n/a", "na", "unknown", "none", "", "null", "not available",
        "not found", "unavailable", "data not available"
    ]


def build_extraction_prompt(
    entities_list: str,
    columns_list: str,
    research_text: str
) -> str:
    """
    Build the LLM prompt for extracting structured data.
    
    Args:
        entities_list: Comma-separated list of entities
        columns_list: Comma-separated list of columns
        research_text: Research data text
        
    Returns:
        Extraction prompt string
    """
    return f"""
            Extract structured data from this research response for multiple entities.
            
            Entities to extract data for: {entities_list}
            Columns to extract: {columns_list}
            
            Research Data (may contain JSON objects per column):
            {research_text}
            
            IMPORTANT: The research data may already contain JSON objects mapping entities to values.
            If you see JSON like {{"Apple": "Tim Cook", "Microsoft": "Satya Nadella"}}, use that directly.
            Otherwise, extract the information from the text.
            
            CRITICAL RULES:
            1. DO NOT use "not disclosed", "N/A", "unknown", or similar placeholder values unless the information is genuinely unavailable after thorough research.
            2. For well-known information (like CEO names of major companies), extract the actual value even if the research data says "not disclosed".
            3. If you see "not disclosed" in the research data but can infer the actual value from context, use the actual value.
            4. Only use "not disclosed" as a last resort when the information truly cannot be determined.
            5. For public companies, information like CEO names, founding dates, and headquarters are typically publicly available - extract them.
            
            Return a JSON object mapping each entity to its column values.
            Format: {{"EntityName": {{"ColumnName": "value", ...}}, ...}}
            
            CRITICAL REQUIREMENTS:
            1. You MUST include data for ALL entities: {entities_list}
            2. You MUST include data for ALL columns: {columns_list}
            3. Every entity in the list MUST appear in your JSON response, even if some columns are empty
            4. Do not skip any entities - if an entity is in the list, it must be in your response
            5. If you cannot find information for a specific entity-column combination, you may leave it empty or use an empty string, but the entity must still be in the response
            
            Example (with actual values, not "not disclosed"):
            {{
                "Apple": {{"CEO": "Tim Cook", "Founded": "1976", "Revenue": "$394.3 billion", "Headquarters": "Cupertino, California"}},
                "Microsoft": {{"CEO": "Satya Nadella", "Founded": "1975", "Revenue": "$211.9 billion", "Headquarters": "Redmond, Washington"}},
                "Google": {{"CEO": "Sundar Pichai", "Founded": "1998", "Revenue": "$307.4 billion", "Headquarters": "Mountain View, California"}},
                "Tesla": {{"CEO": "Elon Musk", "Founded": "2003", "Revenue": "$96.8 billion", "Headquarters": "Austin, Texas"}}
            }}
            
            Return ONLY valid JSON, no other text or explanation:
            """


def parse_direct_json(column_names: List[str], final_result: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Try to parse JSON directly from structured output.
    
    Args:
        column_names: List of column names
        final_result: Final result dictionary from research
        
    Returns:
        Parsed data dictionary or None if not found
    """
    parsed_data = None
    direct_json_found = False
    
    # Check if any column value is already a JSON object string
    for col_name in column_names:
        if col_name in final_result:
            col_value = final_result[col_name]
            if isinstance(col_value, str):
                # Try to parse as JSON
                try:
                    col_json = json.loads(col_value)
                    if isinstance(col_json, dict):
                        logger.info(f"Found direct JSON in column {col_name}: {list(col_json.keys())}")
                        # If we have JSON per column, merge them into per-entity structure
                        if parsed_data is None:
                            parsed_data = {}
                        for entity, value in col_json.items():
                            if entity not in parsed_data:
                                parsed_data[entity] = {}
                            parsed_data[entity][col_name] = value
                        direct_json_found = True
                except:
                    pass
    
    return parsed_data if direct_json_found else None


def extract_entity_data(
    entity: str,
    parsed_data: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find entity data in parsed data using various matching strategies.
    
    Args:
        entity: Entity name to find
        parsed_data: Parsed data dictionary
        
    Returns:
        Tuple of (entity_data, entity_key) or (None, None) if not found
    """
    # Try exact match first
    if entity in parsed_data:
        return parsed_data[entity], entity
    
    # Try case-insensitive match
    entity_lower = entity.lower().strip()
    for key in parsed_data.keys():
        if key.lower().strip() == entity_lower:
            return parsed_data[key], key
    
    # Try partial match (entity name contains key or vice versa)
    for key in parsed_data.keys():
        key_lower = key.lower().strip()
        entity_lower = entity.lower().strip()
        # Check if one contains the other (for cases like "Donald Trump" vs "Trump")
        if entity_lower in key_lower or key_lower in entity_lower:
            return parsed_data[key], key
    
    return None, None


def assign_column_value(
    header: str,
    entity_data: Dict[str, Any],
    empty_columns: List[Tuple[int, str]],
    row_idx: int,
    enriched_table: Dict[str, List[str]]
) -> bool:
    """
    Assign a column value from entity data to the enriched table.
    
    Args:
        header: Column header name
        entity_data: Entity data dictionary
        empty_columns: List of empty columns
        row_idx: Row index
        enriched_table: Enriched table dictionary
        
    Returns:
        True if value was assigned, False otherwise
    """
    placeholder_values = get_placeholder_values()
    value_found = False
    
    # Try exact column name match
    if header in entity_data:
        value = entity_data[header]
        value_str = str(value).strip().lower()
        if value and value_str not in placeholder_values:
            enriched_table[header][row_idx] = str(value).strip()
            logger.info(f"Set {header}[{row_idx}] = {value}")
            value_found = True
        else:
            logger.info(f"Skipping '{value}' for {header}[{row_idx}] (placeholder value)")
    else:
        # Try case-insensitive column match
        header_lower = header.lower().strip()
        for col_key, col_value in entity_data.items():
            if col_key.lower().strip() == header_lower:
                value_str = str(col_value).strip().lower()
                if col_value and value_str not in placeholder_values:
                    enriched_table[header][row_idx] = str(col_value).strip()
                    logger.info(f"Set {header}[{row_idx}] = {col_value} (matched '{col_key}')")
                    value_found = True
                else:
                    logger.info(f"Skipping '{col_value}' for {header}[{row_idx}] (placeholder value)")
                break
    
    return value_found


async def extract_data_from_research(
    final_result: Dict[str, Any],
    context_values: List[str],
    column_names: List[str],
    empty_columns: List[Tuple[int, str]],
    rows: List[List[str]],
    llm_provider: Optional[LLMProvider]
) -> Dict[str, List[str]]:
    """
    Extract structured data from research results using LLM.
    
    Args:
        final_result: Final result dictionary from research
        context_values: List of context values (entities)
        column_names: List of column names
        empty_columns: List of empty columns
        rows: List of rows
        llm_provider: LLM provider instance
        
    Returns:
        Enriched table dictionary
    """
    enriched_table = {header: [""] * len(rows) for _, header in empty_columns}
    placeholder_values = get_placeholder_values()

    if not final_result or not llm_provider:
        # Set all to "Not Found" if no result
        for col_idx, header in empty_columns:
            for row_idx in range(len(rows)):
                enriched_table[header][row_idx] = "Not Found"
        return enriched_table

    # Prepare research text
    entities_list = ', '.join(context_values)
    columns_list = ', '.join(column_names)
    
    research_text = ""
    for key, value in final_result.items():
        if isinstance(value, str):
            # Increase truncation limit significantly - we need all the data for proper extraction
            # Only truncate if it's extremely long (over 50k chars per column)
            if len(value) > 50000:
                research_text += f"{key}: {value[:50000]}... (truncated at 50k chars)\n\n"
                logger.warning(f"Truncated column {key} at 20k characters")
            else:
                research_text += f"{key}: {value}\n\n"
        else:
            research_text += f"{key}: {str(value)}\n\n"
    
    try:
        # Try to parse direct JSON first
        parsed_data = parse_direct_json(column_names, final_result)

        # If no direct JSON, use LLM extraction
        if not parsed_data:
            extraction_prompt = build_extraction_prompt(entities_list, columns_list, research_text)
            
            extracted_json = await llm_provider.generate(extraction_prompt)

            # Parse LLM response
            json_match = re.search(r'\{.*\}', extracted_json, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
        
        if parsed_data:
            # Match entities and assign values
            for row_idx, row in enumerate(rows):
                entity = context_values[row_idx] if row_idx < len(context_values) else None
                if not entity:
                    continue

                entity_data, _ = extract_entity_data(entity, parsed_data)

                if entity_data and isinstance(entity_data, dict):
                    for col_idx, header in empty_columns:
                        value_found = assign_column_value(
                            header, entity_data, empty_columns, row_idx, enriched_table
                        )
                        
                        # If no valid value found, set to "Not Found"
                        if not value_found and not enriched_table[header][row_idx]:
                            enriched_table[header][row_idx] = "Not Found"
                            logger.warning(f"Set {header}[{row_idx}] = 'Not Found' (no valid value for entity '{entity}')")
                else:
                    logger.warning(f"No data found for entity '{entity}' (row {row_idx}). Available keys: {list(parsed_data.keys())}")
                    # Set all columns for this row to "Not Found"
                    for col_idx, header in empty_columns:
                        if not enriched_table[header][row_idx]:
                            enriched_table[header][row_idx] = "Not Found"
                            logger.warning(f"Set {header}[{row_idx}] = 'Not Found' (no entity data for '{entity}')")
    
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        # If extraction fails, set all cells to "Not Found"
        for col_idx, header in empty_columns:
            for row_idx in range(len(rows)):
                if not enriched_table[header][row_idx]:
                    enriched_table[header][row_idx] = "Not Found"
    
    # Set "Not Found" for any remaining empty cells
    for col_idx, header in empty_columns:
        for row_idx in range(len(rows)):
            current_value = enriched_table[header][row_idx]
            if not current_value or str(current_value).strip().lower() in placeholder_values:
                enriched_table[header][row_idx] = "Not Found"
    
    return enriched_table

