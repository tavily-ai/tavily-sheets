"""Functions for preparing table data before enrichment."""

from typing import List, Tuple, Optional


def find_empty_columns(headers: List[str], rows: List[List[str]]) -> List[Tuple[int, str]]:
    """
    Find columns that have at least one empty cell.
    
    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of cell values
        
    Returns:
        List of tuples (column_index, header_name) for columns with empty cells
    """
    empty_columns = []
    for col_idx, header in enumerate(headers):
        if header.strip():
            has_empty = any(not row[col_idx].strip() if col_idx < len(row) else True for row in rows)
            if has_empty:
                empty_columns.append((col_idx, header))
    return empty_columns


def collect_context_values(
    headers: List[str],
    rows: List[List[str]],
    context_column: Optional[str] = None
) -> List[str]:
    """
    Collect context values from the table for enrichment.

    Combines all non-empty column values into a rich context string for each row.
    This provides more context for the research query (e.g., "Name: John, Job: CEO, Company: Acme").

    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of cell values
        context_column: Optional column name to use as the primary identifier

    Returns:
        List of context values (one per row)
    """
    context_values = []

    for row in rows:
        # Build a rich context string combining all non-empty values
        context_parts = []
        primary_value = None

        for col_idx, header in enumerate(headers):
            if col_idx < len(row):
                value = row[col_idx].strip()
                if value:
                    context_parts.append(f"{header}: {value}")
                    # Track the primary value (context_column or first non-empty)
                    if context_column and header == context_column:
                        primary_value = value
                    elif primary_value is None:
                        primary_value = value

        if context_parts:
            # Use the combined context string for better research queries
            context_values.append(", ".join(context_parts))
        elif primary_value:
            # Fallback to just the primary value
            context_values.append(primary_value)

    return context_values

