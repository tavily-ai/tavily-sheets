import requests
from requests.exceptions import RequestException
from typing import List, Dict, Union
from pydantic import BaseModel

TAVILY_API_ENDPOINT = "https://api.tavily.com"

class TableData(BaseModel):
    rows: List[str]  # List of target values to enrich
    context_values: Dict[str, str]
    answer: str = None
    search_result: str = None

def count_enrichable_items(
    data: Union[List[str], Dict[str, TableData]],
    is_batch: bool = False
) -> int:
    """
    Count the number of items that will be enriched in either a batch or table enrichment request.
    Only counts non-empty items.
    
    Args:
        data: Either a list of strings (for batch) or a dict of TableData (for table)
        is_batch: Boolean flag to indicate if this is a batch request (True) or table request (False)
        
    Returns:
        int: Number of items that will be enriched
    """
    if is_batch:
        # For batch requests, data is a List[str]
        return sum(1 for item in data if item.strip())
    else:
        # For table requests, data is a Dict[str, TableData]
        total_cells = 0
        for table_data in data.values():
            total_cells += sum(1 for row_value in table_data.rows if row_value.strip())
        return total_cells

def check_api_key(api_key: str, request_amount: int) -> bool:
    """
    Check if the API key is authorized for the given use case
    
    Args:
        api_key: The API key to check
    
    Returns:
        bool: True if authorized
    """
    try:
        payload = {
            "api_key": api_key,
            "use_case": "data-enrichment",
            "request_amount": request_amount
        }

        response = requests.post(
            f"{TAVILY_API_ENDPOINT}/authorize-use-case",
            json=payload
        )

        response.raise_for_status()
        
        result = response.json()
        
        if not result.get("success"):
            raise requests.exceptions.HTTPError("Authorization failed")
            
        return True

    except requests.exceptions.HTTPError as e:
        raise
    except RequestException as e:
        raise