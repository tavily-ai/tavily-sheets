"""
Clean, simple data extraction for medical professional information.
No over-engineering, just working extraction that's better than the original.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SmartFieldExtractor:
    """
    Simple, effective data extraction that actually works.
    Focuses on practical extraction over complex abstractions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_field_data(self, field: str, raw_response: str, context: Dict[str, Any]) -> str:
        """
        Extract field data using the best strategy for each field type.
        This is the only public method - keeps it simple.
        """
        try:
            if field == "linkedin_url":
                return self._extract_linkedin_url(context, raw_response)
            elif field == "email":
                return self._extract_email(context, raw_response)
            elif field == "credentials":
                return self._extract_credentials(context, raw_response)
            elif field in ["phone", "specialty"]:
                return self._extract_simple_field(field, context, raw_response)
            else:
                return self._validate_generic_response(raw_response)
                
        except Exception as e:
            self.logger.error(f"Error extracting {field}: {str(e)}")
            return "Information not found"
    
    def _extract_linkedin_url(self, context: Dict[str, Any], llm_response: str) -> str:
        """Simple LinkedIn URL extraction - just find the URL, that's it."""
        
        # Check all text sources for LinkedIn URLs
        all_text = f"{llm_response} {context.get('tavily_answer', '')} {context.get('search_content', '')}"
        
        # Add search result URLs and content
        for result in context.get('search_results', []):
            all_text += f" {result.get('url', '')} {result.get('content', '')}"
        
        # Simple regex to find LinkedIn profile URLs - be more flexible
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?'
        matches = re.findall(linkedin_pattern, all_text)
        
        if matches:
            # Clean up the URL and ensure it has https://
            url = matches[0].rstrip('/')
            if not url.startswith('http'):
                url = 'https://' + url
            self.logger.info(f"LinkedIn URL found: {url}")
            return url
        
        return "Information not found"
    
    def _extract_email(self, context: Dict[str, Any], llm_response: str) -> str:
        """Simple email extraction."""
        all_text = f"{llm_response} {context.get('tavily_answer', '')} {context.get('search_content', '')}"
        
        # Simple email regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, all_text)
        
        # Filter out obvious system emails
        for email in matches:
            if not any(bad in email.lower() for bad in ['noreply', 'no-reply', 'admin@']):
                return email
        
        return "Information not found"
    
    def _extract_credentials(self, context: Dict[str, Any], llm_response: str) -> str:
        """Simple credentials extraction."""
        all_text = f"{llm_response} {context.get('tavily_answer', '')} {context.get('search_content', '')}"
        
        # Simple credential patterns
        patterns = [r'\b(MD|DO|PhD|DDS|RN|NP|PA)\b', r'\b(FACS|FRCSC|FRCS)\b']
        
        found = set()
        for pattern in patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            found.update(matches)
        
        return ', '.join(found) if found else "Information not found"
    
    def _extract_simple_field(self, field: str, context: Dict[str, Any], llm_response: str) -> str:
        """Handle phone and specialty with basic validation."""
        return self._validate_generic_response(llm_response)
    
    def _validate_generic_response(self, response: str) -> str:
        """Basic response validation."""
        if not response or not response.strip():
            return "Information not found"
        
        response = response.strip()
        
        # Check for negative indicators
        negative_patterns = [
            'information not found', 'not found', 'not available', 
            'unavailable', 'no information', 'cannot find'
        ]
        
        if any(pattern in response.lower() for pattern in negative_patterns):
            return "Information not found"
        
        return response