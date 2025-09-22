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
        """Extract LinkedIn URL with broader patterns and basic name validation."""
        
        # Check all text sources for LinkedIn URLs
        all_text = f"{llm_response} {context.get('tavily_answer', '')} {context.get('search_content', '')}"
        
        # Add search result URLs and content
        for result in context.get('search_results', []):
            all_text += f" {result.get('url', '')} {result.get('content', '')}"
        
        # Broader regex to find LinkedIn profile URLs
        linkedin_pattern = r'(?:https?://)?(?:[a-z]+\.)?linkedin\.com/(?:in|pub|profile)/[a-zA-Z0-9\-_%]+/?(?:\?[^\s]*)?'
        matches = re.findall(linkedin_pattern, all_text)
        
        if matches:
            # Basic validation against doctor name tokens
            doctor_name = (context.get('doctor_name') or '').strip().lower()
            name_tokens = [t for t in re.split(r'\s+', doctor_name) if t and len(t) > 1]
            scored = []
            for url in matches:
                clean = url if url.startswith('http') else 'https://' + url
                path = clean.split('linkedin.com/')[-1].lower()
                score = 0
                for token in name_tokens:
                    if token in path:
                        score += 1
                # prefer /in/ and medical hints
                if '/in/' in path:
                    score += 1
                if any(h in path for h in ['md', 'dr', 'doctor']):
                    score += 1
                scored.append((score, clean.rstrip('/')))
            scored.sort(reverse=True)
            best = scored[0][1]
            self.logger.info(f"LinkedIn URL found: {best}")
            return best
        
        return "Information not found"
    
    def _extract_email(self, context: Dict[str, Any], llm_response: str) -> str:
        """Robust email extraction including obfuscations and ranking."""
        all_text = f"{llm_response} {context.get('tavily_answer', '')} {context.get('search_content', '')}"
        for result in context.get('search_results', []):
            all_text += f" {result.get('url', '')} {result.get('content', '')} {result.get('raw_content', '')}"

        # Standard emails
        std_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        # Obfuscations: "name at domain dot com", "name[at]domain[dot]org", "name (at) domain (dot) edu"
        obf_pattern = r'\b([A-Za-z0-9._%+-]+)\s*(?:\[?\(?\s*at\s*\)?\]?|@)\s*([A-Za-z0-9.-]+)\s*(?:\[?\(?\s*dot\s*\)?\]?|\.)\s*([A-Za-z]{2,})\b'

        candidates = set(re.findall(std_pattern, all_text))
        for user, domain, tld in re.findall(obf_pattern, all_text, flags=re.IGNORECASE):
            candidates.add(f"{user}@{domain}.{tld}")

        # Clean and rank
        cleaned = []
        doctor_name = (context.get('doctor_name') or '').strip().lower()
        hospital_name = (context.get('hospital_name') or '').strip().lower()
        name_tokens = [t for t in re.split(r'\s+', doctor_name) if t and len(t) > 1]
        bad_tokens = ['noreply', 'no-reply', 'donotreply', 'admin', 'webmaster', 'info@', 'support@']

        for email in candidates:
            e = email.strip().lower()
            if any(b in e for b in bad_tokens):
                continue
            score = 0
            local, _, domain = e.partition('@')
            # Name token match
            for token in name_tokens:
                if token in local:
                    score += 1
            # Hospital/practice domain hint
            if hospital_name:
                for token in hospital_name.split():
                    if token.isalpha() and token in domain:
                        score += 1
            # Prefer edu/org over generic if tied
            if domain.endswith('.edu') or domain.endswith('.org'):
                score += 1
            cleaned.append((score, e))

        if not cleaned:
            return "Information not found"

        cleaned.sort(reverse=True)
        return cleaned[0][1]
    
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