# Analysis and Refactoring Plan

## 1. Executive Summary

The current codebase is functional but suffers from three core issues:
- **Codebase Inflation:** Redundant API endpoints and duplicated logic between the frontend and backend make the system larger and harder to maintain.
- **Hardcoded Logic:** Extensive hardcoding, especially in the backend enrichment pipeline, limits the system's "agentic" potential. It follows rigid, predefined rules instead of dynamically planning and adapting.
- **Inconsistent UX:** The user experience for enriching a single column (streaming) is vastly different from enriching a whole table (blocking), which can be confusing and frustrating.

This plan outlines a series of refactoring steps to address these flaws, aiming to create a leaner, more intelligent, and more user-friendly application.

## 2. Flaw: Codebase Inflation and Redundancy

**Analysis:** The backend has multiple sets of enrichment endpoints (`/api/enrich` vs. `/api/enrich-medical`), and the frontend duplicates logic that exists on the backend (field name canonicalization). This increases complexity and maintenance overhead.

**Instructions for Remediation:**

1.  **Consolidate API Endpoints:**
    - Deprecate and remove the older, generic endpoints: `/api/enrich`, `/api/enrich/batch`, and `/api/enrich-table` from `app.py`.
    - The `enrich-medical` endpoints are more specialized and feature-rich (e.g., streaming) and should be the single source of truth for enrichment.

2.  **Unify Frontend Enrichment Logic:**
    - In `ui/src/components/Spreadsheet.tsx`, modify the `enrichTable` function. Instead of calling the now-deprecated `/api/enrich-table`, it should iterate through the columns that need enrichment and call the existing `enrichColumn` function for each. This will reuse the streaming API and provide a consistent UX.

3.  **Remove Frontend Redundancy:**
    - In `ui/src/components/Spreadsheet.tsx`, remove the `canonicalizeField` utility function. The backend already performs this normalization. The frontend should send the raw column header, simplifying the client-side code.

4.  **Refactor Backend Batch Processing:**
    - In `app.py`, the `SequentialBatchProcessor` class and the custom batching logic within `StreamingMedicalEnricher` are redundant. Refactor them into a single, robust batch processing utility that can be used by both the batch and streaming endpoints to handle rate limiting and concurrency.

## 3. Flaw: Hardcoded Logic and Lack of "Agentic" Feel

**Analysis:** The core enrichment logic in `backend/graph.py` is powerful but constrained by hardcoded prompts, search configurations, and a rigid, linear execution graph. This prevents the system from reasoning about its tasks and adapting its strategy.

**Instructions for Remediation:**

1.  **Introduce Dynamic Field Profiles:**
    - In the backend, create a configuration system (e.g., a `config/` directory with YAML or JSON files) to define "Field Profiles".
    - Replace the hardcoded dictionaries `_get_field_specific_guidance` and `_get_search_config` in `backend/graph.py`. Each field profile should define its own search strategy, prompt guidance, and model complexity.
    - **Example `email.yaml`:**
      ```yaml
      field_name: email
      search_depth: advanced
      complexity: simple
      prompt_guidance: "You are an expert at finding professional contact information..."
      ```
    - This makes the system's knowledge external to the code and easily extensible.

2.  **Implement an Agentic Planner:**
    - In `backend/graph.py`, enhance the `planner` node in the `MedicalEnrichmentPipeline`. Instead of a simple `if` statement, use an LLM call to create a dynamic plan.
    - The planner should receive the full context and decide on a sequence of steps (e.g., `['resolve_hospital_domain', 'primary_email_search', 'extract_email']`). The graph should then execute this plan.

3.  **Enable Self-Correction with a Cyclical Graph:**
    - Modify the `langgraph` structure in `backend/graph.py` to be cyclical.
    - Add a `validate_answer` node after the `extract` node.
    - If `validate_answer` determines the result is poor (e.g., "Information not found"), it should route the process back to the `planner` node to try a new strategy (e.g., use `advanced` search, generate a new query). This creates a powerful self-correction loop.

4.  **Decouple Frontend from Backend Configuration:**
    - In `app.py`, create a new endpoint like `/api/supported-fields`. This endpoint should read the field profiles from the configuration files and return a list of supported field names.
    - In `ui/src/App.tsx`, remove the hardcoded `TARGET_FIELDS` array. Instead, fetch the list of supported fields from the new `/api/supported-fields` endpoint when the application loads.

## 4. Flaw: Inconsistent User Experience

**Analysis:** The UI provides excellent real-time feedback for single-column enrichment via streaming but offers no feedback for full-table enrichment, which appears as a long, blocking operation.

**Instructions for Remediation:**

1.  **Standardize on Streaming Enrichment:**
    - As detailed in Flaw #2, refactor the `enrichTable` function in `ui/src/components/Spreadsheet.tsx` to sequentially call the `enrichColumn` function for every column that needs to be enriched.
    - This ensures that all enrichment tasks, whether for a single column or the whole table, use the streaming API. The user will see cells being populated in real-time, providing a consistent and superior experience. This change also simplifies the backend by removing the need for a separate table enrichment endpoint.
