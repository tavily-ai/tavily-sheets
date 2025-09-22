# Data Enrichment Application Refactoring Plan

## 1. Project Goal

The primary goal of this refactoring is to create a robust, predictable, and intelligent data enrichment application. This involves two main objectives:
1.  **Fix State Management:** Overhaul the current persistence model to create a clean, stateless backend and a predictable frontend user experience.
2.  **Enhance Query Intelligence:** Radically improve the search query generation system to be more effective for all enrichment types, especially complex and abstract fields.

---

## 2. Objective 1: Resolve State and Persistence Issues

### Problem Analysis
The current application suffers from a state management conflict between the backend and frontend, creating a confusing user experience.
- **Backend:** Persists all enrichment results to a `enrichment_results.json` file. This file survives application restarts, preventing a "fresh start" and causing old data to reappear unexpectedly.
- **Frontend:** Uses the browser's `localStorage` to persist *only* enriched cells. When the page is refreshed, the user's original input is lost, but the old enriched data is applied to a new blank grid, creating a disconnected state.

### Solution: A Stateless Backend and Session-Based Frontend

#### 2.1. Backend Refactoring (`backend/graph.py`)
The backend will be made stateless between application runs.

- **Action:** Remove all file-based persistence logic.
  - **Step 1:** Delete the `save_enrichment_result` and `load_enrichment_result` functions.
  - **Step 2:** In the `search_medical_data` function, remove the initial block that calls `load_enrichment_result`. The function must not read from any local files.
  - **Step 3:** In the `extract_field_data` function, remove the call to `save_enrichment_result`. The function must not write to any local files.
  - **Note:** The in-memory `TTLCache` (`_enrichment_cache`) should be kept, as it provides valuable de-duplication for identical requests within a single running session without causing the cross-session state problems.

#### 2.2. Frontend Refactoring (`ui/src/App.tsx`)
The frontend will hold the entire application state in memory for a single browser session. The "Export" button will be the user's method for persisting their work.

- **Action:** Remove `localStorage` and unify the state.
  - **Step 1:** Remove `localStorage` logic. In the `enrichedCells` `useState` declaration, remove the initializer that reads from `localStorage`. Also, delete the `useEffect` hook responsible for writing to `localStorage`.
  - **Step 2:** Unify the React state. The separate `originalData`, `enrichedCells`, and computed `data` states are the source of the UI bugs. They will be replaced by a single state variable, `spreadsheetData`.
    ```javascript
    // Replace the multiple state variables with this single one
    const [spreadsheetData, setSpreadsheetData] = useState<SpreadsheetData>(/* initial empty data */);
    ```
  - **Step 3:** Refactor all component functions (`handleCellUpdate`, `handleEnrichmentUpdate`, `handleConfirmMapping`, etc.) to read from and write to the single `spreadsheetData` state object.
  - **Step 4:** Pass the `spreadsheetData` state directly to the `Spreadsheet` component as its `data` prop.

---

## 3. Objective 2: Enhance Query Generation System

### Problem Analysis
The current query generation is too simplistic. It relies on the column header alone, which is insufficient for abstract concepts like "influence summary" or "strategic summary". The system needs to be taught *what these concepts mean*.

### Solution: Injecting Domain-Specific Intelligence into Query Prompts

The solution is to significantly enhance the `_get_field_specific_guidance` function in `backend/graph.py`. This function will provide the query-generating LLM with a rich, detailed definition, including keywords, synonyms, and examples for each complex field.

#### 3.1. Enhance Field-Specific Guidance (`backend/graph.py`)

- **Action:** Update the `guidance` dictionary within the `_get_field_specific_guidance` function with the detailed instructions below. This will provide the LLM with the necessary context to generate truly intelligent queries.

```python
# This dictionary should be updated in backend/graph.py

guidance = {
    "email": """
TARGET: Professional email address.
KEYWORDS: email, contact, directory, staff, faculty, @.
STRATEGY: Search for the person's name along with keywords like 'email' or 'contact'. Also check for common email formats like 'firstname.lastname@institution.edu'.
""",
    "phone": """
TARGET: Professional phone number (office, clinic, or hospital).
KEYWORDS: phone, telephone, office, clinic, contact, appointment, fax.
STRATEGY: Search for the person's name and their institution along with keywords like 'office phone' or 'contact number'.
""",
    "specialty": """
TARGET: The primary medical specialty.
KEYWORDS: specialty, specializes in, department of, division of, board certified in, clinical focus.
STRATEGY: Search for the surgeon's name along with their hospital and the keyword 'specialty'. The result should be a standard medical specialty (e.g., "Orthopedic Surgery", "Neurosurgery").
""",
    "subspecialty": """
TARGET: The surgeon's specific area of clinical focus beyond their main specialty.
KEYWORDS: subspecialty, focus area, fellowship trained in, clinical interests, specialized procedures, specific conditions (e.g., "pancreatic cancer", "spinal deformities").
EXAMPLES: A "General Surgery" specialty might have a subspecialty of "Surgical Oncology". An "Orthopedic Surgery" specialty might have a subspecialty of "Joint Replacement".
STRATEGY: Search for the surgeon's name and terms like "fellowship", "specializes in", or "clinical focus".
""",
    "credentials": """
TARGET: All medical degrees, board certifications, and professional fellowships.
KEYWORDS: MD, DO, PhD, FACS (Fellow, American College of Surgeons), FRCS (Fellow, Royal College of Surgeons), board certified, residency, fellowship, medical school, education.
STRATEGY: Search for the surgeon's name along with terms like "credentials", "education", or "board certified".
""",
    "linkedin_url": """
TARGET: The URL of the surgeon's professional LinkedIn profile.
KEYWORDS: site:linkedin.com, LinkedIn profile, professional network.
STRATEGY: Perform a site-specific search on LinkedIn for the surgeon's name and their institution.
""",
    "influence_summary": """
TARGET: A summary of the surgeon's academic and professional influence.
KEYWORDS: publications, citations, h-index, research, clinical trials, grants, awards, keynote speaker, conference presentation, editorial board, society leadership.
METRICS: Look for publication counts ("100+ publications"), citation metrics ("cited X times"), h-index values, and grant funding (e.g., "NIH R01 grant").
ROLES: Department Chair, Program Director, President of a medical society (e.g., "President of the American College of Surgeons"), journal editor.
STRATEGY: Generate a query that combines the surgeon's name with several of the keywords and roles above to find evidence of their academic and professional impact.
""",
    "strategic_summary": """
TARGET: A summary of the surgeon's strategic value and role within their institution and the industry.
KEYWORDS: leadership, director, chief, chair, board member, committee, advisory board, consultant, key opinion leader (KOL), industry collaboration, venture capital, startup, founder.
ROLES: Chief of Surgery, Hospital Board Member, Medical Director, Committee Chair (e.g., "Chair of the Quality and Safety Committee"), consultant for medical device companies, scientific advisor.
STRATEGY: Generate a query that looks for the surgeon's name in conjunction with leadership titles and business-oriented keywords to assess their institutional and commercial influence.
""",
    "additional_contacts": """
TARGET: Alternative contact points like an assistant, office manager, or department coordinator.
KEYWORDS: assistant, coordinator, scheduler, department contact, office manager, administrative assistant.
STRATEGY: Search for the surgeon's name or their department name along with keywords like "administrative assistant" or "office contact".
"""
}
```

#### 3.2. Update the Main Query Prompt (`backend/graph.py`)

- **Action:** With the guidance function enhanced, update the main query generation prompt in `_generate_intelligent_query_with_llm` to use this new context effectively. This prompt structure encourages the LLM to create a single, powerful query with a fallback mechanism.

- **New Prompt Template:**

```python
query_prompt = f"""
Generate ONE single, precise, and efficient web search query to find the **{state.target_field}** for the medical professional described below.

**PERSON DETAILS:**
{full_context}

**SEARCH GOAL & CONTEXT:**
{self._get_field_specific_guidance(state.target_field)}

**QUERY CONSTRUCTION RULES:**
1.  **Create a Multi-Part Query:** Use boolean operators (`OR`) and parentheses `()` to create a query with a primary and a secondary search strategy in a single line.
2.  **Primary Search (Person-Specific):** The first part of the query should be highly specific. Combine the person's full name (in quotes) with the most relevant keywords from the "SEARCH GOAL & CONTEXT" section.
3.  **Secondary Search (Institution-Specific):** The second part, connected by `OR`, should be a fallback to search for the information at their primary institution. Combine the institution's name with more general keywords related to the search goal.
4.  **Example Query Structure:** `("Surgeon Name" + specific keywords) OR ("Hospital Name" + general keywords)`
5.  **Be Efficient:** Your goal is to create the single best query string that has the highest probability of finding the correct information in one search.

Return only the single, complete search query string and nothing else.
"""
```
