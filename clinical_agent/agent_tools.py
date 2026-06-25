import re
from clinical_agent.tools import check_drug_interactions

# Define tool schemas for Groq Tool-Calling API
GROQ_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "Check for clinical drug-drug interactions among a list of discharge medications. Crucial to run before finalizing recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of medication names (brand or generic) to cross-reference."
                    }
                },
                "required": ["medications"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_page",
            "description": "Retrieve the full text content of a specific page number from the patient's records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "description": "The 1-indexed page number to retrieve."
                    }
                },
                "required": ["page_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_patient_records",
            "description": "Search for specific clinical topics or keywords (e.g. 'lab results', 'allergies') across all pages of the patient document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term or topic description to look up."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Implement python execution handlers for the tools
def execute_tool(name: str, arguments: dict, parsed_pages: list) -> str:
    """
    Executes a tool by name with arguments and context. Returns a string description of the result.
    """
    if name == "check_drug_interactions":
        meds = arguments.get("medications", [])
        if not meds:
            return "Error: No medications list provided."
        alerts = check_drug_interactions(meds)
        if not alerts:
            return "Success: No critical drug-drug interactions found among checked medications."
        import json
        return f"WARNING: Found drug-drug interactions:\n{json.dumps(alerts, indent=2)}"
        
    elif name == "read_document_page":
        try:
            page_num = int(arguments.get("page_number", 1))
        except (ValueError, TypeError):
            return "Error: Invalid page number format."
            
        for page in parsed_pages:
            if page.get("page_number") == page_num:
                return f"--- Content of Page {page_num} ---\n{page.get('markdown', '')}"
        return f"Error: Page number {page_num} not found in the documents."
        
    elif name == "search_patient_records":
        query = str(arguments.get("query", "")).lower()
        if not query:
            return "Error: Empty search query."
            
        matched_pages = []
        for page in parsed_pages:
            text = page.get("markdown", "").lower()
            if query in text or any(kw in text for kw in query.split()):
                matched_pages.append(page.get("page_number"))
                
        if not matched_pages:
            return f"No direct text matches found for query '{query}'."
            
        # Compile brief context snippets from matched pages
        snippet_output = f"Matches found on pages: {matched_pages}.\n"
        for page_num in matched_pages[:3]: # Cap snippet preview to top 3 pages for token conservation
            for page in parsed_pages:
                if page.get("page_number") == page_num:
                    text = page.get("markdown", "")
                    # Extract up to 1000 characters around matches or just the top of page
                    snippet_output += f"\n--- Page {page_num} snippet ---\n{text[:1000]}\n..."
        return snippet_output
        
    return f"Error: Unknown tool '{name}'."
