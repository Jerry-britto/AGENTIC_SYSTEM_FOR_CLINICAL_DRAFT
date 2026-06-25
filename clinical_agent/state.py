from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Raw parsed pages from PDF
    parsed_pages: List[Dict[str, Any]]
    
    # Accumulated extracted data
    extracted_data: Dict[str, Any]
    
    # Medication reconciliation records
    medication_changes: List[Dict[str, Any]]
    
    # Clinical alerts, drug interactions, or conflicting details
    safety_warnings: List[Dict[str, Any]]
    
    # Specific items flagged for human clinician review
    clinician_review_flags: List[str]
    
    # Current active agent checklist/plan
    current_plan: str
    
    # Step-by-step trace log list
    trace_steps: List[str]
    
    # Iteration counter (cap at 8)
    iteration_count: int
    
    # Track completed phases to prevent infinite loops (e.g. ["read", "reconcile", "verify"])
    completed_phases: List[str]
    
    # Track page numbers retrieved by reader to avoid reading duplicate information
    retrieved_pages: List[int]
    
    # Documented hallucination checks (cross-referencing results)
    hallucination_report: Dict[str, Any]
    
    # Generated reports
    output_markdown: str
    output_json: Dict[str, Any]
    
    # Control flags
    is_finished: bool
