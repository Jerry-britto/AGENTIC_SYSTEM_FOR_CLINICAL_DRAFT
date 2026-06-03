import os
import json
from dotenv import load_dotenv
from clinical_agent.parser import LlamaCloudParser
from clinical_agent.graph import compile_clinical_agent_graph

def main():
    load_dotenv()
    print("=" * 60)
    print("       CLINICAL DISCHARGE SUMMARY AGENTIC SYSTEM       ")
    print("=" * 60)

    pdf_path = "data/patient_records.pdf"
    
    # 1. Parse and extract layout markdown
    parser = LlamaCloudParser()
    try:
        parsed_data = parser.parse_pdf(pdf_path)
    except Exception as e:
        print(f"[FATAL ERROR] PDF parsing failed: {e}")
        return

    pages = parsed_data.get("pages", [])
    print(f"[Success] Loaded {len(pages)} parsed layouts/pages successfully.")

    # 2. Compile LangGraph workflow
    print("[Agent] Compiling LangGraph workflow...")
    workflow = compile_clinical_agent_graph()

    # 3. Initialize Agent State
    initial_state = {
        "parsed_pages": pages,
        "extracted_data": {},
        "medication_changes": [],
        "safety_warnings": [],
        "clinician_review_flags": [],
        "current_plan": "Initialize extraction plan.",
        "trace_steps": ["State initialized. Commencing clinical extraction plan."],
        "iteration_count": 0,
        "output_markdown": "",
        "output_json": {},
        "is_finished": False
    }

    # 4. Execute LangGraph State Machine
    print("[Agent] Initiating Agentic Loop...")
    try:
        final_state = workflow.invoke(initial_state)
    except Exception as e:
        print(f"[FATAL ERROR] Agent loop crashed: {e}")
        return

    # 5. Persist Output Files
    output_dir = "outputs"
    logs_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(output_dir, f"{pdf_base}-discharged_summary.md")
    json_path = os.path.join(output_dir, f"{pdf_base}-discharged_summary.json")
    trace_path = os.path.join(logs_dir, "agent_trace.md")

    # Save Markdown Summary Draft
    markdown_content = final_state.get("output_markdown", "")
    with open(markdown_path, "w", encoding="utf-8") as f_md:
        f_md.write(markdown_content)
    print(f"\n[Success] Persisted Clinician Markdown Draft at: {markdown_path}")

    # Save JSON Draft Data
    json_data = final_state.get("output_json", {})
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2)
    print(f"[Success] Persisted Machine-Readable Structured JSON at: {json_path}")

    # Save detailed Execution Trace
    trace_steps = final_state.get("trace_steps", [])
    trace_markdown = "# Clinical Agentic Loop Observability Trace\n\n"
    trace_markdown += f"**Total Planning Steps Executed:** {final_state.get('iteration_count', 0)}\n\n"
    trace_markdown += "## Reasoning and Decisional History\n\n"
    for i, step in enumerate(trace_steps, 1):
        trace_markdown += f"### Step {i}\n- {step}\n\n"

    trace_markdown += "\n## Safety & Escalation Summary\n"
    trace_markdown += f"- **Clinician Review Flags Raised:** {len(final_state.get('clinician_review_flags', []))}\n"
    trace_markdown += f"- **Clinical Warnings Generated:** {len(final_state.get('safety_warnings', []))}\n\n"
    
    trace_markdown += "### Active Warnings\n"
    for warning in final_state.get("safety_warnings", []):
        trace_markdown += f"- `[{warning.get('type')}]` {warning.get('message')}\n"

    with open(trace_path, "w", encoding="utf-8") as f_trace:
        f_trace.write(trace_markdown)
    print(f"[Success] Persisted detailed Observability Trace at: {trace_path}")
    print("=" * 60)
    print("                    PROCESS COMPLETED                    ")
    print("=" * 60)

if __name__ == "__main__":
    main()
