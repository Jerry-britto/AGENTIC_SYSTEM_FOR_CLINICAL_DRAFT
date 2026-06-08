import os
import json
import time
from dotenv import load_dotenv

# Load env variables before clinical_agent modules are imported to guarantee API keys are loaded
load_dotenv()

import streamlit as st
from clinical_agent.parser import LlamaCloudParser
from clinical_agent.graph import compile_clinical_agent_graph

# Set page configuration with a clinical medical theme
st.set_page_config(
    page_title="Aegis - Clinical Draft Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Styling (CSS)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Navigation Selector on the Sidebar
st.sidebar.title("🏥 Aegis Console")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigate Screen:", ["Summary Generator", "Doctor Feedback & Learning"])
st.sidebar.markdown("---")

if app_mode == "Summary Generator":
    # Sidebar Configuration Layout - Keep ONLY the Document Selection option
    st.sidebar.subheader("📂 Document Selection")

    # Verify and create data directory if missing
    os.makedirs("data", exist_ok=True)

    # Load existing PDFs in data
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if not pdf_files:
        pdf_files = ["patient_records.pdf"]  # Default name placeholder

    # Toggle between upload vs. select
    selection_mode = st.sidebar.radio("Input Source", ["Select Existing Record", "Upload New Record"])

    pdf_path = None
    selected_filename = ""

    if selection_mode == "Select Existing Record":
        if len(os.listdir("data")) > 0:
            pdf_name = st.sidebar.selectbox("Choose patient record:", pdf_files)
            pdf_path = os.path.join("data", pdf_name)
            selected_filename = pdf_name
        else:
            st.sidebar.warning("No PDFs found in `data/` folder. Please upload a file.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Patient PDF", type=["pdf"])
        if uploaded_file is not None:
            pdf_path = os.path.join("data", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_filename = uploaded_file.name
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    # Main Page Action
    if pdf_path and os.path.exists(pdf_path):
        st.info(f"📄 **Selected Document:** `{selected_filename}` (Size: {os.path.getsize(pdf_path) / (1024*1024):.2f} MB)")
        
        # Run Agent Button
        if st.button("🚀 Generate Safe Discharge Draft", type="primary", use_container_width=True):
            st.write("---")
            
            # 1. Parsing Phase
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("🔄 **Phase 1: Ingesting & Parsing PDF Document...**")
            progress_bar.progress(10)
            
            parser = LlamaCloudParser()
            try:
                parsed_data = parser.parse_pdf(pdf_path)
                pages = parsed_data.get("pages", [])
                progress_bar.progress(30)
                status_text.success(f"✅ Ingestion successful! Loaded {len(pages)} parsed layouts from document.")
            except Exception as e:
                st.error(f"❌ PDF ingestion failed: {e}")
                st.stop()
                
            # 2. Compile LangGraph workflow
            status_text.markdown("⚙️ **Phase 2: Compiling Agent Graph Workflow...**")
            progress_bar.progress(40)
            workflow = compile_clinical_agent_graph()
            progress_bar.progress(50)
            
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
            
            # Live Tracing Elements - Keep ONLY the Decisional Trace Log
            st.subheader("📜 Agent Decisional Trace Log (Real-time updates)")
            trace_log_placeholder = st.empty()

            # Run State Machine Stream
            state = initial_state
            progress_bar.progress(60)
            status_text.markdown("🧠 **Phase 3: Running Agentic Clinical reasoning loop...**")
            
            try:
                for event in workflow.stream(state):
                    for node_name, updated_state in event.items():
                        state = updated_state
                        
                        # Update status
                        status_text.markdown(f"🧠 **Running Agentic Loop: Processing node `{node_name.upper()}` (Step {state['iteration_count']}/8)...**")
                        
                        # Update trace log list
                        trace_html = "<ul>"
                        for step in state['trace_steps']:
                            trace_html += f"<li>{step}</li>"
                        trace_html += "</ul>"
                        trace_log_placeholder.markdown(trace_html, unsafe_allow_html=True)
                        
                        # Small wait for layout transitions readability
                        time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.success("🎉 Process Completed Successfully!")
                
                # Persist the output files in output directories
                output_dir = "outputs"
                logs_dir = "logs"
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(logs_dir, exist_ok=True)
                
                pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
                markdown_path = os.path.join(output_dir, f"{pdf_base}-discharged_summary.md")
                json_path = os.path.join(output_dir, f"{pdf_base}-discharged_summary.json")
                trace_path = os.path.join(logs_dir, "agent_trace.md")
                
                # Save files
                markdown_content = state.get("output_markdown", "")
                with open(markdown_path, "w", encoding="utf-8") as f_md:
                    f_md.write(markdown_content)
                    
                json_data = state.get("output_json", {})
                with open(json_path, "w", encoding="utf-8") as f_json:
                    json.dump(json_data, f_json, indent=2)
                    
                # Write trace log
                trace_markdown = "# Clinical Agentic Loop Observability Trace\n\n"
                trace_markdown += f"**Total Planning Steps Executed:** {state.get('iteration_count', 0)}\n\n"
                trace_markdown += "## Reasoning and Decisional History\n\n"
                for i, step in enumerate(state.get('trace_steps', []), 1):
                    trace_markdown += f"### Step {i}\n- {step}\n\n"
                with open(trace_path, "w", encoding="utf-8") as f_trace:
                    f_trace.write(trace_markdown)
                    
            except Exception as e:
                st.error(f"❌ Critical error during workflow execution: {e}")
                st.stop()
                
            # Display Outputs Layout - Present ONLY the markdown draft
            st.write("---")
            st.subheader("🏥 Draft Clinical Discharge Summary")
            
            st.download_button(
                label="💾 Download Markdown File",
                data=markdown_content,
                file_name=f"{pdf_base}-discharged_summary.md",
                mime="text/markdown",
                use_container_width=True
            )
            
            st.markdown(markdown_content)
    else:
        st.warning("Please upload a PDF document or select an existing record from the sidebar to begin.")

else:
    # Doctor Feedback & Learning Dashboard
    st.subheader("🛡️ Clinician Feedback Loop & Preference Learning")
    st.markdown("""
    In production clinical environments, clinicians review and edit AI draft summaries before finalization. 
    Aegis can automatically learn style preferences and terminology corrections from these edits, so future drafts need fewer manual updates.
    
    This simulation runs our agentic pipeline over a training dataset of clinical records, corrects them using a **Simulated Reviewer** (with a hidden composite policy), compares drafts to extract guidelines, and updates the **Correction Memory**.
    """)
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        run_loop = st.button("⚙️ Run Simulated Clinician Feedback Loop", type="primary", use_container_width=True)
    with col_ctrl2:
        clear_cache = st.button("🧹 Clear Correction Memory Cache", use_container_width=True)
        
    if clear_cache:
        from clinical_agent.learning import clear_correction_memory
        clear_correction_memory()
        if os.path.exists("logs/evaluation_results.json"):
            try:
                os.remove("logs/evaluation_results.json")
            except:
                pass
        st.success("Successfully reset correction memory and logs.")
        st.rerun()
        
    if run_loop:
        status_box = st.status("Initializing simulated clinical feedback loop...", expanded=True)
        with status_box:
            st.write("Resetting memory...")
            from clinical_agent.learning import clear_correction_memory
            clear_correction_memory()
            
            st.write("Executing feedback pipeline and training iterations...")
            from scripts.run_learning_eval import run_evaluation_flow
            try:
                run_evaluation_flow()
                status_box.update(label="Feedback Loop Completed Successfully!", state="complete", expanded=False)
                st.success("Feedback loop simulation finished! Results loaded below.")
            except Exception as e:
                status_box.update(label="Evaluation failed!", state="error")
                st.error(f"Error executing feedback loop: {e}")
                
    # Load and display results if logs/evaluation_results.json exists
    eval_path = "logs/evaluation_results.json"
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                
            st.write("---")
            st.subheader("📈 Performance Improvement Curves")
            
            # Prepare data for plotting
            import pandas as pd
            
            steps = eval_data.get("training_steps", [])
            chart_data = []
            for step in steps:
                chart_data.append({
                    "Step": f"Case {step['step']}: Patient {step['patient_id']}",
                    "Similarity": step["similarity"],
                    "Policy Compliance": step["compliance"]["overall_compliance"]
                })
                
            df = pd.DataFrame(chart_data)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Levenshtein Draft-to-Correction Similarity** (Higher is better)")
                st.line_chart(df.set_index("Step")["Similarity"])
            with col_chart2:
                st.markdown("**Clinician Policy Compliance Rate**")
                st.line_chart(df.set_index("Step")["Policy Compliance"])
                
            # Test cases improvement table
            st.subheader("📊 Held-Out Test Cases Evaluation")
            baseline_test = eval_data.get("baseline_test", [])
            final_test = eval_data.get("final_test", [])
            
            test_rows = []
            for b, f in zip(baseline_test, final_test):
                pid = b["patient_id"]
                test_rows.append({
                    "Patient Case": f"Patient {pid} (Held-out Test)",
                    "Baseline Similarity": f"{b['similarity']:.4f}",
                    "Final Similarity": f"{f['similarity']:.4f}",
                    "Similarity Gain": f"{f['similarity'] - b['similarity']:+0.4f}",
                    "Baseline Compliance": f"{b['compliance']['overall_compliance']:.2%}",
                    "Final Compliance": f"{f['compliance']['overall_compliance']:.2%}",
                    "Compliance Gain": f"{f['compliance']['overall_compliance'] - b['compliance']['overall_compliance']:+.2%}"
                })
            st.table(test_rows)
            
            # Display learned preferences guidelines
            st.subheader("💡 Active Correction Memory (Learned Guidelines)")
            learned_rules = eval_data.get("learned_rules", [])
            if learned_rules:
                for rule in learned_rules:
                    st.info(f"👉 {rule}")
            else:
                st.write("No preferences rules learned yet.")
                
        except Exception as e:
            st.error(f"Failed to read evaluation logs: {e}")
            
    st.write("---")
    st.subheader("🧠 System Limitations & Safety Discussion")
    st.markdown("""
    ### 1. The Cold-Start & Limited-Data Problem
    Traditional preference learning loops (like DPO or direct RLHF) require thousands of edited pairs and massive model fine-tuning budgets to update parameters. This creates a severe cold-start problem in clinical settings.
    
    Aegis solves this using **Structured Correction-Memory via In-Context Learning**:
    - Clinician preferences are extracted by comparing draft/corrected pairs in real-time.
    - These rules are generalized into concise, human-readable guidelines.
    - These guidelines are immediately injected as dynamic instructions in subsequent synthesis tasks, reducing edit burden from Patient Case #1.

    ### 2. Preventing "Gaming" and Quality Degradation
    An inherent danger of optimizing systems to "reduce edit distance" is **gaming**:
    - The generator might become increasingly **vague** or omit detailed clinical metrics to avoid errors that trigger doctor edits.
    - The generator might prioritize formatting style over clinical substance.
    
    **Mitigation Safeguards**:
    1. *Pipeline Modular Separation*: The Correction Memory only informs the final *Synthesizer* node. The medical facts are gathered independently by the *Reader*, *Reconciler*, and *Verifier* nodes.
    2. *Absolute Safety Priority*: The Safety Verifier checks the final draft *after* synthesis for clinical contradictions, missing demographics, and safety warnings (e.g. prescribing Loperamide). If a stylistic preference conflicts with safety, the safety rule strictly overrides the preference.
    """)
