import os
import json
import re
from dotenv import load_dotenv

# Load env variables before other modules are imported
load_dotenv()

from clinical_agent.synthetic_data import get_synthetic_patient_data, generate_and_cache_synthetic_patients
from clinical_agent.graph import compile_clinical_agent_graph
from clinical_agent.learning import (
    calculate_normalized_similarity,
    SimulatedReviewer,
    PreferenceLearner,
    save_correction_memory,
    clear_correction_memory,
    load_correction_memory
)
from clinical_agent.logging_utils import logger

def check_policy_compliance(draft: str, patient_id: int) -> dict:
    """
    Evaluates the compliance of the generated draft summary against the doctor's hidden preferences.
    """
    draft_lower = draft.lower()
    
    # Fetch patient details
    patient_data = {
        1: {"meds": ["klavox", "lasix", "nexium"], "labs_count": 3},
        2: {"meds": ["lipitor", "lasix", "ventolin"], "labs_count": 4},
        3: {"meds": ["augmentin", "lipitor", "glucophage"], "labs_count": 4},
        4: {"meds": ["klavox", "nexium"], "labs_count": 4},
        5: {"meds": ["lasix", "lipitor"], "labs_count": 2},
        6: {"meds": ["ventolin", "solu-medrol"], "labs_count": 3}
    }
    
    p_info = patient_data.get(patient_id, {"meds": [], "labs_count": 1})
    
    # 1. Terminology Compliance (fraction of brand names replaced with generic)
    brand_checked = p_info["meds"]
    brand_present = 0
    for brand in brand_checked:
        if brand in draft_lower:
            brand_present += 1
            
    term_score = 1.0
    if brand_checked:
        term_score = (len(brand_checked) - brand_present) / len(brand_checked)
        
    # 2. Lab formatting compliance (checking for qualifiers like normal, elevated, low)
    qualifiers = re.findall(r'\((normal|elevated|low|abnormal)\)', draft_lower)
    lab_score = 0.0
    if p_info["labs_count"] > 0:
        lab_score = min(len(qualifiers) / p_info["labs_count"], 1.0)
    else:
        lab_score = 1.0
        
    # 3. Follow-up structure compliance (checking for subheadings Medications, Appointments, Restrictions)
    required_headers = ["medications", "appointments", "restrictions"]
    header_found = 0
    for header in required_headers:
        if header in draft_lower:
            header_found += 1
            
    followup_score = header_found / len(required_headers)
    
    overall_score = (term_score + lab_score + followup_score) / 3.0
    
    return {
        "term_compliance": term_score,
        "lab_compliance": lab_score,
        "followup_compliance": followup_score,
        "overall_compliance": overall_score
    }

def run_evaluation_flow():
    logger.info("=" * 60)
    logger.info("       CLINICAL AGENT LEARNING EVALUATION ENGINE       ")
    logger.info("=" * 60)
    
    # 1. Initialize dataset
    generate_and_cache_synthetic_patients()
    
    # 2. Reset Correction Memory
    clear_correction_memory()
    
    # Compile Graph
    workflow = compile_clinical_agent_graph()
    reviewer = SimulatedReviewer()
    learner = PreferenceLearner()
    
    test_ids = [5, 6]
    train_ids = [1, 2, 3, 4]
    
    # 3. Baseline Evaluation on Test Set
    logger.info("[Eval] Running BASELINE evaluation on test cases (Patients 5 & 6)...")
    baseline_results = []
    
    for pid in test_ids:
        patient_data = get_synthetic_patient_data(pid)
        initial_state = {
            "parsed_pages": patient_data["pages"],
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
        
        final_state = workflow.invoke(initial_state)
        draft = final_state.get("output_markdown", "")
        
        # simulated correction
        edited = reviewer.review_draft(draft)
        
        # calculate metrics
        similarity = calculate_normalized_similarity(draft, edited)
        compliance = check_policy_compliance(draft, pid)
        
        baseline_results.append({
            "patient_id": pid,
            "similarity": similarity,
            "compliance": compliance
        })
        logger.info(f"Baseline - Patient {pid}: Similarity = {similarity:.4f}, Compliance = {compliance['overall_compliance']:.4f}")
        
    # 4. Sequential Learning Loop on Training Set
    logger.info("[Eval] Starting SEQUENTIAL learning loop across training cases (Patients 1 to 4)...")
    training_steps = []
    accumulated_rules = []
    
    for index, pid in enumerate(train_ids, 1):
        logger.info(f"\n--- Training Step {index}: Processing Patient {pid} ---")
        
        patient_data = get_synthetic_patient_data(pid)
        initial_state = {
            "parsed_pages": patient_data["pages"],
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
        
        final_state = workflow.invoke(initial_state)
        draft = final_state.get("output_markdown", "")
        
        # Simulated edit
        edited = reviewer.review_draft(draft)
        
        # Calculate metrics before training update
        similarity = calculate_normalized_similarity(draft, edited)
        compliance = check_policy_compliance(draft, pid)
        
        # Learn rules from edits
        step_rules = learner.extract_rules(draft, edited)
        accumulated_rules.extend(step_rules)
        
        # Generalize and save preferences
        generalized = learner.generalize_rules(accumulated_rules)
        save_correction_memory(generalized)
        
        training_steps.append({
            "step": index,
            "patient_id": pid,
            "similarity": similarity,
            "compliance": compliance,
            "rules_count": len(generalized)
        })
        logger.info(f"Step {index} - Patient {pid}: Similarity = {similarity:.4f}, Compliance = {compliance['overall_compliance']:.4f}")
        
    # 5. Final Evaluation on Test Set (using learned preferences)
    logger.info("\n[Eval] Running FINAL evaluation on test cases (Patients 5 & 6) with accumulated preferences...")
    final_results = []
    
    for pid in test_ids:
        patient_data = get_synthetic_patient_data(pid)
        initial_state = {
            "parsed_pages": patient_data["pages"],
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
        
        final_state = workflow.invoke(initial_state)
        draft = final_state.get("output_markdown", "")
        
        # simulated correction
        edited = reviewer.review_draft(draft)
        
        # calculate metrics
        similarity = calculate_normalized_similarity(draft, edited)
        compliance = check_policy_compliance(draft, pid)
        
        final_results.append({
            "patient_id": pid,
            "similarity": similarity,
            "compliance": compliance
        })
        logger.info(f"Final - Patient {pid}: Similarity = {similarity:.4f}, Compliance = {compliance['overall_compliance']:.4f}")
        
    # Write evaluation logs
    eval_output = {
        "baseline_test": baseline_results,
        "training_steps": training_steps,
        "final_test": final_results,
        "learned_rules": load_correction_memory()
    }
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
        
    logger.info("\n" + "=" * 50)
    logger.info("               EVALUATION RESULTS REPORT               ")
    logger.info("=" * 50)
    logger.info("Test Cases Comparison:")
    for b, f in zip(baseline_results, final_results):
        pid = b["patient_id"]
        logger.info(f"Patient {pid}:")
        logger.info(f"  Similarity: {b['similarity']:.4f} -> {f['similarity']:.4f} "
                    f"(Improvement: {f['similarity'] - b['similarity']:+.4f})")
        logger.info(f"  Compliance: {b['compliance']['overall_compliance']:.4f} -> {f['compliance']['overall_compliance']:.4f} "
                    f"(Improvement: {f['compliance']['overall_compliance'] - b['compliance']['overall_compliance']:+.4f})")
    logger.info("=" * 50)
    
if __name__ == "__main__":
    run_evaluation_flow()
