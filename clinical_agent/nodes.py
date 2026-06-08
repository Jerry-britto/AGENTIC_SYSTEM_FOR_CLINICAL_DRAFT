import json
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from clinical_agent.config import GROQ_API_KEY, REASONING_MODEL, UTILITY_MODEL
from clinical_agent.state import AgentState
from clinical_agent.logging_utils import logger
from clinical_agent.tools import check_drug_interactions
from clinical_agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    READER_SYSTEM_PROMPT,
    RECONCILER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    SYNTHESIZER_MARKDOWN_PROMPT,
    SYNTHESIZER_JSON_PROMPT
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_groq_llm(system_prompt: str, user_prompt: str, model: str, response_json: bool = False) -> str:
    """
    Calls the Groq LLM with retries on failure. Automatically falls back to a utility/smaller model if 429 occurs.
    """
    selected_model = model
    try:
        kwargs = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
            
        completion = groq_client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "rate limit" in err_msg or "limit exceeded" in err_msg:
            fallback_model = UTILITY_MODEL
            if selected_model == UTILITY_MODEL:
                fallback_model = "gemma2-9b-it"
            
            logger.warning(f"[Rate Limit] Model {selected_model} hit rate limit. Retrying with fallback: {fallback_model}...")
            kwargs = {
                "model": fallback_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            }
            if response_json:
                kwargs["response_format"] = {"type": "json_object"}
                
            completion = groq_client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        else:
            raise e

def parse_json_safely(text: str) -> dict:
    """
    Safely parses JSON even if the model returned markdown code block wrappers.
    """
    text_clean = text.strip()
    # Strip markdown wrappers if present
    if text_clean.startswith("```json"):
        text_clean = text_clean[7:]
    if text_clean.endswith("```"):
        text_clean = text_clean[:-3]
    text_clean = text_clean.strip()
    return json.loads(text_clean)

# ==================== NODE IMPLEMENTATIONS ====================

def planner_node(state: AgentState) -> AgentState:
    """
    Planner Node: Evaluates current state and decides the next action.
    """
    state["iteration_count"] += 1
    trace_steps = state.get("trace_steps", [])
    
    # Cap at 8 iterations
    if state["iteration_count"] > 8:
        logger.warning("[Planner] Hard cap of 8 iterations reached. Forcing synthesis...")
        state["current_plan"] = "Hard cap reached. Forcing SYNTHESIZE_DRAFT."
        trace_steps.append(f"Step {state['iteration_count']}: Planner forced SYNTHESIZE_DRAFT (cap of 8 reached).")
        state["trace_steps"] = trace_steps
        state["is_finished"] = True
        return state

    user_prompt = f"""Current iteration: {state['iteration_count']}
Extracted Data summary: {list(state['extracted_data'].keys())}
Medication Changes: {len(state['medication_changes'])} logged.
Safety Warnings: {len(state['safety_warnings'])} logged.
Current Plan Checklist: {state['current_plan']}"""

    try:
        response_text = call_groq_llm(
            system_prompt=PLANNER_SYSTEM_PROMPT.format(iteration_count=state["iteration_count"]),
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        plan_data = parse_json_safely(response_text)
        
        reasoning = plan_data.get("reasoning", "")
        action = plan_data.get("action", "READ_DOCUMENTS")
        action_params = plan_data.get("action_parameters", {})

        logger.info(f"[Planner] Step {state['iteration_count']} reasoning: {reasoning}")
        logger.info(f"[Planner] Chosen Action: {action} with params: {action_params}")
        
        state["current_plan"] = f"Action: {action} | Params: {action_params} | Reasoning: {reasoning}"
        trace_steps.append(f"Step {state['iteration_count']}: Planner reasoning: '{reasoning}' -> Chosen Action: {action}({action_params})")
        state["trace_steps"] = trace_steps
        
        # We store the selected action and params in state metadata for routing
        state["extracted_data"]["_next_action"] = action
        state["extracted_data"]["_next_action_params"] = action_params

    except Exception as e:
        logger.error(f"[Planner] Error during planning: {e}. Falling back to default extraction.")
        state["extracted_data"]["_next_action"] = "READ_DOCUMENTS"
        state["extracted_data"]["_next_action_params"] = {"search_topic": "all patient details"}
        trace_steps.append(f"Step {state['iteration_count']}: Planner failed: {e}. Fallback to READ_DOCUMENTS.")
        state["trace_steps"] = trace_steps

    return state


def reader_node(state: AgentState) -> AgentState:
    """
    Reader Node: Ingests specific pages or searches topics from raw patient documents.
    """
    action_params = state["extracted_data"].get("_next_action_params", {})
    search_topic = action_params.get("search_topic", "patient details, diagnoses, and medications")
    
    # We combine all parsed pages to provide a focused context
    # If the planner specified certain pages, we read those. Otherwise, we read everything.
    pages_to_read = action_params.get("pages", [])
    
    # Ensure pages_to_read is a list of integers
    use_specified_pages = False
    parsed_pages_list = []
    if isinstance(pages_to_read, list) and len(pages_to_read) > 0:
        try:
            parsed_pages_list = [int(p) for p in pages_to_read]
            use_specified_pages = True
        except (ValueError, TypeError):
            use_specified_pages = False

    doc_context = ""
    if use_specified_pages:
        logger.info(f"[Reader] Reading specified pages: {parsed_pages_list}")
        for page in state["parsed_pages"]:
            if page["page_number"] in parsed_pages_list:
                doc_context += f"\n--- Page {page['page_number']} ---\n{page['markdown']}\n"
    else:
        logger.info(f"[Reader] Dynamically indexing records for topic: '{search_topic}'...")
        # Smart dynamic selector to avoid 12k Groq TPM limit
        keywords = ["diagnosis", "medication", "discharge", "admission", "procedure", "allergies", "history", "course", "lab", "creatinine", "glucose", "insulin"]
        for word in search_topic.lower().split():
            if len(word) > 3:
                keywords.append(word)

        selected_pages = []
        for page in state["parsed_pages"]:
            page_num = page["page_number"]
            # Always include pages 1-4 (discharge summaries/admission records)
            if page_num <= 4:
                selected_pages.append(page)
                continue
            
            # Check for keyword matches in other pages
            text_lower = page["markdown"].lower()
            if any(kw in text_lower for kw in keywords):
                if len(page["markdown"].strip()) > 100:  # Avoid empty/minimal sheets
                    selected_pages.append(page)

        # Sort selected pages by page number
        selected_pages = sorted(selected_pages, key=lambda x: x["page_number"])
        
        # Cap selected pages to a maximum of 10 pages to respect TPM limit
        if len(selected_pages) > 10:
            logger.warning(f"[Reader] Filtered {len(selected_pages)} matching pages down to top 10 to fit within API limits.")
            # Select first 6 pages and last 4 pages (gives optimal temporal summary)
            selected_pages = selected_pages[:6] + selected_pages[-4:]

        logger.info(f"[Reader] Reading key pages: {[p['page_number'] for p in selected_pages]}")
        for page in selected_pages:
            doc_context += f"\n--- Page {page['page_number']} ---\n{page['markdown']}\n"

    # Enforce strict safety cap for Groq TPM (1 token ~ 4 characters, 8000 tokens ~ 32,000 chars)
    max_chars = 32000
    if len(doc_context) > max_chars:
        logger.warning(f"[Reader] Context length ({len(doc_context)} chars) exceeds safety limits. Slicing to {max_chars} chars.")
        doc_context = doc_context[:max_chars] + "\n...[TRUNCATED FOR API CONSERVATISM]..."

    user_prompt = f"Topic to search/extract: {search_topic}\n\nPatient Documents:\n{doc_context}"

    try:
        response_text = call_groq_llm(
            system_prompt=READER_SYSTEM_PROMPT.format(search_topic=search_topic),
            user_prompt=user_prompt,
            model=UTILITY_MODEL,
            response_json=True
        )
        extracted_facts = parse_json_safely(response_text)
        
        # Merge extracted facts into our state's extracted_data
        for k, v in extracted_facts.items():
            if v and v != "None" and v != "Unknown":
                state["extracted_data"][k] = v
                
        logger.info(f"[Reader] Successfully extracted details for topic: {search_topic}")

    except Exception as e:
        logger.error(f"[Reader] Error reading documents: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_READ_ERROR",
            "message": f"Failed to read/extract documents for topic: {search_topic} due to: {e}"
        })

    return state


def reconciler_node(state: AgentState) -> AgentState:
    """
    Medication Reconciler Node: Line-by-line comparison of admission vs discharge medications.
    """
    logger.info("[Reconciler] Running Medication Reconciliation...")
    
    # Identify lists of medications in extracted_data
    # We ask the LLM to pull them or standardise them
    extracted = state["extracted_data"]
    
    admission_meds = extracted.get("Admission Medications", extracted.get("admission_medications", []))
    discharge_meds = extracted.get("Discharge Medications", extracted.get("discharge_medications", []))
    hospital_course = extracted.get("Hospital Course", extracted.get("hospital_course", ""))

    user_prompt = f"""Admission Medications: {admission_meds}
Discharge Medications: {discharge_meds}
Hospital Course: {hospital_course}"""

    try:
        response_text = call_groq_llm(
            system_prompt=RECONCILER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        reconcile_data = parse_json_safely(response_text)
        results = reconcile_data.get("reconciliation_results", [])
        
        state["medication_changes"] = results
        
        # Count discrepancies
        discrepancies = [r for r in results if r.get("severity") == "HIGH WARNING"]
        logger.info(f"[Reconciler] Reconciliation done. {len(results)} items checked. {len(discrepancies)} discrepancies found.")
        
        for d in discrepancies:
            state["safety_warnings"].append({
                "type": "MEDICATION_DISCREPANCY",
                "message": f"Medication change in {d.get('medication_name')} ({d.get('type_of_change')}) has NO documented clinical reason in the hospital notes."
            })
            state["clinician_review_flags"].append(f"Medication Reconciliation: {d.get('medication_name')} ({d.get('type_of_change')}) has no documented reason.")

    except Exception as e:
        logger.error(f"[Reconciler] Reconciliation failed: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_RECONCILE_ERROR",
            "message": f"Medication reconciliation failed due to: {e}"
        })
        state["clinician_review_flags"].append("Medication Reconciliation failed to execute automatically.")

    return state


def safety_verifier_node(state: AgentState) -> AgentState:
    """
    Safety Verifier Node: Performs rigorous safety checks (no-fabrication, conflicts, drug interactions).
    """
    logger.info("[Verifier] Running Safety Verification checks...")
    
    extracted = state["extracted_data"]
    meds_changes = state["medication_changes"]

    # 1. Run local mock drug-drug interactions check
    # Gather list of medications
    discharge_meds_list = []
    # Try to extract discharge medications list
    raw_discharge = extracted.get("Discharge Medications", extracted.get("discharge_medications", []))
    if isinstance(raw_discharge, list):
        for item in raw_discharge:
            if isinstance(item, dict):
                discharge_meds_list.append(item.get("name", ""))
            else:
                discharge_meds_list.append(str(item))
    elif isinstance(raw_discharge, str):
        discharge_meds_list = [raw_discharge]

    # Run check
    local_interaction_alerts = check_drug_interactions(discharge_meds_list)
    for alert in local_interaction_alerts:
        state["safety_warnings"].append({
            "type": "DRUG_INTERACTION",
            "message": f"POTENTIAL {alert['severity']} DRUG-DRUG INTERACTION: {alert['medication_1']} + {alert['medication_2']}. Mechanism: {alert['mechanism']}. Action Required: {alert['action']}."
        })
        state["clinician_review_flags"].append(f"Critical Drug Interaction: {alert['medication_1']} + {alert['medication_2']} ({alert['severity']})")

    # 2. Run LLM clinical verification
    user_prompt = f"""Extracted Data:
{json.dumps(extracted, indent=2)}

Medication Changes:
{json.dumps(meds_changes, indent=2)}"""

    try:
        response_text = call_groq_llm(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        verifier_data = parse_json_safely(response_text)
        
        # Log missing fields
        for mf in verifier_data.get("missing_fields_flags", []):
            state["safety_warnings"].append({
                "type": "MISSING_DATA",
                "message": f"Required clinical field '{mf}' is missing from the raw patient records."
            })
            state["clinician_review_flags"].append(f"Missing clinical field: {mf}")
            
        # Log conflicts
        for conf in verifier_data.get("conflicts", []):
            state["safety_warnings"].append({
                "type": "CLINICAL_CONFLICT",
                "message": f"CONFLICTING INFORMATION: {conf}"
            })
            state["clinician_review_flags"].append(f"Conflicting information: {conf}")

        # Log clinical warnings
        for cw in verifier_data.get("clinical_warnings", []):
            if isinstance(cw, str):
                cw_msg = cw
                cw_type = "CLINICAL_WARNING"
            elif isinstance(cw, dict):
                cw_msg = cw.get("message", "")
                cw_type = cw.get("type", "CLINICAL_WARNING")
            else:
                continue

            # Check if this isn't already logged
            if not any(cw_msg in w.get("message", "") for w in state["safety_warnings"]):
                state["safety_warnings"].append({
                    "type": cw_type,
                    "message": cw_msg
                })
                state["clinician_review_flags"].append(f"Clinical Safety Concern: {cw_msg}")

        if verifier_data.get("escalation_required", False):
            for reason in verifier_data.get("escalation_reasons", []):
                if reason not in state["clinician_review_flags"]:
                    state["clinician_review_flags"].append(f"Safety Escalation: {reason}")

        logger.info(f"[Verifier] Safety verification completed. Total Warnings: {len(state['safety_warnings'])}")

    except Exception as e:
        logger.error(f"[Verifier] Safety verifier failed: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_VERIFY_ERROR",
            "message": f"Safety verification failed due to: {e}"
        })
        state["clinician_review_flags"].append("Safety Verification checks failed to execute automatically.")

    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesizer Node: Compiles final discharge summary report and structured JSON representation.
    """
    logger.info("[Synthesizer] Synthesizing final Discharge Summary draft...")
    
    extracted = state["extracted_data"]
    meds_changes = state["medication_changes"]
    warnings = state["safety_warnings"]
    review_flags = state["clinician_review_flags"]

    user_prompt = f"""Extracted Data:
{json.dumps(extracted, indent=2)}

Medication Changes:
{json.dumps(meds_changes, indent=2)}

Safety Warnings:
{json.dumps(warnings, indent=2)}

Clinician Review Flags:
{json.dumps(review_flags, indent=2)}"""

    try:
        # 1. Synthesize Markdown Draft (raw text)
        from clinical_agent.learning import load_correction_memory
        learned_rules = load_correction_memory()
        
        system_prompt = SYNTHESIZER_MARKDOWN_PROMPT
        if learned_rules:
            logger.info(f"[Synthesizer] Injecting {len(learned_rules)} clinician preference rules from correction memory...")
            preferences_text = "\n".join(f"- {rule}" for rule in learned_rules)
            system_prompt += f"\n\n### CRITICAL ADAPTATION DIRECTIVES - CLINICIAN PREFERENCES:\n" \
                             f"You MUST modify your output formatting, style, and terminology to comply strictly with these preferences:\n" \
                             f"{preferences_text}\n"

        markdown_draft = call_groq_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=False
        )
        state["output_markdown"] = markdown_draft
        logger.info("[Synthesizer] Generated Markdown draft.")

        # 2. Synthesize JSON Draft (structured object)
        json_user_prompt = f"Markdown Draft:\n{markdown_draft}\n\nClinical Facts:\n{user_prompt}"
        json_response = call_groq_llm(
            system_prompt=SYNTHESIZER_JSON_PROMPT,
            user_prompt=json_user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        json_draft = parse_json_safely(json_response)
        
        state["output_json"] = json_draft
        state["is_finished"] = True
        logger.info("[Synthesizer] Synthesized Markdown & JSON drafts successfully.")

    except Exception as e:
        logger.error(f"[Synthesizer] Synthesis failed: {e}")
        state["output_markdown"] = f"# ERROR GENERATING DISCHARGE SUMMARY DRAFT\n\nFailed due to an unexpected error during compilation: {e}"
        state["output_json"] = {"status": "ERROR", "error_message": str(e)}
        state["is_finished"] = True

    return state
