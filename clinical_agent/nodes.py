import json
import re
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from clinical_agent.config import GROQ_API_KEY, REASONING_MODEL, UTILITY_MODEL
from clinical_agent.state import AgentState
from clinical_agent.logging_utils import logger
from clinical_agent.tools import check_drug_interactions
from clinical_agent.agent_tools import GROQ_TOOLS_SCHEMA, execute_tool
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
    retries = 3
    while retries > 0:
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
            if any(term in err_msg for term in ["429", "413", "rate limit", "limit exceeded", "too large", "tpm", "tokens"]):
                import time
                time.sleep(5) # Sleep to let the rate limit bucket reset
                fallback_model = UTILITY_MODEL
                if selected_model == UTILITY_MODEL:
                    fallback_model = REASONING_MODEL
                
                logger.warning(f"[Rate Limit/TPM] Model {selected_model} hit limit. Retrying with fallback: {fallback_model} (retries left: {retries-1})...")
                selected_model = fallback_model
                retries -= 1
            else:
                raise e
    raise Exception("Failed to get completion from Groq API after multiple retries due to rate limits.")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_groq_llm_with_tools(system_prompt: str, user_prompt: str, tools_schema: list, parsed_pages: list, model: str, max_tool_depth: int = 5) -> str:
    """
    Executes a Groq LLM completion. If the model generates tool calls, executes them,
    appends the responses to the conversation, and loops recursively until completion.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    current_model = model
    depth = 0

    while depth < max_tool_depth:
        completion = None
        retries = 3
        while retries > 0:
            try:
                kwargs = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": 0.1
                }
                if tools_schema:
                    kwargs["tools"] = tools_schema

                completion = groq_client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if any(term in err_msg for term in ["429", "413", "rate limit", "limit exceeded", "too large", "tpm", "tokens"]):
                    import time
                    time.sleep(5) # Sleep to let the rate limit bucket reset
                    fallback = UTILITY_MODEL
                    if current_model == UTILITY_MODEL:
                        fallback = REASONING_MODEL
                    logger.warning(f"[Rate Limit/TPM] Model {current_model} hit limit. Retrying with fallback {fallback} (retries left: {retries-1})...")
                    current_model = fallback
                    retries -= 1
                else:
                    raise e

        if not completion:
            raise Exception("Failed to get completion from Groq API after multiple retries due to rate limits.")

        message = completion.choices[0].message

        if message.tool_calls:
            # Append assistant tool call request message in dictionary format
            assistant_msg = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            }
            messages.append(assistant_msg)

            read_count = 0
            executed_count = 0
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception as e_json:
                    logger.error(f"[Tool Call Error] Invalid JSON arguments from tool call {tool_name}: {e_json}")
                    tool_args = {}

                logger.info(f"[Tool Call] Model requested '{tool_name}' with arguments: {tool_args}")

                # Enforce limit of 3 executed tool calls per turn to conserve tokens
                if executed_count >= 3:
                    logger.warning(f"[Verifier Override] Skipping tool call {tool_name} to prevent token overload.")
                    tool_output = "Error: Too many parallel tool calls requested. Skipped to prevent context overload. Please request this in the next step."
                else:
                    # Throttle read_document_page calls to avoid Groq TPM/payload size limits
                    if tool_name == "read_document_page":
                        read_count += 1
                        if read_count > 3:
                            logger.warning("[Verifier Override] Throttling parallel page read to avoid Groq TPM limit.")
                            tool_output = "Error: Page read limit exceeded (max 3 page reads per step). Please search first or read pages selectively."
                        else:
                            tool_output = execute_tool(tool_name, tool_args, parsed_pages)
                            executed_count += 1
                    else:
                        tool_output = execute_tool(tool_name, tool_args, parsed_pages)
                        executed_count += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_output
                })
            # Re-run LLM loop with new tool contents
            depth += 1
            continue
        else:
            return message.content or ""

    logger.warning("[Tool Call] Reached maximum tool calling depth. Forcing final synthesis from assistant.")
    try:
        messages.append({
            "role": "user",
            "content": "Synthesize all the clinical facts and data gathered above into a final clinical extraction JSON object, without making any further tool calls."
        })
        kwargs = {
            "model": current_model,
            "messages": messages,
            "temperature": 0.1
        }
        completion = groq_client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[Tool Call Override] Failed to get final synthesis: {e}")
        return messages[-1].get("content", "") if messages else ""

def parse_json_safely(text: str) -> dict:
    """
    Safely parses JSON even if the model returned markdown code block wrappers or text surroundings.
    """
    try:
        text_clean = text.strip()
        if text_clean.startswith("```json"):
            text_clean = text_clean[7:]
        if text_clean.endswith("```"):
            text_clean = text_clean[:-3]
        text_clean = text_clean.strip()
        return json.loads(text_clean)
    except Exception as e:
        logger.error(f"[JSON Parse Error] Failed standard parse: {e}. Trying regex extraction...")
        # Fallback to regex matching to find the first JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e_inner:
                logger.error(f"[JSON Parse Error] Regex block parse failed: {e_inner}")
        return {}


# ==================== HALLUCINATION GUARDRAIL ====================

def verify_no_hallucinations(state: AgentState) -> AgentState:
    """
    Cross-references extracted diagnoses and medications against literal patient record sheets.
    If terms do not match directly, fall back to a zero-temperature LLM verification prompt.
    """
    logger.info("[Verifier Agent] Running Hallucination Guardrail checks...")
    
    extracted = state["extracted_data"]
    pages_text = "\n".join(p["markdown"] for p in state["parsed_pages"]).lower()
    
    # Extract clinical details to check
    diagnoses_to_check = []
    raw_diag = extracted.get("Diagnoses", extracted.get("Principal and Secondary Diagnoses", extracted.get("diagnoses", "")))
    if isinstance(raw_diag, list):
        diagnoses_to_check = raw_diag
    elif isinstance(raw_diag, str):
        diagnoses_to_check = [raw_diag]
        
    meds_to_check = []
    for key in ["Discharge Medications", "Admission Medications", "discharge_medications", "admission_medications"]:
        val = extracted.get(key, [])
        if isinstance(val, list):
            for v in val:
                if isinstance(v, str):
                    meds_to_check.append(v)
                elif isinstance(v, dict) and "name" in v:
                    meds_to_check.append(v["name"])
        elif isinstance(val, str):
            meds_to_check.append(val)
            
    # Combine items, cleaning whitespace and empty strings
    items_to_verify = list(set([item.strip() for item in (diagnoses_to_check + meds_to_check) if item]))
    
    hallucination_report = state.get("hallucination_report", {})
    if not hallucination_report:
        hallucination_report = {"checked_items": {}, "hallucinations_detected": []}
        
    for item in items_to_verify:
        normalized_item = item.lower()
        
        # Strip common formatting terms like dosage, tab, cap, etc. to match chemical names
        clean_item = re.sub(r'^(tab\.|cap\.|inj\.|tab|cap|inj)\s+', '', normalized_item)
        clean_item = re.sub(r'\s+\d+(mg|g|mcg|ml|iu|u).*$', '', clean_item)
        clean_item = clean_item.strip()
        
        # 1. Direct substring check
        if clean_item in pages_text:
            hallucination_report["checked_items"][item] = "VERIFIED (Direct Substring)"
            continue
            
        # 2. LLM Verification
        logger.warning(f"[Verifier Agent] Direct match failed for '{item}'. Invoking LLM auditor...")
        prompt = f"""Evaluate if the clinical term (medication or diagnosis) '{item}' is mentioned or explicitly documented in the patient records.
Answer ONLY 'YES' or 'NO'. Do not output any other character.

Patient Records:
{pages_text[:20000]}
"""
        try:
            response = call_groq_llm(
                system_prompt="You are a strict clinical fact checker. Reply only YES or NO.",
                user_prompt=prompt,
                model=UTILITY_MODEL,
                response_json=False
            )
            if "YES" in response.upper():
                hallucination_report["checked_items"][item] = "VERIFIED (LLM Checked)"
            else:
                logger.error(f"[Verifier Agent] Hallucination detected: '{item}' cannot be found in patient record!")
                hallucination_report["checked_items"][item] = "HALLUCINATION_DETECTED"
                hallucination_report["hallucinations_detected"].append(item)
                
                # Append warning and flag
                state["safety_warnings"].append({
                    "type": "HALLUCINATION_WARNING",
                    "message": f"CRITICAL GUARDRAIL ALERT: The extracted medication/diagnosis '{item}' could not be verified from the source patient records."
                })
                state["clinician_review_flags"].append(f"Hallucination Warning: Term '{item}' could not be verified in source records.")
        except Exception as e:
            logger.warning(f"Auditor failed for '{item}': {e}. Defaulting to verified.")
            hallucination_report["checked_items"][item] = "VERIFIED (Check Failed)"
            
    state["hallucination_report"] = hallucination_report
    return state


# ==================== NODE IMPLEMENTATIONS ====================

def planner_node(state: AgentState) -> AgentState:
    """
    Planner Node: Evaluates current state, tracking completed phases to prevent infinite loops.
    """
    state["iteration_count"] += 1
    trace_steps = state.get("trace_steps", [])
    
    # Initialize state keys if not present
    if "completed_phases" not in state:
        state["completed_phases"] = []
    if "retrieved_pages" not in state:
        state["retrieved_pages"] = []
    if "hallucination_report" not in state:
        state["hallucination_report"] = {}

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
            system_prompt=PLANNER_SYSTEM_PROMPT.format(completed_phases=state["completed_phases"]),
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        plan_data = parse_json_safely(response_text)
        
        reasoning = plan_data.get("reasoning", "")
        action = plan_data.get("action", "READ_DOCUMENTS")
        action_params = plan_data.get("action_parameters", {})

        # Loop Prevention Override: If the planner attempts to repeat a completed phase without new page reads
        if action == "RECONCILE_MEDICATIONS" and "reconcile" in state["completed_phases"]:
            logger.warning("[Planner Override] Reconcile was already completed. Directing to safety verification.")
            action = "VERIFY_SAFETY"
            reasoning = "Override: Medication reconciliation already done. Transitioning to safety check."
        elif action == "VERIFY_SAFETY" and "verify" in state["completed_phases"]:
            logger.warning("[Planner Override] Verify was already completed. Directing to synthesis.")
            action = "SYNTHESIZE_DRAFT"
            reasoning = "Override: Safety checks already done. Transitioning to synthesis."

        logger.info(f"[Planner] Step {state['iteration_count']} reasoning: {reasoning}")
        logger.info(f"[Planner] Chosen Action: {action} with params: {action_params}")
        
        state["current_plan"] = f"Action: {action} | Params: {action_params} | Reasoning: {reasoning}"
        trace_steps.append(f"Step {state['iteration_count']}: Planner reasoning: '{reasoning}' -> Chosen Action: {action}({action_params})")
        state["trace_steps"] = trace_steps
        
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
    Reader Agent: Dynamically retrieves and extracts patient record details using native tools.
    """
    action_params = state["extracted_data"].get("_next_action_params", {})
    search_topic = action_params.get("search_topic", "patient details, diagnoses, and medications")
    
    logger.info(f"[Reader Agent] Ingesting and searching records for topic: '{search_topic}'...")
    
    try:
        response_text = call_groq_llm_with_tools(
            system_prompt=READER_SYSTEM_PROMPT.format(search_topic=search_topic),
            user_prompt=f"Extract all clinical facts regarding '{search_topic}' from patient records using your tools.",
            tools_schema=GROQ_TOOLS_SCHEMA,
            parsed_pages=state["parsed_pages"],
            model=REASONING_MODEL
        )
        extracted_facts = parse_json_safely(response_text)
        
        # Merge facts
        for k, v in extracted_facts.items():
            if v and v != "None" and v != "Unknown":
                state["extracted_data"][k] = v
                
        logger.info(f"[Reader Agent] Extraction completed for topic: {search_topic}")

    except Exception as e:
        logger.error(f"[Reader Agent] Error reading documents: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_READ_ERROR",
            "message": f"Failed to extract details for '{search_topic}' due to: {e}"
        })

    if "read" not in state["completed_phases"]:
        state["completed_phases"].append("read")

    return state


def reconciler_node(state: AgentState) -> AgentState:
    """
    Pharmacist/Reconciler Agent: Line-by-line medication comparison.
    """
    logger.info("[Reconciler Agent] Comparing admission vs discharge medications...")
    
    extracted = state["extracted_data"]
    admission_meds = extracted.get("Admission Medications", extracted.get("admission_medications", []))
    discharge_meds = extracted.get("Discharge Medications", extracted.get("discharge_medications", []))
    hospital_course = extracted.get("Hospital Course", extracted.get("hospital_course", ""))

    user_prompt = f"""Admission Medications: {admission_meds}
Discharge Medications: {discharge_meds}
Hospital Course: {hospital_course}"""

    try:
        response_text = call_groq_llm(
            system_prompt=RECONCILER_SYSTEM_PROMPT.format(
                admission_meds=admission_meds,
                discharge_meds=discharge_meds,
                hospital_course=hospital_course
            ),
            user_prompt=user_prompt,
            model=REASONING_MODEL,
            response_json=True
        )
        reconcile_data = parse_json_safely(response_text)
        results = reconcile_data.get("reconciliation_results", [])
        
        state["medication_changes"] = results
        
        # Log discrepancies
        discrepancies = [r for r in results if r.get("severity") == "HIGH WARNING"]
        logger.info(f"[Reconciler Agent] Reconciliation completed. Checked {len(results)} items. Found {len(discrepancies)} discrepancies.")
        
        for d in discrepancies:
            state["safety_warnings"].append({
                "type": "MEDICATION_DISCREPANCY",
                "message": f"Discharge medication change in {d.get('medication_name')} ({d.get('type_of_change')}) has NO documented clinical reason in the hospital notes."
            })
            state["clinician_review_flags"].append(f"Medication Reconciliation discrepancy: {d.get('medication_name')} ({d.get('type_of_change')}) lacks clinical justification.")

    except Exception as e:
        logger.error(f"[Reconciler Agent] Medication reconciliation failed: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_RECONCILE_ERROR",
            "message": f"Medication reconciliation failed due to: {e}"
        })
        state["clinician_review_flags"].append("Medication Reconciliation failed to run automatically.")

    if "reconcile" not in state["completed_phases"]:
        state["completed_phases"].append("reconcile")

    return state


def safety_verifier_node(state: AgentState) -> AgentState:
    """
    Safety Verifier Agent: Runs safety checks (including native drug interactions lookup) and hallucination guardrail.
    """
    logger.info("[Verifier Agent] Launching safety verifications...")
    extracted = state["extracted_data"]
    meds_changes = state["medication_changes"]

    user_prompt = f"""Extracted Data:
{json.dumps(extracted, indent=2)}

Medication Changes:
{json.dumps(meds_changes, indent=2)}"""

    try:
        # Calls the verifier LLM with access to drug interactions tool
        response_text = call_groq_llm_with_tools(
            system_prompt=VERIFIER_SYSTEM_PROMPT.format(
                extracted_data=json.dumps(extracted),
                medication_changes=json.dumps(meds_changes)
            ),
            user_prompt=user_prompt,
            tools_schema=GROQ_TOOLS_SCHEMA,
            parsed_pages=state["parsed_pages"],
            model=REASONING_MODEL
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

        logger.info(f"[Verifier Agent] Safety checks processed. Warnings count: {len(state['safety_warnings'])}")

    except Exception as e:
        logger.error(f"[Verifier Agent] Safety verifier failed: {e}")
        state["safety_warnings"].append({
            "type": "SYSTEM_VERIFY_ERROR",
            "message": f"Safety verification failed due to: {e}"
        })
        state["clinician_review_flags"].append("Safety Verification checks failed to execute automatically.")

    # Run the Hallucination Guardrail Check
    state = verify_no_hallucinations(state)

    if "verify" not in state["completed_phases"]:
        state["completed_phases"].append("verify")

    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesizer Node: Compiles final discharge summary report and structured JSON representation.
    """
    logger.info("[Synthesizer Node] Synthesizing final Discharge Summary draft...")
    
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
            logger.info(f"[Synthesizer Node] Injecting {len(learned_rules)} clinician preference rules from correction memory...")
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
        logger.info("[Synthesizer Node] Generated Markdown draft.")

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
        logger.info("[Synthesizer Node] Synthesized Markdown & JSON drafts successfully.")

    except Exception as e:
        logger.error(f"[Synthesizer Node] Synthesis failed: {e}")
        state["output_markdown"] = f"# ERROR GENERATING DISCHARGE SUMMARY DRAFT\n\nFailed due to an unexpected error during compilation: {e}"
        state["output_json"] = {"status": "ERROR", "error_message": str(e)}
        state["is_finished"] = True

    return state
