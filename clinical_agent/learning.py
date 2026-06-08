import os
import json
from clinical_agent.nodes import call_groq_llm, REASONING_MODEL, parse_json_safely
from clinical_agent.logging_utils import logger

# Path for persisting correction memory
CORRECTION_MEMORY_PATH = "cache/correction_memory.json"

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the standard Levenshtein distance between two strings.
    Memory optimized to O(min(M, N)) space.
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_normalized_similarity(s1: str, s2: str) -> float:
    """
    Calculates the normalized similarity: 1.0 - (Levenshtein distance / max_len).
    Returns 1.0 if both strings are empty.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    distance = calculate_levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


# ==================== SIMULATED CLINICAL REVIEWER ====================

SIMULATED_DOCTOR_SYSTEM_PROMPT = """You are a senior, highly meticulous clinical editor (a stand-in "doctor"). 
Your role is to edit the provided clinical discharge summary draft to strictly conform to your preference policy.

Your editing policy is consistent and strict:
1. Terminology: Always replace brand-name medications with their exact generic equivalents.
   - e.g. Replace "Lasix" with "furosemide"
   - e.g. Replace "Ventolin" with "albuterol"
   - e.g. Replace "Lipitor" with "atorvastatin"
   - e.g. Replace "Augmentin" or "Klavox" with "amoxicillin/clavulanic acid"
   - e.g. Replace "Nexium" with "esomeprazole"
   - e.g. Replace "Solu-Medrol" with "methylprednisolone"
2. Lab Results Qualification: Always add status qualifiers in parentheses next to numerical lab values mentioned.
   - e.g. "Creatinine: 1.8 mg/dL" -> "Creatinine: 1.8 mg/dL (Elevated)"
   - e.g. "Hemoglobin: 14 g/dL" -> "Hemoglobin: 14 g/dL (Normal)"
   - e.g. "Sodium: 131 mEq/L" -> "Sodium: 131 mEq/L (Low)"
   - e.g. "Potassium: 5.8 mEq/L" -> "Potassium: 5.8 mEq/L (Elevated)"
3. Follow-up Formatting: Reformat the 'Follow-Up Instructions' section to strictly use a bulleted list with exactly three subheadings:
   - **Medications**
   - **Appointments**
   - **Dietary & Activity Restrictions**

CRITICAL DIRECTIVE: Do not add or remove other clinical information. Keep all other text (names, dates, course) unchanged. Make only these style and terminology modifications.
Output ONLY the corrected Markdown draft directly. Do not write any explanations or chat. Start directly with the discharge summary heading.
"""

class SimulatedReviewer:
    def review_draft(self, draft_markdown: str) -> str:
        """
        Submits the agent's draft to the simulated doctor LLM to apply the hidden editing policy.
        """
        logger.info("[Simulated Reviewer] Applying consistent clinician editing policy to draft...")
        try:
            edited_draft = call_groq_llm(
                system_prompt=SIMULATED_DOCTOR_SYSTEM_PROMPT,
                user_prompt=f"Draft summary to edit:\n\n{draft_markdown}",
                model=REASONING_MODEL,
                response_json=False
            )
            return edited_draft.strip()
        except Exception as e:
            logger.error(f"[Simulated Reviewer] Failed to generate edits: {e}")
            # Fallback to returning the draft itself on failure
            return draft_markdown


# ==================== PREFERENCE LEARNER ====================

PREFERENCE_LEARNER_SYSTEM_PROMPT = """You are a Clinical Preference Rule Extractor. Your task is to analyze an Agent's Draft and a Doctor's Corrected version to extract the doctor's editing rules.
Compare the two documents and identify changes in:
1. Terminology (e.g. brand name to generic name substitutions)
2. Formatting patterns (e.g. follow-up list headings or structures)
3. Laboratory value presentation (e.g. adding qualifiers like 'Normal' or 'Elevated' next to numerical metrics)

Provide your findings as a JSON object with a single key:
- "rules": A list of short, clear, actionable instruction strings summarizing the extracted preferences (e.g., ["Prefer generic furosemide over brand Lasix", "Include qualifiers like (Elevated) or (Normal) next to lab test values", "Structure follow-up instructions into three sections: Medications, Appointments, and Dietary & Activity Restrictions"]).

Ensure you output ONLY the valid JSON object.
"""

class PreferenceLearner:
    def extract_rules(self, draft_markdown: str, edited_markdown: str) -> list:
        """
        Compares draft and edited versions to extract doctor preference rules.
        """
        logger.info("[Preference Learner] Analyzing edits to extract clinician preferences...")
        user_prompt = f"Agent Draft:\n{draft_markdown}\n\nDoctor's Corrected Version:\n{edited_markdown}"
        try:
            response = call_groq_llm(
                system_prompt=PREFERENCE_LEARNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=REASONING_MODEL,
                response_json=True
            )
            rules_data = parse_json_safely(response)
            rules = rules_data.get("rules", [])
            logger.info(f"[Preference Learner] Extracted {len(rules)} preference rules.")
            return rules
        except Exception as e:
            logger.error(f"[Preference Learner] Failed to extract rules: {e}")
            return []

    def generalize_rules(self, all_rules: list) -> list:
        """
        Aggregates and deduplicates rules, cleaning them into a coherent list.
        """
        if not all_rules:
            return []
        
        # We ask the LLM to deduplicate and standardize the rules
        generalizer_prompt = """You are a Clinical Guidelines Standardizer. 
Given a raw list of clinician preference rules, consolidate, deduplicate, and standardize them into a clean, concise list of rules.
Ensure they are actionable for a text generator.

Provide your output as a JSON object with a single key:
- "rules": A list of standardized preference strings.
"""
        user_prompt = f"Raw rules to consolidate:\n{json.dumps(all_rules, indent=2)}"
        try:
            response = call_groq_llm(
                system_prompt=generalizer_prompt,
                user_prompt=user_prompt,
                model=REASONING_MODEL,
                response_json=True
            )
            data = parse_json_safely(response)
            return data.get("rules", [])
        except Exception as e:
            logger.error(f"[Preference Learner] Failed to generalize rules: {e}")
            return list(set(all_rules)) # Simple Python fallback deduplication


# ==================== CORRECTION MEMORY MANAGEMENT ====================

def load_correction_memory() -> list:
    """
    Loads doctor preference rules from cache.
    """
    if not os.path.exists(CORRECTION_MEMORY_PATH):
        return []
    try:
        with open(CORRECTION_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rules", [])
    except Exception as e:
        logger.error(f"[Memory] Failed to load correction memory: {e}")
        return []

def save_correction_memory(rules: list):
    """
    Saves doctor preference rules to cache.
    """
    os.makedirs(os.path.dirname(CORRECTION_MEMORY_PATH), exist_ok=True)
    try:
        with open(CORRECTION_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump({"rules": rules}, f, indent=2)
        logger.info(f"[Memory] Saved {len(rules)} rules to correction memory cache.")
    except Exception as e:
        logger.error(f"[Memory] Failed to save correction memory: {e}")

def clear_correction_memory():
    """
    Clears the stored correction memory.
    """
    if os.path.exists(CORRECTION_MEMORY_PATH):
        try:
            os.remove(CORRECTION_MEMORY_PATH)
            logger.info("[Memory] Cleared correction memory cache.")
        except Exception as e:
            logger.error(f"[Memory] Failed to clear correction memory: {e}")
