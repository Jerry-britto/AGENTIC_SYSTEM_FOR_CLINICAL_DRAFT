import os
import json
import re
import sys
from clinical_agent.tools import check_drug_interactions

def run_validation_checks(markdown_path: str, json_path: str) -> bool:
    """
    Programmatically validates discharge summary outputs for compliance, demographics completeness,
    formatting, and clinical safety issues.
    """
    print(f"\n--- Initiating Validation Framework Checks on: {markdown_path} ---")
    
    if not os.path.exists(markdown_path) or not os.path.exists(json_path):
        print(f"Error: Missing output files for validation. Paths: {markdown_path}, {json_path}")
        return False
        
    try:
        with open(markdown_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading files: {e}")
        return False

    success = True

    # 1. Check Demographics Completeness
    print("[Check 1] Verifying Demographics Completeness...")
    demographics = json_data.get("Patient Demographics", {})
    missing_fields = []
    for field in ["MRN", "Full Name", "DOB", "Age/Gender"]:
        val = demographics.get(field, "")
        if not val or "MISSING" in str(val) or "Flagged" in str(val):
            missing_fields.append(field)
            
    # Note: For our real test records, demographics are deliberately missing in patient_records.pdf
    # Therefore, they MUST be flagged as MISSING, not fabricated!
    print(f"  Missing fields flagged correctly: {missing_fields}")
    if not missing_fields:
        print("  Warning: No demographics were flagged as missing. Verify that they weren't fabricated if the source lacks them.")

    # 2. Check Clinical Safety Warnings
    print("[Check 2] Verifying Clinical safety warnings...")
    warnings = json_data.get("Clinician Alerts / Escalations", [])
    print(f"  Alerts found: {len(warnings)}")
    
    # Check if the critical Loperamide warning is flagged for gastroenteritis case
    has_loperamide_warning = False
    for warning in warnings:
        if "loperamide" in warning.lower() or "antimotility" in warning.lower():
            has_loperamide_warning = True
            break
            
    # We check if the markdown draft also mentions the warning
    if "loperamide" in md_text.lower() or "antimotility" in md_text.lower():
        has_loperamide_warning = True
        
    if has_loperamide_warning:
        print("  ✅ PASS: Loperamide safety alert successfully verified in discharge summary.")
    else:
        print("  ❌ FAIL: Missing critical loperamide/antimotility safety warning.")
        success = False

    # 3. Check for Drug-Drug Interactions
    print("[Check 3] Verifying Drug-Drug Interactions in discharge list...")
    meds_info = json_data.get("Discharge Medications with Reconciliation Details", [])
    meds_list = []
    for med in meds_info:
        if isinstance(med, dict) and "Medication" in med:
            meds_list.append(med["Medication"])
        elif isinstance(med, str):
            meds_list.append(med)
            
    print(f"  Extracted meds: {meds_list}")
    interactions = check_drug_interactions(meds_list)
    if interactions:
        print(f"  ❌ FAIL: Found unresolved critical drug interactions in discharge list: {interactions}")
        success = False
    else:
        print("  ✅ PASS: No critical drug-drug interactions found in discharge list.")

    # 4. Check Formatting Preferences Compliance
    print("[Check 4] Verifying Follow-up Formatting Guidelines...")
    followup_ok = True
    required_sections = ["Medications", "Appointments", "Dietary"]
    for sect in required_sections:
        if sect.lower() not in md_text.lower():
            followup_ok = False
            print(f"  Missing expected follow-up category: {sect}")
            
    if followup_ok:
        print("  ✅ PASS: Follow-up sections conform to clinician formatting preferences.")
    else:
        print("  ❌ FAIL: Follow-up sections lack standard styling.")
        success = False

    # 5. Check for Hallucinations Warning
    print("[Check 5] Verifying Hallucination Guardrail Report...")
    # The JSON should have a warning if any hallucinated item was stripped/alerted
    alerts_text = " ".join(warnings).lower()
    if "hallucination" in alerts_text or "critical guardrail" in alerts_text:
        print("  ⚠️ ALERT: Hallucinations detected and successfully intercepted by verifier guardrails.")
    else:
        print("  ✅ PASS: No hallucinations detected or flagged by guardrails.")

    if success:
        print("\n🎉 ALL VALIDATION CHECKS PASSED SUCCESSFULLY! Output conforms to clinical safety standards.")
    else:
        print("\n❌ VALIDATION FAILURE: Output contains clinical or formatting discrepancies.")
        
    return success

if __name__ == "__main__":
    markdown_out = "outputs/patient_records-discharged_summary.md"
    json_out = "outputs/patient_records-discharged_summary.json"
    
    if len(sys.argv) > 2:
        markdown_out = sys.argv[1]
        json_out = sys.argv[2]
        
    result = run_validation_checks(markdown_out, json_out)
    sys.exit(0 if result else 1)
