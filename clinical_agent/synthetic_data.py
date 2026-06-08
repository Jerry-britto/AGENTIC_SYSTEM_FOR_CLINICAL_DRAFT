import os
import json
from clinical_agent.logging_utils import logger

# Paths for storing synthetic cases
SYNTHETIC_DATA_DIR = "cache/synthetic"

def get_synthetic_patient_data(patient_id: int) -> dict:
    """
    Returns high-fidelity synthetic patient record matching the parser page structure.
    patient_id ranges from 1 to 6.
    """
    # Define clinical records programmatically to preserve rate limits and ensure reproducibility
    patients = {
        1: {
            "name": "David Miller",
            "age": 52,
            "gender": "Male",
            "mrn": "MRN-99827",
            "dob": "1974-04-12",
            "diagnoses": "Acute Gastroenteritis and Acute Kidney Injury",
            "labs": "Creatinine: 2.1 mg/dL. Potassium: 5.4 mEq/L. Sodium: 132 mEq/L.",
            "meds": "Klavox 625mg PO TID, Lasix 40mg PO daily, Nexium 40mg PO daily.",
            "course": "Patient admitted with severe vomiting and diarrhea. Renal function was impaired on admission with creatinine of 2.1 mg/dL. Rehydrated with IV fluids. Antibiotics started for suspected bacterial gastroenteritis.",
            "instructions": "Follow up with family physician in 5 days. Maintain hydration. Continue Nexium and Lasix daily."
        },
        2: {
            "name": "Sarah Jenkins",
            "age": 68,
            "gender": "Female",
            "mrn": "MRN-10492",
            "dob": "1958-09-21",
            "diagnoses": "Congestive Heart Failure Exacerbation",
            "labs": "Potassium: 3.2 mEq/L. Sodium: 135 mEq/L. BUN: 45 mg/dL. Hemoglobin: 10.5 g/dL.",
            "meds": "Lipitor 20mg PO daily, Lasix 80mg PO BID, Ventolin inhaler 2 puffs QID PRN.",
            "course": "Admitted with progressive dyspnea on exertion and bilateral lower extremity pitting edema. Responded well to IV diuresis. Transitioned to oral medications prior to discharge.",
            "instructions": "Cardiology follow up in 1 week. Check blood pressure daily. Maintain low salt diet."
        },
        3: {
            "name": "Robert Chen",
            "age": 45,
            "gender": "Male",
            "mrn": "MRN-55219",
            "dob": "1981-11-03",
            "diagnoses": "Type 2 Diabetes Mellitus with Hyperglycemia",
            "labs": "Glucose: 280 mg/dL. HbA1c: 9.2%. Creatinine: 1.1 mg/dL. Sodium: 137 mEq/L.",
            "meds": "Augmentin 1g PO BID, Lipitor 10mg PO daily, Glucophage 1000mg PO BID.",
            "course": "Patient presented with polyuria and polydipsia. Blood glucose was elevated. Discharged on oral anti-diabetic medications with instruction for close blood sugar monitoring.",
            "instructions": "Check fasting glucose daily. Diet control and exercise are recommended. Follow up with endocrinology in 2 weeks."
        },
        4: {
            "name": "Elena Rostova",
            "age": 34,
            "gender": "Female",
            "mrn": "MRN-33418",
            "dob": "1992-07-15",
            "diagnoses": "Acute Pyelonephritis (UTI)",
            "labs": "WBC: 14.5 x10^3/uL. Hemoglobin: 11.8 g/dL. Creatinine: 0.9 mg/dL. Potassium: 4.1 mEq/L.",
            "meds": "Klavox 1g PO BID, Nexium 20mg PO daily.",
            "course": "Presented with fever and flank pain. Diagnosed with complicated UTI. Treated with IV antibiotics and transitioned to oral medications after defervescing.",
            "instructions": "Take complete course of antibiotics. Drink plenty of water. Return if fever recurs."
        },
        5: {
            "name": "Marcus Vance",
            "age": 59,
            "gender": "Male",
            "mrn": "MRN-88712",
            "dob": "1967-02-28",
            "diagnoses": "Hypertensive Urgency and Dyslipidemia",
            "labs": "Creatinine: 1.7 mg/dL. Potassium: 4.9 mEq/L. Sodium: 136 mEq/L. BUN: 32 mg/dL.",
            "meds": "Lasix 40mg PO daily, Lipitor 40mg PO daily.",
            "course": "Admitted with blood pressure of 195/110 mmHg. Controlled safely with oral antihypertensives. Kidney function was monitored due to elevated creatinine.",
            "instructions": "Monitor blood pressure twice daily. Low sodium diet. Follow up with primary care in 3 days."
        },
        6: {
            "name": "Chloe Thompson",
            "age": 27,
            "gender": "Female",
            "mrn": "MRN-66521",
            "dob": "1999-05-14",
            "diagnoses": "Acute Asthma Exacerbation",
            "labs": "WBC: 9.8 x10^3/uL. Hemoglobin: 13.2 g/dL. Potassium: 3.8 mEq/L.",
            "meds": "Ventolin inhaler 2 puffs q4h, Solu-Medrol 40mg PO daily.",
            "course": "Presented in respiratory distress. Treated with nebulizers and oral steroids. Discharged after pulmonary status stabilized.",
            "instructions": "Use Ventolin inhaler every 4-6 hours as needed. Avoid known triggers. Follow up with pulmonology in 2 weeks."
        }
    }
    
    if patient_id not in patients:
        raise ValueError(f"Patient ID {patient_id} not found in synthetic database.")
        
    p = patients[patient_id]
    
    # Construct Llama Cloud Parser compatible document structure
    raw_markdown = f"""
# CLINICAL RECORD - {p['name']}
## DEMOGRAPHICS
- **Full Name**: {p['name']}
- **MRN/IP No**: {p['mrn']}
- **DOB**: {p['dob']}
- **Age/Gender**: {p['age']} years / {p['gender']}

## CLINICAL FINDINGS
- **Primary Diagnosis**: {p['diagnoses']}
- **Secondary Diagnosis**: None

## LABORATORY TEST RESULTS
- **Labs**: {p['labs']}

## HOSPITAL COURSE
{p['course']}

## DISCHARGE MEDICATIONS
- **Medications**: {p['meds']}

## DISCHARGE INSTRUCTIONS
{p['instructions']}
"""
    
    # Structure it as parsed pages
    pages_data = [
        {
            "page_number": 1,
            "markdown": raw_markdown
        }
    ]
    
    return {
        "num_pages": 1,
        "pages": pages_data
    }

def generate_and_cache_synthetic_patients() -> dict:
    """
    Creates cache directory and writes the 6 patient cases to files for caching.
    """
    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    all_data = {}
    
    for i in range(1, 7):
        filepath = os.path.join(SYNTHETIC_DATA_DIR, f"synthetic_patient_{i}.json")
        data = get_synthetic_patient_data(i)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        all_data[i] = data
        
    logger.info(f"[Synthetic Data] Cached {len(all_data)} synthetic patient profiles successfully.")
    return all_data
