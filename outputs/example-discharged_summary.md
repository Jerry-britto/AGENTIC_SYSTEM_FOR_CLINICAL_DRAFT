# Clinical Discharge Summary Report (Example Output)

> [!NOTE]
> This is a reference example output produced by the Clinical Discharge Summary Agentic System. It demonstrates the structured sections, safety warnings, and clinical summaries automatically compiled by the agent.

---

### Clinician Alerts & Safety Escalations

> [!WARNING]
> **CRITICAL REVIEW REQUIRED: 5 Safety Alerts Raised**
>
> 1. **Medication Discrepancy (High Severity)**: IV fluids, IV antibiotics, IV PPIs, and IV antiemetics were added during the hospital stay but have **no documented justification** in the clinical narrative.
> 2. **Clinical Safety Concern**: The patient was prescribed **Loperamide** (an antimotility agent) despite presenting with symptoms and investigations indicating **active bacterial/infectious gastroenteritis** (blood and pus cells in stool routine).
> 3. **Conflicting Information**: Serum Creatinine levels are reported inconsistently across the notes (elevated at `1.65 mg/dL` vs. normal at `1.17 mg/dL`).
> 4. **Missing Demographics**: Patient Name, Age, Gender, MRN, and DOB are missing from the raw intake forms.
> 5. **Pending Lab Results**: Urine culture & sensitivity reports are marked as pending.

---

### 👤 Patient Demographics
* **Full Name**: `[MISSING - Flagged for Clinician Review]`
* **Age / Gender**: `[MISSING - Flagged for Clinician Review]`
* **MRN / IP No**: `[MISSING - Flagged for Clinician Review]`
* **DOB**: `[MISSING - Flagged for Clinician Review]`

### 📅 Admission & Discharge Details
* **Admission Date**: `[MISSING - Flagged for Clinician Review]`
* **Discharge Date**: `[MISSING - Flagged for Clinician Review]`

### 🩺 Diagnoses
* **Principal Diagnosis**: Acute Gastroenteritis with Dehydration
* **Secondary Diagnosis**: Urinary Tract Infection (UTI)

### 🏥 Hospital Course
The patient presented with multiple episodes of loose stools, 2–3 episodes of vomiting, fatigue lasting 3 days, and a fever since yesterday. 

Initial diagnostic investigations revealed:
- Elevated Serum Creatinine (`1.65 mg/dL`)
- Low Serum Sodium (`128.00 mmol/L`)
- Urine routine showing ketone bodies (+), 10–12/hpf pus cells, 15–20/hpf epithelial cells, and presence of bacteria.

The patient was stabilized and treated with IV fluids, IV antibiotics, IV PPIs, IV antiemetics, and supportive care. An abdominal and pelvic USG noted Grade-I fatty liver changes and a mildly edematous ascending colon up to the hepatic flexure, representing potential colitis.

### 💉 Procedures Performed
* **IV Cannulation**: Left hand (20G)
* **Imaging**: Ultrasonography (USG) of abdomen & pelvis, CT scan of KUB (Kidneys, Ureters, Bladder)

### 💊 Discharge Medications & Reconciliation Log

| Medication Name | Dosage / Route | Frequency | Status | Reconciliation Notes & Justifications |
| :--- | :--- | :--- | :--- | :--- |
| **TAB. RACIPER** | 40 mg | 1-0-0 (Before Food) | `UNCHANGED` | Continued for gastric protection. |
| **TAB. EMESET** | 4 mg | 1-1-1 (3 Days) | `UNCHANGED` | Continued for nausea control. |
| **TAB. OFLOX TZ** | Standard | 1-0-1 (5 Days) | `UNCHANGED` | Antibiotic course for suspected UTI. |
| **TAB. LOPIRAMIDE** | 2 mg | 1-0-1 (5 Days) | `UNCHANGED` | **HIGH WARNING**: Antimotility drug prescribed during potential bacterial diarrhea. |
| **IV FLUIDS** | Intravenous | Continuous | `ADDED` | **HIGH WARNING**: No documented justification for discharge continuation. |
| **IV ANTIBIOTICS** | Intravenous | Standard | `ADDED` | **HIGH WARNING**: No documented justification for discharge continuation. |

### ⚠️ Allergies
* `[MISSING - Flagged for Clinician Review]`

### 🔍 Pending Reports
* Urine culture and sensitivity results (awaited)
* Blood and urine culture results (sent due to fever spike)

### 🚪 Follow-Up Instructions & Discharge Condition
* **Discharge Condition**: Hemodynamically stable, hydration restored.
* **Follow-Up Plan**: Review immediately in case of fever, vomiting, loose stools, or worsening fatigue. Scheduled checkup on **March 9, 2026**, with a repeat CBC.
