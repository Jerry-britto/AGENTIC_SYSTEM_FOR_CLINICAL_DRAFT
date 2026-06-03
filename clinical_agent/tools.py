import re

# Mock drug-drug and drug-condition critical interaction database
CRITICAL_INTERACTIONS = [
    {
        "pair": ["aspirin", "warfarin"],
        "severity": "CRITICAL",
        "mechanism": "Co-administration increases the risk of serious gastrointestinal and systemic hemorrhage due to additive antiplatelet and anticoagulant effects.",
        "action": "Hold one or request urgent cardiologist/hematologist review."
    },
    {
        "pair": ["lisinopril", "potassium"],
        "severity": "CRITICAL",
        "mechanism": "Lisinopril (ACE inhibitor) reduces aldosterone secretion, leading to potassium retention. Co-administration with potassium supplements may cause severe hyperkalemia, potentially leading to cardiac arrhythmias.",
        "action": "Discontinue potassium supplements or monitor serum potassium levels daily."
    },
    {
        "pair": ["sildenafil", "nitroglycerin"],
        "severity": "CRITICAL",
        "mechanism": "Organic nitrates dilate blood vessels via nitric oxide pathways, and PDE5 inhibitors (Sildenafil) prevent degradation of cGMP. Together, they cause synergistic, life-threatening hypotension.",
        "action": "STRICTLY CONTRAINDICATED. Do not administer sildenafil within 24 hours of nitrate use."
    },
    {
        "pair": ["clopidogrel", "omeprazole"],
        "severity": "WARNING",
        "mechanism": "Omeprazole (CYP2C19 inhibitor) decreases the bioactivation of clopidogrel to its active metabolite, potentially reducing antiplatelet efficacy.",
        "action": "Consider alternative proton-pump inhibitors like Pantoprazole or H2 receptor antagonists."
    },
    {
        "pair": ["loperamide", "oflox tz"],
        "severity": "HIGH WARNING",
        "mechanism": "Loperamide (antimotility agent) combined with Oflox TZ (antibiotic) in patients with suspected invasive/bacterial gastroenteritis (blood/pus in stool) can lead to toxic megacolon and prolonged fever by trapping bacterial toxins in the gut.",
        "action": "Flag for clinician review. Loperamide should be avoided if invasive bacterial diarrhea is suspected."
    },
    {
        "pair": ["loperamide", "ofloxacin"],
        "severity": "HIGH WARNING",
        "mechanism": "Loperamide (antimotility agent) combined with Ofloxacin (antibiotic) in patients with suspected invasive/bacterial gastroenteritis (blood/pus in stool) can lead to toxic megacolon and prolonged fever by trapping bacterial toxins in the gut.",
        "action": "Flag for clinician review. Loperamide should be avoided if invasive bacterial diarrhea is suspected."
    }
]

def check_drug_interactions(medications: list[str]) -> list[dict]:
    """
    Checks for mock drug-drug interactions in a list of medications.
    """
    normalized_meds = []
    for med in medications:
        norm = med.lower()
        norm = re.sub(r'^(tab\.|cap\.|inj\.|tab|cap|inj)\s+', '', norm)
        norm = re.sub(r'\s+\d+(mg|g|mcg|ml|iu|u).*$', '', norm)
        norm = norm.strip()
        # Handle clinical typos
        norm = norm.replace("lopiramide", "loperamide")
        normalized_meds.append(norm)

    alerts = []
    # Check all pairs
    for i in range(len(normalized_meds)):
        for j in range(i + 1, len(normalized_meds)):
            med1 = normalized_meds[i]
            med2 = normalized_meds[j]
            
            for interaction in CRITICAL_INTERACTIONS:
                p1, p2 = interaction["pair"]
                # Match substrings to catch brand names or generic variations (e.g. "lopperamide" or "oflox tz")
                match1 = (p1 in med1) or (med1 in p1)
                match2 = (p2 in med2) or (med2 in p2)
                
                # Reverse check
                match1_rev = (p1 in med2) or (med2 in p1)
                match2_rev = (p2 in med1) or (med1 in p2)

                if (match1 and match2) or (match1_rev and match2_rev):
                    alerts.append({
                        "medication_1": medications[i],
                        "medication_2": medications[j],
                        "severity": interaction["severity"],
                        "mechanism": interaction["mechanism"],
                        "action": interaction["action"]
                    })
    return alerts
