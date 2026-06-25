import os
import json
import re
from llama_cloud import LlamaCloud
from clinical_agent.config import LLAMA_CLOUD_API_KEY
from clinical_agent.logging_utils import logger

class LlamaCloudParser:
    def __init__(self, cache_dir: str = "cache"):
        self.client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
        self.cache_dir = cache_dir

    def classify_and_filter_pages(self, pages: list) -> list:
        """
        Classifies each parsed document page and filters out standard medical guidelines, 
        dosing charts, or hospital policies that do not contain patient-specific text.
        """
        logger.info("[Parser Preprocessor] Classifying and filtering pages...")
        filtered_pages = []
        
        # Identify patient names or demographics clues from headers
        patient_clues = set()
        for page in pages:
            text = page.get("markdown", "")
            matches = re.findall(r'(?:name|mrn|dob|patient)\s*:\s*([^\n\r]+)', text, re.IGNORECASE)
            for m in matches:
                clue = m.strip().lower()
                # Exclude boilerplate words
                if len(clue) > 3 and not any(w in clue for w in ["none", "unknown", "na", "n/a"]):
                    patient_clues.add(clue)

        for page in pages:
            text = page.get("markdown", "")
            text_lower = text.lower()
            page_num = page.get("page_number")
            
            # Keywords indicating reference files or administrative guidelines
            reference_keywords = [
                "clinical guideline", "standard dosing protocol", "guideline reference",
                "standard hospital policy", "practice parameters", "dosing tables", 
                "standard operating procedure", "bibliographical references"
            ]
            
            is_reference = any(kw in text_lower for kw in reference_keywords)
            
            # Keywords indicating patient-specific details
            patient_keywords = [
                "mrn", "dob", "history of present illness", "hospital course",
                "diagnoses", "discharge medications", "follow-up", "edema",
                "patient presented", "admission date", "vitals"
            ]
            has_patient_data = any(kw in text_lower for kw in patient_keywords) or any(clue in text_lower for clue in patient_clues)
            
            # Filter if it contains reference patterns and has no patient data
            if is_reference and not has_patient_data:
                logger.info(f"[Parser Filter] Page {page_num} classified as GENERAL GUIDELINE/REFERENCE. Excluding from agent context.")
                continue
                
            filtered_pages.append(page)
            
        logger.info(f"[Parser Preprocessor] Finished. Retained {len(filtered_pages)} / {len(pages)} patient-specific pages.")
        return filtered_pages

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Parses a PDF using Llama Cloud Parser. Uses local cache if available.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        cache_filename = f"patient_records_parsed_{base_name}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Load from cache if it exists
        if os.path.exists(cache_path):
            logger.info(f"[Parser] Loading parsed documents from cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Apply classification filter on cached pages
                    data["pages"] = self.classify_and_filter_pages(data.get("pages", []))
                    return data
            except Exception as e:
                logger.warning(f"[Parser] Failed to read cache: {e}. Re-parsing...")

        # 2. Parse via Llama Cloud
        logger.info(f"[Parser] Cache missing. Uploading and parsing: {pdf_path}...")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found at {pdf_path}")

        try:
            with open(pdf_path, "rb") as f:
                result = self.client.parsing.parse(
                    tier="cost_effective",
                    version="latest",
                    upload_file=f,
                    expand=["markdown"]
                )

            pages_data = []
            for p in result.markdown.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "markdown": p.markdown
                })

            output_data = {
                "num_pages": len(pages_data),
                "pages": pages_data
            }

            # Cache the raw parsed results before filtering (so cache remains complete)
            with open(cache_path, "w", encoding="utf-8") as f_out:
                json.dump(output_data, f_out, indent=2)
            logger.info(f"[Parser] Caching completed successfully at {cache_path}")

            # Apply classification filter for active consumption
            output_data["pages"] = self.classify_and_filter_pages(pages_data)
            return output_data

        except Exception as e:
            logger.error(f"[Parser] Llama Cloud parsing failed: {e}")
            raise e
