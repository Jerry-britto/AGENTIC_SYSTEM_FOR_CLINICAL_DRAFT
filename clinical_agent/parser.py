import os
import json
from llama_cloud import LlamaCloud
from clinical_agent.config import LLAMA_CLOUD_API_KEY
from clinical_agent.logging_utils import logger

class LlamaCloudParser:
    def __init__(self, cache_dir: str = "cache"):
        self.client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
        self.cache_dir = cache_dir

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Parses a PDF using Llama Cloud Parser. Uses local cache if available.
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Determine cache file path
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        cache_filename = f"patient_records_parsed_{base_name}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Load from cache if it exists
        if os.path.exists(cache_path):
            logger.info(f"[Parser] Loading parsed documents from cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
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

            # Cache the parsed results
            with open(cache_path, "w", encoding="utf-8") as f_out:
                json.dump(output_data, f_out, indent=2)

            logger.info(f"[Parser] Caching completed successfully at {cache_path}")
            return output_data

        except Exception as e:
            logger.error(f"[Parser] Llama Cloud parsing failed: {e}")
            raise e
