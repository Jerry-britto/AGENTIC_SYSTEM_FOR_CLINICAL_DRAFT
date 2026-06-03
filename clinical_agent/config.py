import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Groq Model Configurations
# We use Llama 3.3 70B for planning, reconciliation, safety checks, and synthesis
REASONING_MODEL = "llama-3.3-70b-versatile"

# We use Llama 3 8B for utility tasks
UTILITY_MODEL = "llama-3.1-8b-instant"

# Validation
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not configured.")
if not LLAMA_CLOUD_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY environment variable is not configured.")
