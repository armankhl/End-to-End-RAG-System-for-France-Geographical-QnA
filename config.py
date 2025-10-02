# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Data Paths (ensure these match your ingestion.py output) ---
DATA_DIR = "data"
# These paths are from your updated ingestion.py
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "france_land_sections.index")
CHUNKS_DATA_PATH = os.path.join(DATA_DIR, "france_land_sections_chunks_data.json")

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Configuration (Together AI) ---
# Model specified by user

LLM_MODEL_NAME_PRIMARY = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
# LLM_MODEL_NAME_PRIMARY = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# A fast, capable, and cost-effective model for fallback and evaluation
LLM_MODEL_NAME_FALLBACK = "mistralai/Mixtral-8x7B-Instruct-v0.1"


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_BASE_URL = "https://api.together.xyz/v1" # Standard base for v1 API

# --- Retrieval Settings ---
DEFAULT_TOP_K_RETRIEVAL = 5 # Default K for retrieval if not specified in request

# --- Generation Settings ---
DEFAULT_MAX_TOKENS_GENERATE = 1024 # Max tokens for LLM response
DEFAULT_TEMPERATURE_GENERATE = 0.3 # Lower for more factual, less creative outputs
DEFAULT_TOP_P_GENERATE = 0.9     # Nucleus sampling parameter