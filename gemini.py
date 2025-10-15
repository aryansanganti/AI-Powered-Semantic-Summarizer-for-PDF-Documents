import os
try:
    from dotenv import load_dotenv
except ImportError:
    # Minimal .env loader fallback if python-dotenv is not installed
    def load_dotenv(path=None):
        env_path = path or os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from query_faiss import search_faiss
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
genai.configure(api_key=api_key)

# --- New: dynamic model selection and caching ---
_SELECTED_MODEL_ID = None
_GENAI_MODEL = None

def _select_supported_model():
    """Pick a supported model for generateContent from your account."""
    try:
        models = [m for m in genai.list_models() if "generateContent" in getattr(m, "supported_generation_methods", [])]
        # Prefer 1.5 family, then any gemini model
        preferred = [
            "gemini-2.5-flash",
        ]
        # list_models returns names like "models/gemini-1.5-flash-8b"
        for want in preferred:
            for m in models:
                name = getattr(m, "name", "")
                short = name.split("/")[-1] if name else ""
                if name == f"models/{want}" or short == want:
                    return name or f"models/{want}"
        # If none match preferred, return the first with generateContent
        if models:
            return models[0].name
    except Exception:
        # Fall back if listing fails (network/permission)
        pass
    # Conservative fallbacks (try in order)
    return None

def _init_genai_model():
    """Initialize and cache a GenerativeModel with a supported model id."""
    global _GENAI_MODEL, _SELECTED_MODEL_ID
    if _GENAI_MODEL is not None:
        return
    chosen = _select_supported_model()
    candidates = []
    if chosen:
        candidates.append(chosen)  # full name e.g., "models/gemini-1.5-flash-8b"
    # Fallbacks (full names for v1beta compatibility)
    candidates += [
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
    ]
    last_error = None
    for mid in candidates:
        try:
            _GENAI_MODEL = genai.GenerativeModel(mid)
            _SELECTED_MODEL_ID = mid
            return
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"No compatible Gemini model available. Last error: {last_error}")

def get_gemini_response(prompt):
    """Generates a response from Google Gemini"""
    _init_genai_model()
    response = _GENAI_MODEL.generate_content(prompt)
    return response.text
# --- end new ---

def generate_quiz(context, query):
    prompt = f"Based on the following content:\n{context}\n and the query\n{query}\n, create a quiz. Only use your own information to refine the quiz. Do not use it to introduce new topics. Limit the quiz to 3 questions. Also give the answers at the end."
    return get_gemini_response(prompt)

def generate_explanation(context, query):
    prompt = f"Based on the following content:\n{context}\n and the query\n{query}\nExplain it in simple terms. If necessary, use your own information related to the query. Also mention what information is derived from the content given and what is derived from your own knowledge."
    return get_gemini_response(prompt)

def _ensure_index():
    """Build index if missing"""
    if not (os.path.exists("faiss_index.bin") and os.path.exists("metadata.json")):
        from store_embeddings import build_index
        print("No FAISS index found. Building from PDFs in current directory...")
        build_index()

if __name__ == "__main__":
    _ensure_index()
    while True:
        choice = input("Quiz or Explanation (or 'exit'): ").strip().lower()
        if choice in ("exit", "quit"):
            break

        if choice == "quiz":
            query = input("Enter query (or 'exit'): ").strip()
            if query.lower() in ("exit", "quit"):
                break
            context = search_faiss(query)
            if context == "":
                break
            print(generate_quiz(context, query))

        elif choice == "explanation":
            query = input("Enter query (or 'exit'): ").strip()
            if query.lower() in ("exit", "quit"):
                break
            context = search_faiss(query)
            if context == "":
                break
            print(generate_explanation(context, query))
        else:
            print("Unknown command")
            break

