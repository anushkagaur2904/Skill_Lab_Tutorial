"""
=============================================================================
  FastAPI Web App – Word Embedding-Based Prompt Enrichment for GenAI
=============================================================================
  Start the server:
      python3 app.py
  Then open:
      http://localhost:5050

  Endpoints:
    GET  /             → serves index.html
    GET  /api/status   → returns model load status (always JSON)
    POST /api/enrich   → returns enriched prompt + keyword map (JSON)
    POST /api/generate → calls Gemini for both prompts, returns metrics (JSON)
=============================================================================
"""

import os
import re
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn

# ── Optional heavy deps (graceful import) ─────────────────────────────────────
_MISSING_DEP = None
try:
    import numpy as np
    from gensim.downloader import load as gensim_load
    import google.generativeai as genai
    from rouge_score import rouge_scorer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError as _e:
    DEPS_AVAILABLE = False
    _MISSING_DEP = str(_e)

# ── Configuration ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY", "AIzaSyAb8RN6J7FlTxgtbw8GITmHLIK61S7HzQcd0_G6ZEXTzLhSK2Yg")
GEMINI_MODEL    = "gemini-1.5-flash"
EMBEDDING_MODEL = "glove-wiki-gigaword-100"
TOP_N_SIMILAR   = 6
MAX_KEYWORDS    = 5

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Global model holders ───────────────────────────────────────────────────────
wv_model       = None
sbert          = None
gemini_model   = None
rouge_sc       = None
_STARTUP_ERROR = None   # populated if startup loading fails

# ── Startup ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global wv_model, sbert, gemini_model, rouge_sc, _STARTUP_ERROR

    if not DEPS_AVAILABLE:
        _STARTUP_ERROR = (
            f"Missing Python package — {_MISSING_DEP}. "
            "Install everything with:\n"
            "  pip install gensim google-generativeai rouge-score "
            "scikit-learn sentence-transformers numpy"
        )
        log.error(_STARTUP_ERROR)
        yield
        return

    try:
        log.info("Loading GloVe embeddings (%s) – first run ~128 MB download…", EMBEDDING_MODEL)
        wv_model = gensim_load(EMBEDDING_MODEL)
        log.info("GloVe loaded.")

        log.info("Loading Sentence-BERT…")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Sentence-BERT loaded.")

        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        rouge_sc     = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        log.info("✅  All models ready — visit http://localhost:5050")

    except Exception as exc:
        _STARTUP_ERROR = f"Startup failed: {exc}"
        log.error(_STARTUP_ERROR)

    yield
    log.info("Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Prompt Enrichment Studio", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Global exception handlers – always return JSON ────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def json_http_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def json_validation_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})

@app.exception_handler(Exception)
async def json_generic_handler(request, exc):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": f"Server error: {exc}"})


# ── Readiness helper ───────────────────────────────────────────────────────────
def _ready() -> tuple[bool, str]:
    """Returns (is_ready, error_message)."""
    if _STARTUP_ERROR:
        return False, _STARTUP_ERROR
    if not DEPS_AVAILABLE:
        return False, f"Missing package: {_MISSING_DEP}"
    if wv_model is None or sbert is None or gemini_model is None:
        return False, "Models not yet loaded. Server may still be starting up."
    return True, ""


# ── NLP helpers ────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into",
    "through", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "its", "it", "explain", "describe", "please", "tell", "give",
    "your", "our", "their", "there", "here", "also", "just",
}

def extract_keywords(prompt: str) -> list:
    tokens = re.findall(r"\b[a-zA-Z]+\b", prompt.lower())
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 3]
    return list(dict.fromkeys(keywords))[:MAX_KEYWORDS]

def get_similar_words(keyword: str, top_n: int = TOP_N_SIMILAR) -> list:
    try:
        similar = wv_model.most_similar(keyword, topn=top_n)
        return [{"word": word, "score": round(float(score), 4)} for word, score in similar]
    except KeyError:
        return []

def enrich_prompt(original: str) -> tuple:
    keywords = extract_keywords(original)
    keyword_map = {}
    for kw in keywords:
        similars = get_similar_words(kw)
        if similars:
            keyword_map[kw] = similars

    if not keyword_map:
        return original, {}

    context_parts = []
    for kw, sims in keyword_map.items():
        top3 = ", ".join(s["word"] for s in sims[:3])
        context_parts.append(f"{kw} (related: {top3})")

    enrichment = (
        "\n\n[Context enrichment via word embeddings] "
        "Consider these semantically related concepts in your response: "
        + "; ".join(context_parts)
        + ". Please provide a comprehensive, detailed answer covering multiple facets."
    )
    return original + enrichment, keyword_map

def call_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        log.error("Gemini error: %s", exc)
        raise RuntimeError(f"Gemini API error: {exc}") from exc

def evaluate_responses(orig: str, enrch: str) -> dict:
    scores    = rouge_sc.score(orig, enrch)
    orig_vec  = sbert.encode([orig])
    enrch_vec = sbert.encode([enrch])
    cos_sim   = float(cosine_similarity(orig_vec, enrch_vec)[0][0])
    return {
        "rouge1_f":    round(scores["rouge1"].fmeasure, 4),
        "rouge2_f":    round(scores["rouge2"].fmeasure, 4),
        "rougeL_f":    round(scores["rougeL"].fmeasure, 4),
        "cosine_sim":  round(cos_sim, 4),
        "orig_words":  len(orig.split()),
        "enrch_words": len(enrch.split()),
        "word_delta":  len(enrch.split()) - len(orig.split()),
    }


# ── Request model ──────────────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def api_status():
    ready, msg = _ready()
    return JSONResponse({
        "ready": ready,
        "error": msg if not ready else None,
        "models": {
            "glove":  wv_model is not None,
            "sbert":  sbert is not None,
            "gemini": gemini_model is not None,
            "rouge":  rouge_sc is not None,
        }
    })


@app.post("/api/enrich")
async def api_enrich(req: PromptRequest):
    try:
        if not req.prompt.strip():
            return JSONResponse(status_code=400, content={"error": "prompt is required"})

        ready, msg = _ready()
        if not ready:
            return JSONResponse(status_code=503, content={"error": msg})

        original = req.prompt.strip()
        enriched, keyword_map = enrich_prompt(original)
        return JSONResponse({
            "original":    original,
            "enriched":    enriched,
            "keyword_map": keyword_map,
        })
    except Exception as exc:
        log.error("api_enrich error: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/generate")
async def api_generate(req: PromptRequest):
    try:
        if not req.prompt.strip():
            return JSONResponse(status_code=400, content={"error": "prompt is required"})

        ready, msg = _ready()
        if not ready:
            return JSONResponse(status_code=503, content={"error": msg})

        original = req.prompt.strip()

        log.info("Enriching prompt…")
        enriched, keyword_map = enrich_prompt(original)

        log.info("Calling Gemini (original)…")
        orig_response = call_gemini(original)

        log.info("Calling Gemini (enriched)…")
        enrch_response = call_gemini(enriched)

        log.info("Computing metrics…")
        metrics = evaluate_responses(orig_response, enrch_response)

        return JSONResponse({
            "original_prompt":   original,
            "enriched_prompt":   enriched,
            "keyword_map":       keyword_map,
            "original_response": orig_response,
            "enriched_response": enrch_response,
            "metrics":           metrics,
        })
    except Exception as exc:
        log.error("api_generate error: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5050, reload=False)
