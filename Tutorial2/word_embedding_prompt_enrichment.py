"""
=============================================================================
  Word Embedding-Based Prompt Enrichment for Generative AI Models
=============================================================================
  This script demonstrates how word embeddings can be used to:
    1. Retrieve semantically similar words using Word2Vec embeddings
    2. Enrich (augment) original prompts with similar/related words
    3. Generate AI responses for both original & enriched prompts
    4. Compare outputs in terms of detail and relevance (ROUGE + cosine sim)
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import re
import textwrap

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load
import google.generativeai as genai

# ── Rich UI ───────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

# ── Evaluation ────────────────────────────────────────────────────────────────
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =============================================================================
#  CONFIGURATION
# =============================================================================

GOOGLE_API_KEY  = "AQ.Ab8RN6J7FlTxgtbw8GITmHLIK61S7HzQcd0_G6ZEXTzLhSK2Yg"   # ← Replace with your key
GEMINI_MODEL    = "gemini-1.5-flash"
EMBEDDING_MODEL = "glove-wiki-gigaword-100"    # ~128 MB, fast download
TOP_N_SIMILAR   = 5   # similar words to retrieve per keyword
MAX_KEYWORDS    = 4   # max keywords to expand from each prompt

# =============================================================================
#  SAMPLE PROMPTS  (original prompts the user wants to enrich)
# =============================================================================

PROMPTS = [
    "Explain the impact of climate change on biodiversity.",
    "How does machine learning improve healthcare outcomes?",
    "Describe the role of renewable energy in sustainable development.",
]

# =============================================================================
#  INITIALISE COMPONENTS
# =============================================================================

console = Console()

# ── Load GloVe embeddings (glove-wiki-gigaword-100, ~128 MB on first download) ─
console.rule("[bold cyan]Loading GloVe Embeddings[/]")
console.print(f"[yellow]Downloading / loading {EMBEDDING_MODEL} (one-time ~128 MB)…[/]")
wv_model: KeyedVectors = gensim_load(EMBEDDING_MODEL)
console.print("[green]✔ GloVe model loaded.[/]\n")

# ── Sentence-Transformer for semantic similarity scoring ─────────────────────
console.rule("[bold cyan]Loading Sentence Transformer[/]")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
console.print("[green]✔ Sentence-BERT model loaded.[/]\n")

# ── Configure Gemini ──────────────────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── ROUGE scorer ─────────────────────────────────────────────────────────────
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def extract_keywords(prompt: str) -> list[str]:
    """
    Very simple keyword extractor:
      – tokenise on word boundaries
      – filter stop-words and short tokens
      – return at most MAX_KEYWORDS tokens
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "up", "about", "into",
        "through", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "does", "its", "it", "explain", "describe",
    }
    tokens = re.findall(r"\b[a-zA-Z]+\b", prompt.lower())
    keywords = [t for t in tokens if t not in stop_words and len(t) > 3]
    return list(dict.fromkeys(keywords))[:MAX_KEYWORDS]   # deduplicated, ordered


def get_similar_words(keyword: str, top_n: int = TOP_N_SIMILAR) -> list[str]:
    """Return top-N most similar words from Word2Vec vocabulary."""
    try:
        similar = wv_model.most_similar(keyword, topn=top_n)
        return [word for word, _ in similar]
    except KeyError:
        return []   # keyword not in vocabulary


def enrich_prompt(original: str) -> tuple[str, dict[str, list[str]]]:
    """
    Enrich an original prompt by:
      1. Extracting keywords
      2. Finding similar words via Word2Vec
      3. Appending a context sentence listing related terms
    Returns (enriched_prompt, keyword_map).
    """
    keywords = extract_keywords(original)
    keyword_map: dict[str, list[str]] = {}

    for kw in keywords:
        similars = get_similar_words(kw)
        if similars:
            keyword_map[kw] = similars

    if not keyword_map:
        return original, {}

    # Build an "enrichment suffix" that guides the model
    context_parts = []
    for kw, sims in keyword_map.items():
        context_parts.append(f"{kw} (related: {', '.join(sims[:3])})")

    enrichment = (
        "\n\n[Context enrichment via word embeddings] "
        "Consider these semantically related concepts in your response: "
        + "; ".join(context_parts)
        + ". Please provide a comprehensive, detailed answer covering multiple facets."
    )
    return original + enrichment, keyword_map


def call_gemini(prompt: str) -> str:
    """Call Gemini and return the text response."""
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        return f"[ERROR calling Gemini: {exc}]"


def evaluate_responses(
    original_resp: str,
    enriched_resp: str,
) -> dict:
    """
    Compute:
      – ROUGE-1 / ROUGE-2 / ROUGE-L (enriched vs original as pseudo-reference)
      – Cosine similarity between SBERT embeddings
      – Word-count delta
    """
    rouge_scores = rouge.score(original_resp, enriched_resp)

    orig_vec  = sbert.encode([original_resp])
    enrch_vec = sbert.encode([enriched_resp])
    cos_sim   = float(cosine_similarity(orig_vec, enrch_vec)[0][0])

    return {
        "rouge1_f":    round(rouge_scores["rouge1"].fmeasure,  4),
        "rouge2_f":    round(rouge_scores["rouge2"].fmeasure,  4),
        "rougeL_f":    round(rouge_scores["rougeL"].fmeasure,  4),
        "cosine_sim":  round(cos_sim, 4),
        "orig_words":  len(original_resp.split()),
        "enrch_words": len(enriched_resp.split()),
        "word_delta":  len(enriched_resp.split()) - len(original_resp.split()),
    }


def wrap_text(text: str, width: int = 90) -> str:
    """Word-wrap long text for console display."""
    return "\n".join(
        textwrap.fill(line, width=width) for line in text.splitlines()
    )


# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

def show_keyword_table(keyword_map: dict[str, list[str]]) -> None:
    table = Table(
        title="Word Embedding – Similar Words Retrieved",
        box=box.ROUNDED,
        style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Keyword", style="bold yellow", no_wrap=True)
    table.add_column("Top Similar Words (Word2Vec)", style="white")
    for kw, sims in keyword_map.items():
        table.add_row(kw, ", ".join(sims))
    console.print(table)


def show_metrics_table(metrics: dict) -> None:
    table = Table(
        title="Comparison Metrics",
        box=box.SIMPLE_HEAVY,
        style="green",
        header_style="bold green",
    )
    table.add_column("Metric",        style="bold white",  no_wrap=True)
    table.add_column("Value",         style="bold yellow", justify="right")
    table.add_column("Interpretation",style="dim white")

    interp = {
        "rouge1_f":   "Unigram overlap (0–1); higher = more shared content",
        "rouge2_f":   "Bigram overlap (0–1); higher = more coherence overlap",
        "rougeL_f":   "Longest common subsequence; higher = more structural similarity",
        "cosine_sim": "Semantic similarity (0–1); closer to 1 = very similar meaning",
        "orig_words":  "Words in original response",
        "enrch_words": "Words in enriched response",
        "word_delta":  "Extra words added by enrichment",
    }
    for k, v in metrics.items():
        table.add_row(k, str(v), interp.get(k, ""))
    console.print(table)


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_pipeline(prompt_idx: int, original_prompt: str) -> None:
    console.print(Rule(f"[bold white]PROMPT {prompt_idx + 1} / {len(PROMPTS)}"))

    # ── 1. Show original prompt ───────────────────────────────────────────────
    console.print(Panel(original_prompt, title="[bold blue]Original Prompt", border_style="blue"))

    # ── 2. Enrich prompt ──────────────────────────────────────────────────────
    enriched_prompt, keyword_map = enrich_prompt(original_prompt)
    if keyword_map:
        show_keyword_table(keyword_map)
    console.print(Panel(enriched_prompt, title="[bold magenta]Enriched Prompt", border_style="magenta"))

    # ── 3. Generate responses ─────────────────────────────────────────────────
    console.print("\n[bold cyan]⟳  Generating response for original prompt…[/]")
    orig_response = call_gemini(original_prompt)

    console.print("[bold cyan]⟳  Generating response for enriched prompt…[/]\n")
    enrch_response = call_gemini(enriched_prompt)

    # ── 4. Display responses ──────────────────────────────────────────────────
    console.print(Panel(
        wrap_text(orig_response),
        title="[bold blue]Response to Original Prompt",
        border_style="blue",
    ))
    console.print(Panel(
        wrap_text(enrch_response),
        title="[bold magenta]Response to Enriched Prompt",
        border_style="magenta",
    ))

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate_responses(orig_response, enrch_response)
    show_metrics_table(metrics)

    # ── 6. Quick qualitative summary ─────────────────────────────────────────
    delta = metrics["word_delta"]
    cos   = metrics["cosine_sim"]
    rouge1 = metrics["rouge1_f"]

    if delta > 0:
        detail_msg = f"[green]✔ Enriched response is [bold]{delta} words longer[/] – more detailed.[/]"
    else:
        detail_msg = f"[yellow]⚠ Enriched response is [bold]{abs(delta)} words shorter[/] – may be more concise.[/]"

    if cos > 0.90:
        rel_msg = "[green]✔ Very high semantic similarity – both responses are highly relevant.[/]"
    elif cos > 0.75:
        rel_msg = "[yellow]✔ Good semantic overlap – enrichment kept relevance while adding breadth.[/]"
    else:
        rel_msg = "[red]⚠ Lower semantic similarity – enrichment shifted the focus significantly.[/]"

    console.print(Panel(
        f"{detail_msg}\n{rel_msg}\n"
        f"  ROUGE-1 F1: [bold]{rouge1}[/] | Cosine Similarity: [bold]{cos}[/]",
        title="[bold green]Qualitative Summary",
        border_style="green",
    ))
    console.print()


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    console.print(Panel(
        Text(
            "Word Embedding-Based Prompt Enrichment for Generative AI\n"
            "Using GloVe (glove-wiki-gigaword-100) + Gemini 1.5 Flash + ROUGE + Sentence-BERT",
            justify="center",
            style="bold white",
        ),
        border_style="bright_cyan",
        padding=(1, 4),
    ))

    for idx, prompt in enumerate(PROMPTS):
        run_pipeline(idx, prompt)

    console.rule("[bold green]✔ Pipeline Complete[/]")
