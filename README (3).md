# Prompt Enrichment with Word Embeddings

Automatically enrich GenAI prompts using word embeddings, and compare the model's
response to the original prompt vs. the enriched one.

## Idea

LLMs tend to give more detailed, relevant answers when the prompt carries more
context. Instead of manually guessing what extra context to add, this project
uses **pre-trained word embeddings (GloVe)** to find words that are semantically
similar to the key terms in a prompt, and appends them back into the prompt as
extra guidance — a simple, automatic form of prompt engineering.

## Pipeline

1. **Load word embeddings** — pre-trained GloVe vectors (`glove-wiki-gigaword-50`) via `gensim`.
2. **Extract keywords** — POS-tag the prompt with NLTK and keep nouns/adjectives, since those carry most of the topical meaning.
3. **Retrieve similar words** — for each keyword, find its nearest neighbors in embedding space using cosine similarity (`most_similar`).
4. **Enrich the prompt** — append the related words to the original prompt as extra context.
5. **Generate responses** — run both the original and enriched prompt through a text-generation model (GPT-2 by default; swappable for the OpenAI API).
6. **Compare outputs** — word count, vocabulary size, and new words introduced by enrichment.

## Example

```
Original Prompt:
Write a short note about renewable energy.

Enriched Prompt:
Write a short note about renewable energy. (Consider also related concepts
such as: electricity, fuel, power, solar, sustainable.)
```

The enriched prompt's response is typically longer, touches more sub-topics,
and uses a richer vocabulary than the response to the bare original prompt,
since the model has more semantic cues to draw from.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the GloVe vectors (via `gensim`'s downloader) and the
GPT-2 weights (via `transformers`) — this needs an internet connection once;
both are cached locally afterward.

## Usage

Open `prompt_enrichment_word_embeddings.ipynb` in Jupyter or Colab and run all
cells. It walks through each pipeline step and prints the original prompt,
enriched prompt, both model responses, and a side-by-side comparison.

To try it with your own prompt, just change the `original_prompt` variable
in the notebook.

### Using the OpenAI API instead of GPT-2

The notebook uses local GPT-2 by default so it runs without any API key. If
you'd rather use a hosted model, swap in the alternate `generate_response`
function shown in the notebook's "Optional: use OpenAI API" cell (requires
`pip install openai` and an API key).

## Limitations & future work

- Enrichment works at the **word level**, not sentence level — an embedding
  neighbor can occasionally be topically off, since it doesn't account for
  the sentence's full context.
- Using a larger embedding model (e.g. `glove-wiki-gigaword-300` or
  `word2vec-google-news-300`) or filtering similar words by a minimum
  similarity score would likely improve relevance.
- The current comparison metrics (word count, vocabulary size) are simple
  proxies for "detail" — a more rigorous evaluation could use semantic
  similarity scoring or human ratings of relevance.

## Tech stack

- [`gensim`](https://radimrehurek.com/gensim/) — word embeddings (GloVe)
- [`nltk`](https://www.nltk.org/) — tokenization and POS tagging
- [`transformers`](https://huggingface.co/docs/transformers) — GPT-2 text generation
