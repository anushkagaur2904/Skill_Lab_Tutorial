# LangChain Institution Info Extractor

Extracts structured facts about any institution from Wikipedia, using a
Pydantic schema and a **custom** output parser (not the built-in
`PydanticOutputParser`) to enforce the schema.

## How it works

1. `fetch_wikipedia_content()` pulls the institution's Wikipedia page text
   (via the `wikipedia` package).
2. A `ChatPromptTemplate` tells the LLM to return the required fields as
   JSON, with format instructions supplied by the custom parser itself.
3. `ChatOpenAI` generates a raw text response.
4. `InstitutionOutputParser` (subclass of `BaseOutputParser`) strips any
   markdown fences, parses the JSON, and validates it against the
   `InstitutionInfo` Pydantic model — raising `OutputParserException` with a
   clear message if the LLM's output doesn't match the schema.
5. The chain is built with LCEL: `prompt | llm | parser`.

## Schema (`InstitutionInfo`)

| Field           | Type        | Description                          |
|-----------------|-------------|---------------------------------------|
| `founder`       | `str`       | Founder(s) of the institution         |
| `year_founded`  | `str`       | Year founded                          |
| `branches`      | `List[str]` | Current branches / campuses           |
| `num_employees` | `str`       | Approx. number of employees / staff   |
| `summary`       | `str`       | Concise 4-line summary                |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."      # on Windows: set OPENAI_API_KEY=sk-...
```

## Run

```bash
python institution_extractor.py "Ramaiah Institute of Technology"
```

You can pass any institution name as the argument, e.g.:

```bash
python institution_extractor.py "Indian Institute of Science"
python institution_extractor.py "DRDO"
```

If no argument is given, it defaults to "Ramaiah Institute of Technology".

## Sample output shape

```
============================================================
Institution      : Ramaiah Institute of Technology
============================================================
Founder          : Dr. M.S. Ramaiah
Year Founded     : 1962
Branches         : Bengaluru main campus
No. of Employees : Unknown
------------------------------------------------------------
Summary:
Ramaiah Institute of Technology (MSRIT) is an autonomous engineering
college located in Bengaluru, India, founded in 1962.
It offers undergraduate and postgraduate programs across engineering
and technology disciplines.
The institute is affiliated with Visvesvaraya Technological University.
It is known for strong industry partnerships and research initiatives.
```

(Exact values depend on what the LLM extracts from the live Wikipedia
content at run time — the model is instructed to say "Unknown" rather than
guess when a fact isn't present in the fetched text.)

## Why a custom parser instead of `PydanticOutputParser`

The task asks for a hand-implemented parser, so `InstitutionOutputParser`:
- Defines its own `get_format_instructions()` (used to build the prompt).
- Implements `parse()` to clean, JSON-decode, and validate the LLM output.
- Raises a LangChain-native `OutputParserException` with the raw text
  attached, so failures are easy to debug in the lab demo.
