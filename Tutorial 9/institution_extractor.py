"""
LangChain application: Extract structured institution information from Wikipedia
using a Pydantic schema and a custom output parser.

Given an institution's name, the app:
  1. Fetches the institution's Wikipedia page content.
  2. Sends it to an LLM with instructions to return specific structured facts.
  3. Parses and validates the LLM's raw text output against a Pydantic schema
     using a hand-written custom output parser (subclassing BaseOutputParser).

Usage:
    export OPENAI_API_KEY="sk-..."
    python institution_extractor.py "Ramaiah Institute of Technology"
"""

import json
import re
import sys
from typing import List

import wikipedia
from pydantic import BaseModel, Field, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# 1. Pydantic schema for the desired structured output
# ---------------------------------------------------------------------------
class InstitutionInfo(BaseModel):
    founder: str = Field(
        description="Name(s) of the founder(s) of the institution. 'Unknown' if not found."
    )
    year_founded: str = Field(
        description="The year the institution was founded, e.g. '1946'. 'Unknown' if not found."
    )
    branches: List[str] = Field(
        default_factory=list,
        description="Current campuses/branches/locations of the institution. Empty list if not found.",
    )
    num_employees: str = Field(
        description="Approximate number of employees/staff/faculty. 'Unknown' if not found."
    )
    summary: str = Field(
        description="A concise summary of the institution in EXACTLY 4 lines, separated by \\n."
    )


# ---------------------------------------------------------------------------
# 2. Custom output parser enforcing the schema
# ---------------------------------------------------------------------------
class InstitutionOutputParser(BaseOutputParser[InstitutionInfo]):
    """Parses the LLM's raw text response into a validated InstitutionInfo object.

    The LLM is instructed to reply with JSON only. This parser strips any
    accidental markdown code fences, parses the JSON, validates it against
    the Pydantic schema, and raises a clear, catchable error on failure
    instead of silently returning malformed data.
    """

    def get_format_instructions(self) -> str:
        return (
            "Respond with ONLY a single valid JSON object and nothing else "
            "(no markdown fences, no commentary). Use exactly these keys:\n"
            "{\n"
            '  "founder": string,\n'
            '  "year_founded": string,\n'
            '  "branches": [string, ...],\n'
            '  "num_employees": string,\n'
            '  "summary": string   // exactly 4 lines, joined with \\n\n'
            "}"
        )

    def parse(self, text: str) -> InstitutionInfo:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise OutputParserException(
                f"Could not parse LLM output as JSON: {e}\nRaw output:\n{text}"
            )

        try:
            return InstitutionInfo(**data)
        except ValidationError as e:
            raise OutputParserException(
                f"LLM output did not match the InstitutionInfo schema: {e}\nRaw output:\n{text}"
            )

    @property
    def _type(self) -> str:
        return "institution_output_parser"


# ---------------------------------------------------------------------------
# 3. Wikipedia fetch helper
# ---------------------------------------------------------------------------
def fetch_wikipedia_content(institution_name: str, char_limit: int = 4000) -> str:
    wikipedia.set_lang("en")
    try:
        page = wikipedia.page(institution_name, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        page = wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError:
        raise ValueError(f"No Wikipedia page found for '{institution_name}'.")
    return page.content[:char_limit]


# ---------------------------------------------------------------------------
# 4. Build the LangChain chain: prompt | llm | custom_parser
# ---------------------------------------------------------------------------
def build_chain():
    parser = InstitutionOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise research assistant. Extract structured facts "
                "about an institution strictly from the provided Wikipedia text. "
                "If a detail is not present in the text, answer 'Unknown' (or an "
                "empty list for branches). Do not hallucinate or invent facts.\n\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Institution name: {institution_name}\n\n"
                "Wikipedia content:\n{wiki_content}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return prompt | llm | parser


# ---------------------------------------------------------------------------
# 5. Invoke the chain for a given institution
# ---------------------------------------------------------------------------
def get_institution_info(institution_name: str) -> InstitutionInfo:
    wiki_content = fetch_wikipedia_content(institution_name)
    chain = build_chain()
    return chain.invoke(
        {"institution_name": institution_name, "wiki_content": wiki_content}
    )


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Ramaiah Institute of Technology"
    info = get_institution_info(name)

    print("=" * 60)
    print(f"Institution      : {name}")
    print("=" * 60)
    print(f"Founder          : {info.founder}")
    print(f"Year Founded     : {info.year_founded}")
    print(f"Branches         : {', '.join(info.branches) if info.branches else 'Unknown'}")
    print(f"No. of Employees : {info.num_employees}")
    print("-" * 60)
    print("Summary:")
    print(info.summary)
    print("=" * 60)
