"""Search helpers for web context and Lean declarations."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from lean_explore.api.client import ApiClient

import atp.config as cfg

load_dotenv()

search = DuckDuckGoSearchResults(output_format="list")


def get_leanexplore_api_key() -> str:
    """Return the LeanExplore API key or raise a clear error."""
    api_key = os.getenv("LEANEXPLORE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEANEXPLORE_API_KEY is missing. Add it to your environment or .env."
        )
    return api_key


def ddgs_search(query_str: str) -> str:
    """Search the web for the given query string using DuckDuckGo."""
    return search.invoke(query_str)


def summarize_search_results(results: list[cfg.SearchResult]) -> str:
    """Convert retrieved Lean declarations into planner-friendly context."""
    lines: list[str] = []
    for result in results:
        lines.append(f"{result.name} from {result.module}")
        if result.docstring:
            lines.append(f"doc: {result.docstring}")
        lines.append(f"source: {result.source_text}")
    return "\n".join(lines)


async def search_lean_theorem(query_str: str, items: int = 3) -> list[cfg.SearchResult]:
    """Search LeanExplore for declarations related to the query."""
    client = ApiClient(api_key=get_leanexplore_api_key())
    search_response_api = await client.search(query=query_str)
    results: list[cfg.SearchResult] = []
    for item in search_response_api.results[:items]:
        results.append(
            cfg.SearchResult(
                name=item.name,
                module=item.module,
                docstring=item.docstring,
                source_text=item.source_text,
                dependencies=item.dependencies,
                informalization=item.informalization,
            )
        )
    return results


if __name__ == "__main__":
    from pprint import pprint

    pprint(asyncio.run(search_lean_theorem(r"2 is not in \mathbb{Q}")))
