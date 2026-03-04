"""search tool using DuckDuckGo."""

import asyncio
import os

# Load API key (ensure it's configured via CLI or ENV variable)
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from lean_explore.api.client import ApiClient

import atp.config as cfg

load_dotenv()


LEANEXPLORE_API_KEY = os.getenv("LEANEXPLORE_API_KEY")
if LEANEXPLORE_API_KEY is None:
    LEANEXPLORE_API_KEY = input("Please enter your Lean Explore API key: ")
print("API Client initialized.")

search = DuckDuckGoSearchResults(output_format="list")


def ddgs_search(query_str: str) -> str:
    """Search the web for the given query string using DuckDuckGo.

    Args:
        query_str (str): The search query string.

    Returns:
        str: The search results.

    """
    print(f"[+] 搜索查询: {query_str} by DuckDuckGo")
    return search.invoke(query_str)


async def search_lean_theorem(query_str: str, items: int = 3) -> list[cfg.SearchResult]:
    """
    Searches for Lean theorems based on the provided query string and displays the results.
    Args:
        query_str (str): The search query string.
        items (int): The number of items to display.
    Returns:
        list [SearchResult]: A list of search results.

    SearchResult: A dictionary containing search results. Each result includes:
        - name: str "Fully qualified Lean name (e.g., 'Nat.add')."
        - module: str "Module name (e.g., 'Mathlib.Data.List.Basic')."
        - docstring: str | None "Documentation string from the source code, if available."
        - source_text: str "The actual Lean source code for this declaration."
        - dependencies: str | None "JSON array of declaration names this declaration depends on."
        - informalization: str | None "Natural language description of the declaration."
    """
    client = ApiClient(api_key=LEANEXPLORE_API_KEY)
    search_response_api = await client.search(query=query_str)
    ls = []
    for i in search_response_api.results[:items]:
        ls.append(
            cfg.SearchResult(
                name=i.name,
                module=i.module,
                docstring=i.docstring,
                source_text=i.source_text,
                dependencies=i.dependencies,
                informalization=i.informalization,
            )
        )
    return ls


if __name__ == "__main__":
    from pprint import pprint

    res = asyncio.run(search_lean_theorem(r"2 is not in \mathbb{Q}"))
    pprint(res)

    # res = ddgs_search("Lean with machine learning site:arxiv.org/pdf")
    # print(res)
