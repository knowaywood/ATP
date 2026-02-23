"""search tool using DuckDuckGo."""

import asyncio
import os

# Load API key (ensure it's configured via CLI or ENV variable)
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from lean_explore.api.client import ApiClient
from lean_explore.models.search_types import SearchResult

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


async def search_lean_theorem(query_str: str, items: int = 5) -> list[SearchResult]:
    """
    Searches for Lean theorems based on the provided query string and displays the results.
    Args:
        query_str (str): The search query string.
        items (int): The number of items to display.
    Returns:
        list [SearchResponse]

    SearchResponse: A dictionary containing search results. Each result includes:
        - id (int): The unique identifier of the theorem.
        - theorem_name (str): The name of the theorem.
        - display_statement (str): The display statement of the theorem.
        - docstring (str): The dictionary string representation of the theorem.
        - informal_description (str): The informal description of the theorem.
        - source_file (str): The source file where the theorem is defined.
        - range_start_line (int): The starting line number of the theorem's definition.
    """
    client = ApiClient(api_key=LEANEXPLORE_API_KEY)
    search_response_api = await client.search(query=query_str)
    return (search_response_api.results)[:items]


if __name__ == "__main__":
    res = asyncio.run(search_lean_theorem(r"2 is not in \mathbb{Q}"))
    print(res)

    res = ddgs_search("Lean with machine learning site:arxiv.org/pdf")
    print(res)
