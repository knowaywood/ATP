"""For Agent."""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore

from atp import config as cfg

Inmemory_store = InMemoryStore()

def init_agent():
    ...

def main_agent(
    model: str | BaseChatModel,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Agent for taking a atp of paper."""
    # from atp.tools.search import ddgs_search

    main_agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.MAIN_AGENT_PROMPT,
        subagents=subagents,
        store=Inmemory_store,
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memory/": StoreBackend(rt), "/search/": StoreBackend(rt)},
        ),
    )
    return main_agent


def search_agent(model: str | BaseChatModel):
    """Agent for search from internet and return the answer."""
    from deepagents.backends import CompositeBackend, StoreBackend
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents import create_agent

    from atp.tools.Search import ddgs_search

    agent = create_agent(
        model=model,
        tools=[ddgs_search],
        system_prompt=cfg.SEARCH_AGENT_PROMPT,
        middleware=[
            FilesystemMiddleware(
                system_prompt="Use the write your Summarize in /search/ dirctory.",
                backend=lambda rt: CompositeBackend(
                    default=StateBackend(rt),
                    routes={"/search/": StoreBackend(rt)},
                ),
            ),
        ],
        store=Inmemory_store,
    )
    return agent


def search_subagent(model: str | BaseChatModel):
    """Generate subagent which will be used by main agent."""
    agent_graph = search_agent(model)
    agent_subgraph = CompiledSubAgent(
        name="web-search-agent",
        description="An agent that searches the web for information.and the answer will be written to /search/",
        runnable=agent_graph,
    )
    return agent_subgraph


if __name__ == "__main__":
    from pprint import pprint

    from dotenv import load_dotenv
    from langchain_community.chat_models import ChatTongyi

    from atp.tools.FS import read_from_local

    load_dotenv()
    model = ChatTongyi(model="qwen-max")
    subagent = search_subagent(model)
    mainagent = main_agent(model, subagents=[subagent], tools=[read_from_local])
    res = mainagent.invoke(
        {"messages": "读取文件/home/dministrator/git_clone/atp/.env_copy的内容"}
    )
    pprint(res)
