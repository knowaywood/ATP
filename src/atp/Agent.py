"""For Agent."""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore

from atp import config as cfg

Inmemory_store = InMemoryStore()
leanState = cfg.LeanState()


def init_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Init agent."""
    from langchain.agents import create_agent

    from atp.tools.Search import ddgs_search, search_lean_theorem

    if tools is not None:
        tools = [leanState.update, search_lean_theorem, ddgs_search, *tools]
    else:
        tools = [leanState.update, search_lean_theorem, ddgs_search]

    init_agent = create_agent(
        model=model,
        system_prompt=cfg.INIT_AGENT_PROMPT,
        store=Inmemory_store,
        tools=tools,
    )
    return init_agent


def main_agent(
    model: str | BaseChatModel,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Agent for taking a atp"""

    main_agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.MAIN_AGENT_PROMPT,
        subagents=subagents,
        store=Inmemory_store,
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memory/": StateBackend(rt), "/search/": StateBackend(rt)},
        ),
    )
    return main_agent


def search_agent(model: str | BaseChatModel, tools: Sequence[BaseTool] | None = None):
    """Agent for search from internet and return the answer."""
    from deepagents.backends import CompositeBackend
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.SEARCH_AGENT_PROMPT,
        middleware=[
            FilesystemMiddleware(
                system_prompt="Use this tool to write your Summarize in /search/ dirctory.",
                backend=lambda rt: CompositeBackend(
                    default=StateBackend(rt),
                    routes={"/search/": StateBackend(rt)},
                ),
            ),
        ],
        store=Inmemory_store,
    )
    return agent


def _sub_search_agent(
    model: str | BaseChatModel, tools: Sequence[BaseTool] | None = None
):
    """Generate subagent which will be used by main agent."""
    agent_graph = search_agent(model, tools)
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

    from atp.tools.save_memory import save_chat

    load_dotenv()
    model = ChatTongyi(model="qwen-max")
    # model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    subagent = _sub_search_agent(model)
    mainagent = init_agent(model)
    res = asyncio.run(mainagent.ainvoke({"messages": "费马大定理"}))
    save_chat("history.json", res)

    pprint(res)
    pprint(leanState())
