"""Agent builders for the ATP workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore

from atp import config as cfg

INMEMORY_STORE = InMemoryStore()
lean_state = cfg.LeanState()


def _default_init_tools(
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
) -> list[BaseTool | Callable | dict[str, Any]]:
    from atp.tools.Search import ddgs_search, search_lean_theorem

    base_tools: list[BaseTool | Callable | dict[str, Any]] = [
        lean_state.update,
        search_lean_theorem,
        ddgs_search,
    ]
    if tools is not None:
        base_tools.extend(tools)
    return base_tools


def init_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Build the initialization agent."""
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        system_prompt=cfg.INIT_AGENT_PROMPT,
        store=INMEMORY_STORE,
        tools=_default_init_tools(tools),
    )


def planner_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Build the planner agent for proof search proposals."""
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        system_prompt=cfg.PLANNER_AGENT_PROMPT,
        store=INMEMORY_STORE,
        tools=list(tools or ()),
    )


def verifier_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Build the verifier agent."""
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        system_prompt=cfg.VERIFIER_AGENT_PROMPT,
        store=INMEMORY_STORE,
        tools=list(tools or ()),
    )


def main_agent(
    model: str | BaseChatModel,
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Build the main deep ATP agent."""
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=cfg.SUMMARY_AGENT_PROMPT,
        subagents=subagents,
        store=INMEMORY_STORE,
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memory/": StateBackend(rt),
                "/search/": StateBackend(rt),
                "/planner/": StateBackend(rt),
                "/verifier/": StateBackend(rt),
            },
        ),
    )


def search_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
):
    """Build a search sub-agent."""
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=list(tools or ()),
        system_prompt=cfg.SEARCH_AGENT_PROMPT,
        middleware=[
            FilesystemMiddleware(
                system_prompt=(
                    "Use the filesystem to write concise search notes in /search/."
                ),
                backend=lambda rt: CompositeBackend(
                    default=StateBackend(rt),
                    routes={"/search/": StateBackend(rt)},
                ),
            ),
        ],
        store=INMEMORY_STORE,
    )


def compile_search_subagent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
) -> CompiledSubAgent:
    """Generate a web or paper search subagent for the main ATP agent."""
    agent_graph = search_agent(model, tools)
    return CompiledSubAgent(
        name="context-search-agent",
        description=(
            "Searches the web, papers, and Lean context for ATP-relevant notes "
            "and writes them to /search/."
        ),
        runnable=agent_graph,
    )


def build_default_atp_stack(
    model: str | BaseChatModel,
    *,
    init_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    planner_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    verifier_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    search_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the default research-oriented ATP stack."""
    compiled_search = compile_search_subagent(model, search_tools)
    return {
        "state": lean_state,
        "stages": cfg.build_stage_configs(),
        "init": init_agent(model, init_tools),
        "planner": planner_agent(model, planner_tools),
        "verifier": verifier_agent(model, verifier_tools),
        "search": compiled_search,
        "main": main_agent(model, subagents=[compiled_search]),
    }
