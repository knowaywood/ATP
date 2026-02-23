from typing import Annotated, Sequence, TypedDict, Union

from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    initMessages: Annotated[Sequence[BaseMessage], add_messages]
    proverMessages: dict[str, BaseMessage]
    agentMessages: Annotated[Sequence[BaseMessage], add_messages]


class LeanState(TypedDict):
    query: str | None
    background: str | None
    leanQuery: str | None
    leanTheorem: str | None


class Tree(TypedDict):
    leaves: dict[str, list[Union[str, CommandResponse]]]


COT_AGENT_PROMPT = """"""
