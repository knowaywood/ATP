"""Core state models and prompts for the ATP workflow."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class StageName(str, Enum):
    """Named stages in the proving workflow."""

    INIT = "init"
    RETRIEVE = "retrieve"
    PROVE = "prove"
    VERIFY = "verify"
    SUMMARIZE = "summarize"


class BaseState(BaseModel):
    """Base state structure for agent communication."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class SearchResult(BaseModel):
    """Lean or retrieval search result."""

    name: str
    module: str
    docstring: str | None
    source_text: str
    dependencies: str | None
    informalization: str | None


class ProofCandidate(BaseModel):
    """A candidate proof trajectory."""

    candidate_id: str
    theorem_statement: str
    tactic_script: str = ""
    rationale: str = ""
    score: float = 0.0
    status: Literal["draft", "proved", "failed", "stalled"] = "draft"
    source: Literal["planner", "retriever", "prover", "repair", "human"] = "planner"
    notes: list[str] = Field(default_factory=list)


class SearchNode(BaseModel):
    """Node in a lightweight proof-search tree."""

    node_id: str
    parent_id: str | None = None
    depth: int = 0
    priority: float = 0.0
    visits: int = 0
    status: Literal["open", "expanded", "proved", "dead_end"] = "open"
    tactic: str | None = None
    goal: str
    proof_state: str | None = None
    lean_state_id: int | None = None
    error: str | None = None
    messages: list[str] = Field(default_factory=list)
    tactic_history: list[str] = Field(default_factory=list)
    candidate: ProofCandidate | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the node should stop expanding."""
        return self.status in {"proved", "dead_end"}


class LeanState(BaseModel):
    """Research state that can be shared across agents."""

    query: str | None = None
    background: str | None = None
    lean_query: str | None = None
    lean_theorem: str | None = None
    imports: list[str] = Field(default_factory=list)
    retrieved_theorems: list[SearchResult] = Field(default_factory=list)
    candidate_proofs: list[ProofCandidate] = Field(default_factory=list)
    search_tree: list[SearchNode] = Field(default_factory=list)
    verifier_notes: list[str] = Field(default_factory=list)
    solved: bool = False

    def update(self, field_name: str, value: Any) -> bool:
        """Update a field on the shared state."""
        if hasattr(self, field_name):
            setattr(self, field_name, value)
            return True
        return False

    def register_candidate(self, candidate: ProofCandidate) -> None:
        """Append a proof candidate to the state."""
        self.candidate_proofs.append(candidate)

    def add_tree_node(self, node: SearchNode) -> None:
        """Append a search node to the state."""
        self.search_tree.append(node)

    def __call__(self) -> dict[str, Any]:
        return self.model_dump()


class AgentState(BaseModel):
    """Conversation split by major ATP stages."""

    init_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=tuple
    )
    planner_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=tuple
    )
    prover_messages: dict[str, BaseMessage] = Field(default_factory=dict)
    verifier_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=tuple
    )
    summary_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=tuple
    )


class StageConfig(BaseModel):
    """Configuration for each proving stage."""

    name: StageName
    prompt: str
    objective: str


INIT_AGENT_PROMPT = """
You are the initialization agent of an Automated Theorem Proving system for Lean.

Your job is to transform a user theorem request into a clean research state.
You do not write the final proof. You prepare the proving environment.

Workflow:
1. Normalize the natural-language theorem statement.
2. Produce a Lean theorem statement in `lean_query` when possible.
3. Identify core mathematical objects, assumptions, and likely imports.
4. Search for exact Mathlib declarations with `search_lean_theorem`.
5. Save concise background notes and exact theorem references into `LeanState`.

Rules:
- Never invent theorem names. Search first.
- Distinguish informal mathematical intuition from exact Lean artifacts.
- Prefer concise, high-information updates over verbose prose.
"""

PLANNER_AGENT_PROMPT = """
You are the planner in an ATP research pipeline inspired by retrieval-augmented proving,
best-first search, and proof repair loops.

Given the current Lean goal and retrieved context:
1. Propose several distinct proof candidates rather than only one.
2. Explain which candidate is most promising and why.
3. Break the proof into subgoals or tactic milestones.
4. Record each candidate with a score and short rationale.

Good plans diversify search:
- one direct tactic-heavy attempt
- one theorem-application attempt
- one algebraic or rewriting attempt

Avoid pretending a proof is complete unless the evidence supports it.
"""

PROVER_AGENT_PROMPT = """
You are the prover agent.

Input:
- a formal Lean theorem statement
- retrieved Mathlib declarations
- one selected proof candidate

Output:
- a Lean tactic script or term proof
- short notes about what changed in the proof state
- any concrete Lean errors or blockers

Rules:
- Stay close to verified theorem names from retrieval.
- If a tactic fails, report the failure signal clearly for the verifier.
- Prefer short iterative progress over long hallucinated proofs.
"""

VERIFIER_AGENT_PROMPT = """
You are the verifier and repair agent.

Your task is to inspect candidate proofs and search-tree progress.
1. Check whether the candidate is solved, stalled, or invalid.
2. Extract actionable repair feedback.
3. Decide whether to expand, prune, or promote a search branch.
4. Keep the search tree disciplined: prune low-signal branches.

Feedback should be concrete:
- missing import
- wrong theorem arity
- goal shape mismatch
- better rewrite lemma
- branch worth re-expanding with lower priority
"""

SEARCH_AGENT_PROMPT = """
You are a specialized paper-and-context search agent for ATP.

Use search to retrieve:
- theorem proving papers
- Lean/mathlib references
- tactic-specific context
- relevant benchmark or dataset descriptions

Return concise notes that can improve the planner or verifier.
If search fails, report what was tried and why it was insufficient.
"""

SUMMARY_AGENT_PROMPT = """
You are the final summarizer for the ATP workflow.

Summarize:
- the theorem in plain language
- the selected proof strategy
- what retrieval contributed
- whether the proof was completed
- what the next best action is if the proof is incomplete
"""


def build_stage_configs() -> list[StageConfig]:
    """Return the default workflow stage configuration."""
    return [
        StageConfig(
            name=StageName.INIT,
            prompt=INIT_AGENT_PROMPT,
            objective="Translate and ground the problem in Lean and Mathlib.",
        ),
        StageConfig(
            name=StageName.RETRIEVE,
            prompt=PLANNER_AGENT_PROMPT,
            objective="Generate diverse candidate proof directions.",
        ),
        StageConfig(
            name=StageName.PROVE,
            prompt=PROVER_AGENT_PROMPT,
            objective="Attempt Lean proof construction from a chosen candidate.",
        ),
        StageConfig(
            name=StageName.VERIFY,
            prompt=VERIFIER_AGENT_PROMPT,
            objective="Score, prune, and repair proof branches.",
        ),
        StageConfig(
            name=StageName.SUMMARIZE,
            prompt=SUMMARY_AGENT_PROMPT,
            objective="Produce a clean result summary and next actions.",
        ),
    ]
