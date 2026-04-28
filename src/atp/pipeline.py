"""Helpers for managing proof candidates and a lightweight search tree."""

from __future__ import annotations

from collections.abc import Iterable

from atp.config import LeanState, ProofCandidate, SearchNode


def bootstrap_state(query: str, lean_query: str | None = None) -> LeanState:
    """Create a fresh ATP state."""
    return LeanState(query=query, lean_query=lean_query)


def create_root_node(goal: str, proof_state: str | None = None) -> SearchNode:
    """Create the search-tree root."""
    return SearchNode(
        node_id="root",
        goal=goal,
        proof_state=proof_state,
        depth=0,
        priority=1.0,
    )


def score_candidate(
    base_score: float,
    *,
    retrieval_hits: int = 0,
    verifier_bonus: float = 0.0,
    error_penalty: float = 0.0,
) -> float:
    """Compute a simple heuristic score for a proof candidate."""
    return round(base_score + retrieval_hits * 0.15 + verifier_bonus - error_penalty, 4)


def record_candidates(state: LeanState, candidates: Iterable[ProofCandidate]) -> None:
    """Append candidates to the shared state."""
    for candidate in candidates:
        state.register_candidate(candidate)


def record_tree_nodes(state: LeanState, nodes: Iterable[SearchNode]) -> None:
    """Append search nodes to the shared state."""
    for node in nodes:
        state.add_tree_node(node)


def frontier(state: LeanState) -> list[SearchNode]:
    """Return open nodes ordered by search priority."""
    open_nodes = [node for node in state.search_tree if node.status == "open"]
    return sorted(open_nodes, key=lambda node: (-node.priority, node.depth, node.node_id))


def choose_next_node(state: LeanState) -> SearchNode | None:
    """Return the best next node to expand."""
    ordered = frontier(state)
    return ordered[0] if ordered else None


def expand_with_candidates(
    parent: SearchNode,
    candidates: Iterable[ProofCandidate],
) -> list[SearchNode]:
    """Convert proof candidates into child tree nodes."""
    children: list[SearchNode] = []
    for index, candidate in enumerate(candidates, start=1):
        child = SearchNode(
            node_id=f"{parent.node_id}.{index}",
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            priority=max(candidate.score, 0.0),
            tactic=candidate.tactic_script or None,
            goal=candidate.theorem_statement,
            candidate=candidate,
        )
        children.append(child)
    return children


def apply_verifier_feedback(
    node: SearchNode,
    *,
    solved: bool = False,
    dead_end: bool = False,
    note: str | None = None,
    priority_delta: float = 0.0,
) -> SearchNode:
    """Return an updated node after verifier feedback."""
    updates = node.model_copy(deep=True)
    updates.visits += 1
    updates.priority = round(max(0.0, updates.priority + priority_delta), 4)
    if note and updates.candidate is not None:
        updates.candidate.notes.append(note)
    if solved:
        updates.status = "proved"
    elif dead_end:
        updates.status = "dead_end"
    else:
        updates.status = "expanded"
    return updates
