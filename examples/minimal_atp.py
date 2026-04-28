"""Minimal ATP example with InMemoryStore and LeanExecutor."""

from __future__ import annotations

import asyncio

from langgraph.store.memory import InMemoryStore

from atp.config import ProofCandidate
from atp.lean_executor import LeanExecutor


async def main() -> None:
    store = InMemoryStore()
    store.put(
        ("examples",),
        "or_comm",
        {
            "theorem": "forall (p q : Prop), Or p q -> Or q p",
            "tactics": [
                "intro p q h",
                "cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq",
            ],
        },
    )

    item = store.get(("examples",), "or_comm")
    assert item is not None

    theorem = item.value["theorem"]
    tactics = item.value["tactics"]

    candidate = ProofCandidate(
        candidate_id="example-or-comm",
        theorem_statement=theorem,
        score=1.0,
        source="human",
        rationale="Minimal end-to-end Lean execution example.",
    )

    executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
    try:
        result = await executor.run_script_async(
            theorem,
            tactics,
            candidate=candidate,
        )
    finally:
        await executor.close_async()

    print("Solved:", result.solved)
    print("Goal count:", result.goal_count)
    print("Proof state:", result.proof_state)
    print("Tactic history:")
    for tactic in result.node.tactic_history:
        print("-", tactic)


if __name__ == "__main__":
    asyncio.run(main())
