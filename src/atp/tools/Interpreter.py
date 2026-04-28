"""Small local probe for the lean-interact backend."""

from __future__ import annotations

from atp.lean_executor import LeanExecutor


if __name__ == "__main__":
    executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
    try:
        result = executor.run_script(
            "forall (p q : Prop), Or p q -> Or q p",
            [
                "intro p q h",
                "cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq",
            ],
        )
        print(result.proof_state)
    finally:
        executor.close()
