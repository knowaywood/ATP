import asyncio

from atp.config import ProofCandidate
from atp.lean_executor import LeanExecutor


def test_lean_executor_solves_simple_or_commutativity():
    async def scenario():
        executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
        try:
            result = await executor.run_script_async(
                "forall (p q : Prop), Or p q -> Or q p",
                [
                    "intro p q h",
                    "cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq",
                ],
            )
            assert result.solved is True
            assert result.goal_count == 0
            assert result.node.status == "proved"
            assert result.node.proof_state == "no goals"
            assert result.node.tactic_history == [
                "intro p q h",
                "cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq",
            ]
        finally:
            await executor.close_async()

    asyncio.run(scenario())


def test_lean_executor_marks_failed_branch_on_bad_tactic():
    async def scenario():
        executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
        try:
            result = await executor.start_goal_async("forall (p q : Prop), Or p q -> Or q p")
            failed = await executor.run_tactic_async(result.node, "simp [ThisDoesNotExist]")
            assert failed.solved is False
            assert failed.error is not None
            assert failed.node.status == "dead_end"
            assert failed.node.tactic_history == ["simp [ThisDoesNotExist]"]
        finally:
            await executor.close_async()

    asyncio.run(scenario())


def test_lean_executor_updates_candidate_script():
    async def scenario():
        executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
        candidate = ProofCandidate(
            candidate_id="or-comm",
            theorem_statement="forall (p q : Prop), Or p q -> Or q p",
            score=0.8,
        )
        try:
            result = await executor.run_script_async(
                candidate.theorem_statement,
                ["intro p q h", "cases h", "rename_i hp", "exact Or.inr hp", "exact MissingLemma"],
                candidate=candidate,
            )
            assert result.solved is False
            assert result.error is not None
            assert result.node.candidate is not None
            assert result.node.candidate.status == "failed"
            assert "intro p q h" in result.node.candidate.tactic_script
        finally:
            await executor.close_async()

    asyncio.run(scenario())
