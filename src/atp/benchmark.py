"""Benchmark runner for ATP experiments."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from atp.llm_loop import LLMATPResult, LLMLeanProver


class BenchmarkCase(BaseModel):
    """One theorem-proving benchmark case."""

    name: str
    theorem: str
    retrieved_context: str | None = None


class BenchmarkCaseResult(BaseModel):
    """One benchmark case result."""

    name: str
    solved: bool
    rounds: int
    expanded_nodes: int
    successful_node_id: str | None = None
    final_error: str | None = None


class BenchmarkReport(BaseModel):
    """Aggregate benchmark metrics."""

    total: int
    solved: int
    success_rate: float
    pass_at_k: float
    average_rounds: float
    average_expanded_nodes: float
    results: list[BenchmarkCaseResult] = Field(default_factory=list)


@dataclass
class BenchmarkRunner:
    """Run a list of ATP benchmark cases with one prover."""

    prover: LLMLeanProver

    async def run_case_async(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        """Run one benchmark case."""
        result: LLMATPResult = await self.prover.prove_async(
            case.theorem,
            retrieved_context=case.retrieved_context,
        )
        return BenchmarkCaseResult(
            name=case.name,
            solved=result.solved,
            rounds=result.rounds,
            expanded_nodes=result.expanded_nodes,
            successful_node_id=result.successful_node_id,
            final_error=result.final_error,
        )

    async def run_async(self, cases: list[BenchmarkCase]) -> BenchmarkReport:
        """Run all benchmark cases and aggregate metrics."""
        results: list[BenchmarkCaseResult] = []
        for case in cases:
            results.append(await self.run_case_async(case))

        total = len(results)
        solved = sum(1 for item in results if item.solved)
        success_rate = round((solved / total) if total else 0.0, 4)
        average_rounds = round(
            sum(item.rounds for item in results) / total if total else 0.0,
            4,
        )
        average_expanded_nodes = round(
            sum(item.expanded_nodes for item in results) / total if total else 0.0,
            4,
        )
        return BenchmarkReport(
            total=total,
            solved=solved,
            success_rate=success_rate,
            pass_at_k=success_rate,
            average_rounds=average_rounds,
            average_expanded_nodes=average_expanded_nodes,
            results=results,
        )
