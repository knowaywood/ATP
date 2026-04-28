"""LLM-guided ATP search loop with branching and repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from atp.config import ProofCandidate, SearchNode
from atp.lean_executor import LeanExecutionResult, LeanExecutor
from atp.pipeline import score_candidate


class SupportsAInvoke(Protocol):
    """Minimal async chat-model interface used by the ATP loop."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any: ...


class TacticPlan(BaseModel):
    """Structured tactic plan produced by an LLM."""

    theorem: str = ""
    rationale: str = ""
    tactics: list[str] = Field(default_factory=list)
    score: float = 0.0


class TacticPlanBatch(BaseModel):
    """A batch of candidate tactic continuations."""

    theorem: str
    strategy: str = ""
    candidates: list[TacticPlan] = Field(default_factory=list)


class LLMATPResult(BaseModel):
    """Result of an LLM-guided proving attempt."""

    solved: bool
    rounds: int
    expanded_nodes: int
    successful_node_id: str | None = None
    final_error: str | None = None
    plan_history: list[TacticPlanBatch] = Field(default_factory=list)
    tree: list[dict] = Field(default_factory=list)
    execution: dict | None = None


class VerifierFeedback(BaseModel):
    """Heuristic verifier decision for a branch."""

    note: str
    score_delta: float
    status: str


def _extract_json_block(response_text: str) -> str:
    """Extract a JSON object from model output."""
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model did not return JSON: {response_text}")
    return stripped[start : end + 1]


def classify_execution(result: LeanExecutionResult) -> VerifierFeedback:
    """Score a branch using simple Lean-aware heuristics."""
    if result.solved:
        return VerifierFeedback(
            note="branch solved",
            score_delta=1.0,
            status="proved",
        )

    error = (result.error or "").lower()
    proof_state = result.proof_state.lower()

    if "unknown constant" in error or "unknown identifier" in error:
        return VerifierFeedback(
            note="unknown theorem or identifier",
            score_delta=-0.55,
            status="dead_end",
        )
    if "unsolved goals" in error:
        return VerifierFeedback(
            note="partial progress with unsolved goals",
            score_delta=0.1,
            status="open",
        )
    if "tactic" in error and "failed" in error:
        return VerifierFeedback(
            note="tactic mismatch with current goal shape",
            score_delta=-0.2,
            status="dead_end",
        )
    if result.goal_count >= 0 and result.goal_count < 2:
        return VerifierFeedback(
            note="branch reduced to a small number of goals",
            score_delta=0.25,
            status="open",
        )
    if "⊢" in proof_state:
        return VerifierFeedback(
            note="branch still live but not clearly improved",
            score_delta=0.0,
            status="open",
        )
    return VerifierFeedback(
        note="branch stalled",
        score_delta=-0.1,
        status="dead_end" if result.error else "open",
    )


def materialize_candidate(
    parent: SearchNode,
    plan: TacticPlan,
    index: int,
) -> ProofCandidate:
    """Convert a tactic plan into a proof candidate."""
    score = score_candidate(
        plan.score,
        verifier_bonus=max(parent.priority - 0.1, 0.0),
        error_penalty=0.0 if parent.error is None else 0.1,
    )
    return ProofCandidate(
        candidate_id=f"{parent.node_id}-cand-{index}",
        theorem_statement=plan.theorem,
        tactic_script="\n".join(plan.tactics),
        rationale=plan.rationale,
        score=score,
        source="planner",
    )


@dataclass
class LLMLeanProver:
    """A branching planner-executor-repair loop."""

    model: SupportsAInvoke
    executor: LeanExecutor
    max_rounds: int = 3
    beam_width: int = 3
    max_tactics_per_candidate: int = 4

    def _system_prompt(self) -> str:
        return (
            "You are an ATP planner for Lean 4.\n"
            "Given a theorem and the current proof state, propose multiple short tactic continuations.\n"
            "Return strict JSON with keys: theorem, strategy, candidates.\n"
            "Each candidate must contain rationale, score, and tactics.\n"
            "The tactics field must be a JSON array of Lean tactic strings.\n"
            "Prefer diverse branches: direct, theorem-application, simp/rewrite, and cases/induction when relevant.\n"
            "Do not wrap the JSON in markdown."
        )

    def _user_prompt(
        self,
        theorem: str,
        node: SearchNode,
        previous_error: str | None = None,
        retrieved_context: str | None = None,
    ) -> str:
        prompt = [
            f"Theorem:\n{theorem}",
            f"Current proof state:\n{node.proof_state or node.goal}",
            f"Current tactic history:\n{node.tactic_history or []}",
            f"Generate at most {self.beam_width} candidates.",
            f"Each candidate should use at most {self.max_tactics_per_candidate} tactics.",
        ]
        if retrieved_context:
            prompt.append(f"Retrieved Lean or theorem context:\n{retrieved_context}")
        if previous_error:
            prompt.append(f"Previous Lean error or blocker:\n{previous_error}")
            prompt.append("Revise away from the failing branch pattern.")
        return "\n\n".join(prompt)

    def _response_text(self, response: Any) -> str:
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _parse_plan(self, response_text: str, theorem: str) -> TacticPlanBatch:
        payload = json.loads(_extract_json_block(response_text))
        batch = TacticPlanBatch.model_validate(payload)
        if not batch.theorem:
            batch.theorem = theorem
        for candidate in batch.candidates:
            if not candidate.theorem:
                candidate.theorem = batch.theorem
        return batch

    async def propose_plan_batch_async(
        self,
        theorem: str,
        node: SearchNode,
        *,
        previous_error: str | None = None,
        retrieved_context: str | None = None,
    ) -> TacticPlanBatch:
        """Ask the LLM for multiple tactic continuations."""
        response = await self.model.ainvoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(
                    content=self._user_prompt(
                        theorem,
                        node,
                        previous_error=previous_error,
                        retrieved_context=retrieved_context,
                    )
                ),
            ]
        )
        return self._parse_plan(self._response_text(response), theorem)

    async def _execute_plan_from_node(
        self,
        parent: SearchNode,
        candidate: ProofCandidate,
        plan: TacticPlan,
        *,
        child_index: int,
    ) -> LeanExecutionResult:
        """Run a candidate continuation from an existing node."""
        current = parent.model_copy(deep=True)
        current.candidate = candidate
        current.node_id = f"{parent.node_id}.{child_index}"
        current.priority = candidate.score
        result: LeanExecutionResult | None = None
        for tactic in plan.tactics:
            result = await self.executor.run_tactic_async(current, tactic)
            current = result.node
            current.node_id = f"{parent.node_id}.{child_index}"
            current.parent_id = parent.node_id
            current.depth = parent.depth + 1
            if result.solved or result.error is not None:
                break
        assert result is not None
        return result

    async def prove_async(
        self,
        theorem: str,
        *,
        retrieved_context: str | None = None,
    ) -> LLMATPResult:
        """Run a branching LLM -> Lean -> verifier loop."""
        root = await self.executor.start_goal_async(theorem, node_id="root")
        frontier: list[SearchNode] = [root.node]
        explored: list[SearchNode] = [root.node]
        history: list[TacticPlanBatch] = []
        last_error: str | None = None
        expanded_nodes = 0

        if root.solved:
            return LLMATPResult(
                solved=True,
                rounds=0,
                expanded_nodes=0,
                successful_node_id="root",
                tree=[root.node.model_dump()],
                execution=root.node.model_dump(),
            )

        for round_index in range(1, self.max_rounds + 1):
            if not frontier:
                break

            frontier = sorted(frontier, key=lambda node: (-node.priority, node.depth))[
                : self.beam_width
            ]
            next_frontier: list[SearchNode] = []

            for node in frontier:
                batch = await self.propose_plan_batch_async(
                    theorem,
                    node,
                    previous_error=node.error or last_error,
                    retrieved_context=retrieved_context,
                )
                history.append(batch)
                expanded_nodes += 1

                for child_index, plan in enumerate(
                    batch.candidates[: self.beam_width],
                    start=1,
                ):
                    candidate = materialize_candidate(node, plan, child_index)
                    result = await self._execute_plan_from_node(
                        node,
                        candidate,
                        plan,
                        child_index=child_index,
                    )
                    feedback = classify_execution(result)
                    child = result.node.model_copy(deep=True)
                    child.priority = round(max(candidate.score + feedback.score_delta, 0.0), 4)
                    child.status = (
                        "proved"
                        if result.solved
                        else "dead_end"
                        if feedback.status == "dead_end"
                        else "open"
                    )
                    child.messages = [*child.messages, feedback.note]
                    if child.candidate is not None:
                        child.candidate.score = child.priority
                        child.candidate.notes.append(feedback.note)
                        if child.status == "proved":
                            child.candidate.status = "proved"
                    explored.append(child)

                    if result.solved:
                        return LLMATPResult(
                            solved=True,
                            rounds=round_index,
                            expanded_nodes=expanded_nodes,
                            successful_node_id=child.node_id,
                            plan_history=history,
                            tree=[item.model_dump() for item in explored],
                            execution=child.model_dump(),
                        )

                    if child.status == "open":
                        next_frontier.append(child)
                    last_error = result.error or result.proof_state

            frontier = next_frontier

        return LLMATPResult(
            solved=False,
            rounds=self.max_rounds,
            expanded_nodes=expanded_nodes,
            final_error=last_error,
            plan_history=history,
            tree=[item.model_dump() for item in explored],
            execution=explored[-1].model_dump() if explored else None,
        )
