"""Lean execution loop backed by lean-interact.

The synchronous API is primary. Async helpers are thin wrappers around the sync
implementation so the higher-level ATP loop can stay unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lean_interact import AutoLeanServer, Command, LeanREPLConfig, LocalProject, ProofStep
from lean_interact.interface import CommandResponse, LeanError, ProofStepResponse, Sorry

from atp.config import ProofCandidate, SearchNode


def _render_messages(messages) -> list[str]:  # noqa: ANN001
    """Render Lean message objects into strings."""
    rendered: list[str] = []
    for message in messages:
        severity = getattr(message, "severity", "info")
        data = getattr(message, "data", str(message))
        rendered.append(f"{severity}: {data}")
    return rendered


def _find_proof_state(response: CommandResponse) -> Sorry | None:
    """Return the first sorry carrying a proof state."""
    for sorry in response.sorries:
        if sorry.proof_state is not None:
            return sorry
    return None


@dataclass
class LeanExecutionResult:
    """Standardized result for one Lean execution step."""

    node: SearchNode
    solved: bool
    goal_count: int
    state_id: int | None
    proof_state: str
    messages: list[str]
    error: str | None = None


class LeanExecutor:
    """Execute Lean tactics and project the results onto SearchNode."""

    def __init__(
        self,
        *,
        project_path: str = "./ATP",
        imports: list[str] | None = None,
        timeout: int = 60,
    ) -> None:
        self.project_path = project_path
        self.imports = imports or ["ATP"]
        self.timeout = timeout
        self._server: AutoLeanServer | None = None

    def start(self) -> None:
        """Start the Lean REPL server if needed."""
        if self._server is None:
            project = LocalProject(directory=self.project_path, auto_build=False)
            config = LeanREPLConfig(project=project)
            self._server = AutoLeanServer(config)

    async def start_async(self) -> None:
        """Async wrapper for starting the server."""
        await asyncio.to_thread(self.start)

    def close(self) -> None:
        """Close the Lean REPL server."""
        if self._server is not None:
            self._server.kill()
            self._server = None

    async def close_async(self) -> None:
        """Async wrapper for closing the server."""
        await asyncio.to_thread(self.close)

    def _server_or_raise(self) -> AutoLeanServer:
        self.start()
        assert self._server is not None
        return self._server

    def _goal_command(self, theorem: str) -> str:
        imports = "\n".join(f"import {module}" for module in self.imports)
        return f"{imports}\nexample : {theorem} := by\n  sorry"

    def start_goal(
        self,
        theorem: str,
        *,
        node_id: str = "root",
        candidate: ProofCandidate | None = None,
    ) -> LeanExecutionResult:
        """Create a root goal from a theorem statement."""
        server = self._server_or_raise()
        response = server.run(
            Command(
                cmd=self._goal_command(theorem),
                rootGoals=True,
            ),
            timeout=self.timeout,
            add_to_session_cache=True,
        )

        if isinstance(response, LeanError):
            node = SearchNode(
                node_id=node_id,
                goal=theorem,
                proof_state=theorem,
                status="dead_end",
                error=response.message,
                messages=[response.message],
                candidate=candidate,
            )
            return LeanExecutionResult(
                node=node,
                solved=False,
                goal_count=-1,
                state_id=None,
                proof_state=theorem,
                messages=node.messages,
                error=response.message,
            )

        proof_sorry = _find_proof_state(response)
        if proof_sorry is None:
            node = SearchNode(
                node_id=node_id,
                goal=theorem,
                proof_state="no goals",
                status="proved",
                lean_state_id=None,
                messages=_render_messages(response.messages),
                candidate=candidate,
            )
            return LeanExecutionResult(
                node=node,
                solved=True,
                goal_count=0,
                state_id=None,
                proof_state="no goals",
                messages=node.messages,
            )

        goal_text = proof_sorry.goal
        node = SearchNode(
            node_id=node_id,
            goal=goal_text,
            proof_state=goal_text,
            lean_state_id=proof_sorry.proof_state,
            status="open",
            messages=_render_messages(response.messages),
            candidate=candidate,
        )
        return LeanExecutionResult(
            node=node,
            solved=False,
            goal_count=1,
            state_id=proof_sorry.proof_state,
            proof_state=goal_text,
            messages=node.messages,
        )

    async def start_goal_async(
        self,
        theorem: str,
        *,
        node_id: str = "root",
        candidate: ProofCandidate | None = None,
    ) -> LeanExecutionResult:
        """Async wrapper for creating a root goal."""
        return await asyncio.to_thread(
            self.start_goal,
            theorem,
            node_id=node_id,
            candidate=candidate,
        )

    def run_tactic(
        self,
        node: SearchNode,
        tactic: str,
    ) -> LeanExecutionResult:
        """Run one tactic from an existing node."""
        if node.lean_state_id is None:
            raise ValueError("SearchNode is missing lean_state_id.")

        server = self._server_or_raise()
        response = server.run(
            ProofStep(proofState=node.lean_state_id, tactic=tactic),
            timeout=self.timeout,
            add_to_session_cache=True,
        )

        if isinstance(response, LeanError):
            failed = node.model_copy(deep=True)
            failed.tactic_history.append(tactic)
            failed.error = response.message
            failed.status = "dead_end"
            failed.messages = [response.message]
            if failed.candidate is not None:
                failed.candidate.tactic_script = "\n".join(failed.tactic_history)
                failed.candidate.status = "failed"
                failed.candidate.notes.append(response.message)
            return LeanExecutionResult(
                node=failed,
                solved=False,
                goal_count=-1,
                state_id=node.lean_state_id,
                proof_state=node.proof_state or node.goal,
                messages=failed.messages,
                error=response.message,
            )

        assert isinstance(response, ProofStepResponse)
        updated = node.model_copy(deep=True)
        updated.tactic_history.append(tactic)
        updated.lean_state_id = response.proof_state
        updated.messages = _render_messages(response.messages)
        updated.error = None

        solved = response.proof_status.startswith("Completed") or not response.goals
        updated.proof_state = "no goals" if solved else "\n\n".join(response.goals)
        updated.goal = node.goal if solved else response.goals[0]
        updated.status = "proved" if solved else "open"

        if updated.candidate is not None:
            updated.candidate.tactic_script = "\n".join(updated.tactic_history)
            updated.candidate.status = "proved" if solved else "draft"

        response_errors = [msg.data for msg in response.get_errors()]
        if response_errors:
            updated.error = "\n".join(response_errors)
            updated.status = "dead_end"
            if updated.candidate is not None:
                updated.candidate.status = "failed"
                updated.candidate.notes.extend(response_errors)

        return LeanExecutionResult(
            node=updated,
            solved=solved and not response_errors,
            goal_count=0 if solved else len(response.goals),
            state_id=response.proof_state,
            proof_state=updated.proof_state or "",
            messages=updated.messages,
            error=updated.error,
        )

    async def run_tactic_async(
        self,
        node: SearchNode,
        tactic: str,
    ) -> LeanExecutionResult:
        """Async wrapper for one tactic step."""
        return await asyncio.to_thread(self.run_tactic, node, tactic)

    def run_script(
        self,
        theorem: str,
        tactics: list[str],
        *,
        node_id: str = "root",
        candidate: ProofCandidate | None = None,
    ) -> LeanExecutionResult:
        """Run a theorem plus tactic sequence to completion or failure."""
        result = self.start_goal(theorem, node_id=node_id, candidate=candidate)
        current = result.node
        for tactic in tactics:
            result = self.run_tactic(current, tactic)
            current = result.node
            if result.solved or result.error is not None:
                break
        return result

    async def run_script_async(
        self,
        theorem: str,
        tactics: list[str],
        *,
        node_id: str = "root",
        candidate: ProofCandidate | None = None,
    ) -> LeanExecutionResult:
        """Async wrapper for a tactic script."""
        return await asyncio.to_thread(
            self.run_script,
            theorem,
            tactics,
            node_id=node_id,
            candidate=candidate,
        )


async def evolve_node_with_tactic(
    executor: LeanExecutor,
    node: SearchNode,
    tactic: str,
) -> LeanExecutionResult:
    """Evolve a SearchNode with one Lean tactic."""
    return await executor.run_tactic_async(node, tactic)


def execute_candidate(
    executor: LeanExecutor,
    candidate: ProofCandidate,
    tactics: list[str],
    *,
    node_id: str = "root",
) -> LeanExecutionResult:
    """Run a proof candidate through Lean."""
    return executor.run_script(
        candidate.theorem_statement,
        tactics,
        node_id=node_id,
        candidate=candidate,
    )
