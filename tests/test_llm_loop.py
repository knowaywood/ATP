import asyncio

from langchain_core.messages import AIMessage

from atp.benchmark import BenchmarkCase, BenchmarkRunner
from atp.lean_executor import LeanExecutor
from atp.llm_loop import LLMLeanProver, classify_execution


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.index = 0

    async def ainvoke(self, input, **kwargs):  # noqa: ANN001
        response = self.responses[self.index]
        self.index += 1
        return AIMessage(content=response)


def test_parse_plan_extracts_candidates():
    prover = LLMLeanProver(model=FakeModel([]), executor=LeanExecutor())
    batch = prover._parse_plan(
        """
        {
          "theorem": "forall (p q : Prop), Or p q -> Or q p",
          "strategy": "try intro then split by cases",
          "candidates": [
            {
              "theorem": "forall (p q : Prop), Or p q -> Or q p",
              "rationale": "direct cases proof",
              "score": 0.9,
              "tactics": ["intro p q h", "cases h"]
            }
          ]
        }
        """,
        "forall (p q : Prop), Or p q -> Or q p",
    )
    assert batch.strategy == "try intro then split by cases"
    assert batch.candidates[0].tactics == ["intro p q h", "cases h"]


def test_classify_execution_rewards_small_goal_count():
    async def scenario():
        executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
        try:
            root = await executor.start_goal_async("forall (p q : Prop), Or p q -> Or q p")
            result = await executor.run_tactic_async(root.node, "intro p q h")
            feedback = classify_execution(result)
            assert feedback.status == "open"
            assert feedback.score_delta >= 0.0
        finally:
            await executor.close_async()

    asyncio.run(scenario())


def test_branching_llm_loop_solves_simple_theorem():
    async def scenario():
        responses = [
            """
            {
              "theorem": "forall (p q : Prop), Or p q -> Or q p",
              "strategy": "branch over direct proof styles",
              "candidates": [
                {
                  "theorem": "forall (p q : Prop), Or p q -> Or q p",
                  "rationale": "bad simp branch",
                  "score": 0.2,
                  "tactics": ["simp [MissingLemma]"]
                },
                {
                  "theorem": "forall (p q : Prop), Or p q -> Or q p",
                  "rationale": "direct intro/cases branch",
                  "score": 0.9,
                  "tactics": [
                    "intro p q h",
                    "cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq"
                  ]
                }
              ]
            }
            """
        ]
        prover = LLMLeanProver(
            model=FakeModel(responses),
            executor=LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60),
            max_rounds=1,
            beam_width=2,
        )
        try:
            result = await prover.prove_async("forall (p q : Prop), Or p q -> Or q p")
            assert result.solved is True
            assert result.successful_node_id is not None
            assert result.expanded_nodes == 1
            assert len(result.tree) >= 2
        finally:
            await prover.executor.close_async()

    asyncio.run(scenario())


def test_benchmark_runner_aggregates_results():
    async def scenario():
        responses = [
            """
            {
              "theorem": "forall (p q : Prop), And p q -> p",
              "strategy": "one-step intro then exact left hypothesis",
              "candidates": [
                {
                  "theorem": "forall (p q : Prop), And p q -> p",
                  "rationale": "intro then exact hypothesis",
                  "score": 1.0,
                  "tactics": ["intro p q h", "exact h.left"]
                }
              ]
            }
            """
        ]
        prover = LLMLeanProver(
            model=FakeModel(responses),
            executor=LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60),
            max_rounds=1,
        )
        runner = BenchmarkRunner(prover)
        try:
            report = await runner.run_async(
                [BenchmarkCase(name="and_left", theorem="forall (p q : Prop), And p q -> p")]
            )
            assert report.total == 1
            assert report.solved == 1
            assert report.pass_at_k == 1.0
        finally:
            await prover.executor.close_async()

    asyncio.run(scenario())
