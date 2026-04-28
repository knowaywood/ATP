"""LLM-guided ATP example with branching search."""

from __future__ import annotations

import asyncio
import os

from atp.lean_executor import LeanExecutor
from atp.llm_loop import LLMLeanProver
from atp.providers import build_chat_model


async def main() -> None:
    theorem = "forall (p q : Prop), Or p q -> Or q p"
    provider = os.getenv("ATP_PROVIDER", "gemini")
    model_name = os.getenv("ATP_MODEL", "gemini-2.5-flash")

    model = build_chat_model(
        provider=provider,
        model=model_name,
        temperature=0.0,
    )
    executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
    prover = LLMLeanProver(
        model=model,
        executor=executor,
        max_rounds=3,
        beam_width=3,
        max_tactics_per_candidate=4,
    )

    try:
        result = await prover.prove_async(theorem)
    except Exception as exc:
        print(f"Provider: {provider}")
        print(f"Model: {model_name}")
        print(f"LLM request failed: {type(exc).__name__}: {exc}")
        print(
            "Hint: if Gemini is unavailable or blocked, try setting "
            "`ATP_PROVIDER=tongyi` and `ATP_MODEL=qwen-max`."
        )
        raise
    finally:
        await executor.close_async()

    print("Solved:", result.solved)
    print("Rounds:", result.rounds)
    print("Expanded nodes:", result.expanded_nodes)
    print("Final error:", result.final_error)
    print("Plan history:")
    for index, batch in enumerate(result.plan_history, start=1):
        print(f"Round {index}: {batch.strategy}")
        for candidate in batch.candidates:
            print(f"* {candidate.rationale} [score={candidate.score}]")
            for tactic in candidate.tactics:
                print("-", tactic)


if __name__ == "__main__":
    asyncio.run(main())
