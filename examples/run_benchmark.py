"""Run a tiny ATP benchmark."""

from __future__ import annotations

import asyncio
import os

from atp.benchmark import BenchmarkCase, BenchmarkRunner
from atp.lean_executor import LeanExecutor
from atp.llm_loop import LLMLeanProver
from atp.providers import build_chat_model


async def main() -> None:
    provider = os.getenv("ATP_PROVIDER", "gemini")
    model_name = os.getenv("ATP_MODEL", "gemini-2.5-flash")
    cases = [
        BenchmarkCase(
            name="or_comm",
            theorem="forall (p q : Prop), Or p q -> Or q p",
        ),
        BenchmarkCase(
            name="and_left",
            theorem="forall (p q : Prop), And p q -> p",
        ),
    ]

    model = build_chat_model(
        provider=provider,
        model=model_name,
        temperature=0.0,
    )
    executor = LeanExecutor(project_path="./ATP", imports=["ATP"], timeout=60)
    prover = LLMLeanProver(model=model, executor=executor, max_rounds=3, beam_width=3)
    runner = BenchmarkRunner(prover=prover)

    try:
        report = await runner.run_async(cases)
    except Exception as exc:
        print(f"Provider: {provider}")
        print(f"Model: {model_name}")
        print(f"Benchmark LLM request failed: {type(exc).__name__}: {exc}")
        print(
            "Hint: if Gemini is unavailable or blocked, try setting "
            "`ATP_PROVIDER=tongyi` and `ATP_MODEL=qwen-max`."
        )
        raise
    finally:
        await executor.close_async()

    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
