"""Optimization entrypoints for the PAC1 agent.

Model split
-----------
- **task_model** (Qwen, self-hosted): runs the actual PAC1 tasks during training.
  This is the model we're optimizing — its prompts get better after compilation.
- **prompt_model** (Anthropic Claude): generates candidate instruction text in MIPROv2,
  and acts as the teacher in BootstrapFewShot.  A stronger model here yields better
  prompt proposals without changing what runs in production.

Workflow
--------
1. Run ``make optimize`` (or ``uv run python src/optimize.py``) to:
   a. Evaluate the current agent on the benchmark to collect training scores.
   b. Run BootstrapFewShot → MIPROv2 on those examples.
   c. Save the optimized state to ``OPTIMIZED_AGENT_PATH``.
2. Subsequent ``make run`` / ``make eval`` calls load that saved state automatically.

CLI usage
---------
    uv run python src/optimize.py            # optimize on all tasks
    uv run python src/optimize.py t01 t03    # optimize on a subset of tasks
"""
from __future__ import annotations

import sys
from pathlib import Path

import dspy

import config
from agent import PAC1Agent
from harness import make_agent, print_summary, results_to_dspy_examples, run_benchmark
from lm import make_optimize_lm, make_run_lm


def score_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Return the server-side score pre-attached to *example* by the training run."""
    return float(getattr(example, "score", 0.0))


# ---------------------------------------------------------------------------
# Bootstrap (fast first pass)
# ---------------------------------------------------------------------------

def bootstrap_optimize(
    agent: PAC1Agent,
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 8,
) -> PAC1Agent:
    """Optimize *agent* with ``BootstrapFewShot``.

    The optimize model (Claude) acts as teacher, generating demonstrations that
    are distilled into the agent's prompts.  The task model (Qwen) is what runs
    in production and is what the prompts target.
    """
    teacher_lm = make_optimize_lm()
    optimizer = dspy.BootstrapFewShot(
        metric=score_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_settings={"lm": teacher_lm},
    )
    return optimizer.compile(agent, trainset=trainset)


# ---------------------------------------------------------------------------
# MIPROv2 (instruction + demo optimization)
# ---------------------------------------------------------------------------

def mipro_optimize(
    agent: PAC1Agent,
    trainset: list[dspy.Example],
    auto: str = "light",
    max_bootstrapped_demos: int = 2,
    max_labeled_demos: int = 4,
    num_threads: int = 1,
    verbose: bool = True,
) -> PAC1Agent:
    """Optimize *agent* with ``MIPROv2``.

    Claude (``prompt_model``) generates instruction candidates; Qwen (``task_model``)
    evaluates them on the training tasks.  This lets a stronger model craft better
    prompts while keeping the production model as the optimization target.
    """
    prompt_lm = make_optimize_lm()
    task_lm = make_run_lm()

    optimizer = dspy.MIPROv2(
        metric=score_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=auto,  # type: ignore[arg-type]
        num_threads=num_threads,
        verbose=verbose,
    )
    return optimizer.compile(
        agent,
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Collect training data → bootstrap → MIPROv2 → save."""
    task_filter = sys.argv[1:] or None
    out_path = config.OPTIMIZED_AGENT_PATH

    # Step 1: collect training scores (also configures global run LM via run_benchmark)
    print("── Step 1/3: collecting training scores ──────────────────────────────")
    train_results = run_benchmark(task_filter=task_filter, program_path=None)
    print_summary(train_results)

    trainset = results_to_dspy_examples(train_results)
    if not trainset:
        print("No scored examples collected — cannot optimize. Exiting.")
        sys.exit(1)

    successful = [ex for ex in trainset if ex.score >= 1.0]
    print(f"\nTraining set: {len(trainset)} examples, {len(successful)} perfect scores.")

    # Fresh agent for the optimization steps (run_benchmark already configured the global LM)
    agent = make_agent(program_path=None)

    # Step 2: BootstrapFewShot (fast, uses Claude as teacher)
    print("\n── Step 2/3: BootstrapFewShot ────────────────────────────────────────")
    agent = bootstrap_optimize(agent, trainset)

    # Step 3: MIPROv2 (refines instruction text; Claude→prompt, Qwen→task)
    # MIPROv2 runs its own bootstrapping internally, so it needs an uncompiled agent.
    print("\n── Step 3/3: MIPROv2 ─────────────────────────────────────────────────")
    agent = mipro_optimize(make_agent(program_path=None), trainset)

    # Save
    agent.save(out_path)
    print(f"\nOptimized agent saved to {out_path}")
    print(f"Run 'make run' to use it (loads {out_path} automatically).")


if __name__ == "__main__":
    main()
