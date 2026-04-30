"""BitGN harness connection and trial lifecycle management."""

import asyncio
import copy
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import dspy
from bitgn.harness_connect import HarnessServiceClient
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
)
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import AnswerRequest, Outcome
from connectrpc.errors import ConnectError

import config
from agent import OUTCOMES, PAC1Agent
from lm import configure_run_lm
from tools import grounding_snapshot, make_tools

try:
    from dspy.utils.exceptions import AdapterParseError
except ImportError:
    AdapterParseError = Exception  # type: ignore[misc,assignment]

_C = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "blue": "\x1b[34m",
    "clr": "\x1b[0m",
}

_OUTCOME_ENUM = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}


@dataclass
class TrialResult:
    task_id: str
    score: float
    instruction: str = ""
    grounding: str = ""
    detail: list[str] = field(default_factory=list)
    prediction: dspy.Prediction | None = None


def _submit_answer(vm: PcmRuntimeClientSync, prediction: dspy.Prediction) -> None:
    outcome_str = (
        prediction.outcome if prediction.outcome in OUTCOMES else "OUTCOME_ERR_INTERNAL"
    )
    refs = prediction.refs if isinstance(prediction.refs, list) else []
    vm.answer(
        AnswerRequest(
            message=prediction.summary,
            outcome=_OUTCOME_ENUM[outcome_str],
            refs=refs,
        )
    )


def make_agent(program_path: str | None = None, max_iters: int = 30) -> PAC1Agent:
    """Create a PAC1Agent with no-op tools (swapped per trial) and optionally load saved state."""
    agent = PAC1Agent(tools=make_tools(MagicMock()), max_iters=max_iters)
    if program_path and Path(program_path).exists():
        agent.load(program_path)
        print(f"Loaded optimized agent from {program_path}")
    return agent


def run_trial(
    agent: PAC1Agent, harness_url: str, instruction: str, buf: list[str] | None = None
) -> tuple[dspy.Prediction, str]:
    """Bind fresh tools to *agent*, run one trial, submit the answer.

    Returns (prediction, grounding) so callers can store both for optimization.
    If *buf* is provided, all log lines are appended there instead of printed
    immediately — use this when running trials concurrently to avoid interleaving.
    """
    _p = buf.append if buf is not None else print

    vm = PcmRuntimeClientSync(harness_url)
    agent.update_tools(make_tools(vm))

    grounding = grounding_snapshot(vm)
    _p(f"{_C['green']}GROUNDING{_C['clr']}:\n{grounding}\n")

    try:
        prediction = agent(grounding=grounding, instruction=instruction)
    except AdapterParseError as exc:
        _p(f"{_C['red']}parse error:{_C['clr']} {exc}")
        prediction = dspy.Prediction(
            outcome="OUTCOME_ERR_INTERNAL", summary=str(exc)[:200], refs=[]
        )

    _p(f"{_C['blue']}outcome:{_C['clr']} {prediction.outcome}")
    _p(f"{_C['blue']}summary:{_C['clr']} {prediction.summary}")

    _submit_answer(vm, prediction)
    return prediction, grounding


async def run_benchmark(
    task_filter: list[str] | None = None,
    program_path: str | None = None,
    max_iters: int = 30,
    batch_size: int = 8,
) -> list[TrialResult]:
    """Run all benchmark tasks with the configured run model and return scored results."""
    configure_run_lm()
    agent = make_agent(program_path=program_path, max_iters=max_iters)
    return await _run_tasks(agent, batch_size, task_filter)


async def _run_tasks(
    agent: PAC1Agent,
    batch_size: int,
    task_filter: list[str] | None,
) -> list[TrialResult]:
    """Start a tracked run, iterate over trials in parallel batches, then submit."""
    results: list[TrialResult] = []

    async def _run_one(trial) -> tuple:
        buf: list[str] = []
        buf.append(f"\n{'=' * 30} {trial.task_id} {'=' * 30}")
        buf.append(f"{_C['blue']}{trial.instruction}{_C['clr']}\n{'-' * 80}")
        prediction, grounding = None, ""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            trial_agent = copy.deepcopy(agent)
            try:
                prediction, grounding = await asyncio.to_thread(
                    run_trial, trial_agent, trial.harness_url, trial.instruction, buf
                )
                break
            except ConnectError as exc:
                buf.append(
                    f"{_C['red']}connect error (attempt {attempt}/{max_retries}):{_C['clr']} {exc.code}: {exc.message}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)
            except Exception as exc:
                buf.append(f"{_C['red']}agent error:{_C['clr']} {exc}")
                break
        # Return buf unflushed — the main thread appends the score then prints the full group
        return trial, prediction, grounding, buf

    try:
        client = HarnessServiceClient(config.BITGN_URL)
        print("Connecting to BitGN:", await client.status(StatusRequest()))

        res = await client.get_benchmark(
            GetBenchmarkRequest(benchmark_id=config.BENCHMARK_ID)
        )
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{_C['green']}{res.description}{_C['clr']}"
        )

        run = await client.start_run(
            StartRunRequest(
                name=f"khlevon: PAC1 Agent (dspy: {config.RUN_MODEL_ID})",
                benchmark_id=config.BENCHMARK_ID,
                api_key=config.BITGN_API_KEY,
            )
        )

        try:
            all_ids = list(run.trial_ids)
            for batch_start in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[batch_start : batch_start + batch_size]

                # Start trials for this batch sequentially
                batch_trials = []
                for trial_id in batch_ids:
                    trial = await client.start_trial(
                        StartTrialRequest(trial_id=trial_id)
                    )
                    if task_filter and trial.task_id not in task_filter:
                        continue
                    batch_trials.append(trial)

                if not batch_trials:
                    continue

                # Run this batch in parallel
                batch_outcomes = await asyncio.gather(
                    *[_run_one(t) for t in batch_trials]
                )

                # End trials and collect scores sequentially after the batch completes
                for trial, prediction, grounding, buf in batch_outcomes:
                    scored = await client.end_trial(
                        EndTrialRequest(trial_id=trial.trial_id)
                    )
                    if scored.score >= 0:
                        style = _C["green"] if scored.score == 1 else _C["red"]
                        detail_text = textwrap.indent(
                            "\n".join(scored.score_detail), "  "
                        )
                        buf.append(
                            f"\n{style}score: {scored.score:.2f}\n{detail_text}{_C['clr']}"
                        )
                        results.append(
                            TrialResult(
                                task_id=trial.task_id,
                                score=scored.score,
                                instruction=trial.instruction,
                                grounding=grounding,
                                detail=list(scored.score_detail),
                                prediction=prediction,
                            )
                        )
                    print("\n".join(buf))
        finally:
            await client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        print(f"{_C['red']}{exc.code}: {exc.message}{_C['clr']}")
    except KeyboardInterrupt:
        print(f"{_C['red']}interrupted{_C['clr']}")

    return results


def results_to_dspy_examples(results: list[TrialResult]) -> list[dspy.Example]:
    """Convert ``TrialResult`` objects into ``dspy.Example`` instances with scores.

    Useful for feeding eval results into an optimizer's trainset.
    """
    examples = []
    for r in results:
        if r.prediction is None:
            continue
        ex = dspy.Example(
            grounding=r.grounding,
            instruction=r.instruction,
            score=r.score,
        ).with_inputs("grounding", "instruction")
        examples.append(ex)
    return examples


def print_summary(results: list[TrialResult]) -> None:
    if not results:
        return
    for r in results:
        style = _C["green"] if r.score == 1 else _C["red"]
        print(f"{r.task_id}: {style}{r.score:.2f}{_C['clr']}")
    total = sum(r.score for r in results) / len(results) * 100
    print(f"FINAL: {total:.2f}%")
