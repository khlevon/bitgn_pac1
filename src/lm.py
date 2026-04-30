"""DSPy LM factory — one place to configure run vs optimize models."""

import dspy

import config


def make_run_lm(max_tokens: int = 128000) -> dspy.LM:
    """Build the LM used for running and evaluating tasks (self-hosted Qwen)."""
    kwargs: dict = {"max_tokens": max_tokens}
    if config.RUN_MODEL_API_BASE:
        kwargs["api_base"] = config.RUN_MODEL_API_BASE
    if config.RUN_MODEL_API_KEY:
        kwargs["api_key"] = config.RUN_MODEL_API_KEY
    return dspy.LM(config.RUN_MODEL_ID, **kwargs)


def make_optimize_lm(max_tokens: int = 128000) -> dspy.LM:
    """Build the LM used as the prompt_model during optimization (Anthropic Claude)."""
    return dspy.LM(config.OPTIMIZE_MODEL_ID, max_tokens=max_tokens)


def configure_run_lm() -> dspy.LM:
    """Configure DSPy globally with the run model and return it."""
    lm = make_run_lm()
    dspy.configure(lm=lm)
    return lm
