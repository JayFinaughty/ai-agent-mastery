"""
Custom Evaluators for Agent Evaluation

This module provides custom evaluators that extend pydantic-evals.
These are useful when built-in evaluators don't cover your use case.

Usage:
    from evals.evaluators import ContainsAny

    Case(
        name="safety_check",
        inputs={"query": "..."},
        evaluators=[ContainsAny(values=["can't", "cannot", "sorry"])]
    )
"""

from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason


@dataclass
class ContainsAny(Evaluator[dict, str, None]):
    """
    Check if output contains ANY of the specified values (OR logic).

    This is useful when you want to check for multiple possible phrasings.
    For example, LLMs phrase refusals differently:
      - "I can't help with that"
      - "I cannot assist"
      - "Sorry, but I'm unable to"

    With the built-in Contains evaluator, multiple checks use AND logic
    (all must pass). ContainsAny uses OR logic (any one match = pass).

    Example:
        ContainsAny(values=["can't", "cannot", "sorry", "unable"])

    Args:
        values: List of strings to search for (any match = pass)
        case_sensitive: Whether to perform case-sensitive matching (default: False)
    """

    values: list[str]
    case_sensitive: bool = False

    def evaluate(self, ctx: EvaluatorContext[dict, str, None]) -> EvaluationReason:
        output = str(ctx.output)
        check_output = output if self.case_sensitive else output.lower()

        for value in self.values:
            check_value = value if self.case_sensitive else value.lower()
            if check_value in check_output:
                return EvaluationReason(value=True, reason=f"Found '{value}'")

        return EvaluationReason(
            value=False, reason=f"None of {self.values} found in output"
        )
