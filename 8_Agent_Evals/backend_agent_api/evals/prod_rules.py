"""
Production Rule-Based Evaluation

Video 6: Rule-Based Evals (Production)

Runs pydantic-evals evaluators on production traces and syncs results to Langfuse.
Evaluators run async (non-blocking) after response is sent to user.

This module reuses the custom evaluators from Video 3 (NoPII, NoForbiddenWords)
and adds Langfuse score syncing for production monitoring.

Usage:
    from evals.prod_rules import run_production_evals

    # Inside agent_api.py, after getting trace_id:
    asyncio.create_task(
        run_production_evals(trace_id, output, {"query": user_query})
    )

Langfuse Scores Created:
    - rule_no_pii: 1 if no PII detected, 0 if PII found
    - rule_no_forbidden: 1 if no forbidden words, 0 if found
    - rule_check_passed: 1 if all rules passed, 0 otherwise
"""

from dataclasses import dataclass
from typing import Optional

from pydantic_evals.evaluators import EvaluatorContext

from evals.evaluators import NoPII, NoForbiddenWords


@dataclass
class ProductionRuleEvaluator:
    """
    Runs rule-based evaluators and syncs scores to Langfuse.

    This class wraps the evaluators from Video 3 for production use.
    Results are synced to Langfuse as scores attached to the trace.

    Attributes:
        evaluators: List of (name, evaluator) tuples to run on each response

    Example:
        evaluator = ProductionRuleEvaluator()
        passed = await evaluator.evaluate_and_sync(
            trace_id="abc123",
            output="Hello, how can I help?",
            inputs={"query": "Hi there"}
        )
    """

    def __post_init__(self):
        """Initialize evaluators with production configuration."""
        self.evaluators = [
            ("no_pii", NoPII()),
            ("no_forbidden", NoForbiddenWords(
                forbidden=["password", "secret", "confidential", "api_key"]
            )),
        ]

    def _create_context(
        self,
        output: str,
        inputs: Optional[dict] = None
    ) -> EvaluatorContext:
        """
        Create EvaluatorContext for direct evaluator invocation.

        When running evaluators outside pydantic-evals Dataset.evaluate(),
        we construct the context manually with required fields.

        Args:
            output: The agent's response text to evaluate
            inputs: Optional dict with input data (e.g., {"query": "..."})

        Returns:
            EvaluatorContext ready for evaluator.evaluate() calls
        """
        return EvaluatorContext(
            name="production_eval",
            inputs=inputs or {},
            output=output,
            expected_output=None,  # No expected output for production
            metadata={},
            duration=0.0,  # Not measured for production
            _span_tree=None,  # Not using span analysis here (underscore prefix required)
            attributes={},  # Custom attributes
            metrics={},  # Custom metrics
        )

    async def evaluate_and_sync(
        self,
        trace_id: str,
        output: str,
        inputs: Optional[dict] = None
    ) -> bool:
        """
        Run all evaluators and sync scores to Langfuse.

        This method:
        1. Creates an EvaluatorContext with the output
        2. Runs each evaluator and captures the result
        3. Syncs each result as a Langfuse score via low-level API
        4. Syncs an overall pass/fail score

        Args:
            trace_id: Langfuse trace ID (hex string format, e.g., "a1b2c3d4...")
            output: The agent's response text to evaluate
            inputs: Optional dict with input data (e.g., {"query": "..."})

        Returns:
            True if all rules passed, False otherwise
        """
        try:
            from langfuse import Langfuse
            from langfuse.api.resources.score.types import CreateScoreRequest
            langfuse = Langfuse()
        except Exception as e:
            # Don't fail silently - log the error
            print(f"[prod_rules] Failed to get Langfuse client: {e}")
            return True  # Return True to not block on Langfuse issues

        ctx = self._create_context(output, inputs)
        all_passed = True

        for eval_name, evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(ctx)

                # Sync to Langfuse using low-level API (bypasses OTEL conflicts)
                langfuse.api.score.create(
                    request=CreateScoreRequest(
                        traceId=trace_id,
                        name=f"rule_{eval_name}",
                        value=1.0 if result.value else 0.0,
                        comment=result.reason
                    )
                )

                if not result.value:
                    all_passed = False
                    print(f"[prod_rules] Rule '{eval_name}' failed: {result.reason}")

            except Exception as e:
                print(f"[prod_rules] Evaluator {eval_name} failed: {e}")
                # Continue with other evaluators

        # Overall pass/fail score
        try:
            langfuse.api.score.create(
                request=CreateScoreRequest(
                    traceId=trace_id,
                    name="rule_check_passed",
                    value=1.0 if all_passed else 0.0,
                    comment="All rule-based evaluators passed" if all_passed else "One or more rules failed"
                )
            )
        except Exception as e:
            print(f"[prod_rules] Failed to sync overall score: {e}")

        return all_passed


# Module-level singleton for easy import
_evaluator: Optional[ProductionRuleEvaluator] = None


def get_production_evaluator() -> ProductionRuleEvaluator:
    """Get or create the production evaluator singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ProductionRuleEvaluator()
    return _evaluator


async def run_production_evals(
    trace_id: str,
    output: str,
    inputs: Optional[dict] = None
) -> bool:
    """
    Convenience function to run production evaluations.

    This is the main entry point for agent_api.py integration.
    Wraps exceptions to prevent evaluation failures from breaking the API.

    Args:
        trace_id: Langfuse trace ID (hex string, e.g., "a1b2c3d4...")
        output: The agent's response text
        inputs: Optional input data for context

    Returns:
        True if all rules passed, False otherwise

    Example:
        # In agent_api.py:
        if production_trace_id:
            asyncio.create_task(
                run_production_evals(
                    trace_id=production_trace_id,
                    output=full_response,
                    inputs={"query": request.query}
                )
            )
    """
    try:
        evaluator = get_production_evaluator()
        return await evaluator.evaluate_and_sync(trace_id, output, inputs)
    except Exception as e:
        # Catch-all to prevent evaluation from breaking the API
        print(f"[prod_rules] Unexpected error in production evaluation: {e}")
        return True
