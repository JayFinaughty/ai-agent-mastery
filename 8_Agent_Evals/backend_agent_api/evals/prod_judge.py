"""
Production LLM Judge Evaluation

Video 7: LLM-as-Judge (Production)

AI-powered quality scoring on production traces using pydantic-ai.
Follows the same async pattern as prod_rules.py (Video 6).

Model: Uses GPT-5-mini for cost efficiency (same as Video 4 LLMJudge).

Usage:
    from evals.prod_judge import run_production_judge

    # Inside agent_api.py, after getting trace_id:
    asyncio.create_task(
        run_production_judge(trace_id, output, {"query": user_query})
    )

Langfuse Scores Created:
    - llm_judge_score: 0.0-1.0 quality score
    - llm_judge_passed: 1 if score >= 0.7, 0 otherwise
"""

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class JudgeResult(BaseModel):
    """Structured output from the judge."""

    score: float  # 0.0 to 1.0
    passed: bool  # True if score >= 0.7
    reason: str   # Brief explanation (1-2 sentences)


def _create_judge_model() -> OpenAIChatModel:
    """
    Create the judge model with the same configuration as the agent.

    Uses LLM_BASE_URL and LLM_API_KEY environment variables,
    matching the pattern from run_evals.py (Video 4).
    """
    return OpenAIChatModel(
        os.getenv("LLM_CHOICE", "gpt-4o-mini"),  # Use same model choice as agent
        provider=OpenAIProvider(
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_API_KEY"),
        ),
    )


def _create_judge_agent() -> Agent:
    """Create the judge agent with quality evaluation rubric."""
    return Agent(
        model=_create_judge_model(),
        output_type=JudgeResult,
        system_prompt="""You are an AI quality evaluator.

Evaluate the agent's response for:
1. Relevance: Does it address the user's question?
2. Helpfulness: Is it actionable and useful?
3. Accuracy: Does it appear factually correct?

Return:
- score: 0.0 to 1.0 (1.0 = excellent)
- passed: true if score >= 0.7
- reason: Brief explanation (1-2 sentences)

Be concise in your reasoning.""",
    )


@dataclass
class ProductionJudgeEvaluator:
    """
    Runs LLM judge and syncs scores to Langfuse.

    Follows the same pattern as ProductionRuleEvaluator from Video 6.
    Uses pydantic-ai Agent instead of pydantic-evals evaluators for
    more control over the evaluation prompt and output format.

    Attributes:
        judge_agent: The pydantic-ai agent used for evaluation

    Example:
        evaluator = ProductionJudgeEvaluator()
        result = await evaluator.evaluate_and_sync(
            trace_id="abc123",
            output="Hello, how can I help?",
            inputs={"query": "Hi there"}
        )
    """

    def __post_init__(self):
        """Initialize the judge agent."""
        self.judge_agent = _create_judge_agent()

    async def evaluate_and_sync(
        self,
        trace_id: str,
        output: str,
        inputs: Optional[dict] = None,
    ) -> Optional[JudgeResult]:
        """
        Run LLM judge and sync scores to Langfuse.

        This method:
        1. Builds an evaluation prompt from the input/output
        2. Runs the judge agent to get a quality score
        3. Syncs scores to Langfuse via low-level API

        Args:
            trace_id: Langfuse trace ID (hex string format, e.g., "a1b2c3d4...")
            output: The agent's response text to evaluate
            inputs: Optional dict with input data (e.g., {"query": "..."})

        Returns:
            JudgeResult if evaluation succeeded, None otherwise
        """
        try:
            from langfuse import Langfuse
            from langfuse.api.resources.score.types import CreateScoreRequest
            langfuse = Langfuse()
        except Exception as e:
            # Don't fail silently - log the error
            print(f"[prod_judge] Failed to get Langfuse client: {e}")
            return None

        # Build evaluation prompt
        query = inputs.get("query", "Unknown") if inputs else "Unknown"
        prompt = f"""User Query: {query}

Agent Response: {output}

Evaluate this response."""

        try:
            # Run the judge
            result = await self.judge_agent.run(prompt)
            judge_output = result.output

            # Sync scores to Langfuse using low-level API (bypasses OTEL conflicts)
            langfuse.api.score.create(
                request=CreateScoreRequest(
                    traceId=trace_id,
                    name="llm_judge_score",
                    value=judge_output.score,
                    comment=judge_output.reason,
                )
            )

            langfuse.api.score.create(
                request=CreateScoreRequest(
                    traceId=trace_id,
                    name="llm_judge_passed",
                    value=1.0 if judge_output.passed else 0.0,
                    comment="Score >= 0.7" if judge_output.passed else "Score < 0.7",
                )
            )

            if not judge_output.passed:
                print(
                    f"[prod_judge] Low quality detected: {judge_output.score:.2f} - {judge_output.reason}"
                )

            return judge_output

        except Exception as e:
            print(f"[prod_judge] Evaluation failed: {e}")
            return None


# Module-level singleton for easy import
_evaluator: Optional[ProductionJudgeEvaluator] = None


def get_production_judge() -> ProductionJudgeEvaluator:
    """Get or create the production judge singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ProductionJudgeEvaluator()
    return _evaluator


async def run_production_judge(
    trace_id: str,
    output: str,
    inputs: Optional[dict] = None,
) -> Optional[JudgeResult]:
    """
    Convenience function to run production LLM judge.

    This is the main entry point for agent_api.py integration.
    Wraps exceptions to prevent evaluation failures from breaking the API.

    Args:
        trace_id: Langfuse trace ID (hex string, e.g., "a1b2c3d4...")
        output: The agent's response text
        inputs: Optional input data for context

    Returns:
        JudgeResult if evaluation succeeded, None otherwise

    Example:
        # In agent_api.py:
        if production_trace_id and random.random() < JUDGE_SAMPLE_RATE:
            asyncio.create_task(
                run_production_judge(
                    trace_id=production_trace_id,
                    output=full_response,
                    inputs={"query": request.query}
                )
            )
    """
    try:
        evaluator = get_production_judge()
        return await evaluator.evaluate_and_sync(trace_id, output, inputs)
    except Exception as e:
        # Catch-all to prevent evaluation from breaking the API
        print(f"[prod_judge] Unexpected error in production evaluation: {e}")
        return None
