# Strategy: LLM-as-Judge (Production)

> **Video 7** | **Tag:** `module-8-06-llm-judge-prod` | **Phase:** Production

## Overview

**What it is**: AI-powered quality scoring on production traces using Langfuse's built-in evaluators or a custom pydantic-ai judge.

**Philosophy**: You can't manually review every response. Use LLM judges to score at scale, then focus human attention on low-scoring traces.

**Building on Video 4**: You used `LLMJudge` in your golden dataset. Now we run AI evaluation on real production data.

**Building on Video 6**: Like `prod_rules.py`, we run evaluation async (non-blocking) and sync scores to Langfuse using the low-level API.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM-AS-JUDGE (PRODUCTION)                           │
│                                                                         │
│   Production Trace              Judge                  Langfuse         │
│   ────────────────              ─────                  ────────         │
│                                                                         │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐      │
│   │   Agent     │         │  pydantic-ai│         │   Scores    │      │
│   │   Response  │────────►│  Judge      │────────►│   on Trace  │      │
│   │   + Trace   │  async  │  (GPT-5)    │  sync   │             │      │
│   └─────────────┘         └─────────────┘         └─────────────┘      │
│                                                                         │
│   Option 1: Langfuse built-in evaluators (no code, UI config)          │
│   Option 2: Custom pydantic-ai judge (more control, code-based)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What You'll Learn in This Video

1. How to use Langfuse's built-in LLM evaluators (no code)
2. How to build a custom judge with pydantic-ai (like Video 4, but for production)
3. How to run evaluation async (non-blocking) using the pattern from Video 6
4. How to sample traces for cost control
5. How to monitor quality trends in Langfuse dashboard

## Difference from Local (Video 4)

| Aspect | Local (Video 4) | Production (Video 7) |
|--------|-----------------|----------------------|
| **Evaluator** | pydantic-evals `LLMJudge` | pydantic-ai `Agent` with structured output |
| **Data** | Golden dataset (10 cases) | Production traces (1000s) |
| **Output** | Terminal report | Langfuse dashboard scores |
| **Execution** | Sync (blocking) | Async (fire-and-forget) |
| **Cost** | Pay per eval run | Sample to control costs |
| **Model** | `openai:gpt-5-mini` | `openai:gpt-5-mini` (same!) |

---

## Option 1: Langfuse Built-in Evaluators (Recommended Start)

Langfuse provides pre-built evaluators — no code required.

### Available Evaluator Templates

| Evaluator | What it Checks |
|-----------|----------------|
| **Helpfulness** | Is the response useful? |
| **Relevance** | Does it address the query? |
| **Toxicity** | Is there harmful content? |
| **Correctness** | Is information accurate? |
| **Conciseness** | Is it appropriately brief? |
| **Hallucination** | (RAG) Are claims grounded in context? |

### Setting Up in Langfuse UI

1. Go to **Langfuse → Evaluators → "+ Set up Evaluator"**
2. Select a template (e.g., "Helpfulness")
3. Configure:
   - **Model**: GPT-5-mini or GPT-5 (configure in LLM Connections first)
   - **Sample rate**: 10% (to control costs)
   - **Variable mapping** (using JSONPath):
     - `input` → `{{trace.input}}`
     - `output` → `{{trace.output}}`
4. Save and enable

**That's it.** Langfuse will automatically evaluate new traces.

### What You See

Each evaluated trace gets a score:
- `helpfulness`: 0.0-1.0
- Reason explaining the score (chain-of-thought)
- Full trace of the evaluation itself (for debugging)

Filter in Langfuse: `helpfulness < 0.5` to find low-quality responses.

### Execution Monitoring

Langfuse provides visibility into evaluator execution:
- View exact prompts sent to the judge
- See model responses with reasoning
- Track token usage and costs
- Monitor status (Completed, Error, Delayed, Pending)

---

## Option 2: Custom pydantic-ai Judge (Recommended for Control)

For domain-specific evaluation with full control, build your own judge using pydantic-ai.

### Why Custom?

- **Domain-specific rubrics**: Tailor evaluation to your use case
- **Multi-dimension scoring**: Evaluate multiple aspects in one call
- **Consistent with Video 4**: Same model (GPT-5-mini), similar patterns
- **Full visibility**: Debug and iterate on your rubrics

### Step 1: Create Judge Agent

```python
# backend_agent_api/evals/prod_judge.py
"""
Production LLM Judge - Video 7

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


# Configure judge model to use same API as agent (from Video 4 pattern)
judge_model = OpenAIChatModel(
    "gpt-5-mini",  # Cost-effective, same as Video 4
    provider=OpenAIProvider(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY"),
    ),
)

judge_agent = Agent(
    model=judge_model,
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
    Uses pydantic-ai Agent instead of pydantic-evals LLMJudge for
    more control over the evaluation prompt and output format.
    """

    sample_rate: float = 0.1  # Default: evaluate 10% of requests

    async def evaluate_and_sync(
        self,
        trace_id: str,
        output: str,
        inputs: Optional[dict] = None,
    ) -> Optional[JudgeResult]:
        """
        Run LLM judge and sync scores to Langfuse.

        Args:
            trace_id: Langfuse trace ID (hex string)
            output: The agent's response text to evaluate
            inputs: Optional dict with input data (e.g., {"query": "..."})

        Returns:
            JudgeResult if evaluation ran, None if skipped or failed
        """
        try:
            from langfuse import Langfuse
            from langfuse.api.resources.score.types import CreateScoreRequest

            langfuse = Langfuse()
        except Exception as e:
            print(f"[prod_judge] Failed to get Langfuse client: {e}")
            return None

        # Build evaluation prompt
        query = inputs.get("query", "Unknown") if inputs else "Unknown"
        prompt = f"""User Query: {query}

Agent Response: {output}

Evaluate this response."""

        try:
            # Run the judge
            result = await judge_agent.run(prompt)
            judge_output = result.output

            # Sync scores to Langfuse using low-level API (same as Video 6)
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


# Module-level singleton (same pattern as Video 6)
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
        trace_id: Langfuse trace ID (hex string)
        output: The agent's response text
        inputs: Optional input data for context

    Returns:
        JudgeResult if evaluation ran, None otherwise

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
        print(f"[prod_judge] Unexpected error: {e}")
        return None
```

### Step 2: Integrate with Agent API (Sampled)

```python
# backend_agent_api/agent_api.py (additions)

import asyncio
import random
from evals.prod_judge import run_production_judge

JUDGE_SAMPLE_RATE = 0.1  # Evaluate 10% of requests


@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing agent code generates response ...
    # ... existing prod_rules evaluation from Video 6 ...

    trace_id = langfuse_context.get_current_trace_id()

    # VIDEO 7: LLM Judge (sampled)
    # More expensive than rules, so we sample
    if trace_id and random.random() < JUDGE_SAMPLE_RATE:
        asyncio.create_task(
            run_production_judge(
                trace_id=trace_id,
                output=full_response,
                inputs={"query": request.query}
            )
        )

    return {"response": full_response}
```

**Key points:**
- **10% sample rate** keeps costs manageable (~$1-5/day at 1000 traces)
- **Async (fire-and-forget)** so users don't wait
- **GPT-5-mini** is cost-effective (~$0.002/eval)
- **Same pattern as Video 6** — easy to follow

---

## Cost Management

### Model Selection

| Model | Input | Output | ~Cost per Eval | Use Case |
|-------|-------|--------|----------------|----------|
| gpt-5-nano | $0.05/M | $0.40/M | ~$0.0005 | Simple binary checks |
| gpt-5-mini | $0.25/M | $2.00/M | ~$0.002 | **Recommended default** |
| gpt-5 | $1.25/M | $10/M | ~$0.01 | Complex nuanced evaluation |
| claude-sonnet-4.5 | $3/M | $15/M | ~$0.02 | Alternative perspective |

**Note:** GPT-5 offers 90% discount on cached tokens, making repeated evaluations even cheaper.

### Sampling Strategies

```python
# Evaluate all (expensive)
SAMPLE_RATE = 1.0  # ~$20/day at 1000 traces with gpt-5-mini

# Evaluate 10% (balanced) - RECOMMENDED
SAMPLE_RATE = 0.1  # ~$2/day

# Evaluate 1% (budget)
SAMPLE_RATE = 0.01  # ~$0.20/day
```

### Smart Sampling

Evaluate more for interesting traces:

```python
import random

def should_evaluate(trace_metadata: dict, base_rate: float = 0.05) -> bool:
    """
    Smart sampling: evaluate more interesting traces.

    Always evaluate:
    - Negative user feedback (from Video 8)
    - Tool-using traces (more complex)
    - Long conversations

    Random sample the rest at base_rate.
    """
    # Always evaluate negative feedback
    if trace_metadata.get("user_feedback") == "negative":
        return True

    # Always evaluate if multiple tools were used
    tools_used = trace_metadata.get("tools_used", [])
    if len(tools_used) > 1:
        return True

    # Higher rate for RAG queries
    if "retrieve" in str(tools_used):
        return random.random() < 0.2  # 20% for RAG

    # Random sample the rest
    return random.random() < base_rate
```

---

## What You See in Langfuse

After integration, each evaluated trace has these scores:

| Score Name | Type | Meaning |
|------------|------|---------|
| `llm_judge_score` | 0.0-1.0 | Quality score |
| `llm_judge_passed` | 0 or 1 | Passed threshold (>= 0.7) |

### Dashboard Uses

- **Filter**: `llm_judge_passed = 0` to find failures
- **Trend**: Quality score over time (catch regressions)
- **Compare**: Before/after prompt changes
- **Correlate**: Judge score vs user feedback (Video 8)
- **Debug**: Click through to see judge reasoning

---

## Multi-Dimension Evaluation

For more granular insights, evaluate multiple aspects:

```python
class MultiDimensionJudgeResult(BaseModel):
    """Multi-dimension evaluation result."""

    relevance: float      # 0.0-1.0: Does it address the question?
    helpfulness: float    # 0.0-1.0: Is it actionable?
    accuracy: float       # 0.0-1.0: Is it factually correct?
    overall: float        # 0.0-1.0: Overall quality
    reason: str           # Brief explanation


multi_judge_agent = Agent(
    model=judge_model,
    output_type=MultiDimensionJudgeResult,
    system_prompt="""You are an AI quality evaluator.

Evaluate the agent's response on these dimensions (each 0.0-1.0):

1. Relevance: Does it address the user's question?
2. Helpfulness: Is it actionable and useful?
3. Accuracy: Does it appear factually correct?
4. Overall: Weighted average considering all factors

Return scores for each dimension and a brief overall reason.""",
)
```

Then sync each dimension as a separate Langfuse score for filtering/trending.

---

## RAG Faithfulness Evaluation

For RAG systems, evaluate whether responses are grounded in retrieved documents:

```python
class RAGFaithfulnessResult(BaseModel):
    """RAG faithfulness evaluation result."""

    faithfulness: float   # 0.0-1.0: Is response grounded?
    hallucination: bool   # True if significant hallucinations detected
    reason: str           # Explanation


rag_judge_agent = Agent(
    model=judge_model,
    output_type=RAGFaithfulnessResult,
    system_prompt="""You are a RAG faithfulness evaluator.

Check if the response is grounded in the retrieved documents.

Scoring:
- 1.0: Fully grounded, all claims supported
- 0.7-0.9: Mostly grounded, minor inferences acceptable
- 0.5-0.7: Partially grounded, some unsupported claims
- 0.0-0.5: Significant hallucinations

Be strict about hallucinations.""",
)


async def evaluate_rag_trace(
    trace_id: str,
    query: str,
    output: str,
    retrieved_docs: list[str],
) -> Optional[RAGFaithfulnessResult]:
    """Evaluate RAG response faithfulness."""

    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"""User Query: {query}

Retrieved Documents:
{context}

Agent Response: {output}

Is the response faithful to the documents?"""

    result = await rag_judge_agent.run(prompt)
    judge_output = result.output

    # Sync to Langfuse
    # ... same pattern as above ...

    return judge_output
```

---

## Best Practices

### 1. Start with Langfuse Built-in

Use managed evaluators first for common metrics (helpfulness, toxicity). Only build custom when you need domain-specific rubrics.

### 2. Use GPT-5-mini for Production

For most evaluations, GPT-5-mini provides excellent quality at ~$0.002/eval. Reserve GPT-5 for complex nuanced evaluations.

### 3. Sample Strategically

Don't evaluate everything. Prioritize:
- Negative user feedback (always evaluate)
- Tool-using traces (more complex behavior)
- Long conversations (more room for errors)
- Random sample for baseline (5-10%)

### 4. Compare to Human Annotations

Use Video 5 (Manual Annotation) to validate your judge rubrics:
1. Annotate 50 traces manually
2. Run judge on same traces
3. Compare scores
4. Tune rubric based on disagreements
5. Repeat until correlation is high

### 5. Monitor Judge Performance

Track judge metrics over time:
- Average score (should be stable)
- Pass rate (watch for drift)
- Cost per day (budget control)
- Latency (shouldn't block users)

---

## Evolution from Video 4

| Video 4 (Local) | Video 7 (Production) |
|-----------------|----------------------|
| `LLMJudge` evaluator in YAML | `Agent` with structured output |
| Sync execution | Async (fire-and-forget) |
| Terminal report | Langfuse scores |
| All cases evaluated | Sampled for cost control |
| Fixed golden dataset | Real production traces |
| Same rubric patterns | Same rubric patterns! |

**Key insight:** The rubric-writing skills from Video 4 transfer directly. What changes is the execution model (async) and output destination (Langfuse scores).

---

## What's Next

| Video | What You'll Add |
|-------|-----------------|
| **Video 8: User Feedback** | Collect thumbs up/down from real users |

With LLM judge scores AND user feedback, you can:
- Correlate: Do users agree with the judge?
- Calibrate: Tune rubrics based on user disagreement
- Prioritize: Focus human review on disagreements

---

## Resources

- [pydantic-ai Agents](https://ai.pydantic.dev/agents/)
- [pydantic-ai Models (OpenAI)](https://ai.pydantic.dev/models/openai/)
- [pydantic-evals LLMJudge](https://ai.pydantic.dev/evals/evaluators/llm-judge/)
- [Langfuse LLM-as-a-Judge](https://langfuse.com/docs/scores/model-based-evals)
- [Langfuse Custom Scores](https://langfuse.com/docs/scores/custom)
