# Strategy: LLM-as-Judge (Production)

> **Video 7** | **Tag:** `module-8-06-llm-judge-prod` | **Phase:** Production

## Overview

**What it is**: AI-powered quality scoring on production traces using Langfuse's built-in evaluators or a custom pydantic-ai judge.

**Philosophy**: You can't manually review every response. Use LLM judges to score at scale, then focus human attention on low-scoring traces.

**Building on Video 4**: You used `LLMJudge` in your golden dataset. Now we run AI evaluation on real production data.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM-AS-JUDGE (PRODUCTION)                           │
│                                                                         │
│   Production Trace              Langfuse                                │
│   ────────────────              ────────                                │
│                                                                         │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐      │
│   │   Agent     │         │  Built-in   │         │   Scores    │      │
│   │   Response  │────────►│  Evaluator  │────────►│   on Trace  │      │
│   │   + Trace   │         │  (or custom)│         │             │      │
│   └─────────────┘         └─────────────┘         └─────────────┘      │
│                                                                         │
│   Option 1: Langfuse built-in evaluators (no code)                     │
│   Option 2: Custom pydantic-ai judge (more control)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What You'll Learn in This Video

1. How to use Langfuse's built-in LLM evaluators (no code)
2. How to build a custom judge with pydantic-ai
3. How to run evaluation async (non-blocking)
4. How to sample traces for cost control
5. How to monitor quality trends in Langfuse

## Difference from Local

| Aspect | Local (Video 4) | Production (Video 7) |
|--------|-----------------|----------------------|
| **Tool** | pydantic-evals `LLMJudge` | Langfuse built-in or pydantic-ai |
| **Data** | Golden dataset (10 cases) | Production traces (1000s) |
| **Output** | Terminal report | Langfuse dashboard |
| **Cost** | Pay per eval run | Sample to control costs |

---

## Option 1: Langfuse Built-in Evaluators (Recommended Start)

Langfuse provides pre-built evaluators — no code required.

### Available Evaluators

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
   - **Model**: gpt-4o or gpt-4o-mini
   - **Sample rate**: 10% (to control costs)
   - **Variable mapping**:
     - `input` → `{{trace.input}}`
     - `output` → `{{trace.output}}`
4. Save and enable

**That's it.** Langfuse will automatically evaluate new traces.

### What You See

Each evaluated trace gets a score:
- `helpfulness`: 0.0-1.0
- Reason explaining the score

Filter in Langfuse: `helpfulness < 0.5` to find low-quality responses.

---

## Option 2: Custom pydantic-ai Judge

For domain-specific evaluation, build your own judge.

### Step 1: Create Judge Agent

```python
# backend_agent_api/evals/prod_judge.py

from langfuse import Langfuse
from pydantic import BaseModel
from pydantic_ai import Agent

langfuse = Langfuse()


class JudgeResult(BaseModel):
    """Structured output from the judge."""
    score: float  # 0.0 to 1.0
    passed: bool  # True if score >= 0.7
    reason: str   # Explanation


judge_agent = Agent(
    model="openai:gpt-4o-mini",  # Use mini for cost efficiency
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

Be concise in your reasoning."""
)


async def evaluate_trace(trace_id: str):
    """Evaluate a trace and attach score to Langfuse."""

    # Fetch trace
    trace = langfuse.get_trace(trace_id)

    if not trace.output:
        return None

    # Build prompt
    prompt = f"""
User Query: {trace.input}

Agent Response: {trace.output}

Evaluate this response.
"""

    # Run judge
    result = await judge_agent.run(prompt)
    judge_output = result.output

    # Attach scores to trace
    langfuse.score(
        trace_id=trace_id,
        name="llm_judge_score",
        value=judge_output.score,
        comment=judge_output.reason
    )

    langfuse.score(
        trace_id=trace_id,
        name="llm_judge_passed",
        value=1.0 if judge_output.passed else 0.0,
    )

    return judge_output
```

### Step 2: Integrate with Agent API (Sampled)

```python
# backend_agent_api/agent_api.py (additions)

import asyncio
import random
from evals.prod_judge import evaluate_trace

JUDGE_SAMPLE_RATE = 0.1  # Evaluate 10% of requests


@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing agent code generates response ...

    trace_id = langfuse_context.get_current_trace_id()

    # Sample: only evaluate some requests
    if trace_id and random.random() < JUDGE_SAMPLE_RATE:
        # Fire and forget - don't block response
        asyncio.create_task(evaluate_trace(trace_id))

    return {"response": full_response}
```

**Key points:**
- **10% sample rate** keeps costs manageable
- **Async** so users don't wait
- **gpt-4o-mini** is cheap (~$0.001/eval)

---

## Cost Management

### Sampling Strategies

```python
# Evaluate all (expensive)
SAMPLE_RATE = 1.0  # ~$10-50/day at 1000 traces

# Evaluate 10% (balanced)
SAMPLE_RATE = 0.1  # ~$1-5/day

# Evaluate 1% (budget)
SAMPLE_RATE = 0.01  # ~$0.10-0.50/day
```

### Smart Sampling

Evaluate more for interesting traces:

```python
def should_evaluate(trace) -> bool:
    """Smart sampling: evaluate more interesting traces."""

    # Always evaluate negative user feedback
    if trace.metadata.get("user_feedback") == "negative":
        return True

    # Always evaluate if tools were used
    if trace.metadata.get("tools_used"):
        return True

    # Random sample the rest
    return random.random() < 0.05
```

### Model Selection

| Use Case | Model | Cost per Eval |
|----------|-------|---------------|
| High volume | gpt-4o-mini | ~$0.001 |
| Balanced | gpt-4o | ~$0.01 |
| Nuanced | claude-sonnet | ~$0.02 |

---

## What You See in Langfuse

After integration:

| Score Name | Type | Meaning |
|------------|------|---------|
| `llm_judge_score` | 0.0-1.0 | Quality score |
| `llm_judge_passed` | 0 or 1 | Passed threshold (0.7) |

### Dashboard Uses

- **Filter**: `llm_judge_passed = 0` to find failures
- **Trend**: Quality score over time
- **Compare**: Before/after prompt changes
- **Correlate**: Judge score vs user feedback

---

## RAG Evaluation

For RAG systems, add context to your judge:

```python
# Variation: RAG faithfulness judge
judge_agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=JudgeResult,
    system_prompt="""You are a RAG faithfulness evaluator.

Check if the response is grounded in the retrieved documents.
- Score 1.0 if fully grounded
- Score 0.5 if minor unsupported claims
- Score 0.0 if significant hallucinations

Be strict about hallucinations."""
)

async def evaluate_rag_trace(trace_id: str, retrieved_docs: list[str]):
    trace = langfuse.get_trace(trace_id)

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
User Query: {trace.input}

Retrieved Documents:
{context}

Agent Response: {trace.output}

Is the response faithful to the documents?
"""

    result = await judge_agent.run(prompt)
    # ... attach scores ...
```

---

## Best Practices

### 1. Start with Langfuse Built-in

Use managed evaluators first. Only build custom when you need domain-specific rubrics.

### 2. Sample Strategically

Don't evaluate everything. Prioritize:
- Negative user feedback
- Tool-using traces
- Long conversations

### 3. Use gpt-4o-mini

For most evaluations, mini is sufficient and 10x cheaper.

### 4. Compare to Human Annotations

Use Video 5 (Manual Annotation) to validate your judge rubrics:
- Annotate 50 traces manually
- Run judge on same traces
- Compare scores
- Tune rubric based on disagreements

---

## What's Next

| Video | What You'll Add |
|-------|-----------------|
| **Video 8: User Feedback** | Collect thumbs up/down from real users |

---

## Resources

- [Langfuse LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)
- [Langfuse Evaluator Library](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge#langfuse-managed-evaluator-library)
- [pydantic-ai Agents](https://ai.pydantic.dev/agents/)
