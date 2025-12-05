# Strategy: LLM-as-Judge (Production)

> **Video 8** | **Tag:** `module-8-07-llm-judge-prod` | **Phase:** Production

## Overview

**What it is**: Automated quality scoring on production traces using Langfuse's built-in LLM-as-a-Judge or custom async evaluation pipelines.

**Philosophy**: You can't manually review every production response. Use LLM judges to score at scale, then focus human attention on low-scoring traces.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM-AS-JUDGE (PRODUCTION)                           │
│                                                                         │
│   User Request        Agent Response        Langfuse                    │
│        │                    │                   │                       │
│        ▼                    ▼                   ▼                       │
│   ┌─────────┐          ┌─────────┐        ┌─────────────┐              │
│   │  Agent  │─────────►│  Trace  │───────►│  LLM Judge  │              │
│   │         │          │ Created │        │  (Async)    │              │
│   └─────────┘          └─────────┘        └──────┬──────┘              │
│                                                  │                      │
│                                                  ▼                      │
│                                           ┌─────────────┐              │
│                                           │   Scores    │              │
│                                           │  Attached   │              │
│                                           │  to Trace   │              │
│                                           └─────────────┘              │
│                                                                         │
│   Langfuse handles: Model calls, score attachment, execution tracing   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Difference from Local

| Aspect | Local (Video 4) | Production (Video 8) |
|--------|-----------------|----------------------|
| **When** | Offline, golden dataset | Real-time on production traces |
| **Scale** | 10-100 test cases | Thousands of traces |
| **Tool** | pydantic-evals `LLMJudge` | Langfuse built-in evaluators |
| **Cost** | Pay per eval run | Pay per trace evaluated |
| **Results** | Terminal report | Langfuse dashboard |

---

## Langfuse Built-in Evaluators

Langfuse ships pre-built evaluator templates:

| Evaluator | What it Checks |
|-----------|----------------|
| **Hallucination** | Are claims grounded in context? |
| **Helpfulness** | Is the response useful? |
| **Relevance** | Does it address the query? |
| **Toxicity** | Is there harmful content? |
| **Correctness** | Is information accurate? |
| **Conciseness** | Is it appropriately brief? |
| **Context Relevance** | (RAG) Were good docs retrieved? |

### Setting Up a Built-in Evaluator

1. **Navigate** to Langfuse → Evaluators → "+ Set up Evaluator"
2. **Select** a managed evaluator template (e.g., "Helpfulness")
3. **Configure** the LLM model (requires OpenAI or Anthropic connection)
4. **Map variables** to trace properties:
   - `input` → trace.input
   - `output` → trace.output
   - `context` → trace.metadata.retrieved_docs (for RAG)
5. **Set trigger**: New traces, existing traces, or all

### Example: Helpfulness Evaluator

```
Template: Helpfulness

Variables:
  - input: {{trace.input}}
  - output: {{trace.output}}

Trigger: All new traces
Model: gpt-4o
Sample Rate: 10% (to control costs)
```

---

## Custom Evaluators in Langfuse

For domain-specific needs, create custom evaluators:

### Example: RAG Faithfulness

```
Name: RAG Faithfulness

Prompt:
You are evaluating a RAG system response.

User Query: {{input}}
Retrieved Documents: {{context}}
Agent Response: {{output}}

Evaluate faithfulness:
1. Is every claim in the response supported by the documents?
2. Are there any hallucinations (claims not in documents)?
3. Does the response acknowledge when information is missing?

Score 0-1 where:
- 1.0 = Fully grounded, no hallucinations
- 0.5 = Minor unsupported claims
- 0.0 = Significant hallucinations

Variable Mapping:
  - input: trace.input.query
  - context: trace.metadata.retrieved_documents
  - output: trace.output
```

### Example: Domain-Specific Quality

```
Name: Insurance Agent Quality

Prompt:
You are evaluating an insurance agent chatbot.

Customer Question: {{input}}
Agent Response: {{output}}

Evaluate on:
1. Accuracy: Are policy details correct?
2. Compliance: Does it include required disclaimers?
3. Helpfulness: Does it guide the customer to next steps?
4. Tone: Is it professional yet empathetic?

Score 0-1 for overall quality.
```

---

## Implementation: Custom Async Judge

For more control, run your own judge pipeline:

### Step 1: Judge Function

```python
# backend_agent_api/evals/prod_judge.py

from langfuse import Langfuse
from pydantic_ai import Agent
from pydantic import BaseModel

langfuse = Langfuse()

class JudgeResult(BaseModel):
    score: float
    passed: bool
    reason: str

judge_agent = Agent(
    model="openai:gpt-4o",
    output_type=JudgeResult,
    system_prompt="""You are an AI quality evaluator.

Evaluate the response for:
1. Relevance: Does it address the query?
2. Helpfulness: Is it actionable?
3. Accuracy: Does it appear factually correct?

Return a score 0-1, pass/fail (threshold 0.7), and reasoning."""
)

async def evaluate_trace(trace_id: str):
    """Evaluate a single trace and attach scores."""

    # Fetch trace from Langfuse
    trace = langfuse.get_trace(trace_id)

    if not trace.output:
        return

    # Build judge prompt
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
        data_type="BOOLEAN"
    )

    return judge_output
```

### Step 2: Async Integration

```python
# backend_agent_api/agent_api.py (additions)

import asyncio
import random
from evals.prod_judge import evaluate_trace

JUDGE_SAMPLE_RATE = 0.1  # Evaluate 10% of requests

@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing agent code ...

    # After response is complete, maybe evaluate
    if random.random() < JUDGE_SAMPLE_RATE:
        # Fire and forget - don't block response
        asyncio.create_task(evaluate_trace(trace_id))

    return response
```

### Step 3: Batch Evaluation Script

For evaluating historical traces:

```python
# backend_agent_api/evals/batch_judge.py

import asyncio
from langfuse import Langfuse
from prod_judge import evaluate_trace

langfuse = Langfuse()

async def evaluate_recent_traces(hours: int = 24, limit: int = 100):
    """Evaluate recent traces that haven't been judged yet."""

    # Fetch traces without judge scores
    traces = langfuse.fetch_traces(
        limit=limit,
        order_by="timestamp",
        order="desc"
    )

    # Filter to unevaluated
    unevaluated = [
        t for t in traces.data
        if not any(s.name == "llm_judge_score" for s in t.scores)
    ]

    print(f"Found {len(unevaluated)} unevaluated traces")

    # Evaluate in batches
    for trace in unevaluated:
        try:
            result = await evaluate_trace(trace.id)
            print(f"✅ {trace.id}: score={result.score:.2f}")
        except Exception as e:
            print(f"❌ {trace.id}: {e}")

        # Rate limiting
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(evaluate_recent_traces())
```

---

## RAG-Specific Judge

For RAG systems, evaluate faithfulness:

```python
# backend_agent_api/evals/rag_judge.py

from pydantic import BaseModel
from pydantic_ai import Agent

class RAGJudgeResult(BaseModel):
    faithfulness_score: float
    hallucination_detected: bool
    hallucinated_claims: list[str]
    reason: str

rag_judge = Agent(
    model="openai:gpt-4o",
    output_type=RAGJudgeResult,
    system_prompt="""You are a RAG faithfulness evaluator.

Your task: Check if the response is grounded in the retrieved documents.

A hallucination is any claim that:
- Is not supported by the documents
- Contradicts the documents
- Adds information beyond what's in the documents

Be strict. If unsure, mark as potential hallucination."""
)

async def evaluate_rag_trace(trace_id: str, retrieved_docs: list[dict]):
    """Evaluate RAG faithfulness."""

    trace = langfuse.get_trace(trace_id)

    context = "\n\n".join([
        f"[Doc {i+1}]: {doc.get('content', '')}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = f"""
User Query: {trace.input}

Retrieved Documents:
{context}

Agent Response: {trace.output}

Check if the response is faithful to the documents.
"""

    result = await rag_judge.run(prompt)
    output = result.output

    # Attach RAG-specific scores
    langfuse.score(
        trace_id=trace_id,
        name="rag_faithfulness",
        value=output.faithfulness_score
    )

    langfuse.score(
        trace_id=trace_id,
        name="rag_hallucination_free",
        value=0.0 if output.hallucination_detected else 1.0,
        data_type="BOOLEAN",
        comment=", ".join(output.hallucinated_claims) if output.hallucinated_claims else "No hallucinations"
    )

    return output
```

---

## Langfuse Dashboard Analytics

After setting up production judges, you can:

### Filter by Score
- Find low-quality traces (score < 0.5)
- Identify patterns in failures

### Track Trends
- Quality score over time
- Pass rate by day/week
- Compare before/after prompt changes

### Correlate Metrics
- LLM Judge vs User Feedback
- Quality vs latency
- Quality vs tool usage

### Export for Analysis
- Download scored traces
- Build custom dashboards
- Feed into alerting systems

---

## Cost Management

### Sampling Strategy

```python
# Evaluate all: Expensive but comprehensive
SAMPLE_RATE = 1.0  # ~$10-50/day at 1000 traces

# Sample 10%: Good balance
SAMPLE_RATE = 0.1  # ~$1-5/day

# Sample 1%: Budget-friendly
SAMPLE_RATE = 0.01  # ~$0.10-0.50/day
```

### Smart Sampling

```python
def should_evaluate(trace) -> bool:
    """Smart sampling: evaluate more interesting traces."""

    # Always evaluate if user gave negative feedback
    if trace.metadata.get("user_feedback") == "negative":
        return True

    # Always evaluate if tools were used
    if trace.metadata.get("tools_used"):
        return True

    # Random sample the rest
    return random.random() < 0.05
```

### Model Selection

| Use Case | Model | Cost |
|----------|-------|------|
| High-volume basic | gpt-4o-mini | ~$0.001/eval |
| Balanced | gpt-4o | ~$0.01/eval |
| Complex/nuanced | claude-sonnet | ~$0.02/eval |

---

## Monitoring & Alerts

### Webhook for Low Scores

```python
async def on_low_score(trace_id: str, score: float):
    """Alert when judge score is low."""
    if score < 0.5:
        # Send to Slack
        await slack_webhook.send({
            "text": f"⚠️ Low quality trace: {trace_id} (score: {score:.2f})",
            "link": f"https://langfuse.com/trace/{trace_id}"
        })
```

### Daily Quality Report

```python
async def daily_quality_report():
    """Generate daily quality summary."""
    traces = langfuse.fetch_traces(limit=1000)

    scores = [
        s.value for t in traces.data
        for s in t.scores if s.name == "llm_judge_score"
    ]

    return {
        "avg_score": sum(scores) / len(scores),
        "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
        "traces_evaluated": len(scores)
    }
```

---

## Best Practices

### 1. Start with Langfuse Built-in

Use managed evaluators first. Only build custom when needed.

### 2. Sample Strategically

Don't evaluate everything. Focus on:
- Negative user feedback
- Tool-using traces
- Long conversations
- Random baseline sample

### 3. Correlate with User Feedback

Compare LLM judge scores against actual user ratings to validate your rubrics.

### 4. Version Your Prompts

Track evaluator prompt versions. When you change rubrics, note the version.

### 5. Set Alerts

Don't just collect scores—act on them. Alert on sustained quality drops.

---

## What's Different from Local

| Local (Video 4) | Production (Video 8) |
|-----------------|----------------------|
| pydantic-evals `LLMJudge` | Langfuse built-in or custom |
| Golden dataset (10 cases) | Production traces (1000s) |
| Terminal report | Dashboard analytics |
| Run manually | Triggered on new traces |
| All evaluations | Sampled for cost |

---

## Resources

- [Langfuse LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)
- [Langfuse Evaluator Library](https://langfuse.com/changelog/2025-05-24-langfuse-evaluator-library)
- [Custom Evaluators](https://langfuse.com/docs/evaluation/custom-evaluators)
- [Evaluation Best Practices](https://langfuse.com/blog/2025-03-04-llm-evaluation-101-best-practices-and-challenges)
