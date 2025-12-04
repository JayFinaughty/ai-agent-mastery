# Strategy 4: Model-Based Evaluation (LLM Judge)

## Overview

**What it is**: Using a language model to evaluate the quality of another model's responses. The "judge" LLM scores responses against defined criteria, enabling scalable automated evaluation.

**Philosophy**: LLMs can approximate human judgment at scale. While not perfect, they provide consistent, fast, and cost-effective evaluation that correlates well with human assessment when properly calibrated.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM-AS-JUDGE FLOW                                │
│                                                                         │
│  Input ──────► Agent ──────► Response ──────► Judge LLM ──────► Score  │
│    │                            │                 │                     │
│    │                            │                 │                     │
│    ▼                            ▼                 ▼                     │
│  ┌──────┐                 ┌──────────┐      ┌──────────┐               │
│  │Query │                 │ Response │      │ Rubric + │               │
│  │      │                 │ + Context│      │ Chain of │               │
│  └──────┘                 └──────────┘      │ Thought  │               │
│                                             └──────────┘               │
│                                                   │                     │
│                                                   ▼                     │
│                                             ┌──────────┐               │
│                                             │  Score   │               │
│                                             │ + Reason │               │
│                                             └──────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Dimension | Judge Capability | Reliability |
|-----------|------------------|-------------|
| **Relevance** | Excellent | High |
| **Helpfulness** | Good | Medium-High |
| **Clarity** | Excellent | High |
| **Completeness** | Good | Medium |
| **Factual Accuracy** | Limited* | Medium |
| **Safety** | Good | Medium-High |
| **Tone/Style** | Excellent | High |
| **RAG Faithfulness** | Good | Medium-High |

*Factual accuracy requires external verification or RAG context

## When to Use

✅ **Good for:**
- Scaling evaluation (10-100% of requests)
- Rapid iteration on prompts
- Consistent scoring across large datasets
- Detecting subjective quality issues
- Pre-filtering for human review

❌ **Limitations:**
- Judge bias (may prefer verbose responses)
- Self-preference (same-family models)
- Hallucination in reasoning
- Cost for high-volume evaluation
- Requires calibration against human judgment

## Implementation Plan for Dynamous Agent

### Pydantic Evals Integration

```python
# backend_agent_api/evals/evaluators/llm_judge.py

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from typing import Optional
from dataclasses import dataclass

# Define the judge's output structure
class JudgeOutput(BaseModel):
    """Structured output from LLM judge"""

    # Scores (0.0 to 1.0)
    relevance_score: float = Field(..., ge=0, le=1, description="How relevant is the response to the query?")
    helpfulness_score: float = Field(..., ge=0, le=1, description="How helpful is the response?")
    clarity_score: float = Field(..., ge=0, le=1, description="How clear and well-structured is the response?")
    completeness_score: float = Field(..., ge=0, le=1, description="How complete is the response?")
    accuracy_score: float = Field(..., ge=0, le=1, description="How accurate does the response appear?")

    # Overall
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    passed: bool = Field(..., description="Does this response meet quality standards?")

    # Reasoning (critical for transparency)
    reasoning: str = Field(..., description="Chain-of-thought reasoning for the scores")
    strengths: list[str] = Field(default_factory=list, description="What the response did well")
    weaknesses: list[str] = Field(default_factory=list, description="Areas for improvement")

# Judge system prompt with detailed rubric
JUDGE_SYSTEM_PROMPT = """You are an expert AI response evaluator. Your task is to evaluate the quality of an AI assistant's response to a user query.

## Evaluation Criteria

### Relevance (0.0-1.0)
- 1.0: Directly addresses the query with focused, on-topic content
- 0.7: Mostly relevant with minor tangents
- 0.5: Partially relevant, includes off-topic information
- 0.3: Mostly off-topic
- 0.0: Completely irrelevant

### Helpfulness (0.0-1.0)
- 1.0: Exceptionally helpful, provides actionable value
- 0.7: Helpful, user can accomplish their goal
- 0.5: Somewhat helpful but incomplete
- 0.3: Minimally helpful
- 0.0: Not helpful at all

### Clarity (0.0-1.0)
- 1.0: Crystal clear, well-organized, easy to follow
- 0.7: Clear with good structure
- 0.5: Understandable but could be clearer
- 0.3: Confusing or poorly organized
- 0.0: Incomprehensible

### Completeness (0.0-1.0)
- 1.0: Fully addresses all aspects of the query
- 0.7: Addresses main points with minor gaps
- 0.5: Addresses some aspects, missing others
- 0.3: Significant gaps
- 0.0: Does not address the query

### Accuracy (0.0-1.0)
- 1.0: All information appears accurate
- 0.7: Mostly accurate with minor issues
- 0.5: Mix of accurate and questionable content
- 0.3: Significant accuracy concerns
- 0.0: Clearly inaccurate

## Instructions

1. Read the user query carefully
2. Read the AI response carefully
3. Think step-by-step about each criterion
4. Provide scores for each dimension
5. Calculate overall score (weighted average)
6. Determine if response passes (overall >= 0.7)
7. List specific strengths and weaknesses

Be objective and consistent. A passing response should be genuinely helpful to the user."""

# Create the judge agent
judge_agent = Agent(
    model="openai:gpt-4o",  # Use capable model for judging
    output_type=JudgeOutput,
    system_prompt=JUDGE_SYSTEM_PROMPT
)

@dataclass
class LLMJudgeResult:
    """Result from LLM judge evaluation"""
    output: JudgeOutput
    request_id: str
    latency_ms: int
    judge_model: str
    cost_estimate: float

class LLMJudgeEvaluator(Evaluator):
    """
    LLM-as-Judge evaluator using Pydantic AI.

    Evaluates responses using a separate LLM as judge,
    producing structured scores with reasoning.
    """

    def __init__(
        self,
        judge_model: str = "openai:gpt-4o",
        passing_threshold: float = 0.7,
        sample_rate: float = 0.1,  # Evaluate 10% by default
    ):
        self.judge_model = judge_model
        self.passing_threshold = passing_threshold
        self.sample_rate = sample_rate

        # Initialize judge agent
        self.judge = Agent(
            model=judge_model,
            output_type=JudgeOutput,
            system_prompt=JUDGE_SYSTEM_PROMPT
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None
    ) -> LLMJudgeResult:
        """
        Evaluate a single query-response pair.

        Args:
            query: The user's input query
            response: The agent's response
            context: Optional context (e.g., retrieved documents)
            expected_output: Optional expected answer for comparison
        """
        import time

        # Build evaluation prompt
        eval_prompt = f"""## User Query
{query}

## AI Response
{response}"""

        if context:
            eval_prompt += f"""

## Context Provided to AI
{context}"""

        if expected_output:
            eval_prompt += f"""

## Expected/Reference Answer
{expected_output}

Note: Compare the AI response to the reference answer for accuracy assessment."""

        eval_prompt += """

## Your Task
Evaluate the AI response against all criteria. Provide detailed reasoning."""

        # Run judge
        start_time = time.time()
        result = await self.judge.run(eval_prompt)
        latency_ms = int((time.time() - start_time) * 1000)

        # Estimate cost (approximate for GPT-4o)
        input_tokens = len(eval_prompt) // 4
        output_tokens = len(str(result.output)) // 4
        cost_estimate = (input_tokens * 2.5 + output_tokens * 10) / 1_000_000

        return LLMJudgeResult(
            output=result.output,
            request_id="",  # Set by caller
            latency_ms=latency_ms,
            judge_model=self.judge_model,
            cost_estimate=cost_estimate
        )

    async def evaluate_batch(
        self,
        items: list[dict]
    ) -> list[LLMJudgeResult]:
        """Evaluate a batch of items"""
        import asyncio

        tasks = [
            self.evaluate(
                query=item["query"],
                response=item["response"],
                context=item.get("context"),
                expected_output=item.get("expected_output")
            )
            for item in items
        ]

        return await asyncio.gather(*tasks)
```

### RAG-Specific Judge

```python
# backend_agent_api/evals/evaluators/rag_judge.py

class RAGJudgeOutput(BaseModel):
    """Output for RAG-specific evaluation"""

    # RAG-specific scores
    faithfulness_score: float = Field(
        ..., ge=0, le=1,
        description="Is the response grounded in the retrieved documents?"
    )
    context_relevance_score: float = Field(
        ..., ge=0, le=1,
        description="How relevant were the retrieved documents to the query?"
    )
    answer_relevance_score: float = Field(
        ..., ge=0, le=1,
        description="How relevant is the answer to the query?"
    )
    source_attribution_score: float = Field(
        ..., ge=0, le=1,
        description="Are sources properly attributed?"
    )

    # Hallucination detection
    hallucination_detected: bool = Field(
        ...,
        description="Does the response contain information not in the context?"
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="List of claims not supported by context"
    )

    # Overall
    overall_score: float
    passed: bool
    reasoning: str

RAG_JUDGE_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

## Your Task
Evaluate whether the AI's response is properly grounded in the retrieved documents.

## Key Questions
1. **Faithfulness**: Does the response ONLY contain information from the provided context?
2. **Context Relevance**: Were the retrieved documents relevant to the query?
3. **Answer Relevance**: Does the answer address the user's question?
4. **Attribution**: Are claims properly attributed to sources?

## Hallucination Detection
A hallucination is any claim in the response that:
- Is not supported by the retrieved documents
- Contradicts the retrieved documents
- Adds information beyond what's in the documents

## Scoring Guide
- Faithfulness: 1.0 = fully grounded, 0.0 = completely fabricated
- Context Relevance: 1.0 = perfect retrieval, 0.0 = irrelevant documents
- Answer Relevance: 1.0 = directly answers query, 0.0 = off-topic
- Attribution: 1.0 = all sources cited, 0.0 = no attribution

Be strict about hallucinations. If you're unsure if something is in the context, mark it as potential hallucination."""

class RAGJudgeEvaluator:
    """Specialized judge for RAG evaluation"""

    def __init__(self, judge_model: str = "openai:gpt-4o"):
        self.judge = Agent(
            model=judge_model,
            output_type=RAGJudgeOutput,
            system_prompt=RAG_JUDGE_PROMPT
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        retrieved_documents: list[dict]
    ) -> RAGJudgeOutput:
        """Evaluate RAG response faithfulness"""

        # Format retrieved docs
        context = "\n\n".join([
            f"[Document {i+1}: {doc.get('title', 'Untitled')}]\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_documents)
        ])

        eval_prompt = f"""## User Query
{query}

## Retrieved Documents
{context}

## AI Response
{response}

## Evaluation Task
1. Check if every claim in the response is supported by the documents
2. Identify any hallucinations
3. Score each dimension
4. Provide detailed reasoning"""

        result = await self.judge.run(eval_prompt)
        return result.output
```

### Database Schema for Judge Results

```sql
-- LLM Judge evaluation results
CREATE TABLE llm_judge_results (
    id SERIAL PRIMARY KEY,

    -- What was evaluated
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR REFERENCES conversations(session_id),
    message_id INTEGER REFERENCES messages(id),

    -- Judge configuration
    judge_model VARCHAR NOT NULL,
    judge_version VARCHAR,
    evaluation_type VARCHAR NOT NULL,  -- 'response_quality', 'rag_faithfulness', etc.

    -- Scores
    relevance_score FLOAT,
    helpfulness_score FLOAT,
    clarity_score FLOAT,
    completeness_score FLOAT,
    accuracy_score FLOAT,
    overall_score FLOAT NOT NULL,
    passed BOOLEAN NOT NULL,

    -- RAG-specific (nullable)
    faithfulness_score FLOAT,
    context_relevance_score FLOAT,
    hallucination_detected BOOLEAN,

    -- Reasoning
    reasoning TEXT,
    strengths JSONB,
    weaknesses JSONB,

    -- Meta
    latency_ms INTEGER,
    cost_estimate DECIMAL(10, 6),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_llm_judge_request ON llm_judge_results(request_id);
CREATE INDEX idx_llm_judge_score ON llm_judge_results(overall_score);
CREATE INDEX idx_llm_judge_passed ON llm_judge_results(passed);
CREATE INDEX idx_llm_judge_created ON llm_judge_results(created_at);
```

### Integration with Agent API

```python
# backend_agent_api/evals/judge_integration.py

import asyncio
import random
from typing import Optional

class JudgeIntegration:
    """
    Integrates LLM judge evaluation with the agent API.
    Runs evaluations asynchronously to not block responses.
    """

    def __init__(
        self,
        supabase_client,
        sample_rate: float = 0.1,
        enabled: bool = True
    ):
        self.supabase = supabase_client
        self.sample_rate = sample_rate
        self.enabled = enabled

        self.response_judge = LLMJudgeEvaluator()
        self.rag_judge = RAGJudgeEvaluator()

    def should_evaluate(self) -> bool:
        """Determine if this request should be evaluated"""
        if not self.enabled:
            return False
        return random.random() < self.sample_rate

    async def evaluate_response(
        self,
        request_id: str,
        session_id: str,
        message_id: int,
        query: str,
        response: str,
        retrieved_docs: Optional[list[dict]] = None
    ):
        """
        Evaluate a response asynchronously.
        Called after response is sent to user.
        """
        try:
            # Response quality evaluation
            quality_result = await self.response_judge.evaluate(
                query=query,
                response=response,
                context=self._format_docs(retrieved_docs) if retrieved_docs else None
            )

            # Store result
            await self._store_result(
                request_id=request_id,
                session_id=session_id,
                message_id=message_id,
                evaluation_type="response_quality",
                result=quality_result
            )

            # RAG evaluation if documents were used
            if retrieved_docs:
                rag_result = await self.rag_judge.evaluate(
                    query=query,
                    response=response,
                    retrieved_documents=retrieved_docs
                )

                await self._store_rag_result(
                    request_id=request_id,
                    session_id=session_id,
                    message_id=message_id,
                    result=rag_result
                )

            # Send to Langfuse
            await self._sync_to_langfuse(request_id, quality_result)

        except Exception as e:
            print(f"Judge evaluation failed: {e}")

    async def _store_result(self, request_id, session_id, message_id, evaluation_type, result):
        """Store judge result in database"""
        self.supabase.table("llm_judge_results").insert({
            "request_id": request_id,
            "session_id": session_id,
            "message_id": message_id,
            "judge_model": result.judge_model,
            "evaluation_type": evaluation_type,
            "relevance_score": result.output.relevance_score,
            "helpfulness_score": result.output.helpfulness_score,
            "clarity_score": result.output.clarity_score,
            "completeness_score": result.output.completeness_score,
            "accuracy_score": result.output.accuracy_score,
            "overall_score": result.output.overall_score,
            "passed": result.output.passed,
            "reasoning": result.output.reasoning,
            "strengths": result.output.strengths,
            "weaknesses": result.output.weaknesses,
            "latency_ms": result.latency_ms,
            "cost_estimate": result.cost_estimate
        }).execute()

    async def _sync_to_langfuse(self, request_id, result):
        """Send scores to Langfuse"""
        if langfuse_configured:
            langfuse.score(
                trace_id=request_id,
                name="llm_judge_overall",
                value=result.output.overall_score,
                comment=result.output.reasoning[:500]
            )

            for dim in ["relevance", "helpfulness", "clarity", "completeness", "accuracy"]:
                score = getattr(result.output, f"{dim}_score", None)
                if score is not None:
                    langfuse.score(
                        trace_id=request_id,
                        name=f"llm_judge_{dim}",
                        value=score
                    )

    def _format_docs(self, docs: list[dict]) -> str:
        """Format documents for context"""
        return "\n\n".join([
            f"[{doc.get('title', 'Document')}]: {doc.get('content', '')[:500]}"
            for doc in docs
        ])


# Usage in agent_api.py
judge_integration = JudgeIntegration(supabase_client, sample_rate=0.1)

@app.post("/api/pydantic-agent")
async def pydantic_agent(request: AgentRequest, ...):
    # ... existing agent logic ...

    # After response is complete
    if judge_integration.should_evaluate():
        # Fire and forget - don't block response
        asyncio.create_task(
            judge_integration.evaluate_response(
                request_id=request.request_id,
                session_id=request.session_id,
                message_id=message_id,
                query=request.query,
                response=full_response,
                retrieved_docs=retrieved_documents
            )
        )
```

### Judge Calibration

```python
# backend_agent_api/evals/judge_calibration.py

class JudgeCalibration:
    """
    Calibrate LLM judge against human annotations.
    Used to validate and improve judge accuracy.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def calculate_correlation(
        self,
        dimension: str,
        period_days: int = 30
    ) -> dict:
        """
        Calculate correlation between judge scores and human annotations.
        """
        # Get requests with both judge and human scores
        data = self.supabase.rpc(
            "get_judge_human_comparison",
            {"dimension": dimension, "days": period_days}
        ).execute()

        if not data.data or len(data.data) < 20:
            return {"correlation": None, "sample_size": 0, "status": "insufficient_data"}

        judge_scores = [d["judge_score"] for d in data.data]
        human_scores = [d["human_score"] for d in data.data]

        # Calculate Pearson correlation
        correlation = self._pearson_correlation(judge_scores, human_scores)

        # Calculate mean absolute error
        mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / len(data.data)

        return {
            "correlation": correlation,
            "mean_absolute_error": mae,
            "sample_size": len(data.data),
            "status": "calibrated" if correlation > 0.7 else "needs_improvement"
        }

    async def get_judge_failures(self, limit: int = 50) -> list[dict]:
        """
        Get cases where judge significantly disagreed with human.
        Use for judge prompt improvement.
        """
        return self.supabase.rpc(
            "get_judge_human_disagreements",
            {"threshold": 0.3, "limit": limit}
        ).execute().data

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = (
            sum((xi - mean_x) ** 2 for xi in x) ** 0.5 *
            sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        )

        return numerator / denominator if denominator != 0 else 0
```

### Metrics Dashboard

```python
# backend_agent_api/evals/metrics/judge_metrics.py

@dataclass
class JudgeMetrics:
    """Dashboard metrics for LLM judge"""
    period: str

    # Volume
    evaluations_run: int
    sample_rate_actual: float

    # Quality distribution
    avg_overall_score: float
    pass_rate: float
    score_distribution: dict[str, int]  # Score bucket -> count

    # Dimension breakdown
    avg_by_dimension: dict[str, float]
    lowest_dimension: str

    # Common issues
    top_weaknesses: list[str]
    hallucination_rate: float  # For RAG evals

    # Judge performance
    avg_latency_ms: int
    total_cost: float
    cost_per_eval: float

    # Calibration
    human_correlation: Optional[float]
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Human correlation | >0.75 | <0.6 |
| Pass rate | >80% | <65% |
| Avg overall score | >0.75 | <0.6 |
| Hallucination rate (RAG) | <10% | >20% |
| Judge latency | <3s | >10s |
| Cost per eval | <$0.02 | >$0.05 |

## Best Practices

1. **Use Chain-of-Thought**: Always require reasoning before scores
2. **Separate Judge Model**: Don't use the same model that generated the response
3. **Calibrate Regularly**: Compare against human annotations monthly
4. **Version Prompts**: Track judge prompt versions for reproducibility
5. **Monitor Bias**: Check for length bias, verbosity preference
6. **Sample Wisely**: Focus on edge cases, not just random sampling

## Testing

```python
# tests/evals/test_llm_judge.py

import pytest
from evals.evaluators.llm_judge import LLMJudgeEvaluator

@pytest.fixture
def judge():
    return LLMJudgeEvaluator(judge_model="openai:gpt-4o-mini")  # Cheaper for tests

async def test_high_quality_response(judge):
    result = await judge.evaluate(
        query="What is Python?",
        response="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation and supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    )

    assert result.output.overall_score > 0.7
    assert result.output.passed is True

async def test_low_quality_response(judge):
    result = await judge.evaluate(
        query="Explain machine learning",
        response="ML is computers doing stuff."
    )

    assert result.output.overall_score < 0.5
    assert result.output.passed is False
    assert len(result.output.weaknesses) > 0

async def test_rag_hallucination_detection():
    rag_judge = RAGJudgeEvaluator()

    result = await rag_judge.evaluate(
        query="What are our Q3 sales?",
        response="Q3 sales were $5 million, up 20% from Q2. The company also announced plans to expand to Europe.",
        retrieved_documents=[
            {"title": "Q3 Report", "content": "Q3 sales totaled $5 million, representing 20% growth over Q2."}
        ]
    )

    # The Europe expansion is not in the documents
    assert result.hallucination_detected is True
    assert any("Europe" in claim for claim in result.hallucinated_claims)
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ✅ STRONG**

LLM Judge is an excellent fit for Langfuse because:
1. **Score attachment**: Judge scores attach directly to traces
2. **Correlation analysis**: Compare judge vs user feedback
3. **Dashboard visualization**: Track quality trends
4. **Calibration tracking**: Monitor judge performance over time

**Sending Judge Scores to Langfuse:**

```python
from langfuse import Langfuse

langfuse = Langfuse()

async def sync_judge_result_to_langfuse(
    trace_id: str,
    result: LLMJudgeResult
):
    """Sync LLM judge scores to Langfuse trace"""

    # Overall score
    langfuse.score(
        trace_id=trace_id,
        name="llm_judge_overall",
        value=result.output.overall_score,
        comment=result.output.reasoning[:500]
    )

    # Dimension scores
    for dimension in ["relevance", "helpfulness", "clarity", "completeness", "accuracy"]:
        score = getattr(result.output, f"{dimension}_score", None)
        if score is not None:
            langfuse.score(
                trace_id=trace_id,
                name=f"llm_judge_{dimension}",
                value=score
            )

    # Pass/fail
    langfuse.score(
        trace_id=trace_id,
        name="llm_judge_passed",
        value=1.0 if result.output.passed else 0.0,
        data_type="BOOLEAN"
    )

    # RAG-specific scores (if applicable)
    if hasattr(result.output, "faithfulness_score"):
        langfuse.score(
            trace_id=trace_id,
            name="rag_faithfulness",
            value=result.output.faithfulness_score
        )
        langfuse.score(
            trace_id=trace_id,
            name="rag_hallucination",
            value=0.0 if result.output.hallucination_detected else 1.0
        )
```

**Langfuse Dashboard Benefits:**
- Filter traces by quality score
- Compare LLM judge vs user feedback (calibration)
- Track quality trends over time
- Identify systematic issues

### Pydantic AI Support

**Fit Level: ✅ FULL SUPPORT**

Pydantic Evals provides native LLM Judge support via the `LLMJudge` evaluator:

**Built-in LLMJudge Evaluator:**

```python
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import LLMJudge

# Basic usage
dataset = Dataset(
    cases=[
        Case(
            name="quality_test",
            inputs={"query": "What is Python?"},
            expected_output="Python is a programming language..."
        )
    ],
    evaluators=[
        LLMJudge(
            rubric="""Evaluate the response for:
            1. Relevance: Does it address the query? (0-1)
            2. Accuracy: Is the information correct? (0-1)
            3. Clarity: Is it well-structured? (0-1)
            Overall score should reflect these dimensions.""",
            model="openai:gpt-4o",
            include_input=True,
            include_expected_output=True
        )
    ]
)

# Run evaluation
report = await dataset.evaluate(my_agent_function)
print(report)
```

**LLMJudge Configuration Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `rubric` | Evaluation criteria (required) | - |
| `model` | Judge model | `openai:gpt-4o` |
| `include_input` | Show input to judge | `False` |
| `include_expected_output` | Show expected to judge | `False` |
| `model_settings` | Temperature, etc. | `{}` |
| `assertion` | Pass/fail mode | `None` |
| `score` | Numeric score mode | `None` |

**Multiple Dimension Judges:**

```python
from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

# Separate judges for different dimensions
dataset = Dataset(
    cases=[...],
    evaluators=[
        LLMJudge(
            rubric="Is the response relevant to the query?",
            score={"name": "relevance"}
        ),
        LLMJudge(
            rubric="Is the response accurate and factual?",
            score={"name": "accuracy"}
        ),
        LLMJudge(
            rubric="Is the response clear and well-organized?",
            score={"name": "clarity"}
        ),
    ]
)
```

**RAG-Specific Judge:**

```python
rag_judge = LLMJudge(
    rubric="""Evaluate RAG response faithfulness:
    1. Is every claim supported by the retrieved documents?
    2. Are there any hallucinations (claims not in documents)?
    3. Are sources properly attributed?

    Return faithfulness score (0-1) and list any hallucinated claims.""",
    include_input=True,
    include_expected_output=False,  # Use context instead
)
```

**Best Practice - Combine with Deterministic Checks:**

```python
from pydantic_evals.evaluators import LLMJudge, Contains, IsInstance

dataset = Dataset(
    cases=[...],
    evaluators=[
        # Fast deterministic checks first
        Contains("Python", case_sensitive=False),
        IsInstance("str"),

        # Expensive LLM judge only if basics pass
        LLMJudge(
            rubric="Evaluate overall quality...",
            score={"name": "quality"}
        )
    ]
)
```
