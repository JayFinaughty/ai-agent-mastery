# Strategy 8: A/B Comparative Testing

## Overview

**What it is**: Running two or more versions of your agent (different prompts, models, or configurations) against the same inputs and comparing their performance. This enables data-driven decisions about changes before full deployment.

**Philosophy**: Don't guess which prompt is better - measure it. A/B testing provides statistical confidence that a change improves (or degrades) agent quality, letting you iterate safely.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    A/B COMPARATIVE TESTING                              │
│                                                                         │
│  Same Input ─────┬────────────────────────────────────────────────────  │
│                  │                                                      │
│           ┌──────▼──────┐                    ┌──────────────┐           │
│           │  Version A  │                    │  Version B   │           │
│           │ (Baseline)  │                    │ (Candidate)  │           │
│           │             │                    │              │           │
│           │ gpt-4o-mini │                    │ gpt-4o       │           │
│           │ prompt v1.2 │                    │ prompt v1.3  │           │
│           └──────┬──────┘                    └──────┬───────┘           │
│                  │                                  │                   │
│                  ▼                                  ▼                   │
│           ┌─────────────┐                    ┌─────────────┐           │
│           │ Response A  │                    │ Response B  │           │
│           └──────┬──────┘                    └──────┬──────┘           │
│                  │                                  │                   │
│                  └──────────────┬───────────────────┘                   │
│                                 │                                       │
│                          ┌──────▼──────┐                               │
│                          │  EVALUATE   │                               │
│                          │  & COMPARE  │                               │
│                          └──────┬──────┘                               │
│                                 │                                       │
│         ┌───────────────────────┼───────────────────────┐              │
│         ▼                       ▼                       ▼              │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│  │   Quality   │         │    Cost     │         │   Latency   │       │
│  │   Scores    │         │  Comparison │         │   Metrics   │       │
│  │ A: 0.82     │         │ A: $0.002   │         │ A: 1.2s     │       │
│  │ B: 0.89 ✓   │         │ B: $0.025   │         │ B: 2.8s     │       │
│  └─────────────┘         └─────────────┘         └─────────────┘       │
│                                                                         │
│  DECISION: Version B has 8.5% better quality but 12x cost.             │
│            Recommend A for general use, B for high-value queries.      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Dimension | Comparison Metric | Decision Criteria |
|-----------|-------------------|-------------------|
| **Quality** | LLM judge scores | Higher is better |
| **Accuracy** | Golden dataset pass rate | Higher is better |
| **Latency** | P50/P95 response time | Lower is better |
| **Cost** | Tokens × price per request | Lower is better |
| **Tool Usage** | Tool call efficiency | Fewer redundant calls |
| **Safety** | Rule violation rate | Zero is target |
| **User Preference** | Side-by-side human rating | Statistical preference |

## When to Use

✅ **Good for:**
- Prompt engineering iterations
- Model upgrades (gpt-4o-mini → gpt-4o)
- System prompt changes
- Tool configuration changes
- RAG retrieval parameter tuning
- Before production deployments

❌ **Limitations:**
- Doubles (or more) evaluation cost
- Requires sufficient test cases for statistical significance
- Nondeterministic LLMs may need multiple runs
- Can't A/B test in real-time production easily

## Implementation Plan for Dynamous Agent

### Pydantic AI Integration

Pydantic AI's `Agent.override()` makes A/B testing straightforward:

```python
# backend_agent_api/evals/ab_testing/runner.py

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from pydantic_evals import Dataset, Case
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class VersionConfig:
    """Configuration for a test version"""
    name: str
    model: KnownModelName
    system_prompt: str
    temperature: float = 0.7
    description: str = ""

@dataclass
class VersionResult:
    """Results for a single version"""
    version_name: str
    model: str

    # Aggregate metrics
    avg_quality_score: float
    pass_rate: float
    avg_latency_ms: float
    total_tokens: int
    estimated_cost: float

    # Detailed results
    case_results: list[dict]

    # Comparison
    quality_vs_baseline: Optional[float] = None
    cost_vs_baseline: Optional[float] = None
    latency_vs_baseline: Optional[float] = None

@dataclass
class ABTestResult:
    """Complete A/B test results"""
    test_id: str
    dataset_name: str
    num_cases: int

    # Version results
    baseline: VersionResult
    candidates: list[VersionResult]

    # Winner determination
    quality_winner: str
    cost_winner: str
    recommended_version: str
    recommendation_reason: str

class ABTestRunner:
    """
    Runs A/B comparisons between agent versions.
    Uses Pydantic AI's Agent.override() for version switching.
    """

    def __init__(
        self,
        base_agent: Agent,
        supabase_client,
        judge_model: str = "openai:gpt-4o"
    ):
        self.base_agent = base_agent
        self.supabase = supabase_client
        self.judge_model = judge_model

    async def run_ab_test(
        self,
        dataset: Dataset,
        baseline: VersionConfig,
        candidates: list[VersionConfig],
        runs_per_case: int = 1
    ) -> ABTestResult:
        """
        Run A/B test comparing baseline against candidates.

        Args:
            dataset: Test cases to run
            baseline: The current/baseline version
            candidates: One or more candidate versions to compare
            runs_per_case: Number of runs per case (for variance estimation)
        """
        import uuid
        from datetime import datetime

        test_id = str(uuid.uuid4())
        all_versions = [baseline] + candidates
        version_results = {}

        for version in all_versions:
            result = await self._run_version(
                dataset=dataset,
                version=version,
                runs_per_case=runs_per_case
            )
            version_results[version.name] = result

        # Calculate comparisons vs baseline
        baseline_result = version_results[baseline.name]
        for candidate in candidates:
            cand_result = version_results[candidate.name]
            cand_result.quality_vs_baseline = (
                cand_result.avg_quality_score - baseline_result.avg_quality_score
            )
            cand_result.cost_vs_baseline = (
                (cand_result.estimated_cost - baseline_result.estimated_cost)
                / baseline_result.estimated_cost * 100
            )
            cand_result.latency_vs_baseline = (
                (cand_result.avg_latency_ms - baseline_result.avg_latency_ms)
                / baseline_result.avg_latency_ms * 100
            )

        # Determine winners
        all_results = list(version_results.values())
        quality_winner = max(all_results, key=lambda x: x.avg_quality_score).version_name
        cost_winner = min(all_results, key=lambda x: x.estimated_cost).version_name

        # Recommendation logic
        recommended, reason = self._determine_recommendation(
            baseline_result,
            [version_results[c.name] for c in candidates]
        )

        ab_result = ABTestResult(
            test_id=test_id,
            dataset_name="ab_test_dataset",
            num_cases=len(dataset.cases),
            baseline=baseline_result,
            candidates=[version_results[c.name] for c in candidates],
            quality_winner=quality_winner,
            cost_winner=cost_winner,
            recommended_version=recommended,
            recommendation_reason=reason
        )

        # Store results
        await self._store_results(ab_result)

        return ab_result

    async def _run_version(
        self,
        dataset: Dataset,
        version: VersionConfig,
        runs_per_case: int
    ) -> VersionResult:
        """Run all cases for a single version"""
        import time

        case_results = []
        total_tokens = 0
        total_latency = 0
        quality_scores = []

        for case in dataset.cases:
            for run_num in range(runs_per_case):
                # Override agent with version config
                with self.base_agent.override(
                    model=version.model,
                    system_prompt=version.system_prompt
                ):
                    start = time.time()

                    # Run agent
                    result = await self.base_agent.run(
                        case.inputs["query"],
                        model_settings={"temperature": version.temperature}
                    )

                    latency_ms = int((time.time() - start) * 1000)

                    # Get token usage
                    tokens = result.usage().total_tokens if result.usage() else 0

                    # Evaluate quality with LLM judge
                    quality_score = await self._evaluate_quality(
                        query=case.inputs["query"],
                        response=result.output,
                        expected=case.expected_output
                    )

                    case_results.append({
                        "case_id": case.name,
                        "run": run_num,
                        "response": str(result.output)[:500],
                        "latency_ms": latency_ms,
                        "tokens": tokens,
                        "quality_score": quality_score
                    })

                    total_tokens += tokens
                    total_latency += latency_ms
                    quality_scores.append(quality_score)

        # Calculate cost (approximate)
        cost = self._estimate_cost(version.model, total_tokens)

        return VersionResult(
            version_name=version.name,
            model=version.model,
            avg_quality_score=sum(quality_scores) / len(quality_scores),
            pass_rate=sum(1 for s in quality_scores if s >= 0.7) / len(quality_scores),
            avg_latency_ms=total_latency / len(case_results),
            total_tokens=total_tokens,
            estimated_cost=cost,
            case_results=case_results
        )

    async def _evaluate_quality(
        self,
        query: str,
        response: str,
        expected: Optional[str]
    ) -> float:
        """Evaluate response quality using LLM judge"""
        from pydantic_ai import Agent
        from pydantic import BaseModel, Field

        class JudgeScore(BaseModel):
            score: float = Field(..., ge=0, le=1)
            reasoning: str

        judge = Agent(
            model=self.judge_model,
            output_type=JudgeScore,
            system_prompt="""Evaluate the AI response quality on a 0-1 scale.
            Consider: relevance, accuracy, helpfulness, clarity.
            Be consistent and objective."""
        )

        prompt = f"Query: {query}\n\nResponse: {response}"
        if expected:
            prompt += f"\n\nExpected: {expected}"

        result = await judge.run(prompt)
        return result.output.score

    def _estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate cost based on model and tokens"""
        # Approximate costs per 1M tokens (input + output average)
        costs = {
            "openai:gpt-4o": 7.5,  # $2.5 input + $10 output / 2
            "openai:gpt-4o-mini": 0.375,  # $0.15 input + $0.60 output / 2
            "openai:gpt-4-turbo": 15.0,
            "anthropic:claude-3-5-sonnet": 9.0,
        }
        cost_per_m = costs.get(model, 5.0)
        return (tokens / 1_000_000) * cost_per_m

    def _determine_recommendation(
        self,
        baseline: VersionResult,
        candidates: list[VersionResult]
    ) -> tuple[str, str]:
        """Determine which version to recommend"""

        best_candidate = None
        best_score_improvement = 0

        for cand in candidates:
            improvement = cand.avg_quality_score - baseline.avg_quality_score
            if improvement > best_score_improvement:
                best_score_improvement = improvement
                best_candidate = cand

        if best_candidate is None:
            return baseline.version_name, "No candidate showed improvement over baseline"

        # Decision logic
        quality_improvement = best_score_improvement
        cost_increase = best_candidate.cost_vs_baseline or 0
        latency_increase = best_candidate.latency_vs_baseline or 0

        # If quality improved significantly with acceptable cost/latency
        if quality_improvement >= 0.05:  # 5% improvement
            if cost_increase <= 100:  # Cost doesn't more than double
                return best_candidate.version_name, (
                    f"{quality_improvement:.1%} quality improvement with "
                    f"{cost_increase:.1f}% cost increase"
                )
            else:
                return baseline.version_name, (
                    f"Quality improved {quality_improvement:.1%} but cost increased "
                    f"{cost_increase:.1f}% - not recommended"
                )
        else:
            return baseline.version_name, (
                f"Quality improvement ({quality_improvement:.1%}) below threshold"
            )

    async def _store_results(self, result: ABTestResult):
        """Store A/B test results in database"""
        self.supabase.table("ab_test_results").insert({
            "test_id": result.test_id,
            "dataset_name": result.dataset_name,
            "num_cases": result.num_cases,
            "baseline_version": result.baseline.version_name,
            "baseline_model": result.baseline.model,
            "baseline_quality": result.baseline.avg_quality_score,
            "baseline_cost": result.baseline.estimated_cost,
            "quality_winner": result.quality_winner,
            "cost_winner": result.cost_winner,
            "recommended_version": result.recommended_version,
            "recommendation_reason": result.recommendation_reason,
            "full_results": {
                "baseline": result.baseline.__dict__,
                "candidates": [c.__dict__ for c in result.candidates]
            }
        }).execute()
```

### Pydantic Evals Dataset Comparison

```python
# backend_agent_api/evals/ab_testing/pydantic_evals_ab.py

from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import LLMJudge, Contains, MaxDuration

def create_ab_dataset() -> Dataset:
    """Create dataset for A/B testing"""
    return Dataset(
        cases=[
            Case(
                name="greeting",
                inputs={"query": "Hello!"},
                expected_output=None,
                evaluators=[
                    Contains("hello", case_sensitive=False),
                    MaxDuration(seconds=2.0),
                ]
            ),
            Case(
                name="rag_query",
                inputs={"query": "What documents do we have about sales?"},
                expected_output=None,
                evaluators=[
                    LLMJudge(
                        rubric="Response mentions documents and offers to search or list them",
                        include_input=True
                    ),
                ]
            ),
            Case(
                name="calculation",
                inputs={"query": "What is 15 * 23?"},
                expected_output="345",
                evaluators=[
                    Contains("345"),
                    LLMJudge(rubric="Response contains the correct answer: 345"),
                ]
            ),
        ],
        evaluators=[
            LLMJudge(
                rubric="Response is helpful, clear, and appropriate",
                score={"name": "overall_quality"}
            ),
        ]
    )

async def compare_versions(
    agent: Agent,
    versions: list[VersionConfig]
) -> dict:
    """Compare versions using Pydantic Evals"""

    dataset = create_ab_dataset()
    reports = {}

    for version in versions:
        # Create task function with version override
        async def run_task(inputs: dict) -> str:
            with agent.override(
                model=version.model,
                system_prompt=version.system_prompt
            ):
                result = await agent.run(inputs["query"])
                return str(result.output)

        # Run evaluation
        report = await dataset.evaluate(run_task)
        reports[version.name] = report

    # Compare reports
    return {
        "versions": {
            name: {
                "pass_rate": report.pass_rate,
                "avg_score": report.average_score,
                "avg_duration": report.average_duration_seconds,
            }
            for name, report in reports.items()
        }
    }
```

### Head-to-Head LLM Comparison

```python
# backend_agent_api/evals/ab_testing/head_to_head.py

from pydantic import BaseModel, Field
from pydantic_ai import Agent

class ComparisonResult(BaseModel):
    """Result of head-to-head comparison"""
    winner: str = Field(..., description="'A', 'B', or 'tie'")
    winner_score: float = Field(..., ge=0, le=1)
    reasoning: str
    a_strengths: list[str]
    b_strengths: list[str]

COMPARISON_PROMPT = """You are comparing two AI responses to the same query.

Query: {query}

Response A:
{response_a}

Response B:
{response_b}

Compare these responses on:
1. Relevance - Which better addresses the query?
2. Accuracy - Which is more factually correct?
3. Helpfulness - Which would help the user more?
4. Clarity - Which is clearer and better organized?

Determine which response is better overall, or if they're equivalent (tie).
Provide specific strengths for each response."""

class HeadToHeadEvaluator:
    """
    Direct head-to-head comparison of two responses.
    Useful for subjective quality comparisons.
    """

    def __init__(self, judge_model: str = "openai:gpt-4o"):
        self.judge = Agent(
            model=judge_model,
            output_type=ComparisonResult,
            system_prompt="You are an expert at comparing AI responses objectively."
        )

    async def compare(
        self,
        query: str,
        response_a: str,
        response_b: str
    ) -> ComparisonResult:
        """Compare two responses head-to-head"""
        prompt = COMPARISON_PROMPT.format(
            query=query,
            response_a=response_a,
            response_b=response_b
        )

        result = await self.judge.run(prompt)
        return result.output

    async def compare_batch(
        self,
        cases: list[dict]
    ) -> dict:
        """Compare many cases and aggregate results"""
        import asyncio

        results = await asyncio.gather(*[
            self.compare(
                query=case["query"],
                response_a=case["response_a"],
                response_b=case["response_b"]
            )
            for case in cases
        ])

        # Aggregate
        a_wins = sum(1 for r in results if r.winner == "A")
        b_wins = sum(1 for r in results if r.winner == "B")
        ties = sum(1 for r in results if r.winner == "tie")

        return {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "a_win_rate": a_wins / len(results),
            "b_win_rate": b_wins / len(results),
            "statistical_winner": "A" if a_wins > b_wins else ("B" if b_wins > a_wins else "tie"),
            "individual_results": results
        }
```

### Database Schema

```sql
-- A/B test results
CREATE TABLE ab_test_results (
    test_id VARCHAR PRIMARY KEY,

    -- Test configuration
    dataset_name VARCHAR NOT NULL,
    num_cases INTEGER NOT NULL,

    -- Versions
    baseline_version VARCHAR NOT NULL,
    baseline_model VARCHAR NOT NULL,
    candidate_versions JSONB,

    -- Results
    baseline_quality FLOAT,
    baseline_cost DECIMAL(10, 6),
    baseline_latency_ms INTEGER,

    -- Winners
    quality_winner VARCHAR,
    cost_winner VARCHAR,
    latency_winner VARCHAR,

    -- Recommendation
    recommended_version VARCHAR,
    recommendation_reason TEXT,

    -- Full data
    full_results JSONB,

    -- Meta
    triggered_by VARCHAR,  -- 'manual', 'ci', 'scheduled'
    commit_sha VARCHAR,
    branch VARCHAR,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Head-to-head comparisons
CREATE TABLE head_to_head_results (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR REFERENCES ab_test_results(test_id),

    case_id VARCHAR,
    query TEXT,

    -- Responses
    response_a TEXT,
    response_b TEXT,
    version_a VARCHAR,
    version_b VARCHAR,

    -- Result
    winner VARCHAR,  -- 'A', 'B', 'tie'
    winner_score FLOAT,
    reasoning TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_ab_test_created ON ab_test_results(created_at);
CREATE INDEX idx_ab_test_winner ON ab_test_results(recommended_version);
```

---

## Langfuse Integration

A/B testing results should be tracked in Langfuse for:
- Comparing trace quality across versions
- Visualizing score distributions
- Tracking improvements over time

### Sending A/B Results to Langfuse

```python
# backend_agent_api/evals/ab_testing/langfuse_integration.py

from langfuse import Langfuse

class ABTestLangfuseSync:
    """Sync A/B test results to Langfuse"""

    def __init__(self):
        self.langfuse = Langfuse()

    def sync_test_result(self, result: ABTestResult):
        """Create Langfuse dataset run for A/B test"""

        # Create or get dataset
        dataset = self.langfuse.get_or_create_dataset(
            name=f"ab_test_{result.test_id[:8]}"
        )

        # Add items for each version
        for version_result in [result.baseline] + result.candidates:
            for case in version_result.case_results:
                # Create dataset item
                item = dataset.create_item(
                    input={"query": case.get("query", "")},
                    expected_output=case.get("expected", ""),
                    metadata={
                        "version": version_result.version_name,
                        "model": version_result.model,
                        "test_id": result.test_id
                    }
                )

                # Create run with scores
                run = item.create_run(
                    name=f"ab_run_{version_result.version_name}",
                    output=case.get("response", ""),
                    metadata={
                        "latency_ms": case.get("latency_ms"),
                        "tokens": case.get("tokens")
                    }
                )

                # Add scores
                run.score(
                    name="quality",
                    value=case.get("quality_score", 0)
                )
                run.score(
                    name="latency_ms",
                    value=case.get("latency_ms", 0),
                    data_type="NUMERIC"
                )

        # Log summary as event
        self.langfuse.trace(
            name="ab_test_summary",
            metadata={
                "test_id": result.test_id,
                "quality_winner": result.quality_winner,
                "cost_winner": result.cost_winner,
                "recommended": result.recommended_version,
                "reason": result.recommendation_reason
            }
        )

    def compare_traces_by_version(
        self,
        version_a: str,
        version_b: str,
        metric: str = "quality"
    ) -> dict:
        """Compare Langfuse traces between versions"""

        # Query Langfuse for traces with version metadata
        # This would use Langfuse's API to fetch and compare

        # Example aggregation
        return {
            "version_a": {
                "avg_score": 0.82,
                "trace_count": 150
            },
            "version_b": {
                "avg_score": 0.87,
                "trace_count": 150
            },
            "improvement": 0.05
        }
```

### Langfuse Dataset Experiments

```python
# Using Langfuse's native experiment tracking

async def run_langfuse_experiment(
    agent: Agent,
    dataset_name: str,
    experiment_name: str,
    version_config: VersionConfig
):
    """Run experiment using Langfuse datasets"""

    langfuse = Langfuse()
    dataset = langfuse.get_dataset(dataset_name)

    for item in dataset.items:
        # Run with version override
        with agent.override(
            model=version_config.model,
            system_prompt=version_config.system_prompt
        ):
            # This creates a trace automatically
            with langfuse.trace(
                name=experiment_name,
                metadata={"version": version_config.name}
            ) as trace:
                result = await agent.run(item.input["query"])

                # Link to dataset item
                trace.link_to_dataset_item(item)

                # Score the result
                trace.score(
                    name="quality",
                    value=await evaluate_quality(result.output)
                )
```

---

## Pydantic AI Support

| Feature | Pydantic AI Support | Details |
|---------|---------------------|---------|
| Version switching | ✅ `Agent.override()` | Swap model/prompt per run |
| Batch evaluation | ✅ `Dataset.evaluate()` | Run all cases efficiently |
| Quality scoring | ✅ `LLMJudge` | Built-in judge evaluator |
| Performance tracking | ✅ `MaxDuration` | Latency assertions |
| Result comparison | ⚠️ Manual | Compare `EvaluationReport` objects |
| Statistical analysis | ❌ Manual | Implement significance tests |

---

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Minimum test cases | ≥50 | <20 |
| Statistical confidence | >95% | <90% |
| Quality improvement for deploy | ≥3% | <0% (regression) |
| Max cost increase | ≤50% | >100% |
| Latency increase | ≤25% | >50% |

## Best Practices

1. **Use Sufficient Sample Size**: At least 50 cases for meaningful comparison
2. **Control Variables**: Change only one thing at a time (prompt OR model, not both)
3. **Run Multiple Times**: LLMs are non-deterministic; run each case 3+ times
4. **Track All Dimensions**: Quality alone isn't enough; consider cost and latency
5. **Version Control Configs**: Store prompt versions in git
6. **Automate in CI**: Run A/B tests on prompt changes before merge
7. **Document Decisions**: Record why you chose one version over another

## Testing

```python
# tests/evals/test_ab_testing.py

import pytest
from evals.ab_testing.runner import ABTestRunner, VersionConfig

@pytest.fixture
def versions():
    return [
        VersionConfig(
            name="baseline",
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            description="Current production version"
        ),
        VersionConfig(
            name="candidate",
            model="openai:gpt-4o-mini",
            system_prompt="You are an expert helpful assistant. Be concise and accurate.",
            description="Improved prompt"
        )
    ]

async def test_ab_comparison(versions, test_dataset):
    runner = ABTestRunner(agent, supabase_client)

    result = await runner.run_ab_test(
        dataset=test_dataset,
        baseline=versions[0],
        candidates=[versions[1]]
    )

    assert result.baseline is not None
    assert len(result.candidates) == 1
    assert result.recommended_version in ["baseline", "candidate"]
    assert result.recommendation_reason != ""

async def test_head_to_head():
    evaluator = HeadToHeadEvaluator()

    result = await evaluator.compare(
        query="What is Python?",
        response_a="Python is a programming language.",
        response_b="Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991."
    )

    # Response B should win (more detailed)
    assert result.winner == "B"
    assert result.winner_score > 0.6
```
