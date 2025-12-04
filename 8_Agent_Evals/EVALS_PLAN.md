# Agent Evals Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for adding evaluations to the Dynamous AI Agent project. Based on research into agent eval patterns, the Pydantic AI framework's native eval capabilities, and analysis of the current codebase architecture.

---

## Part 1: Research Findings

### 1.1 Common Agent Eval Patterns

#### Types of Evaluations

| Eval Type | Description | Key Metrics |
|-----------|-------------|-------------|
| **Response Quality** | Accuracy, relevance, helpfulness | Factual accuracy %, hallucination rate |
| **Tool Use** | Correct tool selection, proper parameters | Tool call accuracy (0-1), parameter correctness |
| **RAG Retrieval** | Document relevance, context quality | Faithfulness, context precision/recall |
| **Conversation Flow** | Context retention, coherence | Conversation completeness %, coherence score |
| **Safety** | PII handling, harmful content blocking | Detection rate, block rate |
| **Performance** | Latency, throughput | P95/P99 latency, requests/second |
| **Cost** | Token usage, API costs | Cost per query, token utilization |

#### Evaluation Methodologies

1. **LLM-as-Judge**: Use an LLM to score responses against criteria
2. **Ground Truth Comparison**: Compare against expected outputs
3. **Human Feedback**: Collect ratings, use for training
4. **Automated Metrics**: Deterministic checks (exact match, regex, type validation)
5. **Span-Based Evaluation**: Analyze tool calls and execution flow via traces

### 1.2 Pydantic AI Native Eval Features

The project uses Pydantic AI, which provides a dedicated **pydantic-evals** package:

#### Core Components

```
Dataset → Cases → Evaluators → EvaluationReport
```

- **Dataset**: Collection of test cases
- **Case**: Single test scenario (inputs, expected outputs, metadata)
- **Evaluators**: Scoring mechanisms (built-in, LLM judge, custom, span-based)
- **EvaluationReport**: Results with scores and statistics

#### Built-in Evaluators

1. **Deterministic Evaluators**
   - Exact match
   - Substring/contains checks
   - Regex pattern matching
   - JSON schema validation
   - Pydantic model validation
   - Type checking

2. **LLM-as-Judge Evaluators**
   - `judge_input_output()` - Evaluate based on input and output
   - `judge_input_output_expected()` - Include expected output
   - `judge_output()` - Evaluate output only
   - `judge_output_expected()` - Compare output to expected
   - Returns `GradingOutput` with `reason`, `pass_`, `score`

3. **Span-Based Evaluators**
   - Analyze OpenTelemetry traces
   - Evaluate tool calls and execution flow
   - Essential for multi-step agent behavior

#### Testing Utilities

- **TestModel**: Mock model for unit tests (no LLM calls)
- **FunctionModel**: Custom logic for tool execution
- **capture_run_messages()**: Inspect message exchanges
- **Agent.override()**: Replace model/deps for testing

### 1.3 Existing Observability in Project

**Already Configured:**
- Langfuse integration via OpenTelemetry
- Logfire instrumentation
- Trace attributes: user_id, session_id, input, output
- Agent instrumentation via `instrument=True`

**Gaps:**
- No structured eval metrics captured
- Tool execution not instrumented for timing/success
- No token usage tracking
- RAG retrieval quality not measured
- No automated quality scoring
- No cost tracking

---

## Part 2: Integration Points

### 2.1 Agent API Layer (`backend_agent_api/agent_api.py`)

| Location | Line(s) | Integration Point | Eval Types |
|----------|---------|-------------------|------------|
| Pre-execution | ~310 | Before agent.run() | Input validation, intent classification |
| During execution | 316-328 | Streaming loop | Tool call tracking, token counting |
| Post-execution | ~335 | After completion | Response quality, completeness |
| Title generation | ~385 | Title agent call | Title quality eval |
| Memory update | ~250-261 | Mem0 operations | Memory relevance |

### 2.2 Agent Tools (`backend_agent_api/agent.py`)

| Tool | Line(s) | Eval Opportunities |
|------|---------|-------------------|
| `retrieve_relevant_documents` | 76-104 | RAG relevance, faithfulness |
| `list_documents` | 107-120 | Document coverage |
| `get_document_content` | 123-144 | Content completeness |
| `execute_sql_query` | 147-170 | Query correctness, safety |
| `web_search` | 173-203 | Result relevance |
| `image_analysis` | 206-230 | Analysis accuracy |
| `execute_code` | 233-250 | Code correctness, safety |

### 2.3 RAG Pipeline (`backend_rag_pipeline/`)

| Component | File | Eval Opportunities |
|-----------|------|-------------------|
| Text processing | `text_processor.py` | Chunk quality, extraction accuracy |
| Embedding | `db_handler.py` | Embedding similarity distribution |
| Retrieval | Agent tool | Precision, recall, context relevance |

### 2.4 Database Schema

Current tables that support evals:
- `requests` - Query history, rate limiting
- `messages` - Full conversation history with `message_data`
- `conversations` - Session tracking
- `documents` - RAG chunks with embeddings

---

## Part 3: Proposed Eval Architecture

### 3.1 New Database Tables

```sql
-- Evaluation results storage
CREATE TABLE eval_results (
    id SERIAL PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR REFERENCES conversations(session_id),
    eval_type VARCHAR NOT NULL,  -- 'response_quality', 'tool_use', 'rag', etc.
    eval_name VARCHAR NOT NULL,  -- Specific evaluator name
    score FLOAT,                 -- 0.0 to 1.0
    passed BOOLEAN,
    reason TEXT,
    details JSONB,               -- Additional metrics
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Per-request metrics
CREATE TABLE eval_metrics (
    id SERIAL PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    estimated_cost DECIMAL(10, 6),
    tool_calls JSONB,            -- Array of {name, latency_ms, success}
    rag_queries JSONB,           -- Array of {query, results_count, top_score}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Human annotations for training
CREATE TABLE eval_annotations (
    id SERIAL PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    annotator_id UUID,
    annotation_type VARCHAR,     -- 'correctness', 'helpfulness', 'safety'
    rating INTEGER,              -- 1-5 scale
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evaluation datasets for regression testing
CREATE TABLE eval_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    cases JSONB NOT NULL,        -- Array of test cases
    evaluators JSONB,            -- Evaluator configurations
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.2 Eval Module Structure

```
backend_agent_api/
├── evals/
│   ├── __init__.py
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── response_quality.py    # LLM-as-judge for responses
│   │   ├── tool_use.py            # Tool call correctness
│   │   ├── rag_quality.py         # RAG retrieval metrics
│   │   ├── safety.py              # Safety/guardrail checks
│   │   └── custom.py              # Project-specific evaluators
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── golden_dataset.yaml    # Golden test cases
│   │   └── loader.py              # Dataset loading utilities
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py           # Real-time metrics collection
│   │   ├── cost_tracker.py        # Token/cost tracking
│   │   └── latency.py             # Latency measurement
│   ├── runner.py                  # Eval execution engine
│   └── reporter.py                # Results formatting/export
├── tests/
│   ├── evals/
│   │   ├── test_response_quality.py
│   │   ├── test_tool_use.py
│   │   ├── test_rag.py
│   │   └── test_safety.py
│   └── ...existing tests...
```

### 3.3 Integration with Pydantic Evals

```python
# Example: evals/evaluators/response_quality.py
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import LLMJudge, IsInstance

# Define custom evaluator for response quality
response_quality_judge = LLMJudge(
    rubric="""
    Evaluate the agent response for:
    1. Relevance: Does it address the user's question? (0-1)
    2. Accuracy: Is the information factually correct? (0-1)
    3. Completeness: Does it fully answer the question? (0-1)
    4. Clarity: Is the response clear and well-structured? (0-1)

    Return a score from 0.0 to 1.0 and explain your reasoning.
    """,
    model="openai:gpt-4o"
)

# Example: evals/datasets/golden_dataset.yaml
cases:
  - name: "simple_greeting"
    inputs:
      query: "Hello!"
    expected_output_contains: ["Hello", "help"]
    evaluators:
      - response_quality_judge

  - name: "document_search"
    inputs:
      query: "What documents do you have about sales?"
    metadata:
      requires_rag: true
    evaluators:
      - rag_relevance_evaluator
      - response_quality_judge
```

### 3.4 Real-Time Metrics Collection

```python
# Example: evals/metrics/collector.py
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list = None
    rag_queries: list = None

    def __post_init__(self):
        self.tool_calls = []
        self.rag_queries = []

    @property
    def latency_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0

    @property
    def estimated_cost(self) -> float:
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        input_cost = (self.input_tokens / 1_000_000) * 0.15
        output_cost = (self.output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost

class MetricsCollector:
    def __init__(self):
        self.current_metrics: dict[str, RequestMetrics] = {}

    def start_request(self, request_id: str) -> RequestMetrics:
        metrics = RequestMetrics(request_id=request_id, start_time=time.time())
        self.current_metrics[request_id] = metrics
        return metrics

    def record_tool_call(self, request_id: str, tool_name: str,
                         latency_ms: int, success: bool):
        if request_id in self.current_metrics:
            self.current_metrics[request_id].tool_calls.append({
                "name": tool_name,
                "latency_ms": latency_ms,
                "success": success
            })

    def record_rag_query(self, request_id: str, query: str,
                         results_count: int, top_score: float):
        if request_id in self.current_metrics:
            self.current_metrics[request_id].rag_queries.append({
                "query": query,
                "results_count": results_count,
                "top_score": top_score
            })
```

---

## Part 4: Implementation Phases

### Phase 1: Foundation (Metrics & Instrumentation)

**Goal**: Add real-time metrics collection without changing agent behavior

**Tasks**:
1. Add `pydantic-evals` to requirements.txt
2. Create `evals/` module structure
3. Implement `MetricsCollector` class
4. Add metrics capture to `agent_api.py`:
   - Request latency
   - Token counting (from streaming)
   - Tool call timing
5. Create `eval_metrics` table
6. Store metrics after each request
7. Add metrics endpoint: `GET /api/metrics/{request_id}`

**Files to Modify**:
- `requirements.txt`
- `agent_api.py`
- `sql/` (new migration)

### Phase 2: RAG Evaluation

**Goal**: Evaluate retrieval quality for every RAG query

**Tasks**:
1. Implement RAG evaluators:
   - Context relevance (similarity scores)
   - Retrieval precision (relevant docs / retrieved docs)
   - Faithfulness (response grounded in context)
2. Capture RAG metrics in `retrieve_relevant_documents` tool
3. Create RAGAS-style evaluation dataset
4. Add span-based evaluation for RAG flow

**Evaluators to Implement**:
- `ContextRelevanceEvaluator`
- `FaithfulnessEvaluator`
- `RetrievalPrecisionEvaluator`

### Phase 3: Response Quality Evaluation

**Goal**: LLM-as-judge evaluation for response quality

**Tasks**:
1. Implement LLM judge evaluators using Pydantic Evals
2. Define evaluation rubrics:
   - Relevance
   - Accuracy
   - Completeness
   - Helpfulness
3. Run async evaluations (non-blocking)
4. Store results in `eval_results` table
5. Integrate with Langfuse scores

**Configuration**:
- Judge model: `gpt-4o` (separate from agent model)
- Async execution to not block responses
- Sampling rate for production (e.g., 10% of requests)

### Phase 4: Tool Use Evaluation

**Goal**: Evaluate correct tool selection and usage

**Tasks**:
1. Implement tool use evaluators:
   - Tool selection accuracy
   - Parameter correctness
   - Execution success rate
2. Create golden dataset for tool scenarios
3. Add span-based evaluation for tool sequences
4. Track tool-specific error rates

**Evaluators to Implement**:
- `ToolSelectionEvaluator`
- `ToolParameterEvaluator`
- `ToolSequenceEvaluator`

### Phase 5: Safety Evaluation

**Goal**: Evaluate guardrails and safety measures

**Tasks**:
1. Implement safety evaluators:
   - PII detection check
   - Harmful content detection
   - SQL injection prevention (for `execute_sql_query`)
   - Code execution safety
2. Create adversarial test dataset
3. Add pre-response safety scoring
4. Implement safety alerts/blocks

### Phase 6: Regression Testing & CI/CD

**Goal**: Automated evaluation in development workflow

**Tasks**:
1. Create golden dataset with representative cases
2. Implement evaluation runner for datasets
3. Add pytest integration for evals
4. Set up CI/CD quality gates
5. Track eval metrics over time
6. Alert on regression

**Test Structure**:
```python
# tests/evals/test_regression.py
import pytest
from evals.runner import EvalRunner
from evals.datasets import load_golden_dataset

@pytest.mark.eval
async def test_response_quality_regression():
    dataset = load_golden_dataset()
    runner = EvalRunner()
    report = await runner.evaluate(dataset)

    assert report.average_score("response_quality") >= 0.8
    assert report.pass_rate("tool_use") >= 0.95
```

### Phase 7: Human Feedback Loop

**Goal**: Collect and use human annotations

**Tasks**:
1. Add feedback endpoint: `POST /api/feedback`
2. Create annotation UI in frontend
3. Store annotations in `eval_annotations`
4. Export annotations for judge training
5. Track human-vs-automated agreement

---

## Part 5: Key Metrics Dashboard

### Metrics to Track

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| **Latency** | P50 response time | <2s | >5s |
| **Latency** | P99 response time | <10s | >30s |
| **Quality** | Response quality score | >0.8 | <0.6 |
| **Quality** | Hallucination rate | <5% | >15% |
| **RAG** | Retrieval precision | >0.7 | <0.5 |
| **RAG** | Context faithfulness | >0.85 | <0.7 |
| **Tools** | Tool selection accuracy | >0.9 | <0.8 |
| **Tools** | Tool error rate | <5% | >10% |
| **Safety** | PII leak rate | 0% | >0% |
| **Cost** | Avg cost per request | <$0.01 | >$0.05 |

### Langfuse Integration

All metrics and eval results should be sent to Langfuse as scores:

```python
# Example: Sending eval scores to Langfuse
langfuse.score(
    trace_id=trace_id,
    name="response_quality",
    value=0.85,
    comment="High relevance, minor completeness issues"
)

langfuse.score(
    trace_id=trace_id,
    name="rag_faithfulness",
    value=0.92,
    comment="Response well-grounded in retrieved context"
)
```

---

## Part 6: Testing Strategy

### Unit Tests (TestModel)

```python
# tests/test_agent_tools.py
from pydantic_ai.models.test import TestModel
from pydantic_ai import capture_run_messages

async def test_document_retrieval():
    with agent.override(model=TestModel()):
        with capture_run_messages() as messages:
            result = await agent.run("Find sales documents")

        # Assert tool was called
        tool_calls = [m for m in messages if hasattr(m, 'tool_calls')]
        assert any('retrieve_relevant_documents' in str(tc) for tc in tool_calls)
```

### Eval Tests (Pydantic Evals)

```python
# tests/evals/test_quality.py
from pydantic_evals import Dataset

async def test_response_quality_benchmark():
    dataset = Dataset.from_yaml("evals/datasets/golden_dataset.yaml")
    report = await dataset.evaluate_async(run_agent)

    # This is a benchmark, not a pass/fail
    print(report)  # View detailed results

    # But we can set minimum thresholds
    assert report.metrics["response_quality"].mean >= 0.75
```

### Integration Tests

```python
# tests/integration/test_full_flow.py
async def test_end_to_end_with_rag():
    # Upload test document
    # Send query
    # Verify RAG was used
    # Check response quality
    # Verify metrics were captured
    pass
```

---

## Part 7: Dependencies to Add

```
# requirements.txt additions
pydantic-evals>=0.1.0
ragas>=0.1.0  # For RAG evaluation metrics
```

---

## Part 8: Success Criteria

### Phase 1 Complete When:
- [ ] Metrics captured for every request
- [ ] Latency, tokens, tool calls tracked
- [ ] Metrics visible in database
- [ ] Metrics endpoint working

### Phase 2 Complete When:
- [ ] RAG queries capture similarity scores
- [ ] Retrieval precision calculated
- [ ] Faithfulness evaluator implemented
- [ ] RAG metrics in Langfuse

### Phase 3 Complete When:
- [ ] LLM judge evaluating 10% of responses
- [ ] Eval scores stored in database
- [ ] Scores visible in Langfuse
- [ ] Quality dashboard available

### Phase 4 Complete When:
- [ ] Tool accuracy tracked
- [ ] Golden dataset for tools exists
- [ ] Tool sequence validation working
- [ ] Error rates tracked

### Phase 5 Complete When:
- [ ] Safety evaluators implemented
- [ ] Adversarial tests passing
- [ ] Safety alerts configured
- [ ] PII detection verified

### Phase 6 Complete When:
- [ ] CI runs evals on every PR
- [ ] Regression baseline established
- [ ] Quality gates blocking bad changes
- [ ] Eval trends visible

### Phase 7 Complete When:
- [ ] Feedback UI in frontend
- [ ] Annotations stored
- [ ] Human-automated correlation tracked
- [ ] Judge improvement workflow defined

---

## Appendix: Research Sources

- Pydantic AI Evals: https://ai.pydantic.dev/evals/
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- Langfuse Evaluation: https://langfuse.com/docs/evaluation/overview
- RAGAS Framework: https://docs.ragas.io/
- LLM-as-Judge Best Practices: https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
- OpenTelemetry AI Observability: https://opentelemetry.io/blog/2025/ai-agent-observability/
