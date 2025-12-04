# Strategy 7: Span/Trace-Based Evaluation

## Overview

**What it is**: Evaluating agent behavior by analyzing the execution trace - the sequence of steps, tool calls, and decisions the agent made. This goes beyond just evaluating the final output to assess HOW the agent arrived at its answer.

**Philosophy**: For complex agents, the path matters as much as the destination. An agent might produce a correct answer through flawed reasoning, or use inefficient tool sequences. Trace-based evaluation captures these execution quality aspects.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRACE-BASED EVALUATION                               │
│                                                                         │
│  User Query: "What were our Q3 sales?"                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     EXECUTION TRACE                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  SPAN 1: Agent Planning                                         │   │
│  │  ├─ Duration: 250ms                                             │   │
│  │  └─ Decision: Use RAG retrieval                                 │   │
│  │                                                                 │   │
│  │  SPAN 2: Tool Call - retrieve_relevant_documents                │   │
│  │  ├─ Duration: 450ms                                             │   │
│  │  ├─ Query: "Q3 sales figures"                                   │   │
│  │  ├─ Results: 3 documents                                        │   │
│  │  └─ Relevance Scores: [0.92, 0.87, 0.75]                        │   │
│  │                                                                 │   │
│  │  SPAN 3: Response Generation                                    │   │
│  │  ├─ Duration: 380ms                                             │   │
│  │  ├─ Tokens: 150 in, 200 out                                     │   │
│  │  └─ Grounded in: Documents 1, 2                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TRACE EVALUATION:                                                     │
│  ✅ Correct tool selection (RAG for knowledge question)                │
│  ✅ Efficient path (1 tool call, no redundancy)                        │
│  ✅ High relevance retrieval (avg 0.85)                                │
│  ✅ Response grounded in retrieved docs                                │
│  📊 Total latency: 1080ms (within target)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Dimension | Description | Metric |
|-----------|-------------|--------|
| **Tool Selection** | Did agent choose correct tools? | Accuracy % |
| **Tool Sequence** | Is the order logical/efficient? | Sequence score |
| **Redundancy** | Unnecessary or repeated tool calls? | Redundancy count |
| **Efficiency** | Minimal steps to answer? | Step count |
| **Latency Breakdown** | Time per step | ms per span |
| **Error Handling** | Recovery from tool failures? | Recovery rate |
| **Reasoning Quality** | Sound decision-making? | LLM judge score |
| **Grounding** | Response based on retrieved data? | Grounding score |

## When to Use

✅ **Good for:**
- Multi-step agent workflows
- Debugging agent behavior
- Optimizing tool usage
- Detecting inefficient patterns
- Understanding failure modes
- Agent comparison (same task, different approaches)

❌ **Limitations:**
- Requires instrumentation
- Complex to analyze
- Large data volume
- May slow execution if not async
- Interpretation can be subjective

## Implementation Plan for Dynamous Agent

### Trace Data Model

```python
# backend_agent_api/evals/traces/models.py

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

class SpanType(str, Enum):
    AGENT_START = "agent_start"
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    RESPONSE_GENERATION = "response_generation"
    AGENT_END = "agent_end"
    ERROR = "error"

class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"

class Span(BaseModel):
    """A single span in the execution trace"""

    # Identification
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None

    # Type and status
    span_type: SpanType
    status: SpanStatus = SpanStatus.OK

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Content
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Tool-specific
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None

    # LLM-specific
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Error info
    error_message: Optional[str] = None
    error_type: Optional[str] = None

class Trace(BaseModel):
    """Complete execution trace for a request"""

    trace_id: str
    request_id: str
    session_id: str
    user_id: str

    # The request
    query: str
    response: Optional[str] = None

    # All spans
    spans: list[Span] = Field(default_factory=list)

    # Summary
    total_duration_ms: int
    total_tool_calls: int
    total_tokens: int
    tools_used: list[str]

    # Timing
    started_at: datetime
    ended_at: Optional[datetime] = None

    def get_spans_by_type(self, span_type: SpanType) -> list[Span]:
        """Get all spans of a specific type"""
        return [s for s in self.spans if s.span_type == span_type]

    def get_tool_calls(self) -> list[Span]:
        """Get all tool call spans"""
        return self.get_spans_by_type(SpanType.TOOL_CALL)

    @property
    def tool_sequence(self) -> list[str]:
        """Get ordered list of tools called"""
        return [s.tool_name for s in self.get_tool_calls() if s.tool_name]
```

### Trace Capture Integration

```python
# backend_agent_api/evals/traces/capture.py

from contextvars import ContextVar
from typing import Optional
import uuid
from datetime import datetime

# Context variable to hold current trace
_current_trace: ContextVar[Optional[Trace]] = ContextVar("current_trace", default=None)

class TraceCapture:
    """
    Captures execution traces during agent runs.
    Integrates with OpenTelemetry/Logfire.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def start_trace(
        self,
        request_id: str,
        session_id: str,
        user_id: str,
        query: str
    ) -> Trace:
        """Start a new trace for a request"""
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            query=query,
            started_at=datetime.utcnow(),
            total_duration_ms=0,
            total_tool_calls=0,
            total_tokens=0,
            tools_used=[]
        )

        # Store in context
        _current_trace.set(trace)

        # Add start span
        self.add_span(
            span_type=SpanType.AGENT_START,
            name="agent_execution_start",
            attributes={"query": query}
        )

        return trace

    def add_span(
        self,
        span_type: SpanType,
        name: str,
        parent_span_id: Optional[str] = None,
        **kwargs
    ) -> Span:
        """Add a span to the current trace"""
        trace = _current_trace.get()
        if not trace:
            return None

        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace.trace_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            name=name,
            start_time=datetime.utcnow(),
            **kwargs
        )

        trace.spans.append(span)
        return span

    def end_span(self, span_id: str, **kwargs):
        """End a span and calculate duration"""
        trace = _current_trace.get()
        if not trace:
            return

        for span in trace.spans:
            if span.span_id == span_id:
                span.end_time = datetime.utcnow()
                span.duration_ms = int(
                    (span.end_time - span.start_time).total_seconds() * 1000
                )
                for key, value in kwargs.items():
                    setattr(span, key, value)
                break

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        parent_span_id: Optional[str] = None
    ) -> str:
        """Record a tool call, returns span_id for later completion"""
        span = self.add_span(
            span_type=SpanType.TOOL_CALL,
            name=f"tool_call_{tool_name}",
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            tool_input=tool_input
        )

        trace = _current_trace.get()
        if trace:
            trace.total_tool_calls += 1
            if tool_name not in trace.tools_used:
                trace.tools_used.append(tool_name)

        return span.span_id if span else None

    def record_tool_result(
        self,
        span_id: str,
        tool_output: str,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[str] = None
    ):
        """Record tool call completion"""
        self.end_span(
            span_id,
            tool_output=tool_output[:500],  # Truncate
            status=status,
            error_message=error
        )

    def record_llm_call(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        parent_span_id: Optional[str] = None
    ) -> str:
        """Record an LLM call"""
        span = self.add_span(
            span_type=SpanType.LLM_CALL,
            name=f"llm_call_{model_name}",
            parent_span_id=parent_span_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        trace = _current_trace.get()
        if trace:
            trace.total_tokens += input_tokens + output_tokens

        return span.span_id if span else None

    def record_retrieval(
        self,
        query: str,
        num_results: int,
        relevance_scores: list[float],
        parent_span_id: Optional[str] = None
    ) -> str:
        """Record a retrieval operation"""
        span = self.add_span(
            span_type=SpanType.RETRIEVAL,
            name="document_retrieval",
            parent_span_id=parent_span_id,
            attributes={
                "query": query,
                "num_results": num_results,
                "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                "relevance_scores": relevance_scores
            }
        )
        return span.span_id if span else None

    def end_trace(self, response: str) -> Trace:
        """End the current trace"""
        trace = _current_trace.get()
        if not trace:
            return None

        trace.response = response
        trace.ended_at = datetime.utcnow()
        trace.total_duration_ms = int(
            (trace.ended_at - trace.started_at).total_seconds() * 1000
        )

        # Add end span
        self.add_span(
            span_type=SpanType.AGENT_END,
            name="agent_execution_end",
            attributes={"response_length": len(response)}
        )

        # Store trace
        self._store_trace(trace)

        # Clear context
        _current_trace.set(None)

        return trace

    def _store_trace(self, trace: Trace):
        """Store trace in database"""
        self.supabase.table("execution_traces").insert({
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "session_id": trace.session_id,
            "user_id": trace.user_id,
            "query": trace.query,
            "response": trace.response[:1000] if trace.response else None,
            "total_duration_ms": trace.total_duration_ms,
            "total_tool_calls": trace.total_tool_calls,
            "total_tokens": trace.total_tokens,
            "tools_used": trace.tools_used,
            "spans": [s.dict() for s in trace.spans],
            "started_at": trace.started_at.isoformat(),
            "ended_at": trace.ended_at.isoformat() if trace.ended_at else None
        }).execute()


# Global instance
trace_capture = TraceCapture(supabase_client)
```

### Trace Evaluation

```python
# backend_agent_api/evals/evaluators/trace_evaluator.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class TraceEvaluationResult:
    """Result of trace-based evaluation"""

    # Tool selection
    tool_selection_score: float     # 0-1: Were correct tools chosen?
    tool_sequence_score: float      # 0-1: Was sequence logical?
    tool_efficiency_score: float    # 0-1: Minimal tool calls?

    # Execution quality
    redundancy_count: int           # Number of redundant operations
    error_count: int                # Number of errors encountered
    recovery_rate: float            # Recovered from errors?

    # Performance
    total_latency_ms: int
    avg_tool_latency_ms: float
    latency_score: float            # 0-1: Within targets?

    # RAG quality (if applicable)
    retrieval_relevance: Optional[float]
    retrieval_count: int

    # Overall
    overall_score: float
    issues: list[str]
    recommendations: list[str]

class TraceEvaluator:
    """
    Evaluates agent execution traces.
    Analyzes tool usage, efficiency, and execution patterns.
    """

    # Expected tools for different query types
    EXPECTED_TOOLS = {
        "document_search": ["retrieve_relevant_documents"],
        "document_list": ["list_documents"],
        "web_search": ["web_search"],
        "calculation": ["execute_code"],
        "data_query": ["execute_sql_query"],
        "image_analysis": ["image_analysis"],
    }

    # Target latencies (ms)
    TARGET_LATENCIES = {
        "total": 5000,
        "tool_call": 1000,
        "llm_call": 2000,
        "retrieval": 500,
    }

    def __init__(self):
        pass

    def evaluate(self, trace: Trace) -> TraceEvaluationResult:
        """Evaluate a complete trace"""

        issues = []
        recommendations = []

        # Evaluate tool selection
        tool_selection_score = self._evaluate_tool_selection(trace, issues)

        # Evaluate tool sequence
        tool_sequence_score = self._evaluate_tool_sequence(trace, issues)

        # Evaluate efficiency
        tool_efficiency_score, redundancy_count = self._evaluate_efficiency(trace, issues, recommendations)

        # Evaluate errors
        error_count, recovery_rate = self._evaluate_errors(trace, issues)

        # Evaluate latency
        latency_score, avg_tool_latency = self._evaluate_latency(trace, issues, recommendations)

        # Evaluate retrieval (if applicable)
        retrieval_relevance, retrieval_count = self._evaluate_retrieval(trace, issues)

        # Calculate overall score
        weights = {
            "tool_selection": 0.25,
            "tool_sequence": 0.15,
            "efficiency": 0.20,
            "latency": 0.20,
            "retrieval": 0.20,
        }

        overall_score = (
            tool_selection_score * weights["tool_selection"] +
            tool_sequence_score * weights["tool_sequence"] +
            tool_efficiency_score * weights["efficiency"] +
            latency_score * weights["latency"] +
            (retrieval_relevance or 1.0) * weights["retrieval"]
        )

        return TraceEvaluationResult(
            tool_selection_score=tool_selection_score,
            tool_sequence_score=tool_sequence_score,
            tool_efficiency_score=tool_efficiency_score,
            redundancy_count=redundancy_count,
            error_count=error_count,
            recovery_rate=recovery_rate,
            total_latency_ms=trace.total_duration_ms,
            avg_tool_latency_ms=avg_tool_latency,
            latency_score=latency_score,
            retrieval_relevance=retrieval_relevance,
            retrieval_count=retrieval_count,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )

    def _evaluate_tool_selection(self, trace: Trace, issues: list) -> float:
        """Evaluate if correct tools were selected"""
        query_lower = trace.query.lower()
        tools_used = set(trace.tools_used)

        # Infer expected tools from query
        expected = set()
        if any(word in query_lower for word in ["document", "file", "search for", "find"]):
            expected.add("retrieve_relevant_documents")
        if any(word in query_lower for word in ["list", "show all", "what documents"]):
            expected.add("list_documents")
        if any(word in query_lower for word in ["news", "latest", "current", "today"]):
            expected.add("web_search")
        if any(word in query_lower for word in ["calculate", "compute", "run code"]):
            expected.add("execute_code")
        if any(word in query_lower for word in ["query", "sql", "data from", "rows"]):
            expected.add("execute_sql_query")

        if not expected:
            return 1.0  # No specific expectation

        # Check overlap
        correct = expected.intersection(tools_used)
        missing = expected - tools_used
        unexpected = tools_used - expected

        if missing:
            issues.append(f"Missing expected tools: {missing}")
        if unexpected:
            issues.append(f"Unexpected tools used: {unexpected}")

        score = len(correct) / len(expected) if expected else 1.0
        return score

    def _evaluate_tool_sequence(self, trace: Trace, issues: list) -> float:
        """Evaluate if tool sequence is logical"""
        sequence = trace.tool_sequence

        if len(sequence) <= 1:
            return 1.0  # Single or no tools, sequence is trivial

        # Check for anti-patterns
        score = 1.0

        # Anti-pattern: Same tool called multiple times in a row
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                issues.append(f"Repeated tool call: {sequence[i]}")
                score -= 0.2

        # Anti-pattern: web_search after document retrieval (usually redundant)
        if "retrieve_relevant_documents" in sequence and "web_search" in sequence:
            rag_idx = sequence.index("retrieve_relevant_documents")
            web_idx = sequence.index("web_search")
            if web_idx > rag_idx:
                issues.append("Web search after RAG retrieval may be redundant")
                score -= 0.1

        return max(0.0, score)

    def _evaluate_efficiency(self, trace: Trace, issues: list, recommendations: list) -> tuple[float, int]:
        """Evaluate execution efficiency"""
        tool_calls = trace.get_tool_calls()

        # Count redundant calls (same tool, same input)
        seen_calls = set()
        redundant = 0
        for call in tool_calls:
            key = (call.tool_name, str(call.tool_input))
            if key in seen_calls:
                redundant += 1
                issues.append(f"Redundant tool call: {call.tool_name}")
            seen_calls.add(key)

        # Score based on total tool calls
        if trace.total_tool_calls == 0:
            score = 1.0
        elif trace.total_tool_calls <= 3:
            score = 1.0
        elif trace.total_tool_calls <= 5:
            score = 0.8
        elif trace.total_tool_calls <= 8:
            score = 0.6
            recommendations.append("Consider reducing tool calls for efficiency")
        else:
            score = 0.4
            recommendations.append("High tool call count - investigate optimization")

        # Penalize for redundancy
        score -= redundant * 0.15

        return max(0.0, score), redundant

    def _evaluate_errors(self, trace: Trace, issues: list) -> tuple[int, float]:
        """Evaluate error handling"""
        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        error_count = len(error_spans)

        if error_count == 0:
            return 0, 1.0

        # Check for recovery (successful span after error)
        recovered = 0
        for i, span in enumerate(trace.spans):
            if span.status == SpanStatus.ERROR:
                # Check if there's a successful span of same type after
                for later_span in trace.spans[i+1:]:
                    if later_span.span_type == span.span_type and later_span.status == SpanStatus.OK:
                        recovered += 1
                        break

        recovery_rate = recovered / error_count if error_count > 0 else 1.0

        for error_span in error_spans:
            issues.append(f"Error in {error_span.name}: {error_span.error_message}")

        return error_count, recovery_rate

    def _evaluate_latency(self, trace: Trace, issues: list, recommendations: list) -> tuple[float, float]:
        """Evaluate latency performance"""
        # Total latency
        if trace.total_duration_ms > self.TARGET_LATENCIES["total"]:
            issues.append(f"Total latency {trace.total_duration_ms}ms exceeds target {self.TARGET_LATENCIES['total']}ms")

        # Tool call latencies
        tool_calls = trace.get_tool_calls()
        tool_latencies = [t.duration_ms for t in tool_calls if t.duration_ms]

        avg_tool_latency = sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0

        slow_tools = [t for t in tool_calls if t.duration_ms and t.duration_ms > self.TARGET_LATENCIES["tool_call"]]
        if slow_tools:
            for t in slow_tools:
                issues.append(f"Slow tool call: {t.tool_name} took {t.duration_ms}ms")
            recommendations.append("Investigate slow tool calls for optimization")

        # Calculate score
        latency_ratio = trace.total_duration_ms / self.TARGET_LATENCIES["total"]
        if latency_ratio <= 1.0:
            score = 1.0
        elif latency_ratio <= 1.5:
            score = 0.8
        elif latency_ratio <= 2.0:
            score = 0.6
        else:
            score = 0.4

        return score, avg_tool_latency

    def _evaluate_retrieval(self, trace: Trace, issues: list) -> tuple[Optional[float], int]:
        """Evaluate RAG retrieval quality"""
        retrieval_spans = trace.get_spans_by_type(SpanType.RETRIEVAL)

        if not retrieval_spans:
            return None, 0

        relevance_scores = []
        for span in retrieval_spans:
            scores = span.attributes.get("relevance_scores", [])
            relevance_scores.extend(scores)

        if not relevance_scores:
            return None, len(retrieval_spans)

        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        if avg_relevance < 0.7:
            issues.append(f"Low retrieval relevance: {avg_relevance:.2f}")

        return avg_relevance, len(retrieval_spans)
```

### Database Schema

```sql
-- Execution traces
CREATE TABLE execution_traces (
    trace_id VARCHAR PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR,
    user_id UUID,

    -- Request/Response
    query TEXT,
    response TEXT,

    -- Summary
    total_duration_ms INTEGER,
    total_tool_calls INTEGER,
    total_tokens INTEGER,
    tools_used JSONB,

    -- Full spans
    spans JSONB,

    -- Timing
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trace evaluation results
CREATE TABLE trace_evaluation_results (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR REFERENCES execution_traces(trace_id),
    request_id UUID,

    -- Scores
    tool_selection_score FLOAT,
    tool_sequence_score FLOAT,
    tool_efficiency_score FLOAT,
    latency_score FLOAT,
    retrieval_relevance FLOAT,
    overall_score FLOAT,

    -- Metrics
    redundancy_count INTEGER,
    error_count INTEGER,
    recovery_rate FLOAT,
    total_latency_ms INTEGER,
    avg_tool_latency_ms FLOAT,

    -- Analysis
    issues JSONB,
    recommendations JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_traces_request ON execution_traces(request_id);
CREATE INDEX idx_traces_session ON execution_traces(session_id);
CREATE INDEX idx_traces_duration ON execution_traces(total_duration_ms);
CREATE INDEX idx_trace_eval_score ON trace_evaluation_results(overall_score);
```

### Integration with Agent API

```python
# Modifications to backend_agent_api/agent_api.py

from evals.traces.capture import trace_capture

@app.post("/api/pydantic-agent")
async def pydantic_agent(request: AgentRequest, ...):

    # Start trace
    trace = trace_capture.start_trace(
        request_id=request.request_id,
        session_id=request.session_id,
        user_id=request.user_id,
        query=request.query
    )

    try:
        # When tools are called, record them
        # (This would be integrated into the tool execution)

        # ... agent execution ...

        # End trace
        final_trace = trace_capture.end_trace(full_response)

        # Evaluate trace asynchronously
        asyncio.create_task(evaluate_trace(final_trace))

    except Exception as e:
        trace_capture.add_span(
            span_type=SpanType.ERROR,
            name="agent_error",
            error_message=str(e)
        )
        raise
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Tool selection accuracy | >90% | <80% |
| Avg trace score | >0.8 | <0.6 |
| Redundancy rate | <5% | >15% |
| Error recovery rate | >80% | <50% |
| Latency compliance | >90% | <70% |

## Testing

```python
# tests/evals/test_trace_evaluator.py

import pytest
from evals.evaluators.trace_evaluator import TraceEvaluator
from evals.traces.models import Trace, Span, SpanType, SpanStatus

@pytest.fixture
def evaluator():
    return TraceEvaluator()

def test_efficient_trace(evaluator):
    trace = Trace(
        trace_id="test-1",
        request_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        query="What documents about sales?",
        response="I found 3 documents...",
        spans=[
            Span(
                span_id="s1",
                trace_id="test-1",
                span_type=SpanType.TOOL_CALL,
                name="retrieve_docs",
                tool_name="retrieve_relevant_documents",
                start_time=datetime.utcnow(),
                duration_ms=300,
                status=SpanStatus.OK
            )
        ],
        total_duration_ms=1000,
        total_tool_calls=1,
        total_tokens=200,
        tools_used=["retrieve_relevant_documents"],
        started_at=datetime.utcnow()
    )

    result = evaluator.evaluate(trace)

    assert result.overall_score > 0.8
    assert result.redundancy_count == 0
    assert len(result.issues) == 0

def test_inefficient_trace(evaluator):
    # Create trace with redundant calls
    trace = Trace(
        # ... trace with same tool called multiple times
    )

    result = evaluator.evaluate(trace)

    assert result.tool_efficiency_score < 0.8
    assert result.redundancy_count > 0
    assert any("redundant" in issue.lower() for issue in result.issues)
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ⭐ PERFECT**

Span/trace evaluation is exactly what Langfuse was built for:

1. **Native tracing**: Langfuse captures execution traces automatically
2. **Span visualization**: See tool calls, timing, hierarchy
3. **Performance analytics**: Latency percentiles, token usage
4. **No additional code**: Pydantic AI's `instrument=True` sends traces automatically

**Automatic Integration (Already Configured):**

```python
from pydantic_ai import Agent

# This is already in your agent.py!
agent = Agent(
    model="openai:gpt-4o-mini",
    instrument=True  # Sends traces to Langfuse via Logfire/OpenTelemetry
)
```

**What Langfuse Captures Automatically:**
- Agent execution spans
- Tool call spans (name, duration, inputs/outputs)
- LLM call spans (model, tokens, latency)
- Error spans with stack traces
- Span hierarchy (parent-child relationships)

**Adding Custom Span Attributes:**

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Add custom attributes to current trace
with langfuse.trace(name="agent_request") as trace:
    trace.update(
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "query_type": classify_query(query)
        }
    )

    # Agent runs here, spans captured automatically
    result = await agent.run(query)

    # Add evaluation score based on trace analysis
    trace.score(name="trace_efficiency", value=calculate_efficiency(trace))
```

**Langfuse Dashboard Benefits:**
- Trace timeline visualization
- Span duration breakdown
- Tool call frequency analysis
- Error rate tracking
- Performance percentiles (P50, P95, P99)

### Pydantic AI Support

**Fit Level: ✅ FULL SUPPORT**

Pydantic Evals provides native span-based evaluation via `HasMatchingSpan`:

**HasMatchingSpan Evaluator:**

```python
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import HasMatchingSpan

dataset = Dataset(
    cases=[
        Case(
            name="rag_query",
            inputs={"query": "Find sales documents"},
            evaluators=[
                # Verify RAG tool was called
                HasMatchingSpan(
                    query={"name_contains": "retrieve_relevant_documents"},
                    evaluation_name="used_rag_tool"
                ),
                # Verify no dangerous operations
                HasMatchingSpan(
                    query={"not_": {"name_contains": "delete"}},
                    evaluation_name="no_delete_operations"
                ),
                # Verify fast retrieval
                HasMatchingSpan(
                    query={
                        "and_": [
                            {"name_contains": "retrieve"},
                            {"max_duration": 1.0}
                        ]
                    },
                    evaluation_name="fast_retrieval"
                ),
            ]
        ),
    ]
)
```

**SpanQuery Reference:**

| Query Type | Example | Purpose |
|------------|---------|---------|
| Name matching | `{"name_contains": "tool"}` | Find spans by name |
| Attributes | `{"has_attributes": {"key": "val"}}` | Match attributes |
| Duration | `{"max_duration": 1.0}` | Performance SLA |
| Negation | `{"not_": {...}}` | Exclude patterns |
| Logical AND | `{"and_": [cond1, cond2]}` | Multiple conditions |
| Child spans | `{"some_child_has": {...}}` | Hierarchy queries |

**Custom Span Evaluator with SpanTree:**

```python
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

class ToolEfficiencyEvaluator(Evaluator):
    """Evaluate tool call efficiency from spans"""

    async def evaluate(self, ctx: EvaluatorContext) -> dict:
        span_tree = ctx.span_tree

        # Find all tool calls
        tool_spans = span_tree.find(lambda n: "tool" in n.name.lower())

        # Check for redundancy
        tool_names = [s.name for s in tool_spans]
        redundant = len(tool_names) != len(set(tool_names))

        # Calculate efficiency
        if len(tool_spans) <= 3:
            efficiency = 1.0
        elif len(tool_spans) <= 5:
            efficiency = 0.8
        else:
            efficiency = 0.6

        return {
            "tool_count": len(tool_spans),
            "redundant": redundant,
            "efficiency": max(0, efficiency - (0.2 if redundant else 0))
        }
```

**Setup Requirement:**

```python
import logfire

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
# Now span_tree is available in evaluators
```

**SpanTree API:**

| Method | Description |
|--------|-------------|
| `find(predicate)` | Find matching spans |
| `any(predicate)` | Check if any match |
| `count(predicate)` | Count matches |

| SpanNode Property | Description |
|-------------------|-------------|
| `name` | Span name |
| `duration` | Duration (seconds) |
| `attributes` | Key-value dict |
| `children` | Direct children |
| `descendants` | All descendants |
