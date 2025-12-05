# Strategy: Span & Trace Analysis

> **Video 9** | **Tag:** `module-8-08-span-trace` | **Phase:** Production

## Overview

**What it is**: Analyzing the execution trace - the sequence of steps, tool calls, and decisions the agent made. Goes beyond final output to evaluate HOW the agent arrived at its answer.

**Philosophy**: For complex agents, the path matters as much as the destination. An agent might produce a correct answer through flawed reasoning or inefficient tool sequences.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGFUSE TRACE VISUALIZATION                         │
│                                                                         │
│  User Query: "What were our Q3 sales?"                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ TRACE: agent_request                            Total: 1.2s     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │  ├─ GENERATION: gpt-4o (planning)              250ms           │   │
│  │  │   └─ tokens: 150 in, 50 out                                 │   │
│  │  │                                                              │   │
│  │  ├─ TOOL: retrieve_relevant_documents          450ms           │   │
│  │  │   ├─ query: "Q3 sales figures"                              │   │
│  │  │   └─ results: 3 documents                                   │   │
│  │  │                                                              │   │
│  │  └─ GENERATION: gpt-4o (response)              380ms           │   │
│  │      └─ tokens: 800 in, 200 out                                │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Langfuse captures this automatically with instrument=True             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## When to Use

✅ **Good for:**
- Debugging agent behavior
- Understanding failure modes
- Optimizing tool usage
- Performance monitoring
- Detecting inefficient patterns

❌ **Not for:**
- Output quality assessment (use LLM Judge)
- User satisfaction (use User Feedback)
- Simple pass/fail checks (use Rule-Based)

---

## Langfuse Trace Data Model

Langfuse uses a hierarchical model:

```
TRACE (one per request)
├── GENERATION (LLM call)
│   ├── model, tokens, latency
│   └── prompt, completion
├── SPAN (unit of work)
│   ├── name, duration
│   └── attributes
├── TOOL (tool call)
│   ├── name, input, output
│   └── duration, status
└── EVENT (discrete point in time)
```

### Observation Types

| Type | Purpose | Example |
|------|---------|---------|
| **Trace** | Top-level request | User conversation turn |
| **Generation** | LLM API call | GPT-4o completion |
| **Span** | Generic work unit | Custom processing step |
| **Tool** | Tool invocation | `retrieve_relevant_documents` |
| **Event** | Discrete occurrence | Error, state change |

---

## Implementation

### Automatic Trace Capture

Pydantic AI with `instrument=True` sends traces to Langfuse automatically:

```python
# This is already in your agent.py
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    instrument=True  # Sends to Langfuse via OpenTelemetry
)
```

**What's captured automatically:**
- Agent execution spans
- Tool calls (name, inputs, outputs, duration)
- LLM calls (model, tokens, latency)
- Errors with stack traces

### Adding Custom Metadata

Enrich traces with context:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Add metadata to current trace
langfuse.trace(
    name="agent_request",
    user_id=user_id,
    session_id=session_id,
    metadata={
        "query_type": classify_query(query),
        "user_tier": user.tier
    }
)
```

---

## Analyzing Traces in Langfuse

### Dashboard Views

**1. Trace List**
- Filter by time, user, session
- Sort by latency, token usage
- Search by content

**2. Trace Detail**
- Timeline visualization
- Span hierarchy
- Token usage breakdown
- Input/output for each step

**3. Analytics**
- Latency percentiles (P50, P95, P99)
- Token usage trends
- Tool call frequency
- Error rates

### Common Filters

| Filter | Use Case |
|--------|----------|
| `latency > 5s` | Find slow requests |
| `tokens > 2000` | Find expensive requests |
| `status = error` | Find failures |
| `tool = web_search` | Filter by tool usage |

---

## Trace-Based Evaluation

### Pattern 1: Tool Usage Analysis

Check which tools are being called:

```python
from langfuse import Langfuse

langfuse = Langfuse()

def analyze_tool_usage(days: int = 7):
    """Analyze tool usage patterns."""

    traces = langfuse.fetch_traces(limit=1000)

    tool_counts = {}
    tool_latencies = {}

    for trace in traces.data:
        for obs in trace.observations:
            if obs.type == "TOOL":
                tool_name = obs.name
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                if obs.latency:
                    if tool_name not in tool_latencies:
                        tool_latencies[tool_name] = []
                    tool_latencies[tool_name].append(obs.latency)

    # Report
    print("Tool Usage Summary")
    print("=" * 40)
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        avg_latency = (
            sum(tool_latencies.get(tool, [0])) /
            len(tool_latencies.get(tool, [1]))
        )
        print(f"{tool}: {count} calls, avg {avg_latency:.0f}ms")

    return tool_counts, tool_latencies
```

### Pattern 2: Latency Analysis

Find performance bottlenecks:

```python
def analyze_latency(traces):
    """Find slow traces and identify bottlenecks."""

    slow_traces = [t for t in traces if t.latency and t.latency > 5000]

    bottlenecks = []
    for trace in slow_traces:
        # Find slowest observation
        slowest = max(trace.observations, key=lambda o: o.latency or 0)
        bottlenecks.append({
            "trace_id": trace.id,
            "total_latency": trace.latency,
            "bottleneck": slowest.name,
            "bottleneck_latency": slowest.latency
        })

    return bottlenecks
```

### Pattern 3: Error Pattern Detection

Understand failure modes:

```python
def analyze_errors(traces):
    """Categorize and count error patterns."""

    error_traces = [t for t in traces if t.status == "ERROR"]

    error_patterns = {}
    for trace in error_traces:
        # Find error observation
        error_obs = next(
            (o for o in trace.observations if o.status == "ERROR"),
            None
        )
        if error_obs:
            error_type = error_obs.metadata.get("error_type", "unknown")
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

    return error_patterns
```

### Pattern 4: Efficiency Scoring

Score traces based on execution efficiency:

```python
def score_trace_efficiency(trace) -> float:
    """Score a trace's efficiency (0-1)."""

    score = 1.0
    issues = []

    # Count tool calls
    tool_calls = [o for o in trace.observations if o.type == "TOOL"]

    # Penalize excessive tool calls
    if len(tool_calls) > 5:
        score -= 0.2
        issues.append(f"High tool count: {len(tool_calls)}")

    # Check for redundant calls (same tool, same input)
    seen = set()
    for call in tool_calls:
        key = (call.name, str(call.input))
        if key in seen:
            score -= 0.1
            issues.append(f"Redundant call: {call.name}")
        seen.add(key)

    # Penalize slow traces
    if trace.latency and trace.latency > 10000:
        score -= 0.2
        issues.append(f"Slow execution: {trace.latency}ms")

    # Attach score to trace
    langfuse.score(
        trace_id=trace.id,
        name="trace_efficiency",
        value=max(0, score),
        comment="; ".join(issues) if issues else "Efficient execution"
    )

    return max(0, score)
```

---

## Automated Trace Evaluation

### Batch Evaluation Script

```python
# backend_agent_api/evals/evaluate_traces.py

import asyncio
from langfuse import Langfuse

langfuse = Langfuse()

async def evaluate_recent_traces(hours: int = 24):
    """Evaluate traces from the last N hours."""

    traces = langfuse.fetch_traces(limit=500)

    results = {
        "total": 0,
        "efficient": 0,
        "slow": 0,
        "errors": 0
    }

    for trace in traces.data:
        results["total"] += 1

        # Score efficiency
        score = score_trace_efficiency(trace)

        if score >= 0.8:
            results["efficient"] += 1
        elif trace.latency and trace.latency > 5000:
            results["slow"] += 1

        if trace.status == "ERROR":
            results["errors"] += 1

    # Print summary
    print(f"\nTrace Evaluation Summary (last {hours}h)")
    print("=" * 40)
    print(f"Total traces: {results['total']}")
    print(f"Efficient (>0.8): {results['efficient']} ({results['efficient']/results['total']*100:.1f}%)")
    print(f"Slow (>5s): {results['slow']}")
    print(f"Errors: {results['errors']}")

    return results

if __name__ == "__main__":
    asyncio.run(evaluate_recent_traces())
```

---

## What to Look For

### Tool Usage Patterns

| Pattern | Issue | Action |
|---------|-------|--------|
| Same tool called repeatedly | Redundancy | Improve agent logic |
| Wrong tool selected | Misrouting | Improve tool descriptions |
| Tool never used | Underutilization | Review tool discovery |
| Excessive tool calls | Inefficiency | Add planning step |

### Performance Patterns

| Pattern | Issue | Action |
|---------|-------|--------|
| P95 latency > 10s | Slow responses | Optimize bottlenecks |
| Token usage > 5k | Expensive | Reduce context |
| Retry spikes | Reliability | Improve error handling |

### Error Patterns

| Pattern | Issue | Action |
|---------|-------|--------|
| Tool timeouts | External service | Add caching/fallbacks |
| Rate limits | Capacity | Implement backoff |
| Parsing errors | Output format | Improve prompts |

---

## Langfuse Dashboard Tips

### Create Custom Views

1. **Slow Traces**: `latency > 5000`
2. **Tool-Heavy**: `observations.count > 5`
3. **Expensive**: `usage.total > 2000`
4. **Failures**: `status = ERROR`

### Set Up Alerts

Configure alerts for:
- P95 latency exceeds threshold
- Error rate spikes
- Token usage anomalies

### Export for Analysis

```python
# Export traces for deeper analysis
traces = langfuse.fetch_traces(limit=10000)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame([{
    "id": t.id,
    "latency": t.latency,
    "tokens": t.usage.total if t.usage else 0,
    "tool_count": len([o for o in t.observations if o.type == "TOOL"]),
    "status": t.status
} for t in traces.data])

# Analyze
print(df.describe())
```

---

## What NOT to Build

❌ **Custom trace capture** → Pydantic AI + Langfuse handles it
❌ **Custom span models** → Use Langfuse's data model
❌ **Database tables for traces** → Langfuse stores everything
❌ **Custom dashboard** → Use Langfuse UI

**The traces are already being captured. Just analyze them.**

---

## Integration with Other Strategies

### With Rule-Based (Local - Video 3)

Use `HasMatchingSpan` for local golden dataset testing:

```yaml
# Already covered in Video 3
evaluators:
  - HasMatchingSpan:
      query:
        name_contains: "retrieve_relevant_documents"
```

### With LLM Judge (Prod - Video 8)

Score traces based on execution quality:

```python
# Combine trace efficiency with judge score
trace_score = score_trace_efficiency(trace)
judge_score = await run_llm_judge(trace)

overall = 0.5 * trace_score + 0.5 * judge_score
```

### With Manual Annotation (Video 6)

Use traces to inform annotation:

1. Filter traces by efficiency score
2. Add low-scoring traces to annotation queue
3. Expert reviews full trace context

---

## Resources

- [Langfuse Tracing Overview](https://langfuse.com/docs/observability/overview)
- [Langfuse Data Model](https://langfuse.com/docs/observability/data-model)
- [Observation Types](https://langfuse.com/docs/observability/features/observation-types)
- [OpenTelemetry Integration](https://langfuse.com/integrations/native/opentelemetry)
- [pydantic-evals Span-Based Evaluators](https://ai.pydantic.dev/evals/evaluators/span-based/)
