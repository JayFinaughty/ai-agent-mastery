# Agent Evaluation Strategies Overview

## Philosophy: Local First, Then Production

This module teaches agent evaluation through two distinct phases:

1. **Local/Development** — Simple evals with `pydantic-evals`, no external services
2. **Production** — Real user data and observability with Langfuse

This mirrors how a real team adopts evaluations: start simple, add complexity as needed.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT EVALUATION FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: LOCAL DEVELOPMENT                    PHASE 2: PRODUCTION      │
│  ─────────────────────────                     ──────────────────       │
│                                                                         │
│  ┌─────────────┐                               ┌─────────────┐         │
│  │   GOLDEN    │                               │   MANUAL    │         │
│  │   DATASET   │                               │ ANNOTATION  │         │
│  │  (10 cases) │                               │  (Experts)  │         │
│  └──────┬──────┘                               └──────┬──────┘         │
│         │                                             │                 │
│         ▼                                             ▼                 │
│  ┌─────────────┐                               ┌─────────────┐         │
│  │ RULE-BASED  │                               │ RULE-BASED  │         │
│  │   (Local)   │                               │   (Prod)    │         │
│  │  Contains,  │                               │  + Langfuse │         │
│  │  Tool calls │                               │   Scores    │         │
│  └──────┬──────┘                               └──────┬──────┘         │
│         │                                             │                 │
│         ▼                                             ▼                 │
│  ┌─────────────┐                               ┌─────────────┐         │
│  │  LLM JUDGE  │                               │  LLM JUDGE  │         │
│  │   (Local)   │                               │   (Prod)    │         │
│  │  Pydantic   │                               │  Langfuse   │         │
│  │  Evals      │                               │  Built-in   │         │
│  └─────────────┘                               └──────┬──────┘         │
│                                                       │                 │
│  No Langfuse needed!                                  ▼                 │
│  Run from terminal.                            ┌─────────────┐         │
│                                                │    USER     │         │
│                                                │  FEEDBACK   │         │
│                                                │  (Ratings)  │         │
│                                                └─────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Strategy Summary

### Local Phase (Pydantic Evals)

| Strategy | What it Measures | Key Evaluators |
|----------|------------------|----------------|
| **Golden Dataset** | Regression, Benchmarks | `Dataset`, `Case` |
| **Rule-Based** | Tool calls, Content | `Contains`, `HasMatchingSpan` |
| **LLM Judge** | Quality, Relevance | `LLMJudge` with rubrics |

### Production Phase (Langfuse)

| Strategy | What it Measures | Langfuse Feature |
|----------|------------------|------------------|
| **Manual Annotation** | Expert Quality | Annotation UI |
| **Rule-Based (Prod)** | Safety, Compliance | Scores API |
| **LLM Judge (Prod)** | Quality at Scale | Built-in Judge |
| **User Feedback** | Satisfaction | Scores API |

### Honorable Mentions (Not in Core Curriculum)

| Strategy | Status | Notes |
|----------|--------|-------|
| **Span/Trace** | Honorable mention | See [honorable_mention_span_trace.md](./honorable_mention_span_trace.md) |
| **Implicit Feedback** | Honorable mention | See [honorable_mention_implicit_feedback.md](./honorable_mention_implicit_feedback.md) |
| **A/B Testing** | Honorable mention | See [honorable_mention_ab_testing.md](./honorable_mention_ab_testing.md) |
| **Cost/Efficiency** | Honorable mention | See [honorable_mention_cost_efficiency.md](./honorable_mention_cost_efficiency.md) |

---

## Framework Integration Matrix

### Pydantic AI / pydantic-evals Support

| Strategy | Support Level | Key Features |
|----------|---------------|--------------|
| Golden Dataset | ✅ FULL | `Dataset`, `Case`, `EvaluationReport` |
| Rule-Based | ✅ FULL | `Contains`, `IsInstance`, `EqualsExpected`, `HasMatchingSpan` |
| LLM Judge | ✅ FULL | `LLMJudge` evaluator with rubrics |
| Span/Trace | ✅ FULL | `HasMatchingSpan`, `SpanQuery`, `SpanTree` |
| Manual Annotation | ⚠️ Via Logfire | Logfire UI for trace review |
| User Feedback | ❌ Manual | Custom implementation |
| Implicit Feedback | ❌ Manual | Custom implementation |

### Langfuse Integration

| Strategy | Fit Level | Integration Approach |
|----------|-----------|----------------------|
| Golden Dataset | ✅ STRONG | Langfuse Datasets for test cases |
| Rule-Based | ⚠️ PARTIAL | Run locally, sync scores for analytics |
| LLM Judge | ✅ STRONG | Built-in judge, attach scores to traces |
| Span/Trace | ⭐ PERFECT | This IS Langfuse's core purpose |
| Manual Annotation | ✅ STRONG | Built-in annotation UI |
| User Feedback | ✅ STRONG | `langfuse.score(trace_id, "rating", value)` |
| Implicit Feedback | ❌ WEAK | Session-level (keep in app DB) |

### Combined Matrix

```
                          │ Pydantic │ Langfuse │ Phase       │ Video
──────────────────────────┼──────────┼──────────┼─────────────┼───────
Golden Dataset            │  ✅      │  ✅      │ Local       │ 2
Rule-Based (Local)        │  ✅      │  —       │ Local       │ 3
LLM Judge (Local)         │  ✅      │  —       │ Local       │ 4
Manual Annotation         │  ⚠️      │  ✅      │ Production  │ 5
Rule-Based (Prod)         │  ✅      │  ⚠️      │ Production  │ 6
LLM Judge (Prod)          │  ✅      │  ✅      │ Production  │ 7
User Feedback             │  ❌      │  ✅      │ Production  │ 8
```

---

## Implementation Order

### Phase 1: Local Development (Videos 2-4)

Start simple. No external services required.

| Video | Feature | What You Build |
|-------|---------|----------------|
| 2 | Golden Dataset | 10-case YAML dataset, `run_evals.py` script |
| 3 | Rule-Based (Local) | `Contains`, tool call checks |
| 4 | LLM Judge (Local) | `LLMJudge` with quality rubrics |

**After Phase 1, you can evaluate your agent locally with:**
```bash
python run_evals.py
```

### Phase 2: Production (Videos 5-8)

Add Langfuse for real user data and observability.

| Video | Feature | What You Build |
|-------|---------|----------------|
| 5 | Manual Annotation | Expert review workflow via Langfuse UI |
| 6 | Rule-Based (Prod) | Sync rule scores to Langfuse |
| 7 | LLM Judge (Prod) | Async scoring on production traces |
| 8 | User Feedback | Frontend widget, feedback collection |

> **Note:** Langfuse setup was covered in Module 6. Video 5 includes a brief intro to Langfuse for evaluations.

---

## When to Use Each Strategy

```
                    AGENT LIFECYCLE
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  LOCAL DEV            →        PRODUCTION   │
    │  ─────────                     ──────────   │
    │                                              │
    │  ✅ Golden Dataset             ✅ Manual     │
    │  ✅ Rule-Based (local)         ✅ Rule-Based │
    │  ✅ LLM Judge (local)          ✅ LLM Judge  │
    │                                ✅ User       │
    │                                              │
    │  "Does it work?"       →    "Is it good?"   │
    │                                              │
    └──────────────────────────────────────────────┘
```

---

## Strategy Documents

Detailed implementation plans for each strategy:

| File | Strategy | Phase |
|------|----------|-------|
| [01_INTRO_TO_EVALS.md](./01_INTRO_TO_EVALS.md) | Introduction | Conceptual |
| [02_GOLDEN_DATASET.md](./02_GOLDEN_DATASET.md) | Golden Dataset | Local |
| [03_RULE_BASED_LOCAL.md](./03_RULE_BASED_LOCAL.md) | Rule-Based (Local) | Local |
| [04_LLM_JUDGE_LOCAL.md](./04_LLM_JUDGE_LOCAL.md) | LLM Judge (Local) | Local |
| [05_MANUAL_ANNOTATION.md](./05_MANUAL_ANNOTATION.md) | Manual Annotation | Production |
| [06_RULE_BASED_PROD.md](./06_RULE_BASED_PROD.md) | Rule-Based (Prod) | Production |
| [07_LLM_JUDGE_PROD.md](./07_LLM_JUDGE_PROD.md) | LLM Judge (Prod) | Production |
| [08_USER_FEEDBACK.md](./08_USER_FEEDBACK.md) | User Feedback | Production |
| [honorable_mention_span_trace.md](./honorable_mention_span_trace.md) | Span/Trace | (Honorable Mention) |
| [honorable_mention_implicit_feedback.md](./honorable_mention_implicit_feedback.md) | Implicit Feedback | (Honorable Mention) |
| [honorable_mention_ab_testing.md](./honorable_mention_ab_testing.md) | A/B Testing | (Honorable Mention) |
| [honorable_mention_cost_efficiency.md](./honorable_mention_cost_efficiency.md) | Cost/Efficiency | (Honorable Mention) |

---

## Pydantic Evals Reference

### Built-in Evaluators

| Evaluator | Purpose | Example |
|-----------|---------|---------|
| `EqualsExpected` | Exact match | Output must equal expected |
| `Contains` | Substring check | Response contains "hello" |
| `IsInstance` | Type validation | Output is a string |
| `MaxDuration` | Performance SLA | Response under 2 seconds |
| `HasMatchingSpan` | Tool verification | `search_documents` was called |

### LLM Judge

```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric="Response is helpful, accurate, and addresses the user's question",
    include_input=True,
)
```

### Running Evaluations

```python
from pydantic_evals import Dataset

report = dataset.evaluate_sync(my_agent_function)
report.print(include_input=True, include_output=True)
```

---

## Langfuse Features Reference

### Core Features

| Feature | Purpose | Used In |
|---------|---------|---------|
| **Traces** | Capture execution flow | Span/Trace analysis |
| **Scores** | Attach numeric evaluations | User Feedback, LLM Judge, Rule-Based |
| **Datasets** | Store test cases | Golden Dataset (production) |
| **Annotations** | Human review UI | Manual Annotation |

### Score Types

```python
# Binary feedback (thumbs up/down)
langfuse.score(trace_id, name="user_thumbs_up", value=1.0)

# Numeric quality (0-1)
langfuse.score(trace_id, name="llm_judge_quality", value=0.85)

# Categorical with comment
langfuse.score(trace_id, name="rule_safety", value=1.0, comment="All rules passed")
```

---

## Quick Start

### Local Evals (5 minutes)

```bash
pip install pydantic-evals
```

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, LLMJudge

dataset = Dataset(cases=[
    Case(
        name="greeting",
        inputs={"query": "Hello!"},
        evaluators=[Contains("hello", case_sensitive=False)],
    ),
    Case(
        name="quality",
        inputs={"query": "What is Python?"},
        evaluators=[LLMJudge(rubric="Accurately describes Python")],
    ),
])

report = dataset.evaluate_sync(my_agent)
report.print()
```

### Production Evals (after Langfuse setup)

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Scores automatically attached to traces
langfuse.score(
    trace_id=current_trace_id,
    name="user_feedback",
    value=1.0,
    comment="Thumbs up"
)
```
