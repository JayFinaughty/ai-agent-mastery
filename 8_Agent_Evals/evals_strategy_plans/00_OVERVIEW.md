# Agent Evaluation Strategies Overview

## The 8-Strategy Evaluation Framework

This module implements a comprehensive evaluation system using 8 complementary strategies. Each strategy measures different aspects of agent quality and together they provide complete coverage of agent behavior.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT EVALUATION FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   USER      │  │  IMPLICIT   │  │   EXPERT    │  │    LLM      │   │
│  │  FEEDBACK   │  │  BEHAVIORAL │  │  ANNOTATION │  │   JUDGE     │   │
│  │  (Explicit) │  │  (Implicit) │  │  (Manual)   │  │  (Model)    │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │          │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐   │
│  │  Strategy 1 │  │  Strategy 2 │  │  Strategy 3 │  │  Strategy 4 │   │
│  │  Ratings &  │  │  Engagement │  │  Domain     │  │  AI-Powered │   │
│  │  Surveys    │  │  Patterns   │  │  Experts    │  │  Assessment │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │    RULE     │  │   GOLDEN    │  │    SPAN     │  │     A/B     │   │
│  │    BASED    │  │   DATASET   │  │    TRACE    │  │  TESTING    │   │
│  │ (Determin.) │  │  (Ground)   │  │  (Flow)     │  │ (Compare)   │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │          │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐   │
│  │  Strategy 5 │  │  Strategy 6 │  │  Strategy 7 │  │  Strategy 8 │   │
│  │  Safety &   │  │  Regression │  │  Execution  │  │  Version    │   │
│  │  Compliance │  │  Testing    │  │  Analysis   │  │  Comparison │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Framework Integration Summary

### Langfuse Integration

| Strategy | Langfuse Fit | Integration Approach |
|----------|--------------|----------------------|
| 1. User Feedback | ✅ STRONG | `langfuse.score(trace_id, "user_rating", value)` |
| 2. Implicit Feedback | ❌ WEAK | Keep in app DB; session-level, not trace-level |
| 3. Manual Annotation | ✅ STRONG | Use Langfuse's built-in annotation UI |
| 4. LLM Judge | ✅ STRONG | Attach judge scores to traces |
| 5. Rule-Based | ⚠️ PARTIAL | Run locally (real-time), sync scores for analytics |
| 6. Golden Dataset | ✅ STRONG | Use Langfuse Datasets for test cases |
| 7. Span/Trace | ⭐ PERFECT | This IS Langfuse's core purpose |
| 8. A/B Testing | ✅ STRONG | Compare experiments via Langfuse Datasets |

### Pydantic AI Support

| Strategy | Pydantic AI Support | Key Features Used |
|----------|---------------------|-------------------|
| 1. User Feedback | ❌ Not supported | Implement manually |
| 2. Implicit Feedback | ❌ Not supported | Implement manually |
| 3. Manual Annotation | ⚠️ Via Logfire | Logfire UI for trace review |
| 4. LLM Judge | ✅ FULL | `LLMJudge` evaluator with rubrics |
| 5. Rule-Based | ✅ FULL | `Contains`, `IsInstance`, `EqualsExpected`, Pydantic validation |
| 6. Golden Dataset | ✅ FULL | `Dataset`, `Case`, `EvaluationReport` |
| 7. Span/Trace | ✅ FULL | `HasMatchingSpan`, `SpanQuery`, `SpanTree` |
| 8. A/B Testing | ⚠️ Partial | `Agent.override()` + compare `EvaluationReport`s |

### Combined Framework Matrix

```
                          │ Langfuse │ Pydantic AI │ Implementation
──────────────────────────┼──────────┼─────────────┼──────────────────
1. User Feedback          │  ✅      │  ❌         │ Custom + Langfuse scores
2. Implicit Feedback      │  ❌      │  ❌         │ Custom (app DB only)
3. Manual Annotation      │  ✅      │  ⚠️         │ Langfuse UI
4. LLM Judge              │  ✅      │  ✅         │ Pydantic Evals + Langfuse
5. Rule-Based             │  ⚠️      │  ✅         │ Pydantic Evals + sync
6. Golden Dataset         │  ✅      │  ✅         │ Both (Pydantic primary)
7. Span/Trace             │  ✅      │  ✅         │ Both (integrated)
8. A/B Testing            │  ✅      │  ⚠️         │ Agent.override + Langfuse
```

---

## Strategy Summary

| # | Strategy | Type | Measures | Frequency | Cost |
|---|----------|------|----------|-----------|------|
| 1 | User Feedback | Explicit | Satisfaction, Ratings | Per-request (optional) | Free |
| 2 | Implicit Feedback | Behavioral | Engagement, Patterns | Every request | Free |
| 3 | Manual Annotation | Expert | Quality, Accuracy | Sampled (1-5%) | High |
| 4 | Model-Based | LLM Judge | Quality, Relevance | Sampled (10-20%) | Medium |
| 5 | Rule-Based | Deterministic | Safety, Format | Every request | Free |
| 6 | Golden Dataset | Ground Truth | Regression, Benchmarks | CI/CD + Scheduled | Low |
| 7 | Span/Trace | Execution Flow | Tool Use, Efficiency | Every request | Free |
| 8 | A/B Testing | Comparative | Version Comparison | On-demand | Medium |

## When to Use Each Strategy

```
                    AGENT LIFECYCLE
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  DEVELOPMENT        STAGING        PRODUCTION│
    │  ───────────        ───────        ──────────│
    │                                              │
    │  ✅ Golden Dataset  ✅ Golden     ✅ User     │
    │  ✅ Rule-Based      ✅ LLM Judge  ✅ Implicit │
    │  ✅ Span/Trace      ✅ Manual     ✅ LLM Judge│
    │  ✅ A/B Testing     ✅ A/B Test   ✅ Span     │
    │                     ✅ Span/Trace ✅ Rule    │
    └──────────────────────────────────────────────┘
```

## Strategy Interdependencies

```
Golden Dataset ──────► Regression Baseline ──────► A/B Testing
      │                                                 │
      ▼                                                 │
LLM Judge ◄──────────► Manual Annotation (Calibration) │
      │                                                 │
      ▼                                                 │
Rule-Based ──────────► Safety Gates                    │
      │                                                 │
      ▼                                                 │
User Feedback ◄──────► Implicit Feedback (Correlation) │
      │                                                 │
      ▼                                                 │
Span/Trace ─────────► Optimization Insights ◄──────────┘
```

## Implementation Order

**Phase 1: Foundation (Week 1)**
- Strategy 5: Rule-Based (safety gates, required for production)
- Strategy 7: Span/Trace (instrumentation, enables all other analysis)

**Phase 2: Automated Evals (Week 2)**
- Strategy 6: Golden Dataset (regression testing)
- Strategy 4: Model-Based LLM Judge (quality scoring)

**Phase 3: Human-in-Loop (Week 3)**
- Strategy 1: User Feedback (frontend integration)
- Strategy 2: Implicit Feedback (behavioral tracking)
- Strategy 3: Manual Annotation (expert review workflow)

**Phase 4: Optimization (Week 4)**
- Strategy 8: A/B Testing (version comparison)

## Files in This Directory

| File | Strategy | Description |
|------|----------|-------------|
| `01_USER_FEEDBACK.md` | Strategy 1 | Explicit user ratings and feedback |
| `02_IMPLICIT_FEEDBACK.md` | Strategy 2 | Behavioral engagement analysis |
| `03_MANUAL_ANNOTATION.md` | Strategy 3 | Expert review and annotation |
| `04_MODEL_BASED_LLM_JUDGE.md` | Strategy 4 | AI-powered quality assessment |
| `05_RULE_BASED.md` | Strategy 5 | Deterministic safety and validation |
| `06_GOLDEN_DATASET.md` | Strategy 6 | Ground truth regression testing |
| `07_SPAN_TRACE_BASED.md` | Strategy 7 | Execution flow analysis |
| `08_AB_COMPARATIVE_TESTING.md` | Strategy 8 | Version comparison testing |

---

## Pydantic AI Evaluators Reference

### Built-in Evaluators (Rule-Based)

| Evaluator | Purpose | Strategy |
|-----------|---------|----------|
| `EqualsExpected` | Exact match validation | 5, 6 |
| `Contains` | Substring/value presence | 5, 6 |
| `IsInstance` | Type validation | 5, 6 |
| `MaxDuration` | Performance SLA | 5, 7 |

### LLM Judge Evaluators

| Evaluator | Purpose | Strategy |
|-----------|---------|----------|
| `LLMJudge` | Subjective quality assessment | 4 |
| `judge_input_output()` | Evaluate with input context | 4 |
| `judge_output_expected()` | Compare to expected output | 4, 6 |

### Span-Based Evaluators

| Evaluator | Purpose | Strategy |
|-----------|---------|----------|
| `HasMatchingSpan` | Verify span conditions | 7 |
| `SpanQuery` | Complex span matching | 7 |
| `SpanTree` | Custom span traversal | 7 |

### Testing Utilities

| Utility | Purpose | Strategy |
|---------|---------|----------|
| `TestModel` | Mock LLM for unit tests | 6 |
| `FunctionModel` | Custom test logic | 6 |
| `Agent.override()` | Swap model/prompt | 8 |
| `capture_run_messages()` | Inspect message flow | 7 |

---

## Langfuse Features Reference

### Core Features Used

| Feature | Purpose | Strategies |
|---------|---------|------------|
| **Traces** | Capture execution flow | 7 |
| **Scores** | Attach numeric evaluations | 1, 4, 5 |
| **Datasets** | Store test cases | 6, 8 |
| **Annotations** | Human review UI | 3 |
| **Experiments** | Compare dataset runs | 8 |

### Score Types

```python
# Binary feedback
langfuse.score(trace_id, name="user_thumbs_up", value=1.0)

# Numeric quality
langfuse.score(trace_id, name="llm_judge_quality", value=0.85)

# Categorical
langfuse.score(trace_id, name="rule_safety", value=1.0, comment="All rules passed")
```

---

## Combined Scoring Example

When all strategies are implemented, you get comprehensive evaluation:

```
Request: "What documents do you have about Q3 sales?"
Response: "I found 3 documents about Q3 sales..."

┌─────────────────────────────────────────────────────────┐
│ EVALUATION RESULTS                                      │
├─────────────────────────────────────────────────────────┤
│ Strategy 1 - User Feedback:     0.90 (thumbs up)       │
│ Strategy 2 - Implicit:          0.85 (no follow-up)    │
│ Strategy 3 - Expert:            0.88 (sampled review)  │
│ Strategy 4 - LLM Judge:         0.92 (quality score)   │
│ Strategy 5 - Rule-Based:        1.00 (all checks pass) │
│ Strategy 6 - Golden Dataset:    0.95 (matches expected)│
│ Strategy 7 - Span/Trace:        1.00 (correct tool use)│
│ Strategy 8 - A/B Test:          N/A (baseline version) │
├─────────────────────────────────────────────────────────┤
│ WEIGHTED OVERALL SCORE:         0.91                   │
└─────────────────────────────────────────────────────────┘

LANGFUSE TRACE VIEW:
├── Trace: req_abc123
│   ├── Scores:
│   │   ├── user_feedback: 0.90
│   │   ├── llm_judge: 0.92
│   │   └── rule_safety: 1.00
│   ├── Spans:
│   │   ├── agent_run (1200ms)
│   │   │   ├── retrieve_relevant_documents (450ms)
│   │   │   └── response_generation (750ms)
│   └── Annotations: 1 expert review
```

## Weighting Configuration

Different use cases may weight strategies differently:

```python
# Production API (user satisfaction focus)
PRODUCTION_WEIGHTS = {
    "user_feedback": 0.25,
    "implicit_feedback": 0.15,
    "manual_annotation": 0.10,
    "llm_judge": 0.20,
    "rule_based": 0.10,      # Must pass (gate)
    "golden_dataset": 0.05,
    "span_trace": 0.05,
    "ab_testing": 0.10,
}

# Development (accuracy focus)
DEVELOPMENT_WEIGHTS = {
    "user_feedback": 0.05,
    "implicit_feedback": 0.05,
    "manual_annotation": 0.15,
    "llm_judge": 0.25,
    "rule_based": 0.15,
    "golden_dataset": 0.25,
    "span_trace": 0.10,
    "ab_testing": 0.00,
}

# Safety-Critical (compliance focus)
SAFETY_WEIGHTS = {
    "user_feedback": 0.05,
    "implicit_feedback": 0.05,
    "manual_annotation": 0.20,
    "llm_judge": 0.15,
    "rule_based": 0.30,      # Heavy weight on rules
    "golden_dataset": 0.15,
    "span_trace": 0.05,
    "ab_testing": 0.05,
}
```
