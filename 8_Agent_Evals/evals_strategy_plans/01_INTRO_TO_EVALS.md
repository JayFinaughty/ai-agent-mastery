# Introduction to Agent Evaluations

> **Video 1** | **Tag:** _(no code changes)_ | **Phase:** Conceptual

## Overview

**What this video covers**: Why evaluations matter, the two phases of eval adoption, and what you'll learn in this module.

**No code in this video.** This is a conceptual introduction to set up the rest of the module.

## What You'll Learn in This Video

1. Why evaluations are critical for production agents
2. The difference between local and production evaluations
3. The two-phase approach: Local Development → Production
4. Overview of evaluation strategies we'll cover
5. Introduction to pydantic-evals and Langfuse

---

## Why Evaluations Matter

### The Problem

You build an agent. It works in testing. You deploy it. Then:

- Users complain about wrong answers
- The agent calls the wrong tools
- Responses are too long, too short, or off-topic
- You can't tell if your prompt changes helped or hurt

**Without evals, you're flying blind.**

### The Solution

Evaluations let you:

- **Catch regressions** before they hit production
- **Measure quality** objectively, not just "it seems fine"
- **Compare versions** of your agent
- **Build confidence** for deployments

---

## Two Phases of Evaluation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PHASE 1: LOCAL DEVELOPMENT          PHASE 2: PRODUCTION              │
│   ───────────────────────────          ───────────────────              │
│                                                                         │
│   ┌─────────────────────┐              ┌─────────────────────┐         │
│   │  Golden Dataset     │              │  Real User Traces   │         │
│   │  (10 test cases)    │              │  (1000s per day)    │         │
│   └──────────┬──────────┘              └──────────┬──────────┘         │
│              │                                    │                     │
│              ▼                                    ▼                     │
│   ┌─────────────────────┐              ┌─────────────────────┐         │
│   │  pydantic-evals     │              │  Langfuse           │         │
│   │  - Contains         │              │  - Built-in evals   │         │
│   │  - HasMatchingSpan  │              │  - Annotations      │         │
│   │  - LLMJudge         │              │  - User feedback    │         │
│   └──────────┬──────────┘              └──────────┬──────────┘         │
│              │                                    │                     │
│              ▼                                    ▼                     │
│   ┌─────────────────────┐              ┌─────────────────────┐         │
│   │  Terminal Report    │              │  Dashboard          │         │
│   │  "8/10 passed"      │              │  Analytics          │         │
│   └─────────────────────┘              └─────────────────────┘         │
│                                                                         │
│   No external services needed          Requires Langfuse setup         │
│   Fast iteration                       Real user data                   │
│   Before deployment                    After deployment                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Local Development (Videos 2-4)

**When:** Before you deploy or after making changes

**Tool:** pydantic-evals

**What you do:**
1. Create a golden dataset (10 test cases)
2. Run evaluators on your dataset
3. Fix issues, re-run, iterate

**Key insight:** Start simple. Andrew Ng says: "When I set up evals for my agent at first, I do a 10-question golden dataset. That's all you need to start."

### Phase 2: Production (Videos 5-9)

**When:** After deployment, on real user data

**Tool:** Langfuse

**What you do:**
1. Traces flow into Langfuse automatically
2. Evaluators score traces (automated or human)
3. Monitor quality over time
4. Investigate low-scoring traces

**Key insight:** You can't manually review every production response. Automated evals + sampling let you maintain quality at scale.

---

## Evaluation Strategies Overview

### Local Phase

| Video | Strategy | What it Checks |
|-------|----------|----------------|
| 2 | Golden Dataset | Basic response validation |
| 3 | Rule-Based | Tool calls, content patterns |
| 4 | LLM Judge | Subjective quality |

### Production Phase

| Video | Strategy | What it Checks |
|-------|----------|----------------|
| 5 | Manual Annotation | Human expert review (+ Langfuse intro) |
| 6 | Rule-Based (Prod) | PII, safety, compliance |
| 7 | LLM Judge (Prod) | Quality at scale |
| 8 | User Feedback | Real user satisfaction |

---

## The Tools

### pydantic-evals

A Python library for running evaluations locally.

**Key features:**
- YAML-based test cases
- Built-in evaluators: `Contains`, `HasMatchingSpan`, `LLMJudge`
- Custom evaluator support
- Terminal reports

```python
# Example: Run evals from command line
python evals/run_evals.py
```

### Langfuse

An observability platform for LLM applications.

**Key features:**
- Automatic trace capture
- Built-in LLM-as-Judge evaluators
- Human annotation queues
- Dashboards and analytics

```
Langfuse Dashboard
├── Traces (all agent interactions)
├── Scores (evaluation results)
├── Annotations (human review)
└── Analytics (trends over time)
```

---

## What You'll Build in This Module

By the end of this module, you'll have:

1. **A golden dataset** with 10 test cases covering your agent's core functionality

2. **Local evaluation pipeline** that runs in seconds and catches regressions

3. **Production monitoring** with automatic quality scoring

4. **Human review workflow** for calibrating AI judges

5. **User feedback collection** to measure real satisfaction

---

## Key Principles

### 1. Start Simple

Don't over-engineer. A 10-case golden dataset is enough to start.

### 2. Local First

Build and test locally before adding production complexity.

### 3. Rule-Based Before AI

Use fast, free checks first. Only use LLM judges for subjective quality.

### 4. Reuse, Don't Rebuild

Your local evaluators work in production. Just add Langfuse sync.

### 5. Sample in Production

You don't need to evaluate every trace. 10% sampling is often enough.

---

## Module Structure

```
Module 8: Agent Evals
│
├── Video 1: Introduction (this video)
│
├── LOCAL PHASE
│   ├── Video 2: Golden Dataset
│   ├── Video 3: Rule-Based Evaluation
│   └── Video 4: LLM-as-Judge
│
├── PRODUCTION PHASE
│   ├── Video 5: Manual Annotation (+ Langfuse intro)
│   ├── Video 6: Rule-Based (Production)
│   ├── Video 7: LLM Judge (Production)
│   └── Video 8: User Feedback
│
└── (End of module)
```

---

## What's Next

**Video 2: Golden Dataset** — Create your first evaluation dataset with pydantic-evals.

We'll:
1. Set up the `evals/` directory
2. Create a YAML dataset with 10 test cases
3. Write a simple evaluation runner
4. Run your first evaluation

No Langfuse, no external services. Just Python and your agent.

---

## Resources

- [pydantic-evals Documentation](https://ai.pydantic.dev/evals/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Andrew Ng on Evaluations](https://www.deeplearning.ai/) — "Start with 10 test cases"
