# Tag Strategy for Module 8 Implementation

## Overview

Tags mark the codebase state after each video's implementation. The tag number matches the video number (minus the intro), making it easy to checkout the exact state shown in any video.

**Key Principle:** Implementation order = Teaching order = Tag order

## Tag Naming Convention

```
module-8-{number}-{feature-name}
```

Tags are numbered sequentially by implementation/teaching order:

- `module-8-01-golden-dataset`
- `module-8-02-rule-based-local`
- `module-8-03-llm-judge-local`
- etc.

## Implementation Order

### Phase 1: Local Development Evals (Pydantic AI)

No external services. Simple Python scripts with `pydantic-evals`.

| Order | Feature | Tag | What's Added |
|-------|---------|-----|--------------|
| 1 | Golden Dataset | `module-8-01-golden-dataset` | YAML test cases, evaluation runner |
| 2 | Rule-Based (Local) | `module-8-02-rule-based-local` | Deterministic checks (Contains, tool calls) |
| 3 | LLM Judge (Local) | `module-8-03-llm-judge-local` | LLMJudge evaluator with rubrics |

### Phase 2: Production Evals (Langfuse)

Real user data, tracing, and observability.

| Order | Feature | Tag | What's Added |
|-------|---------|-----|--------------|
| 4 | Langfuse Setup | `module-8-04-langfuse-setup` | Tracing integration, score sync |
| 5 | Manual Annotation | `module-8-05-manual-annotation` | Annotation workflow, Langfuse UI |
| 6 | Rule-Based (Prod) | `module-8-06-rule-based-prod` | Score sync to Langfuse, alerts |
| 7 | LLM Judge (Prod) | `module-8-07-llm-judge-prod` | Langfuse built-in judge, async scoring |
| 8 | Span/Trace | `module-8-08-span-trace` | Trace-based evaluation, tool flow analysis |
| 9 | User Feedback | `module-8-09-user-feedback` | Frontend widget, feedback scores |

> **Note:** Tags are cumulative. Each tag includes all previous implementations.

---

## Workflow

### Implementing Each Feature

```bash
# Ensure you're on the prep branch
git checkout module-8-prep-evals

# Implement feature N
# ... make code changes ...

# Commit
git add -A
git commit -m "Implement Video N: {Feature Name}"

# Tag this state
git tag module-8-0N-{feature-name}

# Push commit and tag
git push origin module-8-prep-evals
git push origin module-8-0N-{feature-name}
```

### Checking Out a Specific State

```bash
# List available tags
git tag -l "module-8-*"

# Checkout specific state (detached HEAD)
git checkout module-8-03-llm-judge-local

# Return to latest
git checkout module-8-prep-evals
```

---

## Video to Tag Mapping

| Video # | Video Title | Tag | Strategy Doc |
|---------|-------------|-----|--------------|
| 1 | Introduction to Agent Evals | — (no code) | INDEX.md |
| 2 | Golden Dataset | `module-8-01-golden-dataset` | 06_GOLDEN_DATASET.md |
| 3 | Rule-Based Evals (Local) | `module-8-02-rule-based-local` | 05_RULE_BASED.md |
| 4 | LLM-as-Judge (Local) | `module-8-03-llm-judge-local` | 04_MODEL_BASED_LLM_JUDGE.md |
| 5 | Introduction to Langfuse | `module-8-04-langfuse-setup` | — |
| 6 | Manual Annotation | `module-8-05-manual-annotation` | 03_MANUAL_ANNOTATION.md |
| 7 | Production Rule-Based | `module-8-06-rule-based-prod` | 05_RULE_BASED.md |
| 8 | Production LLM Judge | `module-8-07-llm-judge-prod` | 04_MODEL_BASED_LLM_JUDGE.md |
| 9 | Span & Trace Analysis | `module-8-08-span-trace` | 07_SPAN_TRACE_BASED.md |
| 10 | User Feedback | `module-8-09-user-feedback` | 01_USER_FEEDBACK.md |

---

## Instructor Guide: Recording Videos

### Before Recording

```bash
# Fetch all tags from remote
git fetch --tags

# See all available module 8 tags
git tag -l "module-8-*"
```

### Switching to a Specific Video State

**Example: Preparing to record Video 4 (LLM Judge Local)**

```bash
# Switch to the LLM Judge Local state
git checkout module-8-03-llm-judge-local

# You're now in "detached HEAD" state - this is normal
# The codebase shows: Golden Dataset + Rule-Based + LLM Judge (all local)
```

### After Recording

```bash
# Return to the main working branch
git checkout module-8-prep-evals
```

### Quick Reference

| Video | Tag Command |
|-------|-------------|
| 2 - Golden Dataset | `git checkout module-8-01-golden-dataset` |
| 3 - Rule-Based Local | `git checkout module-8-02-rule-based-local` |
| 4 - LLM Judge Local | `git checkout module-8-03-llm-judge-local` |
| 5 - Langfuse Setup | `git checkout module-8-04-langfuse-setup` |
| 6 - Manual Annotation | `git checkout module-8-05-manual-annotation` |
| 7 - Rule-Based Prod | `git checkout module-8-06-rule-based-prod` |
| 8 - LLM Judge Prod | `git checkout module-8-07-llm-judge-prod` |
| 9 - Span/Trace | `git checkout module-8-08-span-trace` |
| 10 - User Feedback | `git checkout module-8-09-user-feedback` |

---

## Strategy Documents Reference

The strategy documents provide conceptual depth for each evaluation approach. They're numbered by their original category, not implementation order:

| Doc # | Strategy | Used In Videos |
|-------|----------|----------------|
| 01 | User Feedback | Video 10 |
| 02 | Implicit Feedback | (Mentioned, not core) |
| 03 | Manual Annotation | Video 6 |
| 04 | LLM Judge | Videos 4, 8 |
| 05 | Rule-Based | Videos 3, 7 |
| 06 | Golden Dataset | Video 2 |
| 07 | Span/Trace | Video 9 |
| 08 | A/B Testing | (Honorable mention) |
