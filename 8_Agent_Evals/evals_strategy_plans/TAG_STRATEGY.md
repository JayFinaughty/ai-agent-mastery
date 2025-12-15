# Tag Strategy for Module 8 Implementation

## Overview

Tags mark the codebase state after each video's implementation. The tag number corresponds to the implementation order, making it easy to checkout the exact state shown in any video.

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
| 1 | Golden Dataset | `module-8-01-golden-dataset` | YAML test cases, evaluation runner, RAG mock data |
| 2 | Rule-Based (Local) | `module-8-02-rule-based-local` | Deterministic checks (Contains, tool calls) |
| 3 | LLM Judge (Local) | `module-8-03-llm-judge-local` | LLMJudge evaluator with rubrics |

### Phase 2: Production Evals (Langfuse)

Real user data, tracing, and observability.

| Order | Feature | Tag | What's Added |
|-------|---------|-----|--------------|
| 4 | Manual Annotation | `module-8-04-manual-annotation` | Annotation workflow, Langfuse UI intro |
| 5 | Rule-Based (Prod) | `module-8-05-rule-based-prod` | Score sync to Langfuse, alerts |
| 6 | LLM Judge (Prod) | `module-8-06-llm-judge-prod` | Langfuse built-in judge, async scoring |
| 7 | User Feedback | `module-8-07-user-feedback` | Frontend widget, feedback scores |

> **Note:** Tags are cumulative. Each tag includes all previous implementations.
>
> **Note:** Langfuse setup was covered in Module 6. Video 5 (Manual Annotation) serves as the intro to Langfuse for evaluations.

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
| 1 | Introduction to Agent Evals | — (no code) | 01_INTRO_TO_EVALS.md |
| 2 | Golden Dataset | `module-8-01-golden-dataset` | 02_GOLDEN_DATASET.md |
| 3 | Rule-Based Evals (Local) | `module-8-02-rule-based-local` | 03_RULE_BASED_LOCAL.md |
| 4 | LLM-as-Judge (Local) | `module-8-03-llm-judge-local` | 04_LLM_JUDGE_LOCAL.md |
| 5 | Manual Annotation | `module-8-04-manual-annotation` | 05_MANUAL_ANNOTATION.md |
| 6 | Production Rule-Based | `module-8-05-rule-based-prod` | 06_RULE_BASED_PROD.md |
| 7 | Production LLM Judge | `module-8-06-llm-judge-prod` | 07_LLM_JUDGE_PROD.md |
| 8 | User Feedback | `module-8-07-user-feedback` | 08_USER_FEEDBACK.md |

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
| 5 - Manual Annotation | `git checkout module-8-04-manual-annotation` |
| 6 - Rule-Based Prod | `git checkout module-8-05-rule-based-prod` |
| 7 - LLM Judge Prod | `git checkout module-8-06-llm-judge-prod` |
| 8 - User Feedback | `git checkout module-8-07-user-feedback` |

---

## Strategy Documents Reference

Strategy documents are numbered to match video order:

| File | Strategy | Video |
|------|----------|-------|
| 01_INTRO_TO_EVALS.md | Introduction | Video 1 |
| 02_GOLDEN_DATASET.md | Golden Dataset | Video 2 |
| 03_RULE_BASED_LOCAL.md | Rule-Based (Local) | Video 3 |
| 04_LLM_JUDGE_LOCAL.md | LLM Judge (Local) | Video 4 |
| 05_MANUAL_ANNOTATION.md | Manual Annotation | Video 5 |
| 06_RULE_BASED_PROD.md | Rule-Based (Prod) | Video 6 |
| 07_LLM_JUDGE_PROD.md | LLM Judge (Prod) | Video 7 |
| 08_USER_FEEDBACK.md | User Feedback | Video 8 |
| honorable_mention_span_trace.md | Span/Trace | (Honorable mention) |
| honorable_mention_implicit_feedback.md | Implicit Feedback | (Honorable mention) |
| honorable_mention_ab_testing.md | A/B Testing | (Honorable mention) |
| honorable_mention_cost_efficiency.md | Cost/Efficiency | (Honorable mention) |
