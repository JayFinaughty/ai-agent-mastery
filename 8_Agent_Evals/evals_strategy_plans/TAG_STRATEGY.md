# Tag Strategy for Module 8 Implementation

## Overview

Each evaluation strategy is implemented as a sequential commit on the `module-8-prep-evals` branch. After each implementation, create a tag to mark that state of the codebase. This allows easy checkout of any intermediate state for video recording.

## Tag Naming Convention

```
module-8-{number}-{strategy-name}
```

The number corresponds to the strategy number (1-8), not the implementation order.

Examples:

- `module-8-01-user-feedback`
- `module-8-05-rule-based`
- `module-8-07-span-trace`

## Implementation Order

Follow the phased approach from `00_OVERVIEW.md` for practical dependency reasons:

| Phase                        | Strategy              | Tag                             | Rationale                            |
| ---------------------------- | --------------------- | ------------------------------- | ------------------------------------ |
| **Phase 1: Foundation**      |                       |                                 |                                      |
| 1.1                          | Rule-Based (5)        | `module-8-05-rule-based`        | Safety gates required for production |
| 1.2                          | Span/Trace (7)        | `module-8-07-span-trace`        | Instrumentation enables all analysis |
| **Phase 2: Automated Evals** |                       |                                 |                                      |
| 2.1                          | Golden Dataset (6)    | `module-8-06-golden-dataset`    | Regression testing baseline          |
| 2.2                          | LLM Judge (4)         | `module-8-04-llm-judge`         | Quality scoring                      |
| **Phase 3: Human-in-Loop**   |                       |                                 |                                      |
| 3.1                          | User Feedback (1)     | `module-8-01-user-feedback`     | Frontend integration                 |
| 3.2                          | Implicit Feedback (2) | `module-8-02-implicit-feedback` | Behavioral tracking                  |
| 3.3                          | Manual Annotation (3) | `module-8-03-manual-annotation` | Expert review workflow               |
| **Phase 4: Optimization**    |                       |                                 |                                      |
| 4.1                          | A/B Testing (8)       | `module-8-08-ab-testing`        | Version comparison                   |

> **Note:** Tags are cumulative. Each tag includes all previously implemented strategies, regardless of their number.

## Workflow

### Implementing Each Strategy

```bash
# Ensure you're on the prep branch
git checkout module-8-prep-evals

# Implement strategy N
# ... make code changes ...

# Commit
git add -A
git commit -m "Implement Strategy N: {Strategy Name}"

# Tag this state (use strategy number, not implementation order)
git tag module-8-0N-{strategy-name}

# Push commit and tag
git push origin module-8-prep-evals
git push origin module-8-0N-{strategy-name}
```

### Checking Out a Specific State

```bash
# List available tags
git tag -l "module-8-*"

# Checkout specific state (detached HEAD)
git checkout module-8-05-rule-based

# Return to latest
git checkout module-8-prep-evals
```

## Key Points

- Tags are numbered by **strategy number** (1-8), not implementation order
- Each tag represents a **cumulative** state (includes all previously implemented strategies)
- Commits are linear on a single branch
- Tags are immutable snapshots
- If fixes are needed in an earlier state, rebase and re-tag

## Recording Videos

Videos can be recorded in any order. Checkout the appropriate tag for the strategy you're recording:

```bash
git checkout module-8-0N-{strategy}
```

---

## Instructor Guide: Swapping Between States

### Before Recording

```bash
# Fetch all tags from remote
git fetch --tags

# See all available module 8 tags
git tag -l "module-8-*"
```

### Switching to a Specific Video State

**Example: Preparing to record Video for Rule-Based (Strategy 5)**

```bash
# Switch to the Rule-Based state
git checkout module-8-05-rule-based

# You're now in "detached HEAD" state - this is normal
```

### After Recording

```bash
# Return to the main working branch
git checkout module-8-prep-evals
```

### Quick Reference

| Strategy              | Tag                             | Command                                      |
| --------------------- | ------------------------------- | -------------------------------------------- |
| 1 - User Feedback     | `module-8-01-user-feedback`     | `git checkout module-8-01-user-feedback`     |
| 2 - Implicit Feedback | `module-8-02-implicit-feedback` | `git checkout module-8-02-implicit-feedback` |
| 3 - Manual Annotation | `module-8-03-manual-annotation` | `git checkout module-8-03-manual-annotation` |
| 4 - LLM Judge         | `module-8-04-llm-judge`         | `git checkout module-8-04-llm-judge`         |
| 5 - Rule-Based        | `module-8-05-rule-based`        | `git checkout module-8-05-rule-based`        |
| 6 - Golden Dataset    | `module-8-06-golden-dataset`    | `git checkout module-8-06-golden-dataset`    |
| 7 - Span/Trace        | `module-8-07-span-trace`        | `git checkout module-8-07-span-trace`        |
| 8 - A/B Testing       | `module-8-08-ab-testing`        | `git checkout module-8-08-ab-testing`        |
