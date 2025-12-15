# Meta Context: Module 8 Agent Evals Documentation

> **Purpose:** This document primes a new Claude Code session on the Module 8 documentation work. Read this first before making any changes.

## What We're Building

**Module 8: Agent Evals** for the AI Agent Mastery course. This module teaches learners how to evaluate AI agents through two distinct phases:

1. **Local Development (Videos 2-4):** Simple evals with `pydantic-evals`, no external services
2. **Production (Videos 5-8):** Real user data and observability with Langfuse

## The Core Philosophy

From the course instructor (meeting Dec 5, 2024):

> "Start simple. Don't overthink it. Andrew Ng said: 'When I set up evals for my agent at first, I do a 10-question golden dataset. That's all you need to start.'"

**Key principles:**
- **Learner-first:** What's the simplest thing someone would actually do?
- **Local-first:** Videos 2-4 require NO external services (no Langfuse)
- **Cumulative learning:** Each video builds on the previous
- **pydantic-evals native:** Use built-in evaluators, don't reinvent

## Final Video Structure

| Video | Title | Tag | File | Phase |
|-------|-------|-----|------|-------|
| 1 | Introduction to Evals | _(no code)_ | `01_INTRO_TO_EVALS.md` | Conceptual |
| 2 | Golden Dataset | `module-8-01-golden-dataset` | `02_GOLDEN_DATASET.md` | Local |
| 3 | Rule-Based (Local) | `module-8-02-rule-based-local` | `03_RULE_BASED_LOCAL.md` | Local |
| 4 | LLM Judge (Local) | `module-8-03-llm-judge-local` | `04_LLM_JUDGE_LOCAL.md` | Local |
| 5 | Manual Annotation | `module-8-04-manual-annotation` | `05_MANUAL_ANNOTATION.md` | Production |
| 6 | Rule-Based (Prod) | `module-8-05-rule-based-prod` | `06_RULE_BASED_PROD.md` | Production |
| 7 | LLM Judge (Prod) | `module-8-06-llm-judge-prod` | `07_LLM_JUDGE_PROD.md` | Production |
| 8 | User Feedback | `module-8-07-user-feedback` | `08_USER_FEEDBACK.md` | Production |

**Notes:**
- Video 5 (Manual Annotation) is the first production video and includes Langfuse intro
- Langfuse setup was covered in Module 6, so no separate setup video needed
- Span/Trace is now an honorable mention (`honorable_mention_span_trace.md`)

## Documentation Files

Located in `8_Agent_Evals/evals_strategy_plans/`:

| File | Status | Notes |
|------|--------|-------|
| `00_OVERVIEW.md` | ✅ Updated | Framework integration matrix |
| `INDEX.md` | ✅ Updated | Video order, phase breakdown |
| `TAG_STRATEGY.md` | ✅ Updated | Git tag naming, checkout workflow |
| `META_CONTEXT.md` | ✅ Updated | This file — priming context |
| `01_INTRO_TO_EVALS.md` | ✅ Created | Video 1 - conceptual intro |
| `02_GOLDEN_DATASET.md` | ✅ Updated | Video 2 strategy doc |
| `03_RULE_BASED_LOCAL.md` | ✅ Updated | Video 3 strategy doc |
| `04_LLM_JUDGE_LOCAL.md` | ✅ Updated | Video 4 strategy doc |
| `05_MANUAL_ANNOTATION.md` | ✅ Updated | Video 5 strategy doc |
| `06_RULE_BASED_PROD.md` | ✅ Updated | Video 6 strategy doc (simplified) |
| `07_LLM_JUDGE_PROD.md` | ✅ Updated | Video 7 strategy doc (simplified) |
| `08_USER_FEEDBACK.md` | ✅ Updated | Video 8 strategy doc |

**Honorable Mentions:**
- `honorable_mention_span_trace.md`
- `honorable_mention_ab_testing.md`
- `honorable_mention_cost_efficiency.md`
- `honorable_mention_implicit_feedback.md`

## Key Decisions Made

### 0. Production Phase Simplification

**Decision:** Minimize custom code. Reuse local patterns. Use Langfuse/pydantic-ai.

**Video 6 (Rule-Based Prod):**
- Reuse `NoPII`, `NoForbiddenWords` from Video 3
- Add simple async Langfuse score syncing
- ~80 lines instead of 250+ lines of custom RuleEngine

**Video 7 (LLM Judge Prod):**
- Start with Langfuse built-in evaluators (no code)
- ONE custom pydantic-ai judge for domain-specific needs
- RAG evaluation shown as a variation, not a separate system

**Why:** Learners should reuse what they built locally, not learn entirely new systems.

### 1. No LLMJudge in Video 2

**Before:** The golden dataset used `LLMJudge` on 7/10 cases
**After:** Only deterministic evaluators (`Contains`, `IsInstance`, `MaxDuration`)

**Why:** Creates clear learning progression. LLMJudge is introduced in Video 4.

### 2. No Separate Langfuse Setup Video

**Decision:** Langfuse was set up in Module 6. Video 5 (Manual Annotation) includes a brief Langfuse intro for evaluations but doesn't repeat setup.

### 3. Span/Trace Moved to Honorable Mention

**Reason:** To keep the core curriculum focused at 8 videos instead of 10.

### 4. YAML Syntax for pydantic-evals 1.28

**Correct syntax:**
```yaml
# Short form (single parameter)
- Contains: "hello"
- MaxDuration: 5.0
- IsInstance: str

# Full form (multiple parameters)
- Contains:
    value: "hello"
    case_sensitive: false

# HasMatchingSpan
- HasMatchingSpan:
    query:
      name_contains: "retrieve_relevant_documents"
```

**Common mistake:** Using `substring:` instead of `value:` for `Contains`.

### 5. Show "Before → After" Transformations

Each video should show how the dataset evolves from the previous video.

## Codebase Context

### Agent Location
- `8_Agent_Evals/backend_agent_api/agent.py` — Agent definition
- `8_Agent_Evals/backend_agent_api/agent_api.py` — FastAPI endpoint
- `8_Agent_Evals/backend_agent_api/tools.py` — Tool implementations
- `8_Agent_Evals/backend_agent_api/clients.py` — Client helpers

### Agent Tools (for HasMatchingSpan checks)
- `retrieve_relevant_documents`
- `list_documents`
- `execute_sql_query`
- `web_search`
- `execute_code`
- `image_analysis`
- `get_document_content`

### Evals Directory
```
backend_agent_api/
├── evals/
│   ├── __init__.py
│   ├── golden_dataset.yaml          # General agent behavior tests (10 cases)
│   ├── golden_dataset_rag.yaml      # RAG/web search tests (14 cases)
│   ├── run_evals.py                 # Supports --dataset general|rag|all
│   ├── seed_test_data.py            # Seeds basic test documents
│   ├── seed_rag_mock_data.py        # Seeds NeuroVerse RAG documents
│   └── evaluators.py                # Custom evaluators (ContainsAny, etc.)
```

### Mock Data Directory
```
8_Agent_Evals/
├── mock_data/                        # NeuroVerse Studios documents (9 files)
│   ├── NeuroVerse Studios_ Company Overview.md
│   ├── NeuroVerse Studios - Q1 2024 Quarterly Report.md
│   ├── Neural Adaptation Engine (NAE).md
│   └── ... (6 more documents)
```

## Version Requirements

- **pydantic-ai:** 1.28.0+ (updated Dec 2025)
- **pydantic-evals:** 1.28.0+ (updated Dec 2025)
- **langfuse (Python):** 3.10.5+ (updated Dec 2025)
- **langfuse (npm):** 3.38.6+ (for frontend User Feedback)

**Note:** Dependencies were updated to latest versions. The `OpenAIModel` class was renamed to `OpenAIChatModel` in pydantic-ai 1.x.

## Working Style

1. **Read the strategy doc first** before making changes
2. **Check pydantic-evals docs** for correct syntax
3. **Show "before → after"** when evolving the dataset
4. **Don't front-load concepts** — each video introduces specific evaluators
5. **Confirm with user** before major structural changes
6. **Keep consistent structure** across strategy docs

## Document Section Template

```markdown
# Strategy: [Name]

> **Video N** | **Tag:** `module-8-0N-name` | **Phase:** Local/Production

## Overview
[Diagram + philosophy]

## What You'll Learn in This Video
[Numbered list of 4-5 items]

## When to Use
[✅ Good for / ❌ Not for]

## Implementation
### Step 1: [Action]
### Step 2: [Action]

## Best Practices
[3-5 specific tips]

## What's Next
[Table pointing to next videos]

## Resources
[Links to docs]
```
