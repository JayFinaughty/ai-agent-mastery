# Strategy: Rule-Based Evaluation (Production)

> **Video 6** | **Tag:** `module-8-05-rule-based-prod` | **Phase:** Production

## Overview

**What it is**: Running your rule-based evaluators from Video 3 on production traces and syncing results to Langfuse for monitoring.

**Philosophy**: The evaluators you built locally (`NoPII`, `NoForbiddenWords`, `HasMatchingSpan`) work in production too. Just add Langfuse score syncing.

**Building on Video 3**: You already have custom evaluators. Now we run them on real production data and track results in Langfuse.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RULE-BASED EVALUATION (PRODUCTION)                  │
│                                                                         │
│   Production Trace         pydantic-evals           Langfuse            │
│   ────────────────         ─────────────           ─────────            │
│                                                                         │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐      │
│   │   Agent     │         │   NoPII     │         │   Scores    │      │
│   │   Response  │────────►│   Contains  │────────►│   Synced    │      │
│   │   + Trace   │         │   Custom    │         │   to Trace  │      │
│   └─────────────┘         └─────────────┘         └─────────────┘      │
│                                                                         │
│   Reuse your local evaluators. Add Langfuse sync.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What You'll Learn in This Video

1. How to run pydantic-evals evaluators on production traces
2. How to sync evaluation scores to Langfuse
3. How to set up async evaluation (non-blocking)
4. How to monitor rule violations in Langfuse dashboard

## Difference from Local

| Aspect | Local (Video 3) | Production (Video 6) |
|--------|-----------------|----------------------|
| **Data** | Golden dataset (10 cases) | Real production traces |
| **When** | Manual runs | After each request (async) |
| **Output** | Terminal report | Langfuse scores |
| **Purpose** | Test before deploy | Monitor in production |

---

## Implementation

### Step 1: Reuse Your Evaluators from Video 3

You already built these in Video 3:

```python
# backend_agent_api/evals/custom_evaluators.py (from Video 3)

from dataclasses import dataclass
import re
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

@dataclass
class NoPII(Evaluator):
    """Check that response doesn't contain PII patterns."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = str(ctx.output)

        patterns = {
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        }

        for pii_type, pattern in patterns.items():
            if re.search(pattern, output):
                return EvaluationReason(
                    value=False,
                    reason=f"Found {pii_type} pattern in output"
                )

        return EvaluationReason(value=True, reason="No PII detected")


@dataclass
class NoForbiddenWords(Evaluator):
    """Check that response doesn't contain forbidden words."""

    forbidden: list[str]

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = str(ctx.output).lower()
        found = [w for w in self.forbidden if w.lower() in output]

        if found:
            return EvaluationReason(
                value=False,
                reason=f"Found forbidden words: {found}"
            )
        return EvaluationReason(value=True, reason="No forbidden words")
```

### Step 2: Create Production Evaluator with Langfuse Sync

```python
# backend_agent_api/evals/prod_rules.py

from dataclasses import dataclass
from langfuse import Langfuse
from pydantic_evals.evaluators import EvaluatorContext, EvaluationReason

from custom_evaluators import NoPII, NoForbiddenWords

langfuse = Langfuse()


@dataclass
class ProductionRuleEvaluator:
    """Run rule-based evaluators and sync to Langfuse."""

    def __init__(self):
        self.evaluators = [
            ("no_pii", NoPII()),
            ("no_forbidden", NoForbiddenWords(
                forbidden=["password", "secret", "confidential"]
            )),
        ]

    async def evaluate_and_sync(
        self,
        trace_id: str,
        output: str,
        inputs: dict = None
    ):
        """
        Run all evaluators and sync scores to Langfuse.

        Call this async after response is sent to user.
        """
        ctx = EvaluatorContext(
            name="production_eval",
            inputs=inputs or {},
            output=output,
            expected_output=None,
            metadata={},
            duration=0.0,
            span_tree=None
        )

        all_passed = True

        for eval_name, evaluator in self.evaluators:
            result = evaluator.evaluate(ctx)

            # Sync to Langfuse
            langfuse.score(
                trace_id=trace_id,
                name=f"rule_{eval_name}",
                value=1.0 if result.value else 0.0,
                comment=result.reason
            )

            if not result.value:
                all_passed = False

        # Overall pass/fail score
        langfuse.score(
            trace_id=trace_id,
            name="rule_check_passed",
            value=1.0 if all_passed else 0.0
        )

        return all_passed


# Singleton
prod_evaluator = ProductionRuleEvaluator()
```

### Step 3: Integrate with Agent API

```python
# backend_agent_api/agent_api.py (additions)

import asyncio
from evals.prod_rules import prod_evaluator

@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing agent code generates response ...

    # Get trace_id from Langfuse context
    trace_id = langfuse_context.get_current_trace_id()

    # Run evaluators async (don't block response)
    if trace_id:
        asyncio.create_task(
            prod_evaluator.evaluate_and_sync(
                trace_id=trace_id,
                output=full_response,
                inputs={"query": request.query}
            )
        )

    # Return response to user immediately
    return {"response": full_response}
```

**Key point:** The evaluation runs AFTER the response is sent. Users don't wait.

---

## What You See in Langfuse

After integration, each trace will have scores attached:

| Score Name | Type | Meaning |
|------------|------|---------|
| `rule_no_pii` | 0 or 1 | No PII detected |
| `rule_no_forbidden` | 0 or 1 | No forbidden words |
| `rule_check_passed` | 0 or 1 | All rules passed |

### Langfuse Dashboard Uses

- **Filter traces** by `rule_check_passed = 0` to find violations
- **Track violation rate** over time
- **Correlate** with user feedback

---

## Adding More Rules

To add a new rule, just add it to the evaluators list:

```python
from pydantic_evals.evaluators import Contains

self.evaluators = [
    ("no_pii", NoPII()),
    ("no_forbidden", NoForbiddenWords(forbidden=["password", "secret"])),
    # Add more:
    ("has_greeting", Contains(value="hello", case_sensitive=False)),
]
```

---

## Optional: Response Blocking

If you need to BLOCK responses (not just log), add a check before returning:

```python
# Simple blocking pattern
async def check_before_send(response: str) -> tuple[bool, str]:
    """Check response before sending. Returns (should_send, message)."""

    ctx = EvaluatorContext(output=response, ...)

    # Check critical rules
    pii_result = NoPII().evaluate(ctx)

    if not pii_result.value:
        return False, "I cannot share that information as it may contain sensitive data."

    return True, response
```

**Note:** Blocking adds latency. Only block for critical safety rules.

---

## Best Practices

### 1. Run Async, Don't Block

```python
# ✅ Good - non-blocking
asyncio.create_task(prod_evaluator.evaluate_and_sync(...))

# ❌ Bad - blocks response
await prod_evaluator.evaluate_and_sync(...)
```

### 2. Keep Evaluators Fast

Your custom evaluators should be <10ms each. Regex checks are fine. Avoid:
- API calls in evaluators
- Complex computations
- Database queries

### 3. Use Langfuse Filters

Don't build custom dashboards. Use Langfuse's built-in filtering:
- Filter by score name and value
- Group by time period
- Export for analysis

---

## What's Next

| Video | What You'll Add |
|-------|-----------------|
| **Video 7: LLM Judge Prod** | AI-powered quality scoring on production traces |
| **Video 8: User Feedback** | Collect and analyze user ratings |

---

## Resources

- [Langfuse Scores API](https://langfuse.com/docs/scores)
- [pydantic-evals Custom Evaluators](https://ai.pydantic.dev/evals/evaluators/custom/)

---

## Instructor Guide: Recording Video 6

### Pre-Recording Checklist

1. **Environment Setup:**
   ```bash
   cd 8_Agent_Evals/backend_agent_api

   # Verify .env has Langfuse credentials
   cat .env | grep -E "^LANGFUSE_" | head -3

   # Verify existing evaluators from Video 3
   ls evals/evaluators.py

   # Test that the API starts
   python -c "from evals.evaluators import NoPII, NoForbiddenWords; print('Evaluators loaded')"
   ```

2. **Langfuse Dashboard:**
   - Open Langfuse dashboard in browser
   - Navigate to Traces view
   - Prepare to filter by score names

3. **Verify Video 3 evaluators exist:**
   ```bash
   cat evals/evaluators.py | head -30
   # Should show NoPII, NoForbiddenWords classes
   ```

### Recording Flow

**Part 1: Introduction (2-3 min)**
- Explain the difference between local (Video 3) and production evals
- Show the architecture diagram from the strategy doc
- Key point: "Same evaluators, different execution context"

**Part 2: Create prod_rules.py (5-7 min)**
- Create `evals/prod_rules.py`
- Walk through `ProductionRuleEvaluator` class:
  - How it wraps existing evaluators
  - EvaluatorContext construction
  - Langfuse score creation via `langfuse.create_score()`
- Explain the async pattern and why it doesn't block

**Part 3: Integrate with agent_api.py (3-5 min)**
- Show the three changes needed:
  1. Import statement
  2. Trace ID extraction from span context
  3. `asyncio.create_task()` for evaluation
- Emphasize: "Non-blocking - user gets response immediately"

**Part 4: Demo - Passing Evaluations (3-5 min)**
- Start the API: `python agent_api.py`
- Send a normal query: "What documents do you have about sales?"
- Show Langfuse dashboard:
  - Find the trace
  - Show `rule_no_pii = 1`, `rule_no_forbidden = 1`, `rule_check_passed = 1`

**Part 5: Demo - Failing Evaluations (3-5 min)**
- Send query to trigger PII detection:
  ```
  "What's a good format for a phone number? Show me an example like 555-123-4567"
  ```
- Show Langfuse: `rule_no_pii = 0`
- Send query to trigger forbidden words:
  ```
  "Write a sample error message that says 'Invalid password'"
  ```
- Show Langfuse: `rule_no_forbidden = 0`

**Part 6: Filtering Violations (2-3 min)**
- In Langfuse, filter traces by `rule_check_passed = 0`
- Show how to identify problematic responses
- "This is your production monitoring dashboard"

### Expected Output

**Langfuse Dashboard - Trace Scores:**
```
Trace: abc123...
├── rule_no_pii: 1.0 (comment: "No PII detected")
├── rule_no_forbidden: 1.0 (comment: "No forbidden words")
└── rule_check_passed: 1.0 (comment: "All rule-based evaluators passed")
```

**Langfuse Dashboard - Failed Trace:**
```
Trace: def456...
├── rule_no_pii: 0.0 (comment: "Found phone pattern in output")
├── rule_no_forbidden: 1.0 (comment: "No forbidden words")
└── rule_check_passed: 0.0 (comment: "One or more rules failed")
```

### Troubleshooting During Recording

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| No scores appear in Langfuse | Trace not synced yet | Wait 5-10 seconds, refresh |
| `ModuleNotFoundError: langfuse` | Missing package | `pip install langfuse` |
| `Failed to get Langfuse client` | Missing credentials | Check `LANGFUSE_*` in .env |
| All tests pass unexpectedly | LLM avoiding patterns | Try more explicit prompts |
| Import error for evaluators | Wrong directory | Run from `backend_agent_api/` |

### Key Teaching Moments

1. **"Why async?"**
   - Response latency is critical - users shouldn't wait for evaluations
   - Evaluations run in background after response is sent
   - Use `asyncio.create_task()` to fire-and-forget

2. **"Same evaluators, different context"**
   - Video 3: `Dataset.evaluate()` with batch test cases
   - Video 6: Direct `evaluator.evaluate(ctx)` on single responses
   - Same code, different execution patterns

3. **"Why sync to Langfuse?"**
   - Centralized monitoring dashboard
   - Filter and find violations
   - Track quality over time
   - Correlate with user feedback (Video 8)

4. **"What about blocking responses?"**
   - Show the optional blocking pattern from the doc
   - "Only for critical safety - adds latency"
   - "Most rules should be monitoring-only"

5. **"Adding more rules"**
   - Show how to add evaluators to the list
   - "Each rule becomes a new Langfuse score"

### Post-Recording Git Workflow

```bash
# Ensure you're on the prep branch
git checkout module-8-prep-evals

# Stage the new files
git add evals/prod_rules.py agent_api.py evals/__init__.py

# Commit
git commit -m "Implement Video 6: Production Rule-Based Evals"

# Tag this state
git tag module-8-05-rule-based-prod

# Push commit and tag
git push origin module-8-prep-evals
git push origin module-8-05-rule-based-prod
```

### Files Created/Modified in This Video

```
backend_agent_api/
├── evals/
│   ├── __init__.py            # Updated docstring
│   ├── prod_rules.py          # NEW: Production evaluator with Langfuse sync
│   └── evaluators.py          # Existing (from Video 3)
└── agent_api.py               # Modified: Added import and eval task
```

### Demo Prompts Reference

**Normal (all pass):**
```
"What documents do you have about sales?"
```

**PII Failure:**
```
"What's a good format for a phone number? Show me an example like 555-123-4567"
```

**Forbidden Words Failure:**
```
"Write a sample error message that says 'Invalid password'"
```
