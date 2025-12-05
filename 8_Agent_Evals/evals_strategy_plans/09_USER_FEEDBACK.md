# Strategy: User Feedback Collection

> **Video 10** | **Tag:** `module-8-09-user-feedback` | **Phase:** Production

## Overview

**What it is**: Direct, explicit feedback from users about agent responses through thumbs up/down ratings and optional comments.

**Philosophy**: Users are the ultimate judges. Their feedback provides ground truth that no automated system can replicate. Collect it, sync it to Langfuse, and use it to improve.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    USER FEEDBACK FLOW                                   │
│                                                                         │
│   User ──► Agent Response ──► Feedback Widget ──► Langfuse Score       │
│                                     │                                   │
│                                     ▼                                   │
│                              ┌──────────────┐                          │
│                              │  👍  /  👎   │                          │
│                              │              │                          │
│                              │ "Comment..." │                          │
│                              └──────────────┘                          │
│                                     │                                   │
│                                     ▼                                   │
│                              ┌──────────────┐                          │
│                              │   Langfuse   │                          │
│                              │   Dashboard  │                          │
│                              │   ─────────  │                          │
│                              │  Satisfaction│                          │
│                              │  Trends      │                          │
│                              └──────────────┘                          │
│                                                                         │
│   Simple: Widget → Langfuse SDK → Score on Trace                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## When to Use

✅ **Good for:**
- Measuring real user satisfaction
- Identifying issues automated systems miss
- Calibrating LLM judges against human judgment
- Building golden datasets from user-verified examples

❌ **Limitations:**
- Low response rate (typically 1-5%)
- Selection bias (extremes more likely to respond)
- Needs enough production traffic to be meaningful

---

## Implementation

### The Simple Path: LangfuseWeb SDK

The easiest approach is using Langfuse's browser SDK to send feedback directly from the frontend.

### Step 1: Install LangfuseWeb

```bash
npm install langfuse
```

### Step 2: Create Feedback Component

```tsx
// frontend/src/components/chat/FeedbackWidget.tsx

import { LangfuseWeb } from "langfuse";
import { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";

// Initialize Langfuse Web SDK
const langfuse = new LangfuseWeb({
  publicKey: process.env.NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY!,
  baseUrl: process.env.NEXT_PUBLIC_LANGFUSE_HOST,
});

interface FeedbackWidgetProps {
  traceId: string;  // The Langfuse trace ID for this response
}

export function FeedbackWidget({ traceId }: FeedbackWidgetProps) {
  const [submitted, setSubmitted] = useState(false);
  const [selected, setSelected] = useState<"up" | "down" | null>(null);

  const handleFeedback = async (isPositive: boolean) => {
    setSelected(isPositive ? "up" : "down");

    // Send score directly to Langfuse
    await langfuse.score({
      traceId: traceId,
      name: "user_feedback",
      value: isPositive ? 1 : 0,
    });

    setSubmitted(true);
  };

  if (submitted) {
    return (
      <span className="text-sm text-muted-foreground">
        Thanks for your feedback!
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">Was this helpful?</span>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => handleFeedback(true)}
        className={selected === "up" ? "text-green-500" : ""}
      >
        <ThumbsUp className="h-4 w-4" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => handleFeedback(false)}
        className={selected === "down" ? "text-red-500" : ""}
      >
        <ThumbsDown className="h-4 w-4" />
      </Button>
    </div>
  );
}
```

### Step 3: Add Issue Categories (Optional)

For negative feedback, collect more context:

```tsx
// frontend/src/components/chat/DetailedFeedback.tsx

import { useState } from "react";
import { LangfuseWeb } from "langfuse";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const langfuse = new LangfuseWeb({
  publicKey: process.env.NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY!,
});

const ISSUE_CATEGORIES = [
  { id: "wrong", label: "Wrong answer" },
  { id: "incomplete", label: "Incomplete" },
  { id: "irrelevant", label: "Irrelevant" },
  { id: "slow", label: "Too slow" },
  { id: "other", label: "Other" },
];

interface DetailedFeedbackProps {
  traceId: string;
  onComplete: () => void;
}

export function DetailedFeedback({ traceId, onComplete }: DetailedFeedbackProps) {
  const [category, setCategory] = useState<string | null>(null);
  const [comment, setComment] = useState("");

  const submit = async () => {
    // Send numeric score
    await langfuse.score({
      traceId,
      name: "user_feedback",
      value: 0,
      comment: comment || undefined,
    });

    // Send category as separate score
    if (category) {
      await langfuse.score({
        traceId,
        name: "feedback_category",
        value: category,
      });
    }

    onComplete();
  };

  return (
    <div className="flex flex-col gap-3 p-4 border rounded-lg">
      <span className="text-sm font-medium">What went wrong?</span>

      <div className="flex flex-wrap gap-2">
        {ISSUE_CATEGORIES.map((cat) => (
          <Button
            key={cat.id}
            variant={category === cat.id ? "default" : "outline"}
            size="sm"
            onClick={() => setCategory(cat.id)}
          >
            {cat.label}
          </Button>
        ))}
      </div>

      <Textarea
        placeholder="Tell us more (optional)"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />

      <Button onClick={submit}>Submit Feedback</Button>
    </div>
  );
}
```

### Step 4: Pass Trace ID to Frontend

Your backend needs to return the Langfuse trace ID so the frontend can attach feedback:

```python
# backend_agent_api/agent_api.py

from langfuse import Langfuse

langfuse = Langfuse()

@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # Create or get trace
    trace = langfuse.trace(
        name="agent_request",
        user_id=request.user_id,
        session_id=request.session_id,
    )

    # ... agent execution ...

    # Return trace ID with response
    return {
        "response": result.output,
        "trace_id": trace.id  # Frontend uses this for feedback
    }
```

---

## Langfuse Score Types

| Type | Example | Use Case |
|------|---------|----------|
| **Numeric** | `value: 1` or `value: 0` | Thumbs up/down |
| **Numeric** | `value: 0.8` | Star rating (normalized) |
| **Categorical** | `value: "wrong_answer"` | Issue category |

### Thumbs Up/Down

```javascript
langfuse.score({
  traceId: traceId,
  name: "user_feedback",
  value: 1,  // 1 = positive, 0 = negative
});
```

### Star Rating (1-5)

```javascript
langfuse.score({
  traceId: traceId,
  name: "user_rating",
  value: rating / 5,  // Normalize to 0-1
  comment: "4 out of 5 stars"
});
```

### With Comment

```javascript
langfuse.score({
  traceId: traceId,
  name: "user_feedback",
  value: 0,
  comment: "The answer was completely wrong"
});
```

---

## Langfuse Dashboard

### Filtering by Feedback

Find low-rated responses:
```
Scores → user_feedback < 1
```

Find responses with specific issues:
```
Scores → feedback_category = "wrong_answer"
```

### Metrics to Track

| Metric | What to Look For |
|--------|------------------|
| **Feedback Rate** | % of responses with feedback |
| **Satisfaction** | % positive (value = 1) |
| **Issue Distribution** | Most common categories |
| **Trends** | Satisfaction over time |

### Correlating with Other Scores

Compare user feedback vs LLM judge:
- High LLM judge, low user feedback → Judge calibration issue
- Low LLM judge, high user feedback → Judge too strict
- Both low → Real quality problem

---

## Using Feedback Data

### Pattern 1: Find Low-Rated Responses

```python
from langfuse import Langfuse

langfuse = Langfuse()

def get_negative_feedback_traces(limit: int = 100):
    """Get traces with negative user feedback for review."""

    traces = langfuse.fetch_traces(
        filter={
            "scores": {
                "name": "user_feedback",
                "value": {"eq": 0}
            }
        },
        limit=limit
    )

    return traces.data
```

### Pattern 2: Build Golden Dataset from Positive Feedback

```python
def export_positive_examples():
    """Export highly-rated responses as golden dataset cases."""

    traces = langfuse.fetch_traces(
        filter={
            "scores": {
                "name": "user_feedback",
                "value": {"eq": 1}
            }
        },
        limit=100
    )

    cases = []
    for trace in traces.data:
        cases.append({
            "name": f"user_validated_{trace.id[:8]}",
            "inputs": {"query": trace.input},
            "metadata": {
                "source": "user_feedback",
                "user_rating": 1
            }
        })

    return cases
```

### Pattern 3: Calibrate LLM Judge

```python
def compare_judge_to_users():
    """Compare LLM judge scores against user feedback."""

    traces = langfuse.fetch_traces(limit=500)

    comparisons = []
    for trace in traces.data:
        user_score = None
        judge_score = None

        for score in trace.scores:
            if score.name == "user_feedback":
                user_score = score.value
            if score.name == "llm_judge_score":
                judge_score = score.value

        if user_score is not None and judge_score is not None:
            comparisons.append({
                "trace_id": trace.id,
                "user": user_score,
                "judge": judge_score,
                "agree": (user_score >= 0.5) == (judge_score >= 0.5)
            })

    agreement_rate = sum(c["agree"] for c in comparisons) / len(comparisons)
    print(f"User-Judge Agreement: {agreement_rate:.1%}")

    return comparisons
```

---

## Best Practices

### 1. Keep It Simple

```tsx
// ✅ Simple: Two buttons
<Button onClick={() => handleFeedback(true)}>👍</Button>
<Button onClick={() => handleFeedback(false)}>👎</Button>

// ❌ Complex: Too many options upfront
<StarRating />
<Select categories />
<Textarea required />
```

### 2. Ask for Details Only on Negative

```tsx
if (isNegative) {
  showDetailedForm();  // Ask what went wrong
} else {
  submit();  // Just record the thumbs up
}
```

### 3. Don't Ask Too Often

```tsx
// Show feedback widget after every N messages, not every message
const FEEDBACK_INTERVAL = 3;

if (messageIndex % FEEDBACK_INTERVAL === 0) {
  return <FeedbackWidget traceId={traceId} />;
}
```

### 4. Acknowledge Feedback

```tsx
if (submitted) {
  return <span>Thanks for your feedback!</span>;
}
```

---

## What This Enables

With user feedback collected in Langfuse, you can:

1. **Filter traces by satisfaction** → Focus debugging on what users dislike
2. **Compare to LLM judge** → Calibrate automated evaluation
3. **Track trends** → See if changes improve satisfaction
4. **Build golden dataset** → Use validated examples for testing
5. **Prioritize improvements** → Fix most common issue categories

---

## Integration with Other Strategies

### With LLM Judge (Videos 4, 8)

Compare judge scores to user ratings to calibrate rubrics.

### With Golden Dataset (Video 2)

Export positively-rated responses as verified test cases.

### With Manual Annotation (Video 6)

Prioritize annotating traces with negative user feedback.

### With Span/Trace (Video 9)

Correlate user satisfaction with execution patterns (slow = unhappy?).

---

## Environment Variables

```bash
# Frontend (.env.local)
NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY=pk-lf-...
NEXT_PUBLIC_LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
```

---

## Resources

- [Langfuse User Feedback](https://langfuse.com/docs/scores/user-feedback)
- [LangfuseWeb SDK](https://langfuse.com/docs/sdk/typescript/browser)
- [Scores Documentation](https://langfuse.com/docs/scores)
