# Strategy: User Feedback Collection

> **Video 8** | **Tag:** `module-8-07-user-feedback` | **Phase:** Production

## Overview

**What it is**: Two-level feedback system — thumbs up/down on individual messages plus periodic conversation ratings.

**Philosophy**: Users are the ultimate judges. Their feedback provides ground truth that no automated system can replicate. Collect it at multiple levels, sync it to Langfuse, and use it to improve.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TWO-LEVEL FEEDBACK SYSTEM                            │
│                                                                         │
│   LEVEL 1: MESSAGE FEEDBACK              LEVEL 2: CONVERSATION RATING   │
│   ─────────────────────────              ───────────────────────────    │
│                                                                         │
│   ┌─────────────────────────┐           ┌─────────────────────────┐    │
│   │ Agent Response          │           │ Every 5/10/15 responses │    │
│   │ ───────────────────     │           │                         │    │
│   │ "Here's what I found"   │           │ How's this conversation │    │
│   │                         │           │ going so far?           │    │
│   │           [hover]       │           │                         │    │
│   │           👍  👎        │           │ ○ Very good             │    │
│   └─────────────────────────┘           │ ○ Good                  │    │
│                                         │ ○ Not so good           │    │
│   Granular: "Was this                   │ ○ Bad                   │    │
│   specific answer good?"                │                         │    │
│                                         │ [If "Bad" → text input] │    │
│                                         └─────────────────────────┘    │
│                                                                         │
│                                         Holistic: "Is this agent       │
│                                         helping me accomplish my goal?" │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What You'll Learn in This Video

1. How to implement message-level thumbs up/down feedback
2. How to build periodic conversation rating popups
3. How to return trace IDs from backend to frontend
4. How to sync both feedback types to Langfuse
5. How to analyze feedback and connect it to other evaluation strategies

## When to Use

✅ **Good for:**
- Measuring real user satisfaction at multiple granularities
- Identifying issues automated systems miss
- Calibrating LLM judges against human judgment
- Building golden datasets from user-verified examples
- Understanding conversation-level vs response-level quality

❌ **Limitations:**
- Low response rate (typically 1-5%)
- Selection bias (extremes more likely to respond)
- Needs enough production traffic to be meaningful

---

## The Two Feedback Levels

| Level | Trigger | UI | Score Name | Value |
|-------|---------|-----|------------|-------|
| **Message** | Hover on response | 👍 / 👎 | `message_feedback` | 1 or 0 |
| **Conversation** | After 5, 10, 15... responses | 4-option popup | `conversation_rating` | 1.0 / 0.67 / 0.33 / 0.0 |

**Why two levels?**
- A user might thumbs-up individual responses but feel the conversation isn't productive
- Message feedback = "Was this answer good?"
- Conversation feedback = "Is this agent helping me accomplish my goal?"

---

## Prerequisites: Trace ID Setup

Before implementing feedback, the frontend needs access to trace IDs.

### Current State

The codebase already has:
- ✅ Session ID tracking (frontend generates, backend stores)
- ✅ Langfuse tracing via OpenTelemetry
- ❌ Trace ID not returned to frontend (needs to be added)

### Step 1: Return Trace ID from Backend

```python
# backend_agent_api/agent_api.py

from opentelemetry import trace as otel_trace

@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing setup ...

    with tracer.start_as_current_span("Pydantic-Ai-Trace") as span:
        # Get the trace ID from the current span
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')  # Convert to hex string

        # ... existing agent execution ...

        # Include trace_id in completion chunk
        completion_chunk = {
            "type": "completion",
            "session_id": session_id,
            "trace_id": trace_id,  # NEW: Add trace ID
        }
        yield f"data: {json.dumps(completion_chunk)}\n\n"
```

### Step 2: Handle Trace ID in Frontend

```typescript
// frontend/src/lib/api.ts

interface StreamingChunk {
  type: "token" | "completion" | "error";
  content?: string;
  session_id?: string;
  trace_id?: string;  // NEW: Add trace ID
}

// In the streaming handler, capture trace_id from completion chunk
if (chunk.type === "completion" && chunk.trace_id) {
  // Store trace_id for feedback attachment
}
```

### Step 3: Store Trace ID with Messages

```typescript
// frontend/src/components/chat/MessageHandling.tsx

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  trace_id?: string;  // NEW: Associate trace with message
}
```

---

## Implementation: Message-Level Feedback

### FeedbackWidget Component

Shows on hover over any agent message.

```tsx
// frontend/src/components/chat/FeedbackWidget.tsx

import { LangfuseWeb } from "langfuse";
import { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";

const langfuse = new LangfuseWeb({
  publicKey: process.env.NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY!,
  baseUrl: process.env.NEXT_PUBLIC_LANGFUSE_HOST,
});

interface FeedbackWidgetProps {
  traceId: string;
  observationId?: string;  // For message-level, attach to specific generation
}

export function FeedbackWidget({ traceId, observationId }: FeedbackWidgetProps) {
  const [submitted, setSubmitted] = useState(false);
  const [selected, setSelected] = useState<"up" | "down" | null>(null);

  const handleFeedback = async (isPositive: boolean) => {
    setSelected(isPositive ? "up" : "down");

    await langfuse.score({
      traceId,
      observationId,  // Attaches to specific message if provided
      name: "message_feedback",
      value: isPositive ? 1 : 0,
    });

    setSubmitted(true);
  };

  if (submitted) {
    return (
      <span className="text-xs text-muted-foreground">
        {selected === "up" ? "👍" : "👎"}
      </span>
    );
  }

  return (
    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        onClick={() => handleFeedback(true)}
      >
        <ThumbsUp className="h-3 w-3" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        onClick={() => handleFeedback(false)}
      >
        <ThumbsDown className="h-3 w-3" />
      </Button>
    </div>
  );
}
```

### Integration with Message Component

```tsx
// In your message rendering component

<div className="group relative">
  <div className="message-content">
    {message.content}
  </div>

  {message.role === "assistant" && message.trace_id && (
    <div className="absolute right-2 top-2">
      <FeedbackWidget traceId={message.trace_id} />
    </div>
  )}
</div>
```

---

## Implementation: Conversation-Level Feedback

### ConversationRating Component

Popup that appears after 5, 10, 15... agent responses.

```tsx
// frontend/src/components/chat/ConversationRating.tsx

import { LangfuseWeb } from "langfuse";
import { useState } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const langfuse = new LangfuseWeb({
  publicKey: process.env.NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY!,
  baseUrl: process.env.NEXT_PUBLIC_LANGFUSE_HOST,
});

const RATING_OPTIONS = [
  { label: "Very good", value: 1.0, emoji: "😊" },
  { label: "Good", value: 0.67, emoji: "🙂" },
  { label: "Not so good", value: 0.33, emoji: "😕" },
  { label: "Bad", value: 0.0, emoji: "😞" },
];

interface ConversationRatingProps {
  traceId: string;
  sessionId: string;
  onComplete: () => void;
  onDismiss: () => void;
}

export function ConversationRating({
  traceId,
  sessionId,
  onComplete,
  onDismiss,
}: ConversationRatingProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSelect = (value: number) => {
    setSelected(value);

    // Show comment box for "Bad" rating
    if (value === 0.0) {
      setShowComment(true);
    } else {
      // Submit immediately for non-bad ratings
      submitRating(value);
    }
  };

  const submitRating = async (value: number, commentText?: string) => {
    setSubmitting(true);

    await langfuse.score({
      traceId,
      name: "conversation_rating",
      value,
      comment: commentText || undefined,
    });

    // Also store session context
    await langfuse.score({
      traceId,
      name: "conversation_rating_session",
      value: sessionId,  // Categorical: links to session
    });

    setSubmitting(false);
    onComplete();
  };

  return (
    <Card className="w-80 shadow-lg">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">
            How's this conversation going?
          </CardTitle>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onDismiss}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {!showComment ? (
          <div className="space-y-2">
            {RATING_OPTIONS.map((option) => (
              <Button
                key={option.value}
                variant={selected === option.value ? "default" : "outline"}
                className="w-full justify-start"
                onClick={() => handleSelect(option.value)}
                disabled={submitting}
              >
                <span className="mr-2">{option.emoji}</span>
                {option.label}
              </Button>
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Sorry to hear that. What went wrong? (optional)
            </p>
            <Textarea
              placeholder="Tell us more..."
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={3}
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => submitRating(0.0)}
                disabled={submitting}
              >
                Skip
              </Button>
              <Button
                className="flex-1"
                onClick={() => submitRating(0.0, comment)}
                disabled={submitting}
              >
                Submit
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

### Rating Trigger Logic

```tsx
// frontend/src/components/chat/MessageHandling.tsx (or similar)

import { useState, useEffect } from "react";
import { ConversationRating } from "./ConversationRating";

// Decreasing frequency: 5, 10, 15, 20...
const RATING_THRESHOLDS = [5, 10, 15, 20, 25, 30];

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [showRating, setShowRating] = useState(false);
  const [ratingsDismissed, setRatingsDismissed] = useState<Set<number>>(new Set());

  // Count agent responses
  const agentResponseCount = messages.filter(m => m.role === "assistant").length;

  // Check if we should show rating popup
  useEffect(() => {
    const threshold = RATING_THRESHOLDS.find(t =>
      agentResponseCount === t && !ratingsDismissed.has(t)
    );

    if (threshold) {
      setShowRating(true);
    }
  }, [agentResponseCount, ratingsDismissed]);

  const handleRatingComplete = () => {
    setShowRating(false);
    // Mark this threshold as completed
    setRatingsDismissed(prev => new Set([...prev, agentResponseCount]));
  };

  const handleRatingDismiss = () => {
    setShowRating(false);
    // Mark as dismissed so it doesn't show again at this threshold
    setRatingsDismissed(prev => new Set([...prev, agentResponseCount]));
  };

  // Get current trace ID (from most recent assistant message)
  const currentTraceId = messages
    .filter(m => m.role === "assistant")
    .slice(-1)[0]?.trace_id;

  return (
    <div className="relative">
      {/* Chat messages */}
      <div className="messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </div>

      {/* Conversation rating popup */}
      {showRating && currentTraceId && (
        <div className="fixed bottom-24 right-8 z-50">
          <ConversationRating
            traceId={currentTraceId}
            sessionId={sessionId}
            onComplete={handleRatingComplete}
            onDismiss={handleRatingDismiss}
          />
        </div>
      )}
    </div>
  );
}
```

---

## Langfuse Score Structure

### Message Feedback

```javascript
langfuse.score({
  traceId: "abc123",
  observationId: "gen_456",  // Optional: specific generation
  name: "message_feedback",
  value: 1,  // 1 = positive, 0 = negative
});
```

### Conversation Rating

```javascript
langfuse.score({
  traceId: "abc123",
  name: "conversation_rating",
  value: 0.67,  // 1.0 | 0.67 | 0.33 | 0.0
  comment: "Optional user comment for bad ratings",
});
```

### Score Values Reference

| Rating | Label | Numeric Value |
|--------|-------|---------------|
| Very good | 😊 | 1.0 |
| Good | 🙂 | 0.67 |
| Not so good | 😕 | 0.33 |
| Bad | 😞 | 0.0 |

---

## Analyzing Feedback in Langfuse

### Dashboard Filters

```
# Find negative message feedback
Scores → message_feedback = 0

# Find poor conversations
Scores → conversation_rating < 0.5

# Find conversations with comments
Scores → conversation_rating has comment
```

### Metrics to Track

| Metric | What to Look For |
|--------|------------------|
| **Message Feedback Rate** | % of responses with thumbs up/down |
| **Message Satisfaction** | % positive (value = 1) |
| **Conversation Rating Avg** | Mean rating across sessions |
| **Comment Volume** | How many "bad" ratings have comments |

---

## Connecting to Other Strategies

### Feedback Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK ANALYSIS PIPELINE                    │
│                                                                  │
│   User Feedback                                                  │
│   ─────────────                                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────┐                                              │
│   │ All Feedback  │──────► Dashboard: Trends, Distribution       │
│   │ (Track in     │                                              │
│   │  Langfuse)    │                                              │
│   └───────┬───────┘                                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────┐     ┌───────────────┐                       │
│   │   Negative?   │─Yes─►│  LLM Judge    │──► Triage Score       │
│   │               │      │  (Video 7)    │                       │
│   └───────┬───────┘      └───────┬───────┘                       │
│           │ No                   │                               │
│           ▼                      ▼                               │
│   ┌───────────────┐      ┌───────────────┐                       │
│   │ Candidate for │      │ Both Negative │──► Manual Annotation  │
│   │ Golden Dataset│      │ = Priority    │    Queue (Video 5)    │
│   │ (Video 2)     │      └───────────────┘                       │
│   └───────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### With LLM Judge (Videos 4, 7)

Compare judge scores to user ratings to calibrate rubrics:

```python
def compare_judge_to_users():
    """Compare LLM judge scores against user feedback."""

    traces = langfuse.fetch_traces(limit=500)

    comparisons = []
    for trace in traces.data:
        user_score = None
        judge_score = None

        for score in trace.scores:
            if score.name == "message_feedback":
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

    # Find disagreements for calibration
    disagreements = [c for c in comparisons if not c["agree"]]
    print(f"Disagreements to review: {len(disagreements)}")

    return comparisons
```

**Interpretation:**
- User negative + Judge positive → Judge is miscalibrated (too lenient)
- User positive + Judge negative → Judge is too strict
- Both negative → Real problem, high priority

### With Golden Dataset (Video 2)

Export positively-rated conversations as verified test cases:

```python
def export_positive_examples():
    """Export highly-rated responses as golden dataset cases."""

    traces = langfuse.fetch_traces(
        filter={
            "scores": {
                "name": "conversation_rating",
                "value": {"gte": 0.67}  # "Good" or "Very good"
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
                "conversation_rating": next(
                    (s.value for s in trace.scores if s.name == "conversation_rating"),
                    None
                )
            }
        })

    return cases
```

### With Manual Annotation (Video 5)

Prioritize annotating traces with negative feedback:

```python
def get_priority_annotation_queue():
    """Get traces that need manual review, prioritized by user feedback."""

    # Negative conversation ratings
    negative_traces = langfuse.fetch_traces(
        filter={
            "scores": {
                "name": "conversation_rating",
                "value": {"lt": 0.5}  # "Not so good" or "Bad"
            }
        },
        limit=50
    )

    # Sort by rating (worst first)
    sorted_traces = sorted(
        negative_traces.data,
        key=lambda t: next(
            (s.value for s in t.scores if s.name == "conversation_rating"),
            1.0
        )
    )

    return sorted_traces
```

### With Rule-Based (Videos 3, 6)

Correlate rule violations with user feedback:

```python
def correlate_rules_with_feedback():
    """Check if rule failures predict negative user feedback."""

    traces = langfuse.fetch_traces(limit=500)

    correlations = []
    for trace in traces.data:
        rule_passed = None
        user_happy = None

        for score in trace.scores:
            if score.name == "rule_check_passed":
                rule_passed = score.value == 1.0
            if score.name == "conversation_rating":
                user_happy = score.value >= 0.5

        if rule_passed is not None and user_happy is not None:
            correlations.append({
                "rule_passed": rule_passed,
                "user_happy": user_happy,
            })

    # Analyze
    rule_fail_unhappy = sum(1 for c in correlations if not c["rule_passed"] and not c["user_happy"])
    rule_fail_total = sum(1 for c in correlations if not c["rule_passed"])

    if rule_fail_total > 0:
        print(f"When rules fail, user unhappy: {rule_fail_unhappy/rule_fail_total:.1%}")
```

---

## Best Practices

### 1. Keep Message Feedback Minimal

```tsx
// ✅ Simple: Just thumbs, appears on hover
<FeedbackWidget traceId={traceId} />

// ❌ Complex: Too much UI
<StarRating />
<CategorySelect />
<CommentBox />
```

### 2. Ask for Details Only on Negative

```tsx
// Only show comment box for "Bad" rating
if (value === 0.0) {
  setShowComment(true);
}
```

### 3. Make Rating Dismissable

Users should never feel forced to rate. The X button is important.

### 4. Decreasing Frequency

```tsx
// 5, 10, 15... not every 5 messages
const RATING_THRESHOLDS = [5, 10, 15, 20, 25, 30];
```

### 5. Acknowledge Feedback

```tsx
if (submitted) {
  return <span>Thanks! 👍</span>;
}
```

---

## Environment Variables

```bash
# Frontend (.env.local)
NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY=pk-lf-...
NEXT_PUBLIC_LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
```

---

## What's Next

This is the final video in Module 8. With user feedback in place, you now have:

| Strategy | What It Gives You |
|----------|-------------------|
| Golden Dataset (V2) | Regression testing |
| Rule-Based (V3, V6) | Automated safety checks |
| LLM Judge (V4, V7) | Scalable quality scoring |
| Manual Annotation (V5) | Expert insights |
| User Feedback (V8) | Ground truth from real users |

**The feedback loop:**
1. Collect user feedback
2. Find patterns in negative ratings
3. Update golden dataset with validated examples
4. Tune LLM judge rubrics based on user agreement
5. Prioritize manual annotation on user-flagged traces
6. Improve agent → better feedback → repeat

---

## Resources

- [Langfuse User Feedback](https://langfuse.com/docs/scores/user-feedback)
- [LangfuseWeb SDK](https://langfuse.com/docs/sdk/typescript/browser)
- [Scores Documentation](https://langfuse.com/docs/scores)
