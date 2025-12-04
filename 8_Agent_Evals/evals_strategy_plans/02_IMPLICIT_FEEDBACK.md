# Strategy 2: Implicit Feedback (Behavioral Analysis)

## Overview

**What it is**: Analyzing user behavior patterns to infer satisfaction without explicit feedback. Users "vote with their actions" - how they interact after receiving a response reveals quality.

**Philosophy**: Actions speak louder than ratings. A user who immediately asks a follow-up clarification question is implicitly saying "that wasn't helpful." A user who completes their task and leaves is implicitly satisfied.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     IMPLICIT FEEDBACK SIGNALS                           │
│                                                                         │
│   POSITIVE SIGNALS              │    NEGATIVE SIGNALS                  │
│   ──────────────────            │    ─────────────────                 │
│   ✅ Task completion            │    ❌ Immediate follow-up question   │
│   ✅ Session ends naturally     │    ❌ Rephrasing same question       │
│   ✅ Returns to use agent       │    ❌ Session abandonment            │
│   ✅ Long time before next Q    │    ❌ Rapid-fire questions           │
│   ✅ Uses agent recommendation  │    ❌ Ignores agent suggestion       │
│   ✅ Shares/references response │    ❌ Switches to manual search      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Signal | Type | Interpretation | Weight |
|--------|------|----------------|--------|
| **Follow-up Questions** | Count | Clarification needed | High |
| **Rephrase Rate** | Ratio | Original answer unclear | High |
| **Time to Next Query** | Duration | Processing/satisfaction | Medium |
| **Session Duration** | Duration | Engagement level | Medium |
| **Abandonment** | Boolean | Frustration/giving up | High |
| **Return Rate** | Ratio | Long-term satisfaction | High |
| **Tool Adoption** | Boolean | Trusted the suggestion | Medium |
| **Query Complexity Trend** | Direction | Building on answers | Low |

## When to Use

✅ **Good for:**
- 100% coverage (every interaction generates signals)
- No user effort required
- Real-time monitoring
- Detecting patterns explicit feedback misses
- Correlating with explicit feedback to validate

❌ **Limitations:**
- Signals can be ambiguous
- External factors (user got distracted, etc.)
- Requires baseline calibration
- Different user types have different patterns

## Implementation Plan for Dynamous Agent

### Database Schema

```sql
-- Behavioral tracking table
CREATE TABLE user_behavior_signals (
    id SERIAL PRIMARY KEY,

    -- Context
    session_id VARCHAR REFERENCES conversations(session_id),
    user_id UUID REFERENCES user_profiles(id),
    request_id UUID REFERENCES requests(id),
    message_id INTEGER REFERENCES messages(id),

    -- Signal data
    signal_type VARCHAR NOT NULL,        -- 'follow_up', 'rephrase', 'abandonment', etc.
    signal_value FLOAT,                  -- Normalized 0-1 or raw value
    signal_metadata JSONB,               -- Additional context

    -- Timing
    time_since_response_ms INTEGER,      -- Time between response and this signal
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Session-level behavioral metrics
CREATE TABLE session_behavior_metrics (
    session_id VARCHAR PRIMARY KEY REFERENCES conversations(session_id),
    user_id UUID REFERENCES user_profiles(id),

    -- Engagement metrics
    total_messages INTEGER DEFAULT 0,
    user_messages INTEGER DEFAULT 0,
    agent_messages INTEGER DEFAULT 0,

    -- Timing metrics
    session_duration_ms INTEGER,
    avg_response_gap_ms INTEGER,         -- Avg time between user messages
    min_response_gap_ms INTEGER,
    max_response_gap_ms INTEGER,

    -- Quality signals
    follow_up_count INTEGER DEFAULT 0,   -- Questions asking for clarification
    rephrase_count INTEGER DEFAULT 0,    -- Same question rephrased
    topic_switches INTEGER DEFAULT 0,    -- Abrupt topic changes

    -- Outcome signals
    task_completed BOOLEAN,              -- Inferred task completion
    abandoned BOOLEAN DEFAULT FALSE,     -- Session abandoned mid-task
    returned_within_24h BOOLEAN,         -- User came back

    -- Calculated scores
    engagement_score FLOAT,              -- 0-1
    satisfaction_score FLOAT,            -- 0-1 (inferred)

    -- Timestamps
    first_message_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_behavior_signals_session ON user_behavior_signals(session_id);
CREATE INDEX idx_behavior_signals_type ON user_behavior_signals(signal_type);
CREATE INDEX idx_session_metrics_user ON session_behavior_metrics(user_id);
CREATE INDEX idx_session_metrics_score ON session_behavior_metrics(satisfaction_score);
```

### Signal Detection System

```python
# backend_agent_api/evals/evaluators/implicit_feedback.py

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import re

@dataclass
class BehaviorSignal:
    """A detected behavioral signal"""
    signal_type: str
    signal_value: float
    confidence: float
    metadata: dict

@dataclass
class ImplicitFeedbackScore:
    """Aggregated implicit feedback evaluation"""
    score: float                        # 0-1 overall satisfaction inference
    engagement_score: float             # 0-1 engagement level
    signals_detected: list[BehaviorSignal]
    confidence: float                   # How confident in the inference
    interpretation: str                 # Human-readable explanation

class ImplicitFeedbackEvaluator:
    """
    Evaluates agent responses based on implicit user behavior signals.

    Analyzes patterns like:
    - Follow-up questions (indicates incomplete answer)
    - Rephrasing (indicates unclear answer)
    - Session abandonment (indicates frustration)
    - Time gaps (indicates thinking/satisfaction)
    """

    # Patterns that indicate follow-up/clarification questions
    FOLLOW_UP_PATTERNS = [
        r"what do you mean",
        r"can you (explain|clarify|elaborate)",
        r"i don't understand",
        r"what about",
        r"but (what|how|why)",
        r"you didn't (answer|address|mention)",
        r"that's not what i (asked|meant)",
        r"more (detail|specific|information)",
        r"\?\s*\?",  # Multiple question marks
    ]

    # Patterns that indicate rephrasing
    REPHRASE_INDICATORS = [
        r"let me rephrase",
        r"what i meant was",
        r"in other words",
        r"to be (more )?clear",
        r"i'm asking about",
    ]

    # Patterns that indicate task completion
    COMPLETION_PATTERNS = [
        r"(thanks|thank you|thx|ty)",
        r"(perfect|great|awesome|excellent)",
        r"that('s| is) (exactly )?what i needed",
        r"got it",
        r"makes sense",
    ]

    # Patterns that indicate frustration
    FRUSTRATION_PATTERNS = [
        r"(this|you) (is|are) (not |un)?helpful",
        r"never ?mind",
        r"forget it",
        r"i('ll| will) (just )?(do it|figure it out) myself",
        r"useless",
    ]

    def __init__(
        self,
        supabase_client,
        engagement_threshold: float = 0.6,
        follow_up_penalty: float = 0.15,
        rephrase_penalty: float = 0.20,
        abandonment_penalty: float = 0.30,
    ):
        self.supabase = supabase_client
        self.engagement_threshold = engagement_threshold
        self.follow_up_penalty = follow_up_penalty
        self.rephrase_penalty = rephrase_penalty
        self.abandonment_penalty = abandonment_penalty

        # Compile regex patterns
        self.follow_up_regex = re.compile(
            '|'.join(self.FOLLOW_UP_PATTERNS),
            re.IGNORECASE
        )
        self.rephrase_regex = re.compile(
            '|'.join(self.REPHRASE_INDICATORS),
            re.IGNORECASE
        )
        self.completion_regex = re.compile(
            '|'.join(self.COMPLETION_PATTERNS),
            re.IGNORECASE
        )
        self.frustration_regex = re.compile(
            '|'.join(self.FRUSTRATION_PATTERNS),
            re.IGNORECASE
        )

    async def analyze_message_pair(
        self,
        agent_response: str,
        user_followup: str,
        time_gap_ms: int
    ) -> list[BehaviorSignal]:
        """Analyze a single agent response + user follow-up pair"""
        signals = []

        # Check for follow-up question
        if self.follow_up_regex.search(user_followup):
            signals.append(BehaviorSignal(
                signal_type="follow_up_question",
                signal_value=-self.follow_up_penalty,
                confidence=0.8,
                metadata={"pattern": "clarification_request"}
            ))

        # Check for rephrase
        if self.rephrase_regex.search(user_followup):
            signals.append(BehaviorSignal(
                signal_type="rephrase",
                signal_value=-self.rephrase_penalty,
                confidence=0.85,
                metadata={"pattern": "rephrase_indicator"}
            ))

        # Check for completion signals
        if self.completion_regex.search(user_followup):
            signals.append(BehaviorSignal(
                signal_type="task_completion",
                signal_value=0.20,
                confidence=0.75,
                metadata={"pattern": "satisfaction_indicator"}
            ))

        # Check for frustration
        if self.frustration_regex.search(user_followup):
            signals.append(BehaviorSignal(
                signal_type="frustration",
                signal_value=-0.30,
                confidence=0.9,
                metadata={"pattern": "frustration_indicator"}
            ))

        # Analyze time gap
        if time_gap_ms < 3000:  # Less than 3 seconds
            signals.append(BehaviorSignal(
                signal_type="rapid_response",
                signal_value=-0.05,  # Might indicate frustration
                confidence=0.5,
                metadata={"time_gap_ms": time_gap_ms}
            ))
        elif time_gap_ms > 60000:  # More than 1 minute
            signals.append(BehaviorSignal(
                signal_type="considered_response",
                signal_value=0.05,  # User took time to process
                confidence=0.4,
                metadata={"time_gap_ms": time_gap_ms}
            ))

        return signals

    async def analyze_session(
        self,
        session_id: str
    ) -> ImplicitFeedbackScore:
        """Analyze entire session for implicit feedback signals"""

        # Get all messages in session
        messages = self.supabase.table("messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at")\
            .execute()

        if not messages.data or len(messages.data) < 2:
            return ImplicitFeedbackScore(
                score=0.5,
                engagement_score=0.0,
                signals_detected=[],
                confidence=0.0,
                interpretation="Insufficient data for analysis"
            )

        all_signals = []
        msg_list = messages.data

        # Analyze message pairs
        for i in range(len(msg_list) - 1):
            current = msg_list[i]
            next_msg = msg_list[i + 1]

            # Only analyze agent response -> user message pairs
            current_content = current.get("message", {})
            next_content = next_msg.get("message", {})

            if (current_content.get("type") == "agent" and
                next_content.get("type") == "user"):

                time_gap = self._calculate_time_gap(
                    current["created_at"],
                    next_msg["created_at"]
                )

                signals = await self.analyze_message_pair(
                    agent_response=current_content.get("content", ""),
                    user_followup=next_content.get("content", ""),
                    time_gap_ms=time_gap
                )
                all_signals.extend(signals)

        # Check for session abandonment
        last_msg = msg_list[-1]
        last_content = last_msg.get("message", {})

        if last_content.get("type") == "agent":
            # Session ended with agent response - check for abandonment
            time_since_last = self._time_since(last_msg["created_at"])

            if time_since_last > timedelta(hours=1):
                # Check if this looks like abandonment vs completion
                last_text = last_content.get("content", "")
                if not self.completion_regex.search(last_text):
                    all_signals.append(BehaviorSignal(
                        signal_type="potential_abandonment",
                        signal_value=-self.abandonment_penalty,
                        confidence=0.6,
                        metadata={"hours_since_last": time_since_last.total_seconds() / 3600}
                    ))

        # Calculate overall score
        base_score = 0.7  # Start optimistic
        for signal in all_signals:
            base_score += signal.signal_value * signal.confidence

        # Clamp to 0-1
        score = max(0.0, min(1.0, base_score))

        # Calculate engagement score
        user_msg_count = sum(1 for m in msg_list if m.get("message", {}).get("type") == "user")
        engagement_score = min(1.0, user_msg_count / 10)  # Normalize to 10 messages

        # Generate interpretation
        interpretation = self._generate_interpretation(all_signals, score)

        # Calculate confidence based on data quality
        confidence = min(0.9, len(msg_list) / 20)  # More messages = more confidence

        return ImplicitFeedbackScore(
            score=score,
            engagement_score=engagement_score,
            signals_detected=all_signals,
            confidence=confidence,
            interpretation=interpretation
        )

    def _calculate_time_gap(self, time1: str, time2: str) -> int:
        """Calculate milliseconds between two timestamps"""
        t1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
        return int((t2 - t1).total_seconds() * 1000)

    def _time_since(self, timestamp: str) -> timedelta:
        """Calculate time since a timestamp"""
        t = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return datetime.now(t.tzinfo) - t

    def _generate_interpretation(
        self,
        signals: list[BehaviorSignal],
        score: float
    ) -> str:
        """Generate human-readable interpretation"""
        if not signals:
            return "No significant behavioral signals detected."

        signal_types = [s.signal_type for s in signals]

        parts = []
        if "follow_up_question" in signal_types:
            count = signal_types.count("follow_up_question")
            parts.append(f"{count} clarification question(s) asked")

        if "rephrase" in signal_types:
            parts.append("user rephrased their question")

        if "task_completion" in signal_types:
            parts.append("task completion signals detected")

        if "frustration" in signal_types:
            parts.append("frustration signals detected")

        if "potential_abandonment" in signal_types:
            parts.append("session may have been abandoned")

        summary = "; ".join(parts) if parts else "Mixed signals"

        if score > 0.8:
            return f"High satisfaction inferred. {summary}."
        elif score > 0.6:
            return f"Moderate satisfaction. {summary}."
        elif score > 0.4:
            return f"Some issues detected. {summary}."
        else:
            return f"Low satisfaction inferred. {summary}."

    async def update_session_metrics(self, session_id: str):
        """Update aggregated session metrics in database"""
        score = await self.analyze_session(session_id)

        self.supabase.table("session_behavior_metrics").upsert({
            "session_id": session_id,
            "engagement_score": score.engagement_score,
            "satisfaction_score": score.score,
            "follow_up_count": sum(
                1 for s in score.signals_detected
                if s.signal_type == "follow_up_question"
            ),
            "rephrase_count": sum(
                1 for s in score.signals_detected
                if s.signal_type == "rephrase"
            ),
            "abandoned": any(
                s.signal_type == "potential_abandonment"
                for s in score.signals_detected
            ),
            "task_completed": any(
                s.signal_type == "task_completion"
                for s in score.signals_detected
            ),
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
```

### Real-Time Signal Capture

```python
# backend_agent_api/evals/signal_capture.py

from datetime import datetime
from typing import Optional

class BehaviorSignalCapture:
    """
    Captures behavioral signals in real-time as users interact.
    Integrates with the agent API to track signals per-request.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.evaluator = ImplicitFeedbackEvaluator(supabase_client)

    async def on_user_message(
        self,
        session_id: str,
        user_id: str,
        request_id: str,
        message_content: str,
        previous_agent_response: Optional[str],
        time_since_response_ms: Optional[int]
    ):
        """Called when a new user message is received"""

        if previous_agent_response and time_since_response_ms:
            # Analyze the message pair
            signals = await self.evaluator.analyze_message_pair(
                agent_response=previous_agent_response,
                user_followup=message_content,
                time_gap_ms=time_since_response_ms
            )

            # Store each signal
            for signal in signals:
                self.supabase.table("user_behavior_signals").insert({
                    "session_id": session_id,
                    "user_id": user_id,
                    "request_id": request_id,
                    "signal_type": signal.signal_type,
                    "signal_value": signal.signal_value,
                    "signal_metadata": signal.metadata,
                    "time_since_response_ms": time_since_response_ms
                }).execute()

    async def on_session_end(self, session_id: str):
        """Called when a session ends (or times out)"""
        await self.evaluator.update_session_metrics(session_id)
```

### Integration with Agent API

```python
# Modifications to backend_agent_api/agent_api.py

from evals.signal_capture import BehaviorSignalCapture

# In lifespan or startup
signal_capture = BehaviorSignalCapture(supabase_client)

@app.post("/api/pydantic-agent")
async def pydantic_agent(request: AgentRequest, ...):

    # Get previous agent response for signal analysis
    previous_response = None
    time_since_response = None

    if conversation_history:
        last_agent_msg = next(
            (m for m in reversed(conversation_history)
             if m.get("message", {}).get("type") == "agent"),
            None
        )
        if last_agent_msg:
            previous_response = last_agent_msg["message"].get("content")
            time_since_response = calculate_time_gap(
                last_agent_msg["created_at"],
                datetime.utcnow()
            )

    # Capture behavioral signals
    await signal_capture.on_user_message(
        session_id=request.session_id,
        user_id=request.user_id,
        request_id=request.request_id,
        message_content=request.query,
        previous_agent_response=previous_response,
        time_since_response_ms=time_since_response
    )

    # ... rest of agent logic
```

### Metrics Dashboard

```python
# backend_agent_api/evals/metrics/implicit_metrics.py

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ImplicitFeedbackMetrics:
    """Dashboard metrics for implicit feedback"""
    period: str

    # Engagement
    avg_messages_per_session: float
    avg_session_duration_mins: float
    return_rate_24h: float

    # Quality signals
    follow_up_rate: float           # % of responses triggering follow-ups
    rephrase_rate: float            # % of questions rephrased
    abandonment_rate: float         # % of sessions abandoned

    # Inferred satisfaction
    avg_satisfaction_score: float
    satisfaction_trend: float       # Change from previous period

    # Breakdowns
    satisfaction_by_query_type: dict[str, float]
    worst_performing_patterns: list[dict]

async def calculate_implicit_metrics(
    supabase,
    period_days: int = 7
) -> ImplicitFeedbackMetrics:
    """Calculate implicit feedback metrics"""

    end = datetime.utcnow()
    start = end - timedelta(days=period_days)

    # Get session metrics
    sessions = supabase.table("session_behavior_metrics")\
        .select("*")\
        .gte("updated_at", start.isoformat())\
        .execute()

    if not sessions.data:
        return ImplicitFeedbackMetrics(
            period=f"Last {period_days} days",
            avg_messages_per_session=0,
            avg_session_duration_mins=0,
            return_rate_24h=0,
            follow_up_rate=0,
            rephrase_rate=0,
            abandonment_rate=0,
            avg_satisfaction_score=0.5,
            satisfaction_trend=0,
            satisfaction_by_query_type={},
            worst_performing_patterns=[]
        )

    data = sessions.data

    # Calculate aggregates
    avg_satisfaction = sum(s["satisfaction_score"] or 0.5 for s in data) / len(data)
    follow_up_sessions = sum(1 for s in data if s["follow_up_count"] > 0)
    rephrase_sessions = sum(1 for s in data if s["rephrase_count"] > 0)
    abandoned_sessions = sum(1 for s in data if s["abandoned"])

    return ImplicitFeedbackMetrics(
        period=f"Last {period_days} days",
        avg_messages_per_session=sum(s["total_messages"] or 0 for s in data) / len(data),
        avg_session_duration_mins=sum(
            (s["session_duration_ms"] or 0) / 60000 for s in data
        ) / len(data),
        return_rate_24h=sum(1 for s in data if s["returned_within_24h"]) / len(data),
        follow_up_rate=follow_up_sessions / len(data),
        rephrase_rate=rephrase_sessions / len(data),
        abandonment_rate=abandoned_sessions / len(data),
        avg_satisfaction_score=avg_satisfaction,
        satisfaction_trend=0,  # Calculate from previous period
        satisfaction_by_query_type={},
        worst_performing_patterns=[]
    )
```

### Correlation with Explicit Feedback

```python
# backend_agent_api/evals/correlation.py

async def correlate_feedback_types(
    supabase,
    period_days: int = 30
) -> dict:
    """
    Correlate implicit signals with explicit feedback.
    Used to validate and calibrate implicit signal weights.
    """

    # Get sessions with both implicit and explicit feedback
    sessions_with_both = supabase.rpc(
        "get_sessions_with_both_feedback_types",
        {"days": period_days}
    ).execute()

    if not sessions_with_both.data:
        return {"correlation": None, "sample_size": 0}

    # Calculate correlation
    implicit_scores = []
    explicit_scores = []

    for session in sessions_with_both.data:
        implicit_scores.append(session["satisfaction_score"])
        explicit_scores.append(session["avg_explicit_rating"])

    # Pearson correlation
    correlation = calculate_pearson(implicit_scores, explicit_scores)

    return {
        "correlation": correlation,
        "sample_size": len(sessions_with_both.data),
        "interpretation": (
            "Strong correlation" if correlation > 0.7 else
            "Moderate correlation" if correlation > 0.4 else
            "Weak correlation"
        )
    }
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Follow-up rate | <20% | >35% |
| Rephrase rate | <10% | >20% |
| Abandonment rate | <15% | >25% |
| Avg satisfaction score | >0.7 | <0.5 |
| Correlation with explicit | >0.6 | <0.4 |

## Key Insights This Strategy Provides

1. **Early Warning System**: Detect issues before users complain
2. **Pattern Detection**: Find problematic query types or response patterns
3. **Baseline for A/B Testing**: Compare behavior across agent versions
4. **Training Signal**: Use low-satisfaction sessions to improve agent
5. **Validation**: Cross-reference with explicit feedback

## Testing

```python
# tests/evals/test_implicit_feedback.py

import pytest
from evals.evaluators.implicit_feedback import ImplicitFeedbackEvaluator

@pytest.fixture
def evaluator():
    return ImplicitFeedbackEvaluator(mock_supabase())

async def test_follow_up_detection(evaluator):
    signals = await evaluator.analyze_message_pair(
        agent_response="The weather is sunny.",
        user_followup="What do you mean by sunny? Can you be more specific?",
        time_gap_ms=5000
    )

    assert any(s.signal_type == "follow_up_question" for s in signals)

async def test_completion_detection(evaluator):
    signals = await evaluator.analyze_message_pair(
        agent_response="Here are the Q3 sales figures...",
        user_followup="Perfect, thanks! That's exactly what I needed.",
        time_gap_ms=30000
    )

    assert any(s.signal_type == "task_completion" for s in signals)

async def test_frustration_detection(evaluator):
    signals = await evaluator.analyze_message_pair(
        agent_response="I can help you with that...",
        user_followup="This is useless, I'll figure it out myself",
        time_gap_ms=2000
    )

    assert any(s.signal_type == "frustration" for s in signals)
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ❌ WEAK**

Implicit feedback is NOT a good fit for Langfuse because:
1. **Session-level, not trace-level**: Behavioral patterns span multiple traces
2. **Aggregate metrics**: Langfuse focuses on individual trace analysis
3. **Different data model**: Engagement patterns need time-series analysis

**Keep in Application Database:**
- Session duration
- Abandonment events
- Follow-up patterns
- Re-ask detection

**Optional Langfuse Sync (for correlation only):**

```python
# Only sync summary scores, not raw behavioral data
def sync_session_summary_to_langfuse(session_id: str, summary: SessionSummary):
    """Sync session-level engagement score to final trace"""
    if summary.final_trace_id:
        langfuse.score(
            trace_id=summary.final_trace_id,
            name="session_engagement",
            value=summary.engagement_score,
            comment=f"Session duration: {summary.duration_minutes}min"
        )
```

**Why NOT Langfuse for this:**
- Langfuse traces are request-scoped
- Behavioral analysis is user/session-scoped
- Better to use application analytics (PostHog, Mixpanel, custom)

### Pydantic AI Support

**Fit Level: ❌ NOT SUPPORTED**

Pydantic AI provides no built-in support for implicit/behavioral feedback because:

1. **Outside evaluation scope**: Pydantic Evals evaluates outputs, not user behavior
2. **No session concept**: Pydantic AI operates at request level
3. **No behavioral patterns**: No engagement or timing analysis

**What You Need to Build Manually:**
- Session tracking middleware
- Behavioral signal detection
- Time-series database for patterns
- Custom analytics queries

**No Pydantic AI Integration Possible:**
Unlike other strategies, there's no way to use Pydantic AI for implicit feedback analysis. This is purely application-level instrumentation.

```python
# This is custom code, NOT Pydantic AI
class SessionAnalytics:
    """Purely custom - no Pydantic AI integration"""

    async def track_session_event(self, event: SessionEvent):
        # Custom tracking logic
        pass

    async def calculate_engagement_score(self, session_id: str) -> float:
        # Custom calculation
        pass
```
