# Strategy 1: User Feedback (Explicit)

## Overview

**What it is**: Direct, explicit feedback from users about agent responses through ratings, surveys, and feedback forms.

**Philosophy**: Users are the ultimate judges of whether an agent is helpful. Their explicit feedback provides ground truth that no automated system can replicate.

```
┌─────────────────────────────────────────────────────────┐
│                  USER FEEDBACK FLOW                     │
│                                                         │
│   User ──► Agent Response ──► Feedback UI ──► Storage  │
│                                    │                    │
│                                    ▼                    │
│                            ┌──────────────┐            │
│                            │  👍 / 👎     │            │
│                            │  ⭐⭐⭐⭐⭐    │            │
│                            │  "Comment..."│            │
│                            └──────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## What It Measures

| Metric | Type | Description |
|--------|------|-------------|
| **Satisfaction** | Binary | Thumbs up/down on responses |
| **Quality Rating** | 1-5 Scale | Overall response quality |
| **Helpfulness** | 1-5 Scale | Did it help accomplish the task? |
| **Accuracy** | 1-5 Scale | Was the information correct? |
| **Issue Category** | Categorical | What went wrong (if negative) |
| **Free-form Comment** | Text | Detailed user feedback |

## When to Use

✅ **Good for:**
- Measuring real user satisfaction
- Identifying issues automated systems miss
- Training and calibrating LLM judges
- Building golden datasets from user-verified examples

❌ **Limitations:**
- Low response rate (typically 1-5%)
- Selection bias (extremes more likely to respond)
- Subjectivity varies between users
- Can't scale to evaluate every response

## Implementation Plan for Dynamous Agent

### Database Schema

```sql
-- Add to sql/ directory as new migration
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,

    -- Link to request/conversation
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR REFERENCES conversations(session_id),
    message_id INTEGER REFERENCES messages(id),
    user_id UUID REFERENCES user_profiles(id),

    -- Core feedback
    rating_type VARCHAR NOT NULL,        -- 'thumbs', 'stars', 'detailed'
    rating_value INTEGER NOT NULL,       -- 0/1 for thumbs, 1-5 for stars

    -- Detailed ratings (optional)
    helpfulness_score INTEGER,           -- 1-5
    accuracy_score INTEGER,              -- 1-5

    -- Issue categorization (for negative feedback)
    issue_category VARCHAR,              -- 'wrong_answer', 'incomplete', 'irrelevant', 'safety', 'other'

    -- Free-form
    comment TEXT,

    -- Metadata
    feedback_context JSONB,              -- UI context, what was shown
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_thumbs CHECK (
        rating_type != 'thumbs' OR rating_value IN (0, 1)
    ),
    CONSTRAINT valid_stars CHECK (
        rating_type != 'stars' OR rating_value BETWEEN 1 AND 5
    )
);

-- Index for analysis queries
CREATE INDEX idx_user_feedback_session ON user_feedback(session_id);
CREATE INDEX idx_user_feedback_rating ON user_feedback(rating_type, rating_value);
CREATE INDEX idx_user_feedback_created ON user_feedback(created_at);

-- RLS Policy
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can submit feedback"
ON user_feedback FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own feedback"
ON user_feedback FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all feedback"
ON user_feedback FOR SELECT
USING (is_admin());
```

### Backend API Endpoint

```python
# backend_agent_api/feedback_api.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

class FeedbackRequest(BaseModel):
    """User feedback submission"""
    request_id: str
    session_id: str
    message_id: Optional[int] = None

    rating_type: Literal["thumbs", "stars", "detailed"]
    rating_value: int = Field(..., ge=0, le=5)

    # Optional detailed ratings
    helpfulness_score: Optional[int] = Field(None, ge=1, le=5)
    accuracy_score: Optional[int] = Field(None, ge=1, le=5)

    # Issue categorization
    issue_category: Optional[Literal[
        "wrong_answer",
        "incomplete",
        "irrelevant",
        "too_slow",
        "safety_concern",
        "other"
    ]] = None

    comment: Optional[str] = Field(None, max_length=1000)

class FeedbackResponse(BaseModel):
    id: int
    received_at: datetime
    message: str

@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    user: dict = Depends(verify_token),
    supabase = Depends(get_supabase)
):
    """Submit user feedback for an agent response"""

    # Validate rating based on type
    if feedback.rating_type == "thumbs" and feedback.rating_value not in [0, 1]:
        raise HTTPException(400, "Thumbs rating must be 0 or 1")
    if feedback.rating_type == "stars" and not (1 <= feedback.rating_value <= 5):
        raise HTTPException(400, "Star rating must be 1-5")

    # Insert feedback
    result = supabase.table("user_feedback").insert({
        "request_id": feedback.request_id,
        "session_id": feedback.session_id,
        "message_id": feedback.message_id,
        "user_id": user["id"],
        "rating_type": feedback.rating_type,
        "rating_value": feedback.rating_value,
        "helpfulness_score": feedback.helpfulness_score,
        "accuracy_score": feedback.accuracy_score,
        "issue_category": feedback.issue_category,
        "comment": feedback.comment,
        "feedback_context": {
            "user_agent": "...",  # Capture UI context
            "timestamp": datetime.utcnow().isoformat()
        }
    }).execute()

    # Send to Langfuse as score
    if langfuse_configured:
        langfuse.score(
            trace_id=feedback.request_id,
            name="user_feedback",
            value=feedback.rating_value / 5 if feedback.rating_type == "stars" else feedback.rating_value,
            comment=feedback.comment
        )

    return FeedbackResponse(
        id=result.data[0]["id"],
        received_at=datetime.utcnow(),
        message="Feedback received. Thank you!"
    )

@router.get("/stats")
async def get_feedback_stats(
    days: int = 30,
    user: dict = Depends(verify_token)
):
    """Get feedback statistics (admin only)"""
    if not user.get("is_admin"):
        raise HTTPException(403, "Admin access required")

    # Query aggregated stats
    # ... implementation
```

### Frontend Integration

```typescript
// frontend/src/components/chat/FeedbackWidget.tsx

import { useState } from 'react';
import { ThumbsUp, ThumbsDown, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { api } from '@/lib/api';

interface FeedbackWidgetProps {
  requestId: string;
  sessionId: string;
  messageId: number;
}

export function FeedbackWidget({ requestId, sessionId, messageId }: FeedbackWidgetProps) {
  const [submitted, setSubmitted] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [feedback, setFeedback] = useState({
    rating: null as 'positive' | 'negative' | null,
    comment: '',
    issueCategory: null as string | null,
  });

  const submitFeedback = async (isPositive: boolean) => {
    setFeedback(prev => ({ ...prev, rating: isPositive ? 'positive' : 'negative' }));

    if (!isPositive) {
      setShowDetails(true);
      return;
    }

    await api.post('/feedback/submit', {
      request_id: requestId,
      session_id: sessionId,
      message_id: messageId,
      rating_type: 'thumbs',
      rating_value: 1,
    });

    setSubmitted(true);
  };

  const submitDetailedFeedback = async () => {
    await api.post('/feedback/submit', {
      request_id: requestId,
      session_id: sessionId,
      message_id: messageId,
      rating_type: 'thumbs',
      rating_value: 0,
      issue_category: feedback.issueCategory,
      comment: feedback.comment,
    });

    setSubmitted(true);
  };

  if (submitted) {
    return <span className="text-sm text-muted-foreground">Thanks for your feedback!</span>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Was this helpful?</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => submitFeedback(true)}
          className={feedback.rating === 'positive' ? 'text-green-500' : ''}
        >
          <ThumbsUp className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => submitFeedback(false)}
          className={feedback.rating === 'negative' ? 'text-red-500' : ''}
        >
          <ThumbsDown className="h-4 w-4" />
        </Button>
      </div>

      {showDetails && (
        <div className="flex flex-col gap-2 p-3 border rounded-lg">
          <span className="text-sm font-medium">What went wrong?</span>
          <div className="flex flex-wrap gap-2">
            {['wrong_answer', 'incomplete', 'irrelevant', 'too_slow', 'other'].map(category => (
              <Button
                key={category}
                variant={feedback.issueCategory === category ? 'default' : 'outline'}
                size="sm"
                onClick={() => setFeedback(prev => ({ ...prev, issueCategory: category }))}
              >
                {category.replace('_', ' ')}
              </Button>
            ))}
          </div>
          <Textarea
            placeholder="Tell us more (optional)"
            value={feedback.comment}
            onChange={(e) => setFeedback(prev => ({ ...prev, comment: e.target.value }))}
          />
          <Button onClick={submitDetailedFeedback}>Submit Feedback</Button>
        </div>
      )}
    </div>
  );
}
```

### Evaluator Implementation

```python
# backend_agent_api/evals/evaluators/user_feedback.py

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

@dataclass
class UserFeedbackScore:
    """Result of user feedback evaluation"""
    score: float                    # 0.0 to 1.0
    total_feedback: int             # Number of feedback items
    positive_rate: float            # Percentage positive
    average_rating: Optional[float] # Average star rating if available
    common_issues: list[str]        # Most common issue categories
    sample_comments: list[str]      # Recent comments

class UserFeedbackEvaluator:
    """
    Evaluates agent performance based on explicit user feedback.

    This evaluator aggregates user ratings and feedback to produce
    quality scores. It's most useful for:
    - Overall agent health monitoring
    - Identifying problematic response patterns
    - Calibrating other evaluators (LLM judge, golden dataset)
    """

    def __init__(
        self,
        supabase_client,
        satisfaction_threshold: float = 0.75,
        min_feedback_count: int = 10,
    ):
        self.supabase = supabase_client
        self.satisfaction_threshold = satisfaction_threshold
        self.min_feedback_count = min_feedback_count

    async def evaluate_request(
        self,
        request_id: str
    ) -> Optional[UserFeedbackScore]:
        """Get feedback score for a specific request"""
        result = self.supabase.table("user_feedback")\
            .select("*")\
            .eq("request_id", request_id)\
            .execute()

        if not result.data:
            return None  # No feedback yet

        return self._calculate_score(result.data)

    async def evaluate_session(
        self,
        session_id: str
    ) -> Optional[UserFeedbackScore]:
        """Get aggregated feedback score for a session"""
        result = self.supabase.table("user_feedback")\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()

        if not result.data:
            return None

        return self._calculate_score(result.data)

    async def evaluate_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> UserFeedbackScore:
        """Get aggregated feedback score for a time period"""
        result = self.supabase.table("user_feedback")\
            .select("*")\
            .gte("created_at", start.isoformat())\
            .lte("created_at", end.isoformat())\
            .execute()

        return self._calculate_score(result.data)

    def _calculate_score(self, feedback_items: list[dict]) -> UserFeedbackScore:
        """Calculate aggregated score from feedback items"""
        if not feedback_items:
            return UserFeedbackScore(
                score=0.5,  # Neutral when no data
                total_feedback=0,
                positive_rate=0.0,
                average_rating=None,
                common_issues=[],
                sample_comments=[]
            )

        # Separate by rating type
        thumbs = [f for f in feedback_items if f["rating_type"] == "thumbs"]
        stars = [f for f in feedback_items if f["rating_type"] == "stars"]

        # Calculate positive rate from thumbs
        positive_thumbs = sum(1 for f in thumbs if f["rating_value"] == 1)
        positive_rate = positive_thumbs / len(thumbs) if thumbs else 0.5

        # Calculate average star rating
        avg_stars = None
        if stars:
            avg_stars = sum(f["rating_value"] for f in stars) / len(stars)

        # Combine into single score
        if thumbs and stars:
            # Weight thumbs and stars equally
            thumbs_score = positive_rate
            stars_score = (avg_stars - 1) / 4  # Normalize 1-5 to 0-1
            score = (thumbs_score + stars_score) / 2
        elif thumbs:
            score = positive_rate
        elif stars:
            score = (avg_stars - 1) / 4
        else:
            score = 0.5

        # Extract common issues
        issues = [f["issue_category"] for f in feedback_items if f.get("issue_category")]
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:3]

        # Sample recent comments
        comments = [f["comment"] for f in feedback_items if f.get("comment")]
        sample_comments = comments[-5:]  # Last 5 comments

        return UserFeedbackScore(
            score=score,
            total_feedback=len(feedback_items),
            positive_rate=positive_rate,
            average_rating=avg_stars,
            common_issues=common_issues,
            sample_comments=sample_comments
        )

    async def get_low_rated_examples(
        self,
        limit: int = 100
    ) -> list[dict]:
        """
        Get examples with negative feedback for analysis.
        These can be used to build golden datasets and improve the agent.
        """
        result = self.supabase.table("user_feedback")\
            .select("*, requests(*), messages(*)")\
            .eq("rating_type", "thumbs")\
            .eq("rating_value", 0)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return result.data
```

### Integration with Langfuse

```python
# backend_agent_api/evals/langfuse_sync.py

async def sync_feedback_to_langfuse(
    feedback_id: int,
    supabase,
    langfuse
):
    """Sync user feedback to Langfuse as a score"""
    feedback = supabase.table("user_feedback")\
        .select("*")\
        .eq("id", feedback_id)\
        .single()\
        .execute()

    if not feedback.data:
        return

    f = feedback.data

    # Normalize score to 0-1
    if f["rating_type"] == "thumbs":
        score = float(f["rating_value"])
    else:
        score = (f["rating_value"] - 1) / 4

    # Send to Langfuse
    langfuse.score(
        trace_id=f["request_id"],
        name="user_feedback",
        value=score,
        comment=f.get("comment"),
        data_type="NUMERIC"
    )

    # Also send categorical data if present
    if f.get("issue_category"):
        langfuse.score(
            trace_id=f["request_id"],
            name="user_feedback_issue",
            value=f["issue_category"],
            data_type="CATEGORICAL"
        )
```

### Metrics & Dashboard

```python
# backend_agent_api/evals/metrics/feedback_metrics.py

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class FeedbackMetrics:
    """Aggregated feedback metrics for dashboard"""
    period_start: datetime
    period_end: datetime

    # Volume
    total_responses: int
    responses_with_feedback: int
    feedback_rate: float

    # Satisfaction
    overall_satisfaction: float      # 0-1
    thumbs_up_rate: float
    average_star_rating: float

    # Issues breakdown
    issue_distribution: dict[str, int]

    # Trends
    satisfaction_trend: float        # Change from previous period

    # Actionable insights
    worst_performing_queries: list[dict]
    improvement_suggestions: list[str]

async def calculate_feedback_metrics(
    supabase,
    period_days: int = 7
) -> FeedbackMetrics:
    """Calculate feedback metrics for dashboard"""
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)
    prev_start = start - timedelta(days=period_days)

    # Current period
    current = supabase.table("user_feedback")\
        .select("*")\
        .gte("created_at", start.isoformat())\
        .execute()

    # Previous period for trend
    previous = supabase.table("user_feedback")\
        .select("*")\
        .gte("created_at", prev_start.isoformat())\
        .lt("created_at", start.isoformat())\
        .execute()

    # Total responses in period
    total = supabase.table("requests")\
        .select("id", count="exact")\
        .gte("timestamp", start.isoformat())\
        .execute()

    # Calculate metrics
    # ... implementation

    return FeedbackMetrics(...)
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Feedback collection rate | >3% of responses | <1% |
| Overall satisfaction | >80% positive | <70% |
| Average star rating | >4.0 | <3.5 |
| Response time to negative feedback | <24h review | >48h |

## Integration Points in Existing Code

| File | Location | Change |
|------|----------|--------|
| `agent_api.py` | Response streaming | Add feedback widget trigger |
| `frontend/src/components/chat/MessageItem.tsx` | Message display | Add FeedbackWidget component |
| `frontend/src/lib/api.ts` | API client | Add feedback endpoints |
| `sql/` | New migration | Add user_feedback table |

## Testing

```python
# tests/evals/test_user_feedback.py

import pytest
from evals.evaluators.user_feedback import UserFeedbackEvaluator

@pytest.fixture
def mock_supabase():
    # Mock Supabase client with test data
    ...

async def test_positive_feedback_score(mock_supabase):
    evaluator = UserFeedbackEvaluator(mock_supabase)

    # Insert test feedback
    mock_supabase.table("user_feedback").insert([
        {"rating_type": "thumbs", "rating_value": 1},
        {"rating_type": "thumbs", "rating_value": 1},
        {"rating_type": "thumbs", "rating_value": 0},
    ])

    score = await evaluator.evaluate_timerange(...)

    assert score.positive_rate == 2/3
    assert score.score > 0.6

async def test_issue_categorization(mock_supabase):
    evaluator = UserFeedbackEvaluator(mock_supabase)

    mock_supabase.table("user_feedback").insert([
        {"rating_type": "thumbs", "rating_value": 0, "issue_category": "wrong_answer"},
        {"rating_type": "thumbs", "rating_value": 0, "issue_category": "wrong_answer"},
        {"rating_type": "thumbs", "rating_value": 0, "issue_category": "incomplete"},
    ])

    score = await evaluator.evaluate_timerange(...)

    assert score.common_issues[0] == "wrong_answer"
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ✅ STRONG**

User feedback is an excellent fit for Langfuse because:
1. Langfuse's Score API is designed exactly for this use case
2. Scores are attached directly to traces for correlation
3. Dashboard analytics show feedback trends over time
4. Enables comparison with LLM judge scores

**Integration Pattern:**

```python
# When user submits feedback, sync to Langfuse
from langfuse import Langfuse

langfuse = Langfuse()

def on_feedback_submit(feedback: UserFeedback):
    # Normalize to 0-1 scale
    if feedback.rating_type == "thumbs":
        value = 1.0 if feedback.rating_value == 1 else 0.0
    else:  # stars
        value = (feedback.rating_value - 1) / 4  # 1-5 -> 0-1

    # Attach to trace
    langfuse.score(
        trace_id=feedback.request_id,
        name="user_feedback",
        value=value,
        comment=feedback.comment,
        data_type="NUMERIC"
    )

    # Optional: categorical issue tracking
    if feedback.issue_category:
        langfuse.score(
            trace_id=feedback.request_id,
            name="user_issue_category",
            value=feedback.issue_category,
            data_type="CATEGORICAL"
        )
```

**Langfuse Dashboard Benefits:**
- Filter traces by user satisfaction
- Compare user feedback vs LLM judge scores
- Identify systematic quality issues
- Track satisfaction trends over time

### Pydantic AI Support

**Fit Level: ❌ NOT SUPPORTED**

Pydantic AI does not provide built-in support for user feedback collection. This must be implemented manually because:

1. **User feedback is application-level**: It requires UI components, API endpoints, and database storage - all outside Pydantic AI's scope
2. **Real-time human interaction**: Pydantic AI focuses on automated evaluation, not human-in-the-loop feedback
3. **No evaluator equivalent**: There's no `UserFeedbackEvaluator` in pydantic-evals

**What You Need to Build:**
- Frontend feedback widget (React/Vue/etc.)
- Backend API endpoints (FastAPI)
- Database table for feedback storage
- Langfuse sync for analytics

**Possible Future Integration:**
Once you have feedback data, you could use it to:
1. Create golden dataset cases from highly-rated responses
2. Calibrate `LLMJudge` evaluators against human scores
3. Weight evaluation results by user satisfaction

```python
# Example: Using user feedback to calibrate LLM judge
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import LLMJudge

# Load cases with user feedback
async def create_calibration_dataset(supabase) -> Dataset:
    # Get responses with high user ratings
    positive = await supabase.table("user_feedback")\
        .select("*, requests(*), messages(*)")\
        .eq("rating_value", 1)\
        .limit(100)\
        .execute()

    cases = [
        Case(
            name=f"user_positive_{i}",
            inputs={"query": r["requests"]["user_query"]},
            expected_output=r["messages"]["content"],
            metadata={"user_rating": 1.0}
        )
        for i, r in enumerate(positive.data)
    ]

    return Dataset(
        cases=cases,
        evaluators=[
            LLMJudge(
                rubric="Rate this response quality. User rated it positively.",
                include_input=True,
                include_expected_output=True
            )
        ]
    )
```
