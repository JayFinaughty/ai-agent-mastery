# Strategy 3: Manual Annotation (Expert Review)

## Overview

**What it is**: Human experts systematically review and score agent responses against defined criteria. This provides high-quality, trustworthy evaluation data.

**Philosophy**: Humans are still the gold standard for nuanced quality assessment. Expert annotations provide ground truth for training and calibrating automated evaluators.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MANUAL ANNOTATION WORKFLOW                           │
│                                                                         │
│  Sample Selection         Expert Review           Analysis              │
│  ────────────────         ─────────────           ────────              │
│                                                                         │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│  │ Random      │         │ Rubric-Based│         │ Inter-Rater │       │
│  │ Sampling    │────────►│ Scoring     │────────►│ Agreement   │       │
│  │ (5-10%)     │         │ (1-5 Scale) │         │ Analysis    │       │
│  └─────────────┘         └─────────────┘         └─────────────┘       │
│         │                       │                       │               │
│         ▼                       ▼                       ▼               │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│  │ Stratified  │         │ Multi-Dim   │         │ Calibration │       │
│  │ by Query    │         │ Assessment  │         │ & Training  │       │
│  │ Type        │         │             │         │ Data        │       │
│  └─────────────┘         └─────────────┘         └─────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Dimension | Description | Scale |
|-----------|-------------|-------|
| **Correctness** | Is the information factually accurate? | 1-5 |
| **Completeness** | Does it fully address the question? | 1-5 |
| **Relevance** | Is the response on-topic? | 1-5 |
| **Clarity** | Is it easy to understand? | 1-5 |
| **Helpfulness** | Does it help the user accomplish their goal? | 1-5 |
| **Safety** | Are there any safety concerns? | Pass/Fail |
| **Tone** | Is the tone appropriate? | 1-5 |
| **Source Attribution** | Are sources properly cited (for RAG)? | 1-5 |

## When to Use

✅ **Good for:**
- Creating gold-standard evaluation data
- Calibrating LLM judges
- Auditing for compliance/safety
- Building training datasets
- Validating automated evaluators

❌ **Limitations:**
- Expensive ($5-20 per annotation)
- Slow (hours to days)
- Limited scale (1-10% of data)
- Annotator bias/variability
- Requires trained annotators

## Implementation Plan for Dynamous Agent

### Database Schema

```sql
-- Annotation tasks queue
CREATE TABLE annotation_tasks (
    id SERIAL PRIMARY KEY,

    -- What to annotate
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR REFERENCES conversations(session_id),
    message_id INTEGER REFERENCES messages(id),

    -- Task metadata
    task_type VARCHAR NOT NULL,          -- 'response_quality', 'safety', 'rag_quality'
    priority INTEGER DEFAULT 0,          -- Higher = more urgent
    status VARCHAR DEFAULT 'pending',    -- 'pending', 'assigned', 'completed', 'disputed'

    -- Sampling info
    sampling_reason VARCHAR,             -- 'random', 'low_confidence', 'user_flagged', 'safety_concern'

    -- Assignment
    assigned_to UUID REFERENCES user_profiles(id),
    assigned_at TIMESTAMPTZ,
    due_at TIMESTAMPTZ,

    -- Completion
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Annotation results
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,

    -- Link to task
    task_id INTEGER REFERENCES annotation_tasks(id),
    annotator_id UUID REFERENCES user_profiles(id),

    -- What was annotated
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR,
    message_id INTEGER,

    -- Scores (using rubric)
    correctness_score INTEGER CHECK (correctness_score BETWEEN 1 AND 5),
    completeness_score INTEGER CHECK (completeness_score BETWEEN 1 AND 5),
    relevance_score INTEGER CHECK (relevance_score BETWEEN 1 AND 5),
    clarity_score INTEGER CHECK (clarity_score BETWEEN 1 AND 5),
    helpfulness_score INTEGER CHECK (helpfulness_score BETWEEN 1 AND 5),
    tone_score INTEGER CHECK (tone_score BETWEEN 1 AND 5),

    -- Safety assessment
    safety_passed BOOLEAN,
    safety_issues TEXT[],                -- Array of identified issues

    -- RAG-specific (if applicable)
    source_attribution_score INTEGER CHECK (source_attribution_score BETWEEN 1 AND 5),
    faithfulness_score INTEGER CHECK (faithfulness_score BETWEEN 1 AND 5),

    -- Overall
    overall_score INTEGER CHECK (overall_score BETWEEN 1 AND 5),

    -- Qualitative feedback
    strengths TEXT,
    weaknesses TEXT,
    suggested_improvement TEXT,

    -- Metadata
    annotation_time_seconds INTEGER,     -- How long it took
    confidence VARCHAR,                  -- 'high', 'medium', 'low'

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Annotation rubrics (for consistency)
CREATE TABLE annotation_rubrics (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    version INTEGER DEFAULT 1,

    -- Rubric definition
    dimensions JSONB NOT NULL,           -- Array of {name, description, scale, examples}

    -- Metadata
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Inter-annotator agreement tracking
CREATE TABLE annotation_agreements (
    id SERIAL PRIMARY KEY,
    request_id UUID,
    dimension VARCHAR,

    annotator_1_id UUID,
    annotator_1_score INTEGER,
    annotator_2_id UUID,
    annotator_2_score INTEGER,

    agreement BOOLEAN,                   -- Within 1 point = agree
    difference INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_annotation_tasks_status ON annotation_tasks(status);
CREATE INDEX idx_annotation_tasks_assigned ON annotation_tasks(assigned_to);
CREATE INDEX idx_annotations_request ON annotations(request_id);
CREATE INDEX idx_annotations_annotator ON annotations(annotator_id);
```

### Annotation Rubric Definition

```python
# backend_agent_api/evals/rubrics/response_quality.py

RESPONSE_QUALITY_RUBRIC = {
    "name": "response_quality_v1",
    "version": 1,
    "dimensions": [
        {
            "name": "correctness",
            "description": "Is the information factually accurate?",
            "scale": {
                1: "Completely incorrect or misleading information",
                2: "Mostly incorrect with some accurate elements",
                3: "Mix of correct and incorrect information",
                4: "Mostly correct with minor inaccuracies",
                5: "Completely accurate and verified"
            },
            "examples": {
                1: "Stated Python was created in 2010 (actually 1991)",
                5: "Correctly explained Python's creation in 1991 by Guido van Rossum"
            }
        },
        {
            "name": "completeness",
            "description": "Does the response fully address the user's question?",
            "scale": {
                1: "Does not address the question at all",
                2: "Addresses only a small part of the question",
                3: "Addresses main points but missing important details",
                4: "Addresses most aspects with minor omissions",
                5: "Fully addresses all aspects of the question"
            }
        },
        {
            "name": "relevance",
            "description": "Is the response on-topic and relevant?",
            "scale": {
                1: "Completely off-topic or irrelevant",
                2: "Mostly off-topic with some relevant content",
                3: "Partially relevant with tangential information",
                4: "Mostly relevant with minor tangents",
                5: "Entirely relevant and focused"
            }
        },
        {
            "name": "clarity",
            "description": "Is the response easy to understand?",
            "scale": {
                1: "Incomprehensible or extremely confusing",
                2: "Difficult to understand, poorly structured",
                3: "Understandable but could be clearer",
                4: "Clear with minor clarity issues",
                5: "Exceptionally clear and well-structured"
            }
        },
        {
            "name": "helpfulness",
            "description": "Does the response help the user accomplish their goal?",
            "scale": {
                1: "Not helpful at all, may hinder user",
                2: "Minimally helpful",
                3: "Somewhat helpful but user needs more",
                4: "Helpful for user's needs",
                5: "Extremely helpful, exceeds expectations"
            }
        }
    ]
}

RAG_QUALITY_RUBRIC = {
    "name": "rag_quality_v1",
    "version": 1,
    "dimensions": [
        {
            "name": "source_attribution",
            "description": "Are sources properly referenced?",
            "scale": {
                1: "No sources cited when needed",
                2: "Sources mentioned but not properly attributed",
                3: "Some sources cited, some missing",
                4: "Most sources properly attributed",
                5: "All sources clearly and properly attributed"
            }
        },
        {
            "name": "faithfulness",
            "description": "Is the response grounded in the retrieved documents?",
            "scale": {
                1: "Response contradicts or ignores retrieved documents",
                2: "Mostly fabricated with minimal grounding",
                3: "Mix of grounded and fabricated content",
                4: "Mostly grounded with minor extrapolations",
                5: "Fully grounded in retrieved documents"
            }
        }
    ]
}
```

### Sampling Strategy

```python
# backend_agent_api/evals/annotation/sampler.py

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import random

@dataclass
class SamplingConfig:
    """Configuration for annotation sampling"""
    random_sample_rate: float = 0.05     # 5% random sampling
    low_confidence_threshold: float = 0.6
    safety_keywords: list[str] = None
    max_daily_samples: int = 100

class AnnotationSampler:
    """
    Selects requests for manual annotation using multiple strategies.
    """

    def __init__(self, supabase_client, config: SamplingConfig = None):
        self.supabase = supabase_client
        self.config = config or SamplingConfig()

    async def should_sample(
        self,
        request_id: str,
        query: str,
        response: str,
        llm_judge_score: Optional[float] = None,
        user_feedback: Optional[int] = None
    ) -> tuple[bool, str]:
        """
        Determine if a request should be sampled for annotation.
        Returns (should_sample, reason)
        """

        # Check daily limit
        today_count = await self._get_today_sample_count()
        if today_count >= self.config.max_daily_samples:
            return False, "daily_limit_reached"

        # Strategy 1: Random sampling
        if random.random() < self.config.random_sample_rate:
            return True, "random"

        # Strategy 2: Low LLM judge confidence
        if llm_judge_score and llm_judge_score < self.config.low_confidence_threshold:
            return True, "low_confidence"

        # Strategy 3: Negative user feedback
        if user_feedback is not None and user_feedback == 0:
            return True, "negative_feedback"

        # Strategy 4: Safety-sensitive queries
        if self.config.safety_keywords:
            query_lower = query.lower()
            for keyword in self.config.safety_keywords:
                if keyword in query_lower:
                    return True, "safety_concern"

        # Strategy 5: Long responses (more room for error)
        if len(response) > 2000:
            if random.random() < 0.1:  # 10% of long responses
                return True, "long_response"

        return False, "not_selected"

    async def create_annotation_task(
        self,
        request_id: str,
        session_id: str,
        message_id: int,
        task_type: str,
        sampling_reason: str,
        priority: int = 0
    ):
        """Create an annotation task in the queue"""
        self.supabase.table("annotation_tasks").insert({
            "request_id": request_id,
            "session_id": session_id,
            "message_id": message_id,
            "task_type": task_type,
            "sampling_reason": sampling_reason,
            "priority": priority,
            "status": "pending",
            "due_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }).execute()

    async def _get_today_sample_count(self) -> int:
        """Get number of samples created today"""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        result = self.supabase.table("annotation_tasks")\
            .select("id", count="exact")\
            .gte("created_at", today_start.isoformat())\
            .execute()
        return result.count or 0

    async def get_stratified_sample(
        self,
        sample_size: int = 100,
        period_days: int = 7
    ) -> list[dict]:
        """
        Get a stratified sample for batch annotation.
        Ensures coverage across query types, time periods, and confidence levels.
        """
        start = datetime.utcnow() - timedelta(days=period_days)

        # Get all requests in period
        requests = self.supabase.table("requests")\
            .select("*, messages(*)")\
            .gte("timestamp", start.isoformat())\
            .execute()

        if not requests.data:
            return []

        # Stratify by various dimensions
        # ... implementation for stratified sampling

        return random.sample(requests.data, min(sample_size, len(requests.data)))
```

### Annotation Interface API

```python
# backend_agent_api/annotation_api.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

router = APIRouter(prefix="/api/annotations", tags=["annotations"])

class AnnotationScores(BaseModel):
    """Annotation scores following the rubric"""
    correctness_score: int = Field(..., ge=1, le=5)
    completeness_score: int = Field(..., ge=1, le=5)
    relevance_score: int = Field(..., ge=1, le=5)
    clarity_score: int = Field(..., ge=1, le=5)
    helpfulness_score: int = Field(..., ge=1, le=5)
    tone_score: Optional[int] = Field(None, ge=1, le=5)

    # Safety
    safety_passed: bool
    safety_issues: Optional[List[str]] = None

    # RAG-specific
    source_attribution_score: Optional[int] = Field(None, ge=1, le=5)
    faithfulness_score: Optional[int] = Field(None, ge=1, le=5)

    # Overall
    overall_score: int = Field(..., ge=1, le=5)

    # Qualitative
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    suggested_improvement: Optional[str] = None

    # Meta
    confidence: str = "medium"  # 'high', 'medium', 'low'

class AnnotationTask(BaseModel):
    """Annotation task for the queue"""
    id: int
    request_id: str
    session_id: str
    task_type: str
    priority: int
    status: str
    due_at: datetime

    # The content to annotate
    user_query: str
    agent_response: str
    retrieved_documents: Optional[List[dict]] = None

@router.get("/tasks", response_model=List[AnnotationTask])
async def get_annotation_tasks(
    status: str = "pending",
    limit: int = 10,
    user: dict = Depends(verify_token),
    supabase = Depends(get_supabase)
):
    """Get annotation tasks assigned to current user or unassigned"""
    if not user.get("is_admin"):
        raise HTTPException(403, "Annotator access required")

    tasks = supabase.table("annotation_tasks")\
        .select("*, requests(user_query), messages(message)")\
        .eq("status", status)\
        .or_(f"assigned_to.is.null,assigned_to.eq.{user['id']}")\
        .order("priority", desc=True)\
        .order("created_at")\
        .limit(limit)\
        .execute()

    return tasks.data

@router.post("/tasks/{task_id}/claim")
async def claim_task(
    task_id: int,
    user: dict = Depends(verify_token),
    supabase = Depends(get_supabase)
):
    """Claim an annotation task"""
    result = supabase.table("annotation_tasks")\
        .update({
            "assigned_to": user["id"],
            "assigned_at": datetime.utcnow().isoformat(),
            "status": "assigned"
        })\
        .eq("id", task_id)\
        .eq("status", "pending")\
        .execute()

    if not result.data:
        raise HTTPException(400, "Task not available")

    return {"message": "Task claimed", "task_id": task_id}

@router.post("/tasks/{task_id}/submit")
async def submit_annotation(
    task_id: int,
    scores: AnnotationScores,
    annotation_time_seconds: int,
    user: dict = Depends(verify_token),
    supabase = Depends(get_supabase)
):
    """Submit annotation for a task"""

    # Get task
    task = supabase.table("annotation_tasks")\
        .select("*")\
        .eq("id", task_id)\
        .eq("assigned_to", user["id"])\
        .single()\
        .execute()

    if not task.data:
        raise HTTPException(404, "Task not found or not assigned to you")

    # Insert annotation
    annotation = supabase.table("annotations").insert({
        "task_id": task_id,
        "annotator_id": user["id"],
        "request_id": task.data["request_id"],
        "session_id": task.data["session_id"],
        "message_id": task.data["message_id"],
        **scores.dict(),
        "annotation_time_seconds": annotation_time_seconds
    }).execute()

    # Update task status
    supabase.table("annotation_tasks")\
        .update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        })\
        .eq("id", task_id)\
        .execute()

    # Sync to Langfuse
    await sync_annotation_to_langfuse(annotation.data[0])

    return {"message": "Annotation submitted", "annotation_id": annotation.data[0]["id"]}

@router.get("/rubrics/{rubric_name}")
async def get_rubric(
    rubric_name: str,
    supabase = Depends(get_supabase)
):
    """Get annotation rubric for reference"""
    rubric = supabase.table("annotation_rubrics")\
        .select("*")\
        .eq("name", rubric_name)\
        .eq("is_active", True)\
        .single()\
        .execute()

    if not rubric.data:
        raise HTTPException(404, "Rubric not found")

    return rubric.data
```

### Inter-Annotator Agreement

```python
# backend_agent_api/evals/annotation/agreement.py

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class AgreementMetrics:
    """Inter-annotator agreement metrics"""
    dimension: str
    exact_agreement: float      # % exact matches
    within_one_agreement: float # % within 1 point
    cohens_kappa: float         # Cohen's Kappa
    krippendorffs_alpha: float  # Krippendorff's Alpha
    sample_size: int

class InterAnnotatorAgreement:
    """
    Calculate and track inter-annotator agreement.
    Used to ensure annotation quality and consistency.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def calculate_agreement(
        self,
        dimension: str,
        period_days: int = 30
    ) -> AgreementMetrics:
        """Calculate agreement for a specific dimension"""

        # Get annotations where multiple annotators scored same request
        pairs = self.supabase.table("annotation_agreements")\
            .select("*")\
            .eq("dimension", dimension)\
            .execute()

        if not pairs.data or len(pairs.data) < 10:
            return AgreementMetrics(
                dimension=dimension,
                exact_agreement=0,
                within_one_agreement=0,
                cohens_kappa=0,
                krippendorffs_alpha=0,
                sample_size=0
            )

        scores_1 = [p["annotator_1_score"] for p in pairs.data]
        scores_2 = [p["annotator_2_score"] for p in pairs.data]

        # Exact agreement
        exact = sum(1 for s1, s2 in zip(scores_1, scores_2) if s1 == s2)
        exact_agreement = exact / len(pairs.data)

        # Within 1 point agreement
        within_one = sum(1 for s1, s2 in zip(scores_1, scores_2) if abs(s1 - s2) <= 1)
        within_one_agreement = within_one / len(pairs.data)

        # Cohen's Kappa
        kappa = self._cohens_kappa(scores_1, scores_2)

        # Krippendorff's Alpha
        alpha = self._krippendorffs_alpha(scores_1, scores_2)

        return AgreementMetrics(
            dimension=dimension,
            exact_agreement=exact_agreement,
            within_one_agreement=within_one_agreement,
            cohens_kappa=kappa,
            krippendorffs_alpha=alpha,
            sample_size=len(pairs.data)
        )

    def _cohens_kappa(self, scores_1: List[int], scores_2: List[int]) -> float:
        """Calculate Cohen's Kappa"""
        # Implementation of Cohen's Kappa
        n = len(scores_1)
        if n == 0:
            return 0

        # Observed agreement
        po = sum(1 for s1, s2 in zip(scores_1, scores_2) if s1 == s2) / n

        # Expected agreement by chance
        categories = set(scores_1) | set(scores_2)
        pe = 0
        for cat in categories:
            p1 = sum(1 for s in scores_1 if s == cat) / n
            p2 = sum(1 for s in scores_2 if s == cat) / n
            pe += p1 * p2

        if pe == 1:
            return 1

        return (po - pe) / (1 - pe)

    def _krippendorffs_alpha(self, scores_1: List[int], scores_2: List[int]) -> float:
        """Calculate Krippendorff's Alpha"""
        # Simplified implementation for ordinal data
        # ... full implementation
        return 0.0

    async def flag_low_agreement_annotations(self):
        """Flag annotations where agreement is low for review"""
        dimensions = ["correctness", "completeness", "relevance", "clarity", "helpfulness"]

        for dim in dimensions:
            metrics = await self.calculate_agreement(dim)
            if metrics.within_one_agreement < 0.7:
                # Log or alert
                print(f"Low agreement on {dim}: {metrics.within_one_agreement:.2%}")
```

### Annotation Quality Dashboard

```python
# backend_agent_api/evals/metrics/annotation_metrics.py

@dataclass
class AnnotationMetrics:
    """Dashboard metrics for manual annotation"""
    period: str

    # Volume
    total_tasks_created: int
    tasks_completed: int
    tasks_pending: int
    completion_rate: float

    # Quality
    avg_annotation_time_seconds: int
    inter_annotator_agreement: dict[str, float]

    # Annotator performance
    annotators_active: int
    avg_annotations_per_annotator: float

    # Content insights
    avg_overall_score: float
    score_distribution: dict[int, int]  # Score -> count
    common_weaknesses: List[str]
    safety_failure_rate: float

async def calculate_annotation_metrics(
    supabase,
    period_days: int = 30
) -> AnnotationMetrics:
    """Calculate annotation program metrics"""
    # ... implementation
    pass
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Inter-annotator agreement (within 1) | >80% | <70% |
| Cohen's Kappa | >0.6 | <0.4 |
| Task completion rate | >90% | <75% |
| Avg annotation time | <5 min | >10 min |
| Annotation coverage | >5% of requests | <2% |

## Use Cases for Annotations

1. **LLM Judge Calibration**: Compare judge scores to expert annotations
2. **Golden Dataset Creation**: Export high-confidence annotations as test cases
3. **Training Data**: Use annotated examples for fine-tuning
4. **Compliance Auditing**: Document human review for regulated industries
5. **Failure Analysis**: Deep-dive into low-scoring responses

## Testing

```python
# tests/evals/test_manual_annotation.py

import pytest
from evals.annotation.sampler import AnnotationSampler, SamplingConfig

@pytest.fixture
def sampler():
    config = SamplingConfig(random_sample_rate=0.1)
    return AnnotationSampler(mock_supabase(), config)

async def test_low_confidence_sampling(sampler):
    should_sample, reason = await sampler.should_sample(
        request_id="test-123",
        query="What is Python?",
        response="Python is a programming language.",
        llm_judge_score=0.4  # Below threshold
    )

    assert should_sample is True
    assert reason == "low_confidence"

async def test_safety_keyword_sampling(sampler):
    sampler.config.safety_keywords = ["password", "hack"]

    should_sample, reason = await sampler.should_sample(
        request_id="test-123",
        query="How do I hack into a system?",
        response="I cannot help with that.",
        llm_judge_score=0.9
    )

    assert should_sample is True
    assert reason == "safety_concern"
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ✅ STRONG**

Manual annotation is an excellent fit for Langfuse because:
1. **Built-in annotation UI**: Experts can review traces directly in Langfuse
2. **Score attachment**: Annotations become scores on traces
3. **No custom UI needed**: Use Langfuse's existing interface
4. **Collaboration**: Multiple annotators can work in Langfuse

**Using Langfuse's Annotation UI:**

Instead of building a custom annotation interface, use Langfuse:

1. **Traces appear automatically** (from Pydantic AI instrumentation)
2. **Annotators log into Langfuse** and review traces
3. **Add scores directly** via Langfuse UI
4. **Export for analysis** via Langfuse API

```python
# No custom code needed for basic annotation!
# Langfuse UI handles it.

# For programmatic access to annotations:
from langfuse import Langfuse

langfuse = Langfuse()

def get_annotated_traces(min_score: float = None) -> list:
    """Fetch traces that have been annotated"""
    # Use Langfuse API to query traces with scores
    traces = langfuse.get_traces(
        filter={
            "scores": {"name": "expert_review", "exists": True}
        }
    )
    return traces

def export_annotations_for_training() -> list[dict]:
    """Export annotated examples for LLM judge calibration"""
    annotated = langfuse.get_traces(
        filter={"scores": {"name": "expert_accuracy"}}
    )

    return [
        {
            "query": t.input,
            "response": t.output,
            "expert_score": t.scores["expert_accuracy"],
            "expert_comment": t.scores.get("expert_comment", "")
        }
        for t in annotated
    ]
```

**Langfuse Annotation Workflow:**
1. Configure annotation queue in Langfuse
2. Experts receive notifications for traces to review
3. Review trace details (input, output, tool calls)
4. Add scores and comments via Langfuse UI
5. Scores automatically attached to traces

### Pydantic AI Support

**Fit Level: ⚠️ PARTIAL (via Logfire)**

Pydantic AI doesn't have direct annotation support, but:

1. **Logfire integration**: Pydantic AI traces go to Logfire
2. **Logfire UI**: Similar to Langfuse, has trace review capabilities
3. **No pydantic-evals equivalent**: No `ManualAnnotationEvaluator`

**Using Logfire for Annotation:**

```python
import logfire

# Traces automatically captured
logfire.configure()
logfire.instrument_pydantic_ai()

# Annotators use Logfire UI to review traces
# No code needed - it's a manual UI-based workflow
```

**For Custom Annotation (if not using Langfuse/Logfire):**

```python
from pydantic import BaseModel
from pydantic_evals import Dataset, Case

class AnnotatedCase(BaseModel):
    """Case with expert annotation"""
    query: str
    response: str
    expert_score: float
    expert_reasoning: str

def create_dataset_from_annotations(annotations: list[AnnotatedCase]) -> Dataset:
    """Convert expert annotations to Pydantic Evals dataset"""
    return Dataset(
        cases=[
            Case(
                name=f"annotated_{i}",
                inputs={"query": a.query},
                expected_output=a.response,
                metadata={
                    "expert_score": a.expert_score,
                    "expert_reasoning": a.expert_reasoning
                }
            )
            for i, a in enumerate(annotations)
        ]
    )
```

**Recommendation:** Use Langfuse's annotation UI rather than building custom. It's already integrated with your observability stack.
