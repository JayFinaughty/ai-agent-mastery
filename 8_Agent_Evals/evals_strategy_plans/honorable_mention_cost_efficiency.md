# Strategy 8: Cost/Efficiency Evaluation

## Overview

**What it is**: Tracking and evaluating the economic efficiency of agent operations - token usage, API costs, latency, and tying these to business outcomes. This ensures the agent delivers value proportional to its cost.

**Philosophy**: Quality without cost awareness is unsustainable. A perfect agent that costs $10 per query isn't viable. Cost evaluation ensures you optimize for the best quality at an acceptable cost, and can measure ROI.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COST/EFFICIENCY TRACKING                             │
│                                                                         │
│  Request ──► Agent ──► Response                                        │
│     │          │          │                                            │
│     ▼          ▼          ▼                                            │
│  ┌──────┐  ┌──────┐  ┌──────┐                                         │
│  │Input │  │LLM   │  │Output│                                         │
│  │Tokens│  │Calls │  │Tokens│                                         │
│  │ 150  │  │  2   │  │ 300  │                                         │
│  └──────┘  └──────┘  └──────┘                                         │
│     │          │          │                                            │
│     └──────────┼──────────┘                                            │
│                ▼                                                        │
│  ┌─────────────────────────────────────────┐                           │
│  │           COST CALCULATION               │                           │
│  ├─────────────────────────────────────────┤                           │
│  │  Input:  150 tokens × $2.50/1M = $0.000375│                          │
│  │  Output: 300 tokens × $10.00/1M = $0.003  │                          │
│  │  ─────────────────────────────────────── │                          │
│  │  Total Cost: $0.003375 per request        │                          │
│  │                                           │                          │
│  │  Business Context:                        │                          │
│  │  - Support ticket saved: ~$15            │                          │
│  │  - ROI: 4,444x                           │                          │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Metric | Description | Unit |
|--------|-------------|------|
| **Token Usage** | Input/output tokens per request | Tokens |
| **API Cost** | Direct cost of LLM API calls | USD |
| **Tool Cost** | Cost of external API tools (web search, etc.) | USD |
| **Total Cost** | All costs per request | USD |
| **Cost per Session** | Aggregate across conversation | USD |
| **Latency** | Time to response | ms |
| **Throughput** | Requests processed per time | req/min |
| **Cost per Outcome** | Cost to achieve business goal | USD |
| **ROI** | Value delivered / cost | Ratio |

## When to Use

✅ **Good for:**
- Budget planning and monitoring
- Optimizing model selection
- A/B testing cost-efficiency
- Identifying expensive query patterns
- Business justification
- Preventing runaway costs

❌ **Limitations:**
- Value is hard to quantify
- Costs vary by provider/model
- Caching complicates calculations
- Doesn't capture all value

## Implementation Plan for Dynamous Agent

### Cost Configuration

```python
# backend_agent_api/evals/cost/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelPricing:
    """Pricing for a specific model"""
    model_id: str
    input_cost_per_1m: float      # Cost per 1M input tokens
    output_cost_per_1m: float     # Cost per 1M output tokens
    cached_input_cost_per_1m: Optional[float] = None  # Discounted cached input

# Current OpenAI pricing (as of 2024)
MODEL_PRICING = {
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        cached_input_cost_per_1m=1.25
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        cached_input_cost_per_1m=0.075
    ),
    "gpt-4-turbo": ModelPricing(
        model_id="gpt-4-turbo",
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00
    ),
    "claude-3-5-sonnet": ModelPricing(
        model_id="claude-3-5-sonnet",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00
    ),
    "claude-3-opus": ModelPricing(
        model_id="claude-3-opus",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00
    ),
    # Embedding models
    "text-embedding-3-small": ModelPricing(
        model_id="text-embedding-3-small",
        input_cost_per_1m=0.02,
        output_cost_per_1m=0.0  # Embeddings have no output cost
    ),
}

# External tool costs
TOOL_COSTS = {
    "web_search": 0.005,        # Per search (Brave API)
    "image_analysis": 0.002,    # Approximate per image
}

@dataclass
class CostConfig:
    """Cost tracking configuration"""
    model_id: str = "gpt-4o-mini"
    embedding_model_id: str = "text-embedding-3-small"
    budget_per_request: float = 0.10      # Alert threshold
    budget_per_session: float = 1.00
    budget_daily: float = 100.00
    budget_monthly: float = 2000.00

    def get_model_pricing(self) -> ModelPricing:
        return MODEL_PRICING.get(self.model_id, MODEL_PRICING["gpt-4o-mini"])

    def get_embedding_pricing(self) -> ModelPricing:
        return MODEL_PRICING.get(self.embedding_model_id, MODEL_PRICING["text-embedding-3-small"])
```

### Cost Tracker

```python
# backend_agent_api/evals/cost/tracker.py

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

@dataclass
class RequestCost:
    """Cost breakdown for a single request"""
    request_id: str

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0

    # LLM costs
    llm_input_cost: float = 0.0
    llm_output_cost: float = 0.0
    embedding_cost: float = 0.0

    # Tool costs
    tool_costs: dict[str, float] = field(default_factory=dict)

    # Totals
    total_tokens: int = 0
    total_cost: float = 0.0

    # Timing
    latency_ms: int = 0
    started_at: datetime = None
    ended_at: datetime = None

    # Context
    model_id: str = ""
    query_type: str = ""

    def calculate_totals(self):
        """Recalculate totals"""
        self.total_tokens = self.input_tokens + self.output_tokens + self.embedding_tokens
        self.total_cost = (
            self.llm_input_cost +
            self.llm_output_cost +
            self.embedding_cost +
            sum(self.tool_costs.values())
        )

@dataclass
class SessionCost:
    """Aggregate cost for a session"""
    session_id: str
    user_id: str

    # Aggregates
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # Breakdowns
    request_costs: list[RequestCost] = field(default_factory=list)

    # Timing
    first_request_at: datetime = None
    last_request_at: datetime = None
    total_latency_ms: int = 0

@dataclass
class CostAlert:
    """Alert for cost threshold exceeded"""
    alert_type: str  # 'request', 'session', 'daily', 'monthly'
    threshold: float
    actual: float
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None

class CostTracker:
    """
    Tracks costs in real-time as requests are processed.
    """

    def __init__(self, supabase_client, config: CostConfig = None):
        self.supabase = supabase_client
        self.config = config or CostConfig()
        self.pricing = self.config.get_model_pricing()
        self.embedding_pricing = self.config.get_embedding_pricing()

        # In-memory tracking for current requests
        self._active_requests: dict[str, RequestCost] = {}
        self._session_costs: dict[str, SessionCost] = {}

    def start_request(self, request_id: str, session_id: str, query_type: str = ""):
        """Start tracking a new request"""
        cost = RequestCost(
            request_id=request_id,
            started_at=datetime.utcnow(),
            model_id=self.config.model_id,
            query_type=query_type
        )
        self._active_requests[request_id] = cost
        return cost

    def record_llm_usage(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_input: bool = False
    ):
        """Record LLM token usage"""
        cost = self._active_requests.get(request_id)
        if not cost:
            return

        cost.input_tokens += input_tokens
        cost.output_tokens += output_tokens

        # Calculate costs
        if cached_input:
            input_rate = self.pricing.cached_input_cost_per_1m or self.pricing.input_cost_per_1m
        else:
            input_rate = self.pricing.input_cost_per_1m

        cost.llm_input_cost += (input_tokens / 1_000_000) * input_rate
        cost.llm_output_cost += (output_tokens / 1_000_000) * self.pricing.output_cost_per_1m
        cost.calculate_totals()

    def record_embedding_usage(self, request_id: str, tokens: int):
        """Record embedding token usage"""
        cost = self._active_requests.get(request_id)
        if not cost:
            return

        cost.embedding_tokens += tokens
        cost.embedding_cost += (tokens / 1_000_000) * self.embedding_pricing.input_cost_per_1m
        cost.calculate_totals()

    def record_tool_usage(self, request_id: str, tool_name: str):
        """Record external tool usage"""
        cost = self._active_requests.get(request_id)
        if not cost:
            return

        tool_cost = TOOL_COSTS.get(tool_name, 0)
        cost.tool_costs[tool_name] = cost.tool_costs.get(tool_name, 0) + tool_cost
        cost.calculate_totals()

    def end_request(self, request_id: str, session_id: str) -> tuple[RequestCost, list[CostAlert]]:
        """End tracking and return final cost"""
        cost = self._active_requests.pop(request_id, None)
        if not cost:
            return None, []

        cost.ended_at = datetime.utcnow()
        cost.latency_ms = int((cost.ended_at - cost.started_at).total_seconds() * 1000)
        cost.calculate_totals()

        # Check for alerts
        alerts = self._check_alerts(cost, session_id)

        # Update session cost
        self._update_session_cost(session_id, cost)

        # Store in database
        self._store_cost(cost, session_id)

        return cost, alerts

    def _check_alerts(self, cost: RequestCost, session_id: str) -> list[CostAlert]:
        """Check for cost threshold violations"""
        alerts = []

        # Request-level alert
        if cost.total_cost > self.config.budget_per_request:
            alerts.append(CostAlert(
                alert_type="request",
                threshold=self.config.budget_per_request,
                actual=cost.total_cost,
                request_id=cost.request_id,
                timestamp=datetime.utcnow()
            ))

        # Session-level alert
        session_cost = self._session_costs.get(session_id)
        if session_cost and session_cost.total_cost > self.config.budget_per_session:
            alerts.append(CostAlert(
                alert_type="session",
                threshold=self.config.budget_per_session,
                actual=session_cost.total_cost,
                session_id=session_id,
                timestamp=datetime.utcnow()
            ))

        return alerts

    def _update_session_cost(self, session_id: str, cost: RequestCost):
        """Update session-level aggregates"""
        if session_id not in self._session_costs:
            self._session_costs[session_id] = SessionCost(
                session_id=session_id,
                user_id="",
                first_request_at=cost.started_at
            )

        session = self._session_costs[session_id]
        session.total_requests += 1
        session.total_tokens += cost.total_tokens
        session.total_cost += cost.total_cost
        session.total_latency_ms += cost.latency_ms
        session.last_request_at = cost.ended_at
        session.request_costs.append(cost)

    def _store_cost(self, cost: RequestCost, session_id: str):
        """Store cost in database"""
        self.supabase.table("request_costs").insert({
            "request_id": cost.request_id,
            "session_id": session_id,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "embedding_tokens": cost.embedding_tokens,
            "total_tokens": cost.total_tokens,
            "llm_input_cost": cost.llm_input_cost,
            "llm_output_cost": cost.llm_output_cost,
            "embedding_cost": cost.embedding_cost,
            "tool_costs": cost.tool_costs,
            "total_cost": cost.total_cost,
            "latency_ms": cost.latency_ms,
            "model_id": cost.model_id,
            "query_type": cost.query_type,
            "started_at": cost.started_at.isoformat(),
            "ended_at": cost.ended_at.isoformat() if cost.ended_at else None
        }).execute()

    async def get_daily_cost(self) -> float:
        """Get total cost for today"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = self.supabase.table("request_costs")\
            .select("total_cost")\
            .gte("started_at", today.isoformat())\
            .execute()

        return sum(r["total_cost"] for r in result.data) if result.data else 0

    async def get_monthly_cost(self) -> float:
        """Get total cost for this month"""
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = self.supabase.table("request_costs")\
            .select("total_cost")\
            .gte("started_at", month_start.isoformat())\
            .execute()

        return sum(r["total_cost"] for r in result.data) if result.data else 0
```

### Cost Evaluator

```python
# backend_agent_api/evals/evaluators/cost_evaluator.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class CostEvaluationResult:
    """Result of cost/efficiency evaluation"""

    # Cost metrics
    total_cost: float
    cost_per_token: float
    cost_vs_budget: float          # Ratio to budget

    # Efficiency metrics
    tokens_per_response_char: float  # Higher = less efficient
    latency_per_token: float       # ms per token

    # Scores
    cost_score: float              # 0-1: Lower cost = higher score
    efficiency_score: float        # 0-1: Better efficiency = higher score
    overall_score: float

    # Context
    within_budget: bool
    percentile: Optional[float]    # How this compares to other requests

    # Recommendations
    recommendations: list[str]

class CostEvaluator:
    """
    Evaluates cost efficiency of agent requests.
    """

    def __init__(
        self,
        supabase_client,
        budget_per_request: float = 0.05,
        target_tokens_per_char: float = 0.5,
        target_latency_per_token: float = 10  # ms
    ):
        self.supabase = supabase_client
        self.budget_per_request = budget_per_request
        self.target_tokens_per_char = target_tokens_per_char
        self.target_latency_per_token = target_latency_per_token

    def evaluate(
        self,
        cost: RequestCost,
        response_length: int
    ) -> CostEvaluationResult:
        """Evaluate cost efficiency of a request"""

        recommendations = []

        # Cost metrics
        cost_per_token = cost.total_cost / cost.total_tokens if cost.total_tokens else 0
        cost_vs_budget = cost.total_cost / self.budget_per_request

        # Efficiency metrics
        tokens_per_char = cost.total_tokens / response_length if response_length else float('inf')
        latency_per_token = cost.latency_ms / cost.total_tokens if cost.total_tokens else 0

        # Cost score (lower cost = higher score)
        if cost.total_cost <= self.budget_per_request * 0.5:
            cost_score = 1.0
        elif cost.total_cost <= self.budget_per_request:
            cost_score = 0.8
        elif cost.total_cost <= self.budget_per_request * 1.5:
            cost_score = 0.6
            recommendations.append("Request cost approaching budget limit")
        elif cost.total_cost <= self.budget_per_request * 2:
            cost_score = 0.4
            recommendations.append("Request cost exceeds budget")
        else:
            cost_score = 0.2
            recommendations.append("Request cost significantly over budget - investigate")

        # Efficiency score
        efficiency_issues = []

        # Token efficiency
        if tokens_per_char > self.target_tokens_per_char * 2:
            efficiency_issues.append("high_token_ratio")
            recommendations.append("High token-to-response ratio - consider prompt optimization")
        elif tokens_per_char > self.target_tokens_per_char * 1.5:
            efficiency_issues.append("moderate_token_ratio")

        # Latency efficiency
        if latency_per_token > self.target_latency_per_token * 2:
            efficiency_issues.append("high_latency")
            recommendations.append("High latency per token - check for bottlenecks")
        elif latency_per_token > self.target_latency_per_token * 1.5:
            efficiency_issues.append("moderate_latency")

        efficiency_score = 1.0 - (len(efficiency_issues) * 0.2)
        efficiency_score = max(0.2, efficiency_score)

        # Overall score
        overall_score = (cost_score * 0.6 + efficiency_score * 0.4)

        return CostEvaluationResult(
            total_cost=cost.total_cost,
            cost_per_token=cost_per_token,
            cost_vs_budget=cost_vs_budget,
            tokens_per_response_char=tokens_per_char,
            latency_per_token=latency_per_token,
            cost_score=cost_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            within_budget=cost.total_cost <= self.budget_per_request,
            percentile=None,  # Calculate from historical data
            recommendations=recommendations
        )

    async def evaluate_with_percentile(
        self,
        cost: RequestCost,
        response_length: int
    ) -> CostEvaluationResult:
        """Evaluate with historical percentile comparison"""
        result = self.evaluate(cost, response_length)

        # Get percentile from historical data
        historical = self.supabase.table("request_costs")\
            .select("total_cost")\
            .order("total_cost")\
            .execute()

        if historical.data:
            costs = [r["total_cost"] for r in historical.data]
            below = sum(1 for c in costs if c < cost.total_cost)
            result.percentile = (below / len(costs)) * 100

        return result
```

### Business Value Tracking

```python
# backend_agent_api/evals/cost/business_value.py

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class OutcomeType(str, Enum):
    QUESTION_ANSWERED = "question_answered"
    DOCUMENT_FOUND = "document_found"
    TASK_COMPLETED = "task_completed"
    SUPPORT_DEFLECTED = "support_deflected"
    SALE_ASSISTED = "sale_assisted"

@dataclass
class BusinessOutcome:
    """Business value of an agent interaction"""
    outcome_type: OutcomeType
    estimated_value: float        # USD value of this outcome
    confidence: float             # How confident in this attribution

# Estimated value per outcome type
OUTCOME_VALUES = {
    OutcomeType.QUESTION_ANSWERED: 5.00,      # Saves user time
    OutcomeType.DOCUMENT_FOUND: 10.00,        # Information retrieval value
    OutcomeType.TASK_COMPLETED: 15.00,        # Full task automation
    OutcomeType.SUPPORT_DEFLECTED: 25.00,     # Avoided human support cost
    OutcomeType.SALE_ASSISTED: 50.00,         # Sales assistance value
}

@dataclass
class ROICalculation:
    """ROI calculation for agent usage"""
    period: str
    total_cost: float
    total_value: float
    roi_ratio: float
    cost_per_outcome: dict[str, float]
    value_per_outcome: dict[str, float]

class BusinessValueTracker:
    """
    Tracks business value and calculates ROI.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def record_outcome(
        self,
        request_id: str,
        outcome_type: OutcomeType,
        custom_value: Optional[float] = None,
        confidence: float = 0.8
    ):
        """Record a business outcome"""
        value = custom_value or OUTCOME_VALUES.get(outcome_type, 0)

        self.supabase.table("business_outcomes").insert({
            "request_id": request_id,
            "outcome_type": outcome_type.value,
            "estimated_value": value,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

    async def calculate_roi(self, days: int = 30) -> ROICalculation:
        """Calculate ROI for a period"""
        from datetime import datetime, timedelta

        start = datetime.utcnow() - timedelta(days=days)

        # Get costs
        costs = self.supabase.table("request_costs")\
            .select("total_cost, query_type")\
            .gte("started_at", start.isoformat())\
            .execute()

        total_cost = sum(r["total_cost"] for r in costs.data) if costs.data else 0

        # Get outcomes
        outcomes = self.supabase.table("business_outcomes")\
            .select("outcome_type, estimated_value, confidence")\
            .gte("created_at", start.isoformat())\
            .execute()

        total_value = sum(
            r["estimated_value"] * r["confidence"]
            for r in outcomes.data
        ) if outcomes.data else 0

        # Calculate ROI
        roi_ratio = total_value / total_cost if total_cost > 0 else 0

        # Breakdown by outcome type
        value_by_outcome = {}
        for outcome in (outcomes.data or []):
            otype = outcome["outcome_type"]
            value_by_outcome[otype] = value_by_outcome.get(otype, 0) + outcome["estimated_value"]

        return ROICalculation(
            period=f"Last {days} days",
            total_cost=total_cost,
            total_value=total_value,
            roi_ratio=roi_ratio,
            cost_per_outcome={},  # Calculate from data
            value_per_outcome=value_by_outcome
        )

    async def infer_outcome_from_behavior(
        self,
        request_id: str,
        session_id: str,
        query: str,
        response: str,
        user_feedback: Optional[int] = None,
        follow_up_count: int = 0
    ) -> Optional[OutcomeType]:
        """Infer business outcome from behavior signals"""

        # Positive signals indicate task completion
        if user_feedback == 1 and follow_up_count == 0:
            # Positive feedback, no follow-ups = task likely completed

            # Determine outcome type from query
            query_lower = query.lower()

            if any(word in query_lower for word in ["document", "file", "find", "search"]):
                return OutcomeType.DOCUMENT_FOUND

            if any(word in query_lower for word in ["how", "help", "support", "issue"]):
                return OutcomeType.SUPPORT_DEFLECTED

            return OutcomeType.QUESTION_ANSWERED

        return None
```

### Database Schema

```sql
-- Request cost tracking
CREATE TABLE request_costs (
    id SERIAL PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR,

    -- Token counts
    input_tokens INTEGER,
    output_tokens INTEGER,
    embedding_tokens INTEGER,
    total_tokens INTEGER,

    -- Costs (USD)
    llm_input_cost DECIMAL(10, 6),
    llm_output_cost DECIMAL(10, 6),
    embedding_cost DECIMAL(10, 6),
    tool_costs JSONB,
    total_cost DECIMAL(10, 6),

    -- Performance
    latency_ms INTEGER,

    -- Context
    model_id VARCHAR,
    query_type VARCHAR,

    -- Timing
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Business outcomes
CREATE TABLE business_outcomes (
    id SERIAL PRIMARY KEY,
    request_id UUID REFERENCES requests(id),
    outcome_type VARCHAR NOT NULL,
    estimated_value DECIMAL(10, 2),
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily cost summary (for dashboards)
CREATE TABLE daily_cost_summary (
    date DATE PRIMARY KEY,
    total_requests INTEGER,
    total_tokens INTEGER,
    total_cost DECIMAL(10, 2),
    avg_cost_per_request DECIMAL(10, 6),
    avg_tokens_per_request INTEGER,
    peak_hourly_cost DECIMAL(10, 2),
    model_breakdown JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_request_costs_session ON request_costs(session_id);
CREATE INDEX idx_request_costs_date ON request_costs(started_at);
CREATE INDEX idx_request_costs_total ON request_costs(total_cost);
CREATE INDEX idx_business_outcomes_type ON business_outcomes(outcome_type);
```

### Metrics Dashboard

```python
# backend_agent_api/evals/metrics/cost_metrics.py

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CostMetrics:
    """Dashboard metrics for cost tracking"""
    period: str

    # Volume
    total_requests: int
    total_tokens: int

    # Costs
    total_cost: float
    avg_cost_per_request: float
    max_cost_request: float
    cost_by_model: dict[str, float]
    cost_by_query_type: dict[str, float]

    # Efficiency
    avg_tokens_per_request: int
    avg_latency_ms: int
    cost_efficiency_score: float

    # Budget
    daily_spend: float
    monthly_spend: float
    budget_utilization: float

    # Trends
    cost_trend_7d: float      # % change
    efficiency_trend_7d: float

    # Alerts
    over_budget_requests: int
    anomalies_detected: int

async def calculate_cost_metrics(
    supabase,
    period_days: int = 7,
    config: CostConfig = None
) -> CostMetrics:
    """Calculate cost metrics for dashboard"""
    config = config or CostConfig()
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)

    # Get cost data
    costs = supabase.table("request_costs")\
        .select("*")\
        .gte("started_at", start.isoformat())\
        .execute()

    if not costs.data:
        return CostMetrics(
            period=f"Last {period_days} days",
            total_requests=0,
            total_tokens=0,
            total_cost=0,
            avg_cost_per_request=0,
            max_cost_request=0,
            cost_by_model={},
            cost_by_query_type={},
            avg_tokens_per_request=0,
            avg_latency_ms=0,
            cost_efficiency_score=1.0,
            daily_spend=0,
            monthly_spend=0,
            budget_utilization=0,
            cost_trend_7d=0,
            efficiency_trend_7d=0,
            over_budget_requests=0,
            anomalies_detected=0
        )

    data = costs.data
    total_cost = sum(r["total_cost"] or 0 for r in data)
    total_tokens = sum(r["total_tokens"] or 0 for r in data)

    # Calculate metrics
    return CostMetrics(
        period=f"Last {period_days} days",
        total_requests=len(data),
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_cost_per_request=total_cost / len(data) if data else 0,
        max_cost_request=max(r["total_cost"] or 0 for r in data),
        cost_by_model={},  # Aggregate
        cost_by_query_type={},
        avg_tokens_per_request=total_tokens // len(data) if data else 0,
        avg_latency_ms=sum(r["latency_ms"] or 0 for r in data) // len(data) if data else 0,
        cost_efficiency_score=0.8,  # Calculate
        daily_spend=total_cost / period_days,
        monthly_spend=total_cost / period_days * 30,
        budget_utilization=(total_cost / period_days * 30) / config.budget_monthly if config.budget_monthly else 0,
        cost_trend_7d=0,
        efficiency_trend_7d=0,
        over_budget_requests=sum(1 for r in data if (r["total_cost"] or 0) > config.budget_per_request),
        anomalies_detected=0
    )
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Avg cost per request | <$0.02 | >$0.05 |
| Budget utilization | <80% | >95% |
| ROI ratio | >10x | <5x |
| Over-budget requests | <5% | >15% |
| Cost efficiency score | >0.7 | <0.5 |

## Testing

```python
# tests/evals/test_cost_tracker.py

import pytest
from evals.cost.tracker import CostTracker, RequestCost
from evals.cost.config import CostConfig

@pytest.fixture
def tracker():
    return CostTracker(mock_supabase(), CostConfig())

def test_cost_calculation(tracker):
    request_id = "test-123"

    tracker.start_request(request_id, "session-1", "general")
    tracker.record_llm_usage(request_id, input_tokens=100, output_tokens=200)

    cost, alerts = tracker.end_request(request_id, "session-1")

    # GPT-4o-mini pricing: $0.15/1M in, $0.60/1M out
    expected_input_cost = (100 / 1_000_000) * 0.15
    expected_output_cost = (200 / 1_000_000) * 0.60

    assert cost.llm_input_cost == pytest.approx(expected_input_cost)
    assert cost.llm_output_cost == pytest.approx(expected_output_cost)

def test_budget_alert(tracker):
    tracker.config.budget_per_request = 0.001  # Very low budget

    request_id = "test-456"
    tracker.start_request(request_id, "session-1", "general")
    tracker.record_llm_usage(request_id, input_tokens=10000, output_tokens=20000)  # High usage

    cost, alerts = tracker.end_request(request_id, "session-1")

    assert len(alerts) > 0
    assert alerts[0].alert_type == "request"
```
