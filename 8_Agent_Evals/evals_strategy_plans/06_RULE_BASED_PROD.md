# Strategy: Rule-Based Evaluation (Production)

> **Video 7** | **Tag:** `module-8-06-rule-based-prod` | **Phase:** Production

## Overview

**What it is**: Real-time response gating with rule-based checks, plus syncing results to Langfuse for analytics.

**Philosophy**: In production, safety rules must run on every request. Block unsafe responses before they reach users. Track violations for monitoring.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RULE-BASED EVALUATION (PRODUCTION)                  │
│                                                                         │
│   User Request                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  Agent  │─────►│  Rule Gate  │─────►│  Response   │                │
│   │  Output │      │  (<10ms)    │      │  to User    │                │
│   └─────────┘      └──────┬──────┘      └─────────────┘                │
│                           │                                             │
│                           │ Async                                       │
│                           ▼                                             │
│                    ┌─────────────┐                                      │
│                    │  Langfuse   │                                      │
│                    │  Score Sync │                                      │
│                    └─────────────┘                                      │
│                                                                         │
│   Block: PII, SQL injection, credit cards                              │
│   Flag: Profanity, length limits, encoding                             │
│   Log: All violations for monitoring                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Difference from Local

| Aspect | Local (Video 3) | Production (Video 7) |
|--------|-----------------|----------------------|
| **When** | Offline, batch | Real-time, every request |
| **Purpose** | Test cases | Response gating |
| **Blocking** | N/A | Yes - prevent unsafe responses |
| **Langfuse** | Not used | Scores synced for analytics |
| **Speed** | Not critical | Must be <10ms |

---

## Implementation

### Step 1: Rule Engine

```python
# backend_agent_api/evals/rule_engine.py

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

class Severity(Enum):
    CRITICAL = "critical"  # Block response
    WARNING = "warning"    # Flag but allow
    INFO = "info"          # Log only

class Action(Enum):
    BLOCK = "block"
    FLAG = "flag"
    LOG = "log"

@dataclass
class Violation:
    """A rule violation"""
    rule_name: str
    severity: Severity
    action: Action
    message: str
    details: dict = field(default_factory=dict)

@dataclass
class RuleResult:
    """Result of rule evaluation"""
    passed: bool
    blocked: bool
    violations: list[Violation]
    score: float  # 1.0 = clean, 0.0 = blocked

@dataclass
class Rule:
    """A single rule definition"""
    name: str
    check: Callable[[str, dict], Optional[Violation]]
    enabled: bool = True


class RuleEngine:
    """Fast rule engine for production use."""

    def __init__(self):
        self.rules: list[Rule] = []
        self._register_default_rules()

    def evaluate(self, response: str, context: dict = None) -> RuleResult:
        """Evaluate response against all rules. Must be <10ms."""
        context = context or {}
        violations = []

        for rule in self.rules:
            if not rule.enabled:
                continue
            violation = rule.check(response, context)
            if violation:
                violations.append(violation)

        blocked = any(v.action == Action.BLOCK for v in violations)
        score = 0.0 if blocked else (1.0 - len(violations) * 0.1)

        return RuleResult(
            passed=len(violations) == 0,
            blocked=blocked,
            violations=violations,
            score=max(0.0, score)
        )

    def _register_default_rules(self):
        """Register production safety rules."""

        # PII: Credit Card (BLOCK)
        self.rules.append(Rule(
            name="pii_credit_card",
            check=self._check_credit_card
        ))

        # PII: SSN (BLOCK)
        self.rules.append(Rule(
            name="pii_ssn",
            check=self._check_ssn
        ))

        # PII: Phone (FLAG)
        self.rules.append(Rule(
            name="pii_phone",
            check=self._check_phone
        ))

        # PII: Email (FLAG)
        self.rules.append(Rule(
            name="pii_email",
            check=self._check_email
        ))

        # Empty Response (FLAG)
        self.rules.append(Rule(
            name="empty_response",
            check=self._check_empty
        ))

    # === Rule Check Functions ===

    def _check_credit_card(self, response: str, ctx: dict) -> Optional[Violation]:
        """Detect credit card numbers using Luhn algorithm."""
        pattern = r'\b(?:\d[-\s]?){13,19}\b'
        for match in re.findall(pattern, response):
            digits = re.sub(r'\D', '', match)
            if self._luhn_check(digits):
                return Violation(
                    rule_name="pii_credit_card",
                    severity=Severity.CRITICAL,
                    action=Action.BLOCK,
                    message="Credit card number detected",
                    details={"pattern": "****" + digits[-4:]}
                )
        return None

    def _luhn_check(self, num: str) -> bool:
        """Validate using Luhn algorithm."""
        if not num.isdigit() or len(num) < 13:
            return False
        total = sum(
            (d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 else d
            for i, d in enumerate(int(x) for x in reversed(num))
        )
        return total % 10 == 0

    def _check_ssn(self, response: str, ctx: dict) -> Optional[Violation]:
        """Detect SSN patterns."""
        pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        for match in re.findall(pattern, response):
            digits = re.sub(r'\D', '', match)
            # Valid SSN ranges
            if not (digits.startswith('000') or digits.startswith('666') or
                    int(digits[:3]) >= 900):
                return Violation(
                    rule_name="pii_ssn",
                    severity=Severity.CRITICAL,
                    action=Action.BLOCK,
                    message="Potential SSN detected",
                    details={}
                )
        return None

    def _check_phone(self, response: str, ctx: dict) -> Optional[Violation]:
        """Detect phone numbers."""
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\+\d{1,3}[-.]?\d{3,4}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
        ]
        for pattern in patterns:
            if re.search(pattern, response):
                return Violation(
                    rule_name="pii_phone",
                    severity=Severity.WARNING,
                    action=Action.FLAG,
                    message="Phone number detected",
                    details={}
                )
        return None

    def _check_email(self, response: str, ctx: dict) -> Optional[Violation]:
        """Detect email addresses (excluding examples)."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = [m for m in re.findall(pattern, response)
                   if not m.endswith('@example.com')]
        if matches:
            return Violation(
                rule_name="pii_email",
                severity=Severity.WARNING,
                action=Action.FLAG,
                message=f"Email address detected: {len(matches)} found",
                details={}
            )
        return None

    def _check_empty(self, response: str, ctx: dict) -> Optional[Violation]:
        """Detect empty responses."""
        if not response or not response.strip():
            return Violation(
                rule_name="empty_response",
                severity=Severity.WARNING,
                action=Action.FLAG,
                message="Response is empty",
                details={}
            )
        return None
```

### Step 2: Response Gate

```python
# backend_agent_api/evals/response_gate.py

from typing import Tuple, Optional
from rule_engine import RuleEngine, RuleResult

class ResponseGate:
    """
    Pre-response safety gate.
    Blocks unsafe responses before they reach users.
    """

    def __init__(self):
        self.engine = RuleEngine()

    def check(self, response: str, context: dict = None) -> Tuple[bool, Optional[str], RuleResult]:
        """
        Check response against safety rules.

        Returns:
            (should_send, replacement_message, result)
        """
        result = self.engine.evaluate(response, context)

        if result.blocked:
            replacement = self._safe_replacement(result)
            return False, replacement, result

        return True, None, result

    def _safe_replacement(self, result: RuleResult) -> str:
        """Generate safe replacement for blocked response."""
        for v in result.violations:
            if v.rule_name.startswith("pii_"):
                return (
                    "I apologize, but I cannot share that information "
                    "as it may contain sensitive data. How else can I help?"
                )
        return "I apologize, but I cannot provide that response."


# Singleton for use in API
response_gate = ResponseGate()
```

### Step 3: Langfuse Sync

```python
# backend_agent_api/evals/langfuse_sync.py

from langfuse import Langfuse
from rule_engine import RuleResult

langfuse = Langfuse()

async def sync_rule_scores(trace_id: str, result: RuleResult):
    """
    Sync rule evaluation results to Langfuse.
    Run async - don't block response.
    """

    # Overall rule score
    langfuse.score(
        trace_id=trace_id,
        name="rule_safety_score",
        value=result.score,
        comment=f"Violations: {len(result.violations)}"
    )

    # Pass/fail boolean
    langfuse.score(
        trace_id=trace_id,
        name="rule_passed",
        value=1.0 if result.passed else 0.0,
        data_type="BOOLEAN"
    )

    # Blocked status
    if result.blocked:
        langfuse.score(
            trace_id=trace_id,
            name="rule_blocked",
            value=1.0,
            comment="Response blocked by safety rules"
        )

    # Individual violations
    for v in result.violations:
        langfuse.score(
            trace_id=trace_id,
            name=f"rule_violation_{v.rule_name}",
            value=0.0,
            comment=v.message
        )
```

### Step 4: Integration with Agent API

```python
# backend_agent_api/agent_api.py (additions)

import asyncio
from evals.response_gate import response_gate
from evals.langfuse_sync import sync_rule_scores

@app.post("/api/pydantic-agent")
async def pydantic_agent_endpoint(request: AgentRequest):
    # ... existing agent code generates response ...

    # Check rules before sending (BLOCKING - must be fast)
    should_send, replacement, rule_result = response_gate.check(
        response=full_response,
        context={
            "query": request.query,
            "user_id": request.user_id
        }
    )

    # Sync to Langfuse (ASYNC - don't wait)
    if trace_id:
        asyncio.create_task(sync_rule_scores(trace_id, rule_result))

    # Return response or safe replacement
    if not should_send:
        yield {"text": replacement, "blocked": True}
        return

    yield {"text": full_response, "complete": True}
```

---

## Langfuse Dashboard

After integrating, you'll see in Langfuse:

### Scores on Traces
- `rule_safety_score`: 0.0-1.0 overall safety
- `rule_passed`: Boolean pass/fail
- `rule_blocked`: 1.0 if response was blocked
- `rule_violation_*`: Individual rule violations

### Analytics You Can Build
- **Block rate over time**: Are safety issues increasing?
- **Common violations**: What rules trigger most?
- **User correlation**: Do certain users trigger more flags?

---

## Production Configuration

### Environment Variables

```bash
# .env
RULE_BLOCKING_ENABLED=true       # Enable blocking in production
RULE_SYNC_TO_LANGFUSE=true       # Sync scores to Langfuse
RULE_LOG_VIOLATIONS=true          # Log all violations
```

### Enabling/Disabling Rules

```python
# Disable a rule
engine.rules["pii_phone"].enabled = False

# Add custom rule
engine.rules.append(Rule(
    name="custom_check",
    check=my_custom_check_function
))
```

---

## Testing

```python
# tests/test_rule_engine.py

import pytest
from evals.rule_engine import RuleEngine

@pytest.fixture
def engine():
    return RuleEngine()

def test_clean_response_passes(engine):
    result = engine.evaluate("Hello! How can I help you today?")
    assert result.passed
    assert not result.blocked
    assert result.score == 1.0

def test_credit_card_blocked(engine):
    result = engine.evaluate("Your card: 4111-1111-1111-1111")
    assert result.blocked
    assert any(v.rule_name == "pii_credit_card" for v in result.violations)

def test_ssn_blocked(engine):
    result = engine.evaluate("SSN: 123-45-6789")
    assert result.blocked

def test_phone_flagged_not_blocked(engine):
    result = engine.evaluate("Call me at 555-123-4567")
    assert not result.blocked  # Flagged, not blocked
    assert not result.passed   # But didn't pass clean
    assert any(v.rule_name == "pii_phone" for v in result.violations)

def test_example_email_allowed(engine):
    result = engine.evaluate("Use format: user@example.com")
    assert result.passed  # example.com is allowed
```

---

## What's Different from Local

| Local (Video 3) | Production (Video 7) |
|-----------------|----------------------|
| `pydantic-evals` evaluators | Custom `RuleEngine` class |
| Batch evaluation | Real-time per-request |
| Test cases | Live responses |
| Terminal output | Langfuse scores |
| No blocking | Response blocking |

The local version uses pydantic-evals for testing. The production version uses a custom engine optimized for speed and real-time blocking.

---

## Resources

- [Langfuse Scores API](https://langfuse.com/docs/scores)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
