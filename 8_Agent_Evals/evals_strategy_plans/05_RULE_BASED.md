# Strategy 5: Rule-Based Evaluation (Deterministic)

## Overview

**What it is**: Deterministic checks using predefined rules, patterns, and validation logic. These evaluations produce the same result every time for the same input - no randomness, no LLM judgment.

**Philosophy**: Some quality criteria can and should be checked deterministically. Safety rules, format requirements, and compliance checks don't need AI judgment - they need reliable, fast, consistent enforcement.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RULE-BASED EVALUATION                               │
│                                                                         │
│  Response ──► Rule Engine ──► Pass/Fail ──► Block or Flag              │
│                   │                                                     │
│                   ▼                                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        RULE CATEGORIES                          │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │  🛡️ SAFETY          │  📋 FORMAT           │  ✅ VALIDATION    │    │
│  │  ───────────        │  ──────────          │  ────────────     │    │
│  │  • PII Detection    │  • Length Limits     │  • JSON Schema    │    │
│  │  • Toxicity Check   │  • Structure Check   │  • Type Check     │    │
│  │  • Jailbreak Detect │  • Encoding Valid    │  • Pydantic       │    │
│  │  • SQL Injection    │  • No Empty Response │  • Required Fields│    │
│  │  • Code Safety      │  • Language Check    │  • Constraints    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What It Measures

| Rule Type | Examples | Action |
|-----------|----------|--------|
| **Safety Gates** | PII exposure, toxic content | Block response |
| **Format Rules** | Length, structure, encoding | Warn or fix |
| **Validation** | Schema compliance, types | Reject or retry |
| **Compliance** | Required disclaimers, audit | Flag for review |
| **Quality Floors** | Non-empty, relevant tool use | Warn |

## When to Use

✅ **Good for:**
- 100% coverage (every request)
- Real-time blocking (pre-response)
- Compliance requirements
- Reproducible results
- Zero additional cost
- Fast execution (<10ms)

❌ **Limitations:**
- Can't assess nuanced quality
- Pattern-based (can be gamed)
- Requires maintenance as patterns evolve
- Binary results (no gradations)

## Implementation Plan for Dynamous Agent

### Core Rule Engine

```python
# backend_agent_api/evals/evaluators/rule_based.py

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum
import re

class RuleSeverity(Enum):
    CRITICAL = "critical"   # Block response
    WARNING = "warning"     # Flag but allow
    INFO = "info"           # Log only

class RuleAction(Enum):
    BLOCK = "block"         # Prevent response from being sent
    FLAG = "flag"           # Allow but flag for review
    LOG = "log"             # Just log the violation
    FIX = "fix"             # Attempt to fix automatically

@dataclass
class RuleViolation:
    """A single rule violation"""
    rule_name: str
    rule_category: str
    severity: RuleSeverity
    action: RuleAction
    message: str
    details: dict = field(default_factory=dict)
    location: Optional[str] = None  # Where in response

@dataclass
class RuleEvaluationResult:
    """Result of rule-based evaluation"""
    passed: bool
    violations: list[RuleViolation]
    blocked: bool
    score: float  # 1.0 if no violations, decreases with violations

    @property
    def critical_violations(self) -> list[RuleViolation]:
        return [v for v in self.violations if v.severity == RuleSeverity.CRITICAL]

    @property
    def should_block(self) -> bool:
        return any(v.action == RuleAction.BLOCK for v in self.violations)

@dataclass
class Rule:
    """Definition of a single rule"""
    name: str
    category: str
    description: str
    check_fn: Callable[[str, dict], Optional[RuleViolation]]
    severity: RuleSeverity = RuleSeverity.WARNING
    action: RuleAction = RuleAction.FLAG
    enabled: bool = True

class RuleBasedEvaluator:
    """
    Deterministic rule-based evaluator.
    Runs all enabled rules against responses.
    """

    def __init__(self):
        self.rules: list[Rule] = []
        self._register_default_rules()

    def register_rule(self, rule: Rule):
        """Register a new rule"""
        self.rules.append(rule)

    def evaluate(
        self,
        response: str,
        context: dict = None
    ) -> RuleEvaluationResult:
        """
        Evaluate response against all rules.

        Args:
            response: The agent's response text
            context: Additional context (query, tools used, etc.)

        Returns:
            RuleEvaluationResult with all violations
        """
        context = context or {}
        violations = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            violation = rule.check_fn(response, context)
            if violation:
                violations.append(violation)

        # Calculate score
        if not violations:
            score = 1.0
        else:
            # Deduct based on severity
            deductions = {
                RuleSeverity.CRITICAL: 0.5,
                RuleSeverity.WARNING: 0.1,
                RuleSeverity.INFO: 0.02
            }
            total_deduction = sum(deductions[v.severity] for v in violations)
            score = max(0.0, 1.0 - total_deduction)

        blocked = any(v.action == RuleAction.BLOCK for v in violations)

        return RuleEvaluationResult(
            passed=len(violations) == 0,
            violations=violations,
            blocked=blocked,
            score=score
        )

    def _register_default_rules(self):
        """Register default safety and quality rules"""

        # === SAFETY RULES ===

        # PII Detection
        self.register_rule(Rule(
            name="pii_email",
            category="safety",
            description="Detects exposed email addresses in response",
            check_fn=self._check_pii_email,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.FLAG
        ))

        self.register_rule(Rule(
            name="pii_phone",
            category="safety",
            description="Detects phone numbers in response",
            check_fn=self._check_pii_phone,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.FLAG
        ))

        self.register_rule(Rule(
            name="pii_ssn",
            category="safety",
            description="Detects SSN patterns in response",
            check_fn=self._check_pii_ssn,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.BLOCK
        ))

        self.register_rule(Rule(
            name="pii_credit_card",
            category="safety",
            description="Detects credit card numbers in response",
            check_fn=self._check_pii_credit_card,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.BLOCK
        ))

        # Toxicity
        self.register_rule(Rule(
            name="profanity",
            category="safety",
            description="Detects profane language",
            check_fn=self._check_profanity,
            severity=RuleSeverity.WARNING,
            action=RuleAction.FLAG
        ))

        # SQL Injection (for execute_sql tool)
        self.register_rule(Rule(
            name="sql_injection",
            category="safety",
            description="Detects SQL injection patterns",
            check_fn=self._check_sql_injection,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.BLOCK
        ))

        # === FORMAT RULES ===

        self.register_rule(Rule(
            name="empty_response",
            category="format",
            description="Detects empty or whitespace-only responses",
            check_fn=self._check_empty_response,
            severity=RuleSeverity.CRITICAL,
            action=RuleAction.FLAG
        ))

        self.register_rule(Rule(
            name="max_length",
            category="format",
            description="Response exceeds maximum length",
            check_fn=self._check_max_length,
            severity=RuleSeverity.WARNING,
            action=RuleAction.LOG
        ))

        self.register_rule(Rule(
            name="encoding",
            category="format",
            description="Detects encoding issues",
            check_fn=self._check_encoding,
            severity=RuleSeverity.WARNING,
            action=RuleAction.FIX
        ))

        # === QUALITY RULES ===

        self.register_rule(Rule(
            name="repetition",
            category="quality",
            description="Detects excessive repetition",
            check_fn=self._check_repetition,
            severity=RuleSeverity.WARNING,
            action=RuleAction.FLAG
        ))

        self.register_rule(Rule(
            name="incomplete_sentence",
            category="quality",
            description="Detects incomplete sentences at end",
            check_fn=self._check_incomplete,
            severity=RuleSeverity.INFO,
            action=RuleAction.LOG
        ))

    # === RULE CHECK FUNCTIONS ===

    def _check_pii_email(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, response)

        # Filter out generic examples
        real_emails = [m for m in matches if not m.endswith('@example.com')]

        if real_emails:
            return RuleViolation(
                rule_name="pii_email",
                rule_category="safety",
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.FLAG,
                message=f"Found {len(real_emails)} email address(es) in response",
                details={"emails_found": real_emails}
            )
        return None

    def _check_pii_phone(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for phone numbers"""
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\+\d{1,3}[-.]?\d{3,4}[-.]?\d{4}\b',  # International
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',  # (XXX) XXX-XXXX
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return RuleViolation(
                    rule_name="pii_phone",
                    rule_category="safety",
                    severity=RuleSeverity.CRITICAL,
                    action=RuleAction.FLAG,
                    message=f"Found {len(matches)} phone number(s) in response",
                    details={"phones_found": matches}
                )
        return None

    def _check_pii_ssn(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for SSN patterns"""
        pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        matches = re.findall(pattern, response)

        # Filter obvious non-SSNs (dates, etc.)
        potential_ssns = []
        for m in matches:
            digits = re.sub(r'\D', '', m)
            # SSNs don't start with 000, 666, or 900-999
            if not (digits.startswith('000') or digits.startswith('666') or
                    int(digits[:3]) >= 900):
                potential_ssns.append(m)

        if potential_ssns:
            return RuleViolation(
                rule_name="pii_ssn",
                rule_category="safety",
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.BLOCK,
                message="Potential SSN detected in response",
                details={"patterns_found": len(potential_ssns)}
            )
        return None

    def _check_pii_credit_card(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for credit card numbers using Luhn algorithm"""
        # Find potential card numbers (13-19 digits)
        pattern = r'\b(?:\d[-\s]?){13,19}\b'
        potential_cards = re.findall(pattern, response)

        valid_cards = []
        for card in potential_cards:
            digits = re.sub(r'\D', '', card)
            if self._luhn_check(digits):
                valid_cards.append(card)

        if valid_cards:
            return RuleViolation(
                rule_name="pii_credit_card",
                rule_category="safety",
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.BLOCK,
                message="Credit card number detected in response",
                details={"count": len(valid_cards)}
            )
        return None

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        if not card_number.isdigit() or len(card_number) < 13:
            return False

        total = 0
        for i, digit in enumerate(reversed(card_number)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    def _check_profanity(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for profane language"""
        # Basic profanity list (expand as needed)
        profanity_patterns = [
            r'\bf+u+c+k+\b',
            r'\bs+h+i+t+\b',
            r'\ba+s+s+h+o+l+e+\b',
            r'\bb+i+t+c+h+\b',
            r'\bd+a+m+n+\b',
        ]

        response_lower = response.lower()
        for pattern in profanity_patterns:
            if re.search(pattern, response_lower):
                return RuleViolation(
                    rule_name="profanity",
                    rule_category="safety",
                    severity=RuleSeverity.WARNING,
                    action=RuleAction.FLAG,
                    message="Profane language detected in response"
                )
        return None

    def _check_sql_injection(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for SQL injection patterns in SQL tool usage"""
        if context.get("tool_name") != "execute_sql_query":
            return None

        sql_query = context.get("tool_input", "")

        # Dangerous patterns
        dangerous_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)',
            r'--\s*$',  # SQL comment at end
            r'/\*.*\*/',  # Block comments
            r'UNION\s+SELECT',
            r'OR\s+1\s*=\s*1',
            r"OR\s+'[^']*'\s*=\s*'[^']*'",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return RuleViolation(
                    rule_name="sql_injection",
                    rule_category="safety",
                    severity=RuleSeverity.CRITICAL,
                    action=RuleAction.BLOCK,
                    message="Potential SQL injection detected",
                    details={"pattern": pattern}
                )
        return None

    def _check_empty_response(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for empty responses"""
        if not response or not response.strip():
            return RuleViolation(
                rule_name="empty_response",
                rule_category="format",
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.FLAG,
                message="Response is empty or whitespace only"
            )
        return None

    def _check_max_length(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check response length"""
        max_length = 10000  # Characters
        if len(response) > max_length:
            return RuleViolation(
                rule_name="max_length",
                rule_category="format",
                severity=RuleSeverity.WARNING,
                action=RuleAction.LOG,
                message=f"Response exceeds maximum length ({len(response)} > {max_length})",
                details={"length": len(response), "max": max_length}
            )
        return None

    def _check_encoding(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for encoding issues"""
        # Check for replacement characters
        if '\ufffd' in response:
            return RuleViolation(
                rule_name="encoding",
                rule_category="format",
                severity=RuleSeverity.WARNING,
                action=RuleAction.FIX,
                message="Response contains encoding errors"
            )
        return None

    def _check_repetition(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for excessive repetition"""
        words = response.lower().split()
        if len(words) < 10:
            return None

        # Check for repeated phrases
        for n in [3, 4, 5]:  # Check 3, 4, 5-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = {}
            for ng in ngrams:
                ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

            max_repeat = max(ngram_counts.values()) if ngram_counts else 0
            if max_repeat > 3:
                return RuleViolation(
                    rule_name="repetition",
                    rule_category="quality",
                    severity=RuleSeverity.WARNING,
                    action=RuleAction.FLAG,
                    message=f"Excessive repetition detected ({n}-gram repeated {max_repeat} times)"
                )
        return None

    def _check_incomplete(self, response: str, context: dict) -> Optional[RuleViolation]:
        """Check for incomplete sentences"""
        response = response.strip()
        if not response:
            return None

        # Check if ends with sentence-ending punctuation
        if not response[-1] in '.!?:"\')':
            # Check if it's a list item or code
            lines = response.split('\n')
            last_line = lines[-1].strip()
            if not (last_line.startswith('-') or last_line.startswith('*') or
                    last_line.startswith('```') or last_line.endswith('```')):
                return RuleViolation(
                    rule_name="incomplete_sentence",
                    rule_category="quality",
                    severity=RuleSeverity.INFO,
                    action=RuleAction.LOG,
                    message="Response may be incomplete (no ending punctuation)"
                )
        return None
```

### Database Schema

```sql
-- Rule evaluation results
CREATE TABLE rule_evaluation_results (
    id SERIAL PRIMARY KEY,

    -- What was evaluated
    request_id UUID REFERENCES requests(id),
    session_id VARCHAR,
    message_id INTEGER,

    -- Results
    passed BOOLEAN NOT NULL,
    blocked BOOLEAN DEFAULT FALSE,
    score FLOAT NOT NULL,
    violations_count INTEGER DEFAULT 0,

    -- Violations detail
    violations JSONB,  -- Array of violation objects

    -- Timing
    evaluation_time_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_rule_eval_request ON rule_evaluation_results(request_id);
CREATE INDEX idx_rule_eval_blocked ON rule_evaluation_results(blocked);
CREATE INDEX idx_rule_eval_passed ON rule_evaluation_results(passed);
```

### Integration with Agent API (Pre-Response)

```python
# backend_agent_api/evals/rule_integration.py

from typing import Optional, Tuple

class RuleGate:
    """
    Pre-response rule gate.
    Evaluates responses BEFORE sending to user.
    Can block unsafe responses.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.evaluator = RuleBasedEvaluator()

    async def check_response(
        self,
        response: str,
        context: dict = None
    ) -> Tuple[bool, Optional[str], RuleEvaluationResult]:
        """
        Check response against rules.

        Returns:
            (should_send, replacement_message, evaluation_result)
        """
        result = self.evaluator.evaluate(response, context)

        # Store result
        await self._store_result(context.get("request_id"), result)

        if result.blocked:
            # Generate safe replacement message
            replacement = self._generate_safe_response(result)
            return False, replacement, result

        return True, None, result

    def _generate_safe_response(self, result: RuleEvaluationResult) -> str:
        """Generate safe replacement for blocked response"""
        critical = result.critical_violations
        if any(v.rule_name.startswith("pii_") for v in critical):
            return "I apologize, but I cannot share that information as it may contain sensitive personal data. Please let me know how else I can help you."
        if any(v.rule_name == "sql_injection" for v in critical):
            return "I cannot execute that query as it contains potentially unsafe patterns. Please rephrase your request."
        return "I apologize, but I cannot provide that response. Please try rephrasing your question."

    async def _store_result(self, request_id: str, result: RuleEvaluationResult):
        """Store evaluation result"""
        if not request_id:
            return

        self.supabase.table("rule_evaluation_results").insert({
            "request_id": request_id,
            "passed": result.passed,
            "blocked": result.blocked,
            "score": result.score,
            "violations_count": len(result.violations),
            "violations": [
                {
                    "rule_name": v.rule_name,
                    "category": v.rule_category,
                    "severity": v.severity.value,
                    "message": v.message,
                    "details": v.details
                }
                for v in result.violations
            ]
        }).execute()


# Usage in agent_api.py
rule_gate = RuleGate(supabase_client)

@app.post("/api/pydantic-agent")
async def pydantic_agent(request: AgentRequest, ...):
    # ... agent generates response ...

    # Check rules before sending
    should_send, replacement, rule_result = await rule_gate.check_response(
        response=full_response,
        context={
            "request_id": request.request_id,
            "query": request.query,
            "tools_used": tools_used
        }
    )

    if not should_send:
        # Send safe replacement instead
        yield {"text": replacement, "blocked": True}
        return

    # Continue with normal response
    yield {"text": full_response, "complete": True}
```

### Tool-Specific Rules

```python
# backend_agent_api/evals/evaluators/tool_rules.py

class ToolRules:
    """Rules specific to tool execution"""

    @staticmethod
    def validate_sql_query(query: str) -> Optional[RuleViolation]:
        """Validate SQL query before execution"""
        query_upper = query.upper().strip()

        # Only allow SELECT
        if not query_upper.startswith("SELECT"):
            return RuleViolation(
                rule_name="sql_read_only",
                rule_category="safety",
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.BLOCK,
                message="Only SELECT queries are allowed"
            )

        # Block dangerous keywords
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC", "EXECUTE"]
        for keyword in forbidden:
            if keyword in query_upper:
                return RuleViolation(
                    rule_name="sql_forbidden_keyword",
                    rule_category="safety",
                    severity=RuleSeverity.CRITICAL,
                    action=RuleAction.BLOCK,
                    message=f"Forbidden SQL keyword: {keyword}"
                )

        return None

    @staticmethod
    def validate_code_execution(code: str) -> Optional[RuleViolation]:
        """Validate code before execution"""
        dangerous_patterns = [
            (r'import\s+os', "os module import"),
            (r'import\s+subprocess', "subprocess import"),
            (r'import\s+sys', "sys module import"),
            (r'open\s*\(', "file operations"),
            (r'exec\s*\(', "exec function"),
            (r'eval\s*\(', "eval function"),
            (r'__import__', "dynamic import"),
            (r'requests\.(get|post|put|delete)', "network requests"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                return RuleViolation(
                    rule_name="code_forbidden_pattern",
                    rule_category="safety",
                    severity=RuleSeverity.CRITICAL,
                    action=RuleAction.BLOCK,
                    message=f"Forbidden code pattern: {description}"
                )

        return None

    @staticmethod
    def validate_web_search(query: str) -> Optional[RuleViolation]:
        """Validate web search query"""
        # Check for attempts to search for PII
        pii_patterns = [
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        ]

        for pattern in pii_patterns:
            if re.search(pattern, query):
                return RuleViolation(
                    rule_name="search_pii_query",
                    rule_category="safety",
                    severity=RuleSeverity.WARNING,
                    action=RuleAction.FLAG,
                    message="Search query contains potential PII"
                )

        return None
```

## Success Criteria

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Rule evaluation latency | <10ms | >50ms |
| Block rate (critical) | <1% | >5% |
| PII detection accuracy | >95% | <90% |
| False positive rate | <1% | >3% |
| Coverage | 100% | <100% |

## Testing

```python
# tests/evals/test_rule_based.py

import pytest
from evals.evaluators.rule_based import RuleBasedEvaluator, RuleSeverity

@pytest.fixture
def evaluator():
    return RuleBasedEvaluator()

def test_pii_email_detection(evaluator):
    result = evaluator.evaluate(
        "Contact us at john.doe@company.com for more info"
    )

    assert not result.passed
    assert any(v.rule_name == "pii_email" for v in result.violations)

def test_pii_example_email_allowed(evaluator):
    result = evaluator.evaluate(
        "Use format like user@example.com"
    )

    # example.com should be allowed
    assert not any(v.rule_name == "pii_email" for v in result.violations)

def test_credit_card_blocked(evaluator):
    result = evaluator.evaluate(
        "Your card number is 4111-1111-1111-1111"
    )

    assert result.blocked
    assert any(v.rule_name == "pii_credit_card" for v in result.violations)

def test_sql_injection_blocked(evaluator):
    result = evaluator.evaluate(
        "",
        context={
            "tool_name": "execute_sql_query",
            "tool_input": "SELECT * FROM users; DROP TABLE users;"
        }
    )

    assert result.blocked

def test_clean_response_passes(evaluator):
    result = evaluator.evaluate(
        "Python is a high-level programming language known for its simplicity and readability."
    )

    assert result.passed
    assert result.score == 1.0
    assert len(result.violations) == 0
```

---

## Framework Integration

### Langfuse Integration

**Fit Level: ⚠️ PARTIAL**

Rule-based evaluation has a specific integration pattern with Langfuse:

1. **Run locally first** (real-time, pre-response blocking)
2. **Sync results to Langfuse** (for analytics only)

**Why Run Locally:**
- Real-time blocking required (<10ms latency)
- No network dependency for safety gates
- Deterministic (no LLM API needed)

**Sync Pattern:**

```python
from langfuse import Langfuse

langfuse = Langfuse()

async def sync_rule_result_to_langfuse(
    trace_id: str,
    result: RuleEvaluationResult
):
    """Sync rule evaluation results to Langfuse for analytics"""

    # Overall rule score
    langfuse.score(
        trace_id=trace_id,
        name="rule_safety_score",
        value=result.score,
        comment=f"Violations: {len(result.violations)}"
    )

    # Pass/fail
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

    # Individual rule violations (categorical)
    for violation in result.violations:
        langfuse.score(
            trace_id=trace_id,
            name=f"rule_violation_{violation.rule_category}",
            value=violation.rule_name,
            data_type="CATEGORICAL"
        )
```

**Langfuse Dashboard Uses:**
- Track rule violation rates over time
- Identify common violation patterns
- Correlate safety issues with other metrics

### Pydantic AI Support

**Fit Level: ✅ FULL SUPPORT**

Pydantic Evals provides built-in deterministic evaluators perfect for rule-based checks:

**Built-in Evaluators:**

```python
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import (
    EqualsExpected,
    Contains,
    IsInstance,
    MaxDuration
)

dataset = Dataset(
    cases=[
        Case(
            name="format_check",
            inputs={"query": "Hello"},
            expected_output="Hello!",
        )
    ],
    evaluators=[
        # Exact match
        EqualsExpected(),

        # Substring check
        Contains("Hello", case_sensitive=False),

        # Type validation
        IsInstance("str"),

        # Performance SLA
        MaxDuration(seconds=2.0),
    ]
)
```

**Evaluator Reference:**

| Evaluator | Purpose | Parameters |
|-----------|---------|------------|
| `EqualsExpected` | Exact match to expected | None |
| `Equals(value)` | Exact match to value | `value` |
| `Contains(value)` | Substring/value present | `value`, `case_sensitive` |
| `IsInstance(type)` | Type check | `type_name` |
| `MaxDuration(s)` | Latency SLA | `seconds` |

**Custom Rule Evaluator:**

```python
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

class PIIDetector(Evaluator):
    """Custom evaluator for PII detection"""

    def __init__(self):
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        }

    async def evaluate(self, ctx: EvaluatorContext) -> dict:
        import re
        output = str(ctx.output)

        violations = []
        for pii_type, pattern in self.patterns.items():
            if re.search(pattern, output):
                violations.append(pii_type)

        return {
            "pii_detected": len(violations) > 0,
            "pii_types": violations,
            "score": 0.0 if violations else 1.0
        }

# Usage
dataset = Dataset(
    cases=[...],
    evaluators=[
        PIIDetector(),
        Contains("error", case_sensitive=False),  # Should NOT contain
    ]
)
```

**Pydantic Model Validation:**

Pydantic AI's type system provides automatic validation:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class SafeResponse(BaseModel):
    """Response model with built-in validation"""
    content: str = Field(..., min_length=1, max_length=10000)
    sources: list[str] = Field(default_factory=list)

    @validator('content')
    def no_pii(cls, v):
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', v):
            raise ValueError("Response contains SSN pattern")
        return v

# Agent with built-in validation
agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=SafeResponse  # Pydantic validates automatically
)
```

**Best Practice - Pre-Response Gate:**

```python
from pydantic_evals.evaluators import Contains

# Create evaluators for pre-response check
safety_evaluators = [
    Contains("DROP TABLE", case_sensitive=False),  # SQL injection
    Contains("password", case_sensitive=False),    # Credential leak
]

async def check_response_safety(response: str) -> bool:
    """Check response before sending to user"""
    for evaluator in safety_evaluators:
        result = await evaluator.evaluate({"output": response})
        if result.get("contains", False):
            return False  # Block response
    return True
```
