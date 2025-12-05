# Video Showcase Guide: Strategy 5 - Rule-Based Evaluation

## Video Overview
- **Duration**: ~8-10 minutes
- **Key Takeaway**: Rule-based evaluation provides fast (<10ms), deterministic safety checks on every agent response - no LLM cost, 100% coverage
- **Prerequisites**: Viewers should understand what agent evaluations are and why they matter

---

## Opening (1-2 min)

### Hook
*"What if your AI agent accidentally outputs a customer's credit card number? Or an SSN? Today we're implementing the first line of defense - rule-based evaluation."*

### Context Setting
- This is Strategy 5 in our 8-strategy evaluation framework
- Rule-based evaluation is the **foundation layer** - runs on every single request
- Key characteristics:
  - **Deterministic**: Same input = same result, every time
  - **Fast**: Target <10ms (we achieve ~0.01ms!)
  - **Free**: No LLM API calls
  - **100% Coverage**: Evaluates every response

### Why Start Here?
- Foundation strategies (Rule-Based, Span/Trace) should be implemented first
- Safety gates are non-negotiable for production
- Provides data for all other strategies to build on

---

## Problem Statement (1-2 min)

### The Risks
- LLMs can inadvertently output sensitive data they've seen in context
- PII leakage: emails, phone numbers, SSNs, credit cards
- Toxic/profane content in responses
- SQL injection in generated queries
- Format issues: empty responses, truncation

### What We Need
- Automated checks that catch these issues
- Run on every response without slowing down the user
- Log violations for monitoring and analytics
- Option to block critical violations (though we use log-only here)

### Show the Pain
*"Imagine a support bot that accidentally includes a customer's SSN in a response. By the time a human reviews it, the damage is done."*

---

## Solution Demo (3-5 min)

### Step 1: Show Normal Operation
1. Open the chat UI at localhost:8082
2. Send: `"What is Python?"`
3. Show the response comes through normally
4. Open Supabase → `rule_evaluation_results` table
5. Show the record: `passed=true`, `score=1.0`, `violations_count=0`
6. **Key point**: "Every response gets evaluated. This one passed all 11 rules."

### Step 2: Trigger PII Detection
1. Send: `"My contact email is john.smith@company.com"`
2. Response still comes through (log-only mode)
3. Open Supabase → show the new record
4. Highlight: `passed=false`, `score < 1.0`, violations shows `pii_email`
5. **Key point**: "The violation was detected and logged, but we didn't block the response. This is configurable."

### Step 3: Show Langfuse Integration
1. Open Langfuse dashboard
2. Find the latest traces
3. Show the scores attached: `rule_safety_score`, `rule_passed`
4. **Key point**: "All this data flows into Langfuse automatically. We can track safety trends over time."

### Step 4: Show Performance
1. Point to `evaluation_time_ms` in the database
2. "Notice this took less than 1ms. We're not slowing down the user experience at all."

---

## Code Walkthrough (3-4 min)

### File 1: `evals/evaluators/rule_based.py` (Core Engine)

**Show the data structures:**
```python
class RuleSeverity(Enum):
    CRITICAL = "critical"  # Would block if enabled
    WARNING = "warning"    # Flag but allow
    INFO = "info"          # Log only

class RuleAction(Enum):
    BLOCK = "block"
    FLAG = "flag"
    LOG = "log"
```
*"We have severity levels and actions. Critical+Block means we'd stop this response."*

**Show a rule definition:**
```python
@dataclass
class Rule:
    name: str
    category: str
    description: str
    check_fn: Callable[[str, dict], Optional[RuleViolation]]
    severity: RuleSeverity
    action: RuleAction
```
*"Each rule is a simple data class with a check function."*

**Show a check function (PII Email):**
```python
def _check_pii_email(self, response: str, context: dict):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    matches = re.findall(pattern, response)
    real_emails = [m for m in matches if not m.endswith("@example.com")]

    if real_emails:
        return RuleViolation(
            rule_name="pii_email",
            # ... details
        )
```
*"Simple regex pattern matching. We whitelist example.com for documentation."*

**Show Luhn algorithm (Credit Card):**
```python
def _luhn_check(self, card_number: str) -> bool:
    # Industry-standard credit card validation
    # Not just pattern matching - actually validates the number
```
*"For credit cards, we use the Luhn algorithm - the same validation used by banks."*

### File 2: `evals/rule_integration.py` (Integration)

**Show the service class:**
```python
class RuleEvaluationService:
    async def evaluate_response(self, response, context):
        result = self.evaluator.evaluate(response, context)
        await self._store_result(context, result)  # Supabase
        await self._sync_to_langfuse(context, result)  # Langfuse
        return result
```
*"The service coordinates evaluation, storage, and Langfuse sync."*

### File 3: `agent_api.py` (Hook Point)

**Show the integration:**
```python
# After storing the message...
asyncio.create_task(
    rule_evaluation_service.evaluate_response(
        response=full_response,
        context={"request_id": ..., "session_id": ...}
    )
)
```
*"We use asyncio.create_task so evaluation runs in the background. The user doesn't wait for it."*

---

## Results & Metrics (1-2 min)

### What We Track
- **Per-request**: score, pass/fail, violations, evaluation time
- **Aggregated**: Safety trends over time, common violation types
- **Langfuse dashboard**: Filter by `rule_safety_score` to find problematic traces

### Success Criteria
| Metric | Target | What We Achieved |
|--------|--------|------------------|
| Evaluation time | <10ms | ~0.01ms |
| Coverage | 100% | 100% (every response) |
| False positive rate | <1% | Pattern-tuned |

### 11 Rules Implemented
- **Safety**: pii_email, pii_phone, pii_ssn, pii_credit_card, profanity, sql_injection
- **Format**: empty_response, max_length, encoding
- **Quality**: repetition, incomplete_sentence

---

## Wrap-up (1 min)

### What We Built
- Deterministic rule engine with 11 default rules
- Async integration that doesn't slow down responses
- Database storage for audit trail
- Langfuse integration for analytics

### What's Next
*"Rule-based evaluation is our foundation. In the next strategy, we'll add Span/Trace evaluation to analyze the agent's execution flow - which tools it called, how long each step took."*

### Call to Action
*"Try adding your own custom rules. The pattern is simple: write a function that takes the response and returns a violation or None."*

---

## B-Roll Suggestions
- Database table view with scrolling records
- Langfuse dashboard with score distributions
- Code editor showing rule definitions
- Terminal showing test run (51 tests passing)
- Diagram of the evaluation flow (Response → Rules → DB + Langfuse)

## Common Questions to Address

1. **"Why not block violations?"**
   - Log-only mode is safer for initial deployment
   - Avoids false positive blocking
   - Can enable blocking later once rules are tuned

2. **"What about false positives?"**
   - Example.com emails are whitelisted
   - SSN validation excludes invalid prefixes (000, 666, 900+)
   - Credit cards use Luhn algorithm, not just pattern matching

3. **"How do I add custom rules?"**
   - Create a check function: `def check_fn(response, context) -> Optional[RuleViolation]`
   - Register it: `evaluator.register_rule(Rule(...))`

4. **"Does this slow down responses?"**
   - No! Runs as async background task
   - Evaluation takes ~0.01ms (target was <10ms)

---

## Manual Testing Checklist

Before recording, verify:

- [ ] Docker services running (agent-api healthy)
- [ ] Frontend accessible at localhost:8082
- [ ] Supabase `rule_evaluation_results` table exists
- [ ] Langfuse configured and receiving scores

### Test Scenarios to Demo

1. **Clean response**: Send `"What is Python?"` → `score=1.0, passed=true`
2. **PII email**: Send message with email → `pii_email` violation logged
3. **Example.com exemption**: `test@example.com` should NOT trigger violation

### Database Query for Demo
```sql
SELECT id, created_at, passed, score, violations_count, evaluation_time_ms
FROM rule_evaluation_results
ORDER BY created_at DESC
LIMIT 5;
```
