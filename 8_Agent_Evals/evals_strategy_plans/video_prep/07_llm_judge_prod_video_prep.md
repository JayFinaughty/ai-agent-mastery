# Video Preparation: LLM-as-Judge (Production)

> Source: `07_LLM_JUDGE_PROD.md` | Phase: Production | Video 7

---

## Meta Information

### Why This Video Exists

Learners have used `LLMJudge` for local golden dataset testing. Now they need to scale that same capability to production, where they can't manually review thousands of responses. This video bridges that gap, showing how the rubric-writing skills they've already developed translate directly to production AI-powered quality monitoring.

### Target Learner Profile

Someone who has:
- Completed Videos 1-6 (especially Video 4: LLM Judge Local and Video 6: Rule-Based Prod)
- Written LLM rubrics for golden dataset evaluation
- Implemented async evaluation with Langfuse score sync
- Understands the cost implications of LLM API calls

### Learning Outcomes

By the end of this video, learners will be able to:
1. Build a custom pydantic-ai judge agent with structured output for production use
2. Implement sampled async evaluation to control costs while maintaining quality monitoring
3. Sync judge scores to Langfuse for trend analysis and filtering
4. Configure smart sampling strategies that prioritize interesting traces

### Key Takeaway (One Sentence)

The rubric-writing skills you learned locally transfer directly to production — what changes is the execution model (async) and output destination (Langfuse scores).

---

## Instructor Notes: Timing & Delays

### Important: Scores Appear with Delay

Both evaluation methods have delays before scores appear in Langfuse:

1. **Langfuse Built-in Evaluators** (configured in UI)
   - Run on a schedule (every few minutes)
   - Expect **5-7 minute delay** before scores appear on traces
   - Scores show up as separate "Execute evaluator" entries in Langfuse

2. **Custom pydantic-ai Judge** (`prod_judge.py`)
   - Runs immediately after agent response (async)
   - Scores sync within seconds, but with **10% sampling** by default
   - Only ~1 in 10 requests will have `llm_judge_score`

### Changing Sample Rate for Demo

For video recording, you may want to evaluate 100% of requests so scores appear reliably.

**Location:** `backend_agent_api/agent_api.py` line 31-33

```python
# Default (production):
JUDGE_SAMPLE_RATE = 0.1  # 10% of requests

# For demo/testing:
JUDGE_SAMPLE_RATE = 1.0  # 100% of requests
```

After changing, restart the container:
```bash
docker compose restart agent-api
```

**Remember to change back to 0.1 after recording** to avoid unnecessary API costs.

### Demo Strategy

1. **Before recording:** Set `JUDGE_SAMPLE_RATE = 1.0` and restart
2. **During recording:** Send messages, scores appear immediately
3. **Explain:** "In production, you'd sample 10% to control costs"
4. **After recording:** Revert to 0.1

---

## Narrative Arc

### The Hook (Opening)

"You've been running your agent in production for a week. Users are happy, but you have 10,000 traces. How do you know if the quality is good? You can't manually review them all, and rule-based checks only catch obvious problems. What if your agent is giving technically correct but unhelpful responses?"

### The Problem

Rule-based evaluation (from the previous lesson) catches clear violations — PII leaks, forbidden words, format issues. But it can't assess subjective quality: Is this response actually helpful? Does it address what the user was really asking? Is it concise or rambling?

Manual annotation scales to hundreds of traces, not thousands. You need automation that can make nuanced quality judgments.

### The Solution (Conceptual)

Use the same LLM-as-Judge pattern from local testing, but adapted for production:
- **Same rubrics** you already wrote
- **Async execution** so users don't wait
- **Sampled** to control costs (10% default)
- **Scores to Langfuse** for filtering and trending

The judge runs in the background after each response, evaluates quality, and syncs scores. You can then filter Langfuse for `llm_judge_passed = 0` to find problems worth investigating.

### The "Aha" Moment

When learners realize that the `LLMJudge` rubrics they wrote for their golden dataset can be reused almost verbatim in production. The skills transfer — only the plumbing changes. This should click when we show the `system_prompt` of the judge agent and compare it to their Video 4 rubrics.

---

## Content Outline

### Introduction (2-3 minutes)

- Key talking points:
  - "We've been building up evaluation layers: golden dataset, rule-based, LLM judge locally. Now we add the final piece — LLM judge in production."
  - Briefly recap Video 4 (local LLM judge with rubrics) and Video 6 (async pattern with Langfuse scores)
  - Preview three options: Langfuse built-in evaluators, custom Langfuse evaluators, and pydantic-ai judges

- Avoid:
  - Deep dive into Langfuse setup (covered in earlier videos)
  - Explaining what LLM-as-Judge is conceptually (they know from Video 4)

### Concept Explanation (3-4 minutes)

- Core concepts to cover:
  - **Three options for production LLM judges:**
    1. **Langfuse built-in evaluators** — Use prebuilt templates (Helpfulness, Relevance, etc.) with no code
    2. **Custom Langfuse evaluators** — Create your own evaluation prompts in Langfuse UI with JSONPath variable mapping
    3. **pydantic-ai judge** — Full code control with structured output and complex logic
  - **When to use each:**
    - Built-in: Quick start, common metrics
    - Custom Langfuse: Domain-specific rubrics without code, team collaboration
    - pydantic-ai: Structured output, version control, complex evaluation logic
  - **Sampling for cost control**: At 1000 requests/day with 100% evaluation, you'd pay ~$2/day. At 10%, it's $0.20/day. Show the math.
  - **What "async" means here**: Fire-and-forget. User gets their response immediately; judge runs in background.

- Analogies or mental models:
  - "Think of it like a restaurant: the customer gets served immediately, while a quality inspector randomly samples dishes in the kitchen. The inspector doesn't slow down service."

### Live Demo (8-10 minutes)

- Setup to show:
  - Terminal with Docker logs visible (`docker compose logs -f agent-api`)
  - Browser with chat frontend open
  - Langfuse dashboard in another tab

- Demo flow:
  1. Show the existing `prod_judge.py` file — walk through the `JudgeResult` model and `ProductionJudgeEvaluator` class
  2. Point out the system prompt (the rubric) — compare to Video 4's `LLMJudge` rubric
  3. Show the integration in `agent_api.py` — the sampling logic and `asyncio.create_task()`
  4. Send a few chat messages in the frontend
  5. Watch Docker logs for `[prod_judge]` output on low-quality detections
  6. Switch to Langfuse — find traces with `llm_judge_score` and `llm_judge_passed` scores
  7. Filter for `llm_judge_passed = 0` to show how you'd find problem responses

- Key moments to highlight:
  - When the `[prod_judge] Low quality detected` message appears in logs
  - When you see the score and reason on a Langfuse trace
  - The sampling in action — most traces won't have judge scores (that's intentional)

- Potential issues to address live:
  - "If you don't see judge scores, remember we're only evaluating 10% of requests. Send more messages!"
  - "The judge runs async, so scores might take a few seconds to appear in Langfuse"

### Code Walkthrough (5-6 minutes)

- Files to show:
  - `evals/prod_judge.py` — Focus on:
    - `JudgeResult` model (structured output)
    - `_create_judge_agent()` — the rubric/system prompt
    - `evaluate_and_sync()` — the Langfuse score sync using low-level API
  - `agent_api.py` — Focus on:
    - The import and `JUDGE_SAMPLE_RATE` constant
    - The sampling logic (`random.random() < JUDGE_SAMPLE_RATE`)
    - The `asyncio.create_task()` call (same pattern as Video 6)

- Code concepts to explain:
  - Why we use `langfuse.api.score.create()` (low-level API) instead of decorator — avoids OTEL conflicts
  - The singleton pattern for the evaluator (lazy initialization)
  - Why `passed` is a separate score (easier filtering in Langfuse)

- Skip/minimize:
  - Import statements
  - Exception handling details (mention it exists, don't dwell)
  - The `_create_judge_model()` function (just say "uses same config as agent")

### Results & Interpretation (3-4 minutes)

- What to show:
  - Langfuse trace with both `llm_judge_score` (0.0-1.0) and `llm_judge_passed` (0/1)
  - The comment field showing the judge's reasoning
  - Langfuse filtering: `llm_judge_passed = 0` to find failures
  - Optional: Langfuse analytics showing score distribution over time

- How to interpret:
  - Score < 0.7 means the judge found quality issues
  - The `comment` field explains why — use this to improve your agent's prompts
  - Trend down over time? Something changed (model update, prompt regression, etc.)

- Connect to real-world value:
  - "Instead of manually reviewing 10,000 traces, you can filter to the ~50 that the judge flagged as problems"
  - "Track quality trends after prompt changes — did your 'improvement' actually help?"

### Wrap-up (2 minutes)

- Recap key points:
  - LLM judge scales subjective quality assessment to production
  - Same rubrics as local testing, different execution model
  - Sample to control costs, async to avoid user latency

- What comes next:
  - "We have automated evaluation. But what do users actually think? In the next video, we'll collect explicit feedback — thumbs up/down — and correlate it with our judge scores."

- Call to action:
  - "Try adjusting the sampling rate or adding a second rubric (e.g., conciseness). See how the scores distribute across your production traffic."

---

## Teaching Tips

### Common Misconceptions

1. **"I need to evaluate every request"**
   - Reality: 10% sampling gives statistically meaningful data at 10x lower cost. Increase only if you're debugging specific issues.

2. **"The judge score is ground truth"**
   - Reality: LLM judges have biases and can be wrong. Use them for triage, not final judgment. Validate with human review.

3. **"I should use GPT-4/Claude Opus for the judge"**
   - Reality: GPT-4o-mini is usually sufficient for quality evaluation. Save the expensive models for the actual agent responses.

### Questions Learners Might Ask

- Q: "What if the judge and user disagree?"
  - A: "Great question! In the next lesson, we'll collect user feedback and can compare. Disagreements are learning opportunities for your rubrics."

- Q: "How do I tune the 0.7 threshold?"
  - A: "Start with 0.7, then look at borderline cases (0.6-0.8). If too many false positives, raise it. If missing real problems, lower it."

- Q: "Can I use different judges for different query types?"
  - A: "Absolutely! You could have one for RAG queries (checking faithfulness) and another for general chat (checking helpfulness)."

### Pacing Notes

- **Slow down**: During the rubric/system prompt explanation — this is where skills transfer from Video 4
- **Speed up**: Through the Langfuse score sync code — it's the same pattern from Video 6
- **Pause for effect**: When showing the `[prod_judge] Low quality detected` log message — this is the payoff moment

---

## Visual Aids

### Diagrams to Include

**Evaluation Flow Diagram:**
```
User Request → Agent Response → [User sees response immediately]
                    ↓
              [Background]
                    ↓
            Sampling (10%)
                    ↓
              LLM Judge
                    ↓
            Langfuse Scores
                    ↓
           Dashboard/Filtering
```

**Cost Comparison Table:**
| Sampling Rate | Daily Cost (1K requests) | Coverage |
|---------------|-------------------------|----------|
| 100% | ~$2.00 | All traces |
| 10% | ~$0.20 | Representative sample |
| 1% | ~$0.02 | Spot check only |

### Screen Recording Notes

- Terminal font size: 16pt minimum
- Windows to have open:
  - VS Code with `prod_judge.py` open
  - Terminal with `docker compose logs -f agent-api`
  - Browser: Chat frontend
  - Browser: Langfuse dashboard (Traces view)
- Hide/close: Any windows with API keys or personal data

### B-Roll Suggestions

- Langfuse filter UI showing score-based filtering
- Optional: Graph showing score distribution (if you have enough data)

---

## Checklist Before Recording

- [ ] `prod_judge.py` exists and imports correctly
- [ ] `agent_api.py` has the judge integration with sampling
- [ ] Docker services are running and healthy
- [ ] LLM API key is configured (judge can make calls)
- [ ] Langfuse is connected and receiving traces
- [ ] Test: High-quality response gets score > 0.7
- [ ] Test: Low-quality response gets score < 0.7 and appears in logs
- [ ] Clear browser history/cookies for clean UI
- [ ] Test chat session generates judge scores in Langfuse
- [ ] Prepare 2-3 test queries that will showcase different quality levels

---

## Sample Test Queries for Demo

**High Quality (should pass):**
- "What is machine learning?"
- "How do I read a CSV file in Python?"

**Low Quality Triggers (should fail):**
- Ask a specific question, then manually send a response like "I don't know" through a test trace
- Or: Ask about something the agent clearly can't help with to trigger a deflection response

**Edge Case (borderline):**
- "What's the weather?" — Agent might deflect appropriately but judge could see it as unhelpful
