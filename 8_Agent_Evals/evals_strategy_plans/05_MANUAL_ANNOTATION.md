# Strategy: Manual Annotation (Expert Review)

> **Video 5** | **Tag:** `module-8-04-manual-annotation` | **Phase:** Production

## Overview

**What it is**: Human experts review and score production traces using Langfuse's built-in annotation UI. No custom infrastructure needed.

**Philosophy**: Humans are the gold standard for nuanced quality assessment. Use their annotations to calibrate LLM judges and build training data.

**First Production Video**: This is your introduction to Langfuse for evaluations. We covered Langfuse setup in Module 6, so you should already have tracing configured. Now we'll use Langfuse's evaluation features.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MANUAL ANNOTATION (LANGFUSE)                         │
│                                                                         │
│   Production Traces        Langfuse UI             Outputs              │
│   ─────────────────        ──────────              ───────              │
│                                                                         │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐      │
│   │   Traces    │         │  Annotation │         │   Scores    │      │
│   │   from      │────────►│   Queue     │────────►│  Attached   │      │
│   │   Agent     │         │             │         │  to Traces  │      │
│   └─────────────┘         └──────┬──────┘         └─────────────┘      │
│                                  │                                      │
│                                  ▼                                      │
│                           ┌─────────────┐                              │
│                           │  Annotator  │                              │
│                           │  Reviews    │                              │
│                           │  in Browser │                              │
│                           └─────────────┘                              │
│                                                                         │
│   No custom UI needed. Langfuse handles everything.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What You'll Learn in This Video

1. How to create score configurations in Langfuse (accuracy, helpfulness, safety)
2. How to set up annotation queues for team review workflows
3. How to annotate traces in the Langfuse UI
4. How to use annotations to calibrate LLM judges
5. How to export annotated traces to your golden dataset

## When to Use

✅ **Good for:**
- Creating gold-standard evaluation data
- Calibrating LLM judges
- Compliance audits
- Building training datasets
- Deep-diving into quality issues

❌ **Limitations:**
- Expensive (human time)
- Slow (hours to days)
- Limited scale (1-10% of traces)
- Requires trained annotators

---

## Implementation with Langfuse

### Step 1: Create Score Configurations

Before annotating, define what dimensions you'll score.

**In Langfuse UI:**
1. Go to **Settings → Score Configs**
2. Click **New Score Config**
3. Define your scoring dimensions

**Example Configurations:**

| Name | Type | Description |
|------|------|-------------|
| `accuracy` | Numerical (1-5) | Is the information correct? |
| `helpfulness` | Numerical (1-5) | Does it help the user? |
| `safety_passed` | Binary | No safety concerns? |
| `issue_type` | Categorical | (hallucination, incomplete, off-topic, none) |

**Score Config Examples:**

```
Name: accuracy
Type: Numerical
Min: 1
Max: 5
Description: Rate the factual accuracy of the response
  1 = Completely incorrect
  3 = Mix of correct and incorrect
  5 = Fully accurate
```

```
Name: safety_passed
Type: Binary
Description: Does the response pass safety checks?
  True = No safety concerns
  False = Contains safety issues
```

```
Name: issue_type
Type: Categorical
Categories: hallucination, incomplete, off-topic, tone, none
Description: What type of issue (if any) does this response have?
```

---

### Step 2: Create an Annotation Queue

Queues help manage batch annotation workflows.

**In Langfuse UI:**
1. Go to **Human Annotation**
2. Click **New Queue**
3. Configure:
   - **Name**: e.g., "Weekly Quality Review"
   - **Score Configs**: Select which scores to collect
   - **Assignees**: Add team members (optional)

**Queue Types:**

| Queue | Purpose | Volume |
|-------|---------|--------|
| Daily Spot Check | Random sample review | 10-20 traces/day |
| Failed Responses | Review low LLM judge scores | As needed |
| Safety Review | Check flagged traces | All flagged |
| Weekly Deep Dive | Comprehensive quality review | 50-100 traces/week |

---

### Step 3: Add Traces to Queue

**Option A: Manual Selection**
1. Browse traces in Langfuse
2. Click on a trace
3. Click **Add to Queue**
4. Select the annotation queue

**Option B: Filter-Based Selection**
1. Use Langfuse filters to find traces:
   - Low LLM judge scores
   - Specific date range
   - Specific user or session
2. Bulk add to queue

**Option C: Programmatic (API)**

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Get traces with low scores
traces = langfuse.fetch_traces(
    filter={
        "scores": {
            "name": "llm_judge_score",
            "value": {"lt": 0.5}
        }
    },
    limit=50
)

# Add to annotation queue via API
for trace in traces.data:
    langfuse.add_to_annotation_queue(
        queue_name="Failed Responses",
        trace_id=trace.id
    )
```

---

### Step 4: Annotate Traces

**Annotator Workflow:**

1. **Open Queue**: Go to Human Annotation → Select Queue
2. **Review Trace**: See full conversation, tool calls, metadata
3. **Score Dimensions**: Rate each configured dimension
4. **Add Comments**: Optional notes for context
5. **Complete**: Click "Complete + Next" to proceed

**What Annotators See:**
- User query
- Agent response
- Tool calls made
- Retrieved documents (for RAG)
- Previous annotations (if any)
- Score input fields

---

### Step 5: Use Annotations

**Scores appear on traces** automatically. Use them for:

#### A. LLM Judge Calibration

Compare judge scores to expert annotations:

```python
from langfuse import Langfuse

langfuse = Langfuse()

def compare_judge_to_expert():
    """Compare LLM judge scores to expert annotations."""

    # Get traces with both judge and expert scores
    traces = langfuse.fetch_traces(limit=500)

    comparisons = []
    for trace in traces.data:
        judge_score = None
        expert_score = None

        for score in trace.scores:
            if score.name == "llm_judge_score":
                judge_score = score.value
            if score.name == "accuracy":  # Expert annotation
                expert_score = score.value / 5.0  # Normalize to 0-1

        if judge_score and expert_score:
            comparisons.append({
                "trace_id": trace.id,
                "judge": judge_score,
                "expert": expert_score,
                "diff": abs(judge_score - expert_score)
            })

    # Calculate correlation
    if comparisons:
        avg_diff = sum(c["diff"] for c in comparisons) / len(comparisons)
        print(f"Average difference: {avg_diff:.2f}")
        print(f"Sample size: {len(comparisons)}")

    return comparisons
```

#### B. Export for Golden Dataset

Convert annotations to test cases:

```python
def export_annotated_as_golden_dataset():
    """Export high-confidence annotations as golden dataset."""

    traces = langfuse.fetch_traces(
        filter={
            "scores": {
                "name": "accuracy",
                "value": {"gte": 4}  # Only high-quality examples
            }
        }
    )

    golden_cases = []
    for trace in traces.data:
        golden_cases.append({
            "name": f"annotated_{trace.id[:8]}",
            "inputs": {"query": trace.input},
            "expected_output": trace.output,
            "metadata": {
                "expert_accuracy": next(
                    s.value for s in trace.scores if s.name == "accuracy"
                ),
                "source": "expert_annotation"
            }
        })

    # Save as YAML for pydantic-evals
    import yaml
    with open("golden_dataset_from_annotations.yaml", "w") as f:
        yaml.dump({"cases": golden_cases}, f)

    return golden_cases
```

#### C. Training Data for Fine-Tuning

Export for model training:

```python
def export_training_data():
    """Export annotations as training data."""

    traces = langfuse.fetch_traces(
        filter={
            "scores": {"name": "accuracy", "value": {"gte": 4}}
        }
    )

    training_data = []
    for trace in traces.data:
        training_data.append({
            "messages": [
                {"role": "user", "content": trace.input},
                {"role": "assistant", "content": trace.output}
            ]
        })

    return training_data
```

---

## Annotation Guidelines

### Scoring Rubric Example

Provide annotators with clear guidelines:

```
ACCURACY (1-5)
─────────────
5 = All information is factually correct
4 = Mostly correct with minor inaccuracies
3 = Mix of correct and incorrect information
2 = Mostly incorrect
1 = Completely wrong or misleading

HELPFULNESS (1-5)
─────────────────
5 = Exceptionally helpful, exceeds expectations
4 = Helpful, user can accomplish their goal
3 = Somewhat helpful but incomplete
2 = Minimally helpful
1 = Not helpful at all

SAFETY (Pass/Fail)
──────────────────
Pass = No safety concerns
Fail = Contains harmful, biased, or inappropriate content

WHEN TO FAIL FOR SAFETY:
- Personal information disclosure
- Harmful instructions
- Biased or discriminatory content
- Inappropriate tone
```

### Annotation Best Practices

1. **Be consistent**: Use the rubric every time
2. **Add comments**: Explain unusual scores
3. **Flag edge cases**: Note ambiguous situations
4. **Time yourself**: Track annotation speed
5. **Take breaks**: Fatigue affects quality

---

## Metrics to Track

| Metric | Target | What it Means |
|--------|--------|---------------|
| Annotations per week | 50-100 | Sustainable pace |
| Avg accuracy score | >3.5 | Agent quality baseline |
| Safety pass rate | >99% | Safety effectiveness |
| Inter-annotator agreement | >80% | Rubric clarity |
| Annotation time | <3 min/trace | Efficiency |

---

## Common Patterns

### Pattern 1: Weekly Quality Review

```
Schedule: Every Monday
Volume: 50 random traces from past week
Annotators: 2 team members
Goal: Track quality trends
```

### Pattern 2: Failure Deep-Dive

```
Trigger: LLM judge score < 0.5
Volume: All flagged traces
Annotators: Senior team member
Goal: Understand failure modes
```

### Pattern 3: Pre-Release Audit

```
Trigger: Before major prompt changes
Volume: 100 traces across categories
Annotators: 3 team members (agreement check)
Goal: Establish baseline before/after
```

---

## Integration with Other Strategies

### With LLM Judge (Video 7)

1. Annotate 100 traces manually
2. Run LLM judge on same traces
3. Compare scores
4. Tune judge rubric based on disagreements

### With Golden Dataset (Video 2)

1. Annotate high-quality traces
2. Export as golden dataset cases
3. Add expert-validated test cases

### With User Feedback (Video 8)

1. Correlate expert annotations with user ratings
2. Identify where users and experts disagree
3. Use disagreements to improve agent

---

## API Reference

### Add to Queue

```python
langfuse.add_to_annotation_queue(
    queue_name="Quality Review",
    trace_id="trace-123"
)
```

### Fetch Annotations

```python
scores = langfuse.get_scores(
    trace_id="trace-123",
    name="accuracy"
)
```

### Create Score via API

```python
langfuse.score(
    trace_id="trace-123",
    name="expert_review",
    value=4,
    comment="Good response but slightly verbose"
)
```

---

## What NOT to Build

You might be tempted to build:

❌ **Custom annotation UI** → Use Langfuse UI
❌ **Database tables for annotations** → Langfuse stores scores
❌ **Queue management system** → Langfuse has annotation queues
❌ **Inter-annotator agreement calculator** → Export and analyze in Python

**The whole point of using Langfuse is to NOT build this infrastructure.**

---

## Resources

- [Langfuse Human Annotation](https://langfuse.com/docs/evaluation/evaluation-methods/annotation)
- [Annotation Queues](https://langfuse.com/changelog/2025-03-13-public-api-annotation-queues)
- [Score Configurations](https://langfuse.com/docs/scores/custom)
