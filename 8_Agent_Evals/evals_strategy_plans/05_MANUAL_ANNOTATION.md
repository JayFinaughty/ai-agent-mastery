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

**Option C: Programmatic (REST API)**

> **Note:** As of 2025, Langfuse annotation queues are managed via REST API.
> There's no dedicated SDK method yet, but you can use the API directly.

```python
import os
import base64
import requests
from langfuse import get_client

langfuse = get_client()

# Step 1: Fetch traces (filter by scores client-side)
traces = langfuse.api.trace.list(limit=100)

# Step 2: Filter for low-scoring traces
low_score_traces = []
for trace in traces.data:
    for score in (trace.scores or []):
        if score.name == "llm_judge_score" and score.value < 0.5:
            low_score_traces.append(trace)
            break

# Step 3: Add to annotation queue via REST API
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
auth = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

queue_id = "your-queue-id"  # Get from Langfuse UI or API

for trace in low_score_traces[:50]:
    response = requests.post(
        f"{LANGFUSE_HOST}/api/public/annotation-queues/{queue_id}/items",
        headers={"Authorization": f"Basic {auth}"},
        json={"traceId": trace.id}
    )
    print(f"Added trace {trace.id}: {response.status_code}")
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
from langfuse import get_client

langfuse = get_client()

def compare_judge_to_expert(limit: int = 500) -> dict:
    """Compare LLM judge scores to expert annotations."""

    # Fetch traces using the SDK
    traces = langfuse.api.trace.list(limit=limit)

    comparisons = []
    for trace in traces.data:
        judge_score = None
        expert_score = None

        for score in (trace.scores or []):
            if score.name == "llm_judge_score":
                judge_score = score.value
            elif score.name == "accuracy":  # Expert annotation
                expert_score = score.value / 5.0  # Normalize to 0-1

        if judge_score is not None and expert_score is not None:
            comparisons.append({
                "trace_id": trace.id,
                "judge": judge_score,
                "expert": expert_score,
                "diff": abs(judge_score - expert_score)
            })

    # Calculate metrics
    if not comparisons:
        return {"error": "No traces with both judge and expert scores"}

    avg_diff = sum(c["diff"] for c in comparisons) / len(comparisons)
    return {
        "sample_size": len(comparisons),
        "average_difference": round(avg_diff, 3),
        "comparisons": comparisons[:10]  # Sample for review
    }
```

#### B. Export for Golden Dataset

Convert annotations to test cases:

```python
import yaml
from pathlib import Path
from langfuse import get_client

langfuse = get_client()

def export_annotated_as_golden_dataset(min_accuracy: int = 4, limit: int = 100):
    """Export high-confidence annotations as golden dataset."""

    # Fetch traces and filter by accuracy score client-side
    traces = langfuse.api.trace.list(limit=limit)

    golden_cases = []
    for trace in traces.data:
        # Find accuracy score
        accuracy_score = None
        for score in (trace.scores or []):
            if score.name == "accuracy":
                accuracy_score = score.value
                break

        # Only include high-quality examples
        if accuracy_score and accuracy_score >= min_accuracy:
            golden_cases.append({
                "name": f"annotated_{trace.id[:8]}",
                "inputs": {"query": trace.input or ""},
                "metadata": {
                    "expert_accuracy": accuracy_score,
                    "source": "expert_annotation",
                    "trace_id": trace.id
                }
            })

    # Save as YAML for pydantic-evals
    output_path = Path("golden_dataset_from_annotations.yaml")
    with open(output_path, "w") as f:
        yaml.dump({"cases": golden_cases}, f, default_flow_style=False)

    print(f"Exported {len(golden_cases)} cases to {output_path}")
    return golden_cases
```

#### C. Training Data for Fine-Tuning

Export for model training:

```python
import json
from langfuse import get_client

langfuse = get_client()

def export_training_data(min_accuracy: int = 4, limit: int = 500):
    """Export annotations as training data in JSONL format."""

    traces = langfuse.api.trace.list(limit=limit)

    training_data = []
    for trace in traces.data:
        # Check if trace has high accuracy score
        accuracy_score = None
        for score in (trace.scores or []):
            if score.name == "accuracy":
                accuracy_score = score.value
                break

        if accuracy_score and accuracy_score >= min_accuracy:
            training_data.append({
                "messages": [
                    {"role": "user", "content": trace.input or ""},
                    {"role": "assistant", "content": trace.output or ""}
                ]
            })

    # Save as JSONL (standard format for fine-tuning)
    with open("training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"Exported {len(training_data)} examples to training_data.jsonl")
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

### Add to Queue (REST API)

```python
import requests
import base64
import os

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
auth = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

# Add trace to annotation queue
response = requests.post(
    f"{LANGFUSE_HOST}/api/public/annotation-queues/{queue_id}/items",
    headers={"Authorization": f"Basic {auth}"},
    json={"traceId": "trace-123"}
)
```

### Fetch Traces with Scores

```python
from langfuse import get_client

langfuse = get_client()

# List traces (scores are included in response)
traces = langfuse.api.trace.list(limit=100)

# Access scores on each trace
for trace in traces.data:
    for score in (trace.scores or []):
        print(f"{trace.id}: {score.name} = {score.value}")
```

### Create Score via SDK

```python
from langfuse import get_client

langfuse = get_client()

# Attach a score to a trace
langfuse.create_score(
    trace_id="trace-123",
    name="expert_review",
    value=4,
    data_type="NUMERIC",
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

---

## Instructor Guide: Recording Video 5

### Pre-Recording Checklist

1. **Langfuse Account Ready:**
   ```bash
   # Verify credentials in .env
   cd 8_Agent_Evals/backend_agent_api
   cat .env | grep -E "^LANGFUSE_"
   # Should show: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
   ```

2. **Generate Test Traces:**
   ```bash
   # Start the agent API
   cd 8_Agent_Evals/backend_agent_api
   python agent_api.py &

   # Send a few test queries via the frontend or curl
   # This creates traces in Langfuse for annotation
   ```

3. **Verify Traces Exist:**
   - Open Langfuse dashboard
   - Navigate to Traces
   - Confirm you see recent traces from your agent

4. **Open Browser Tabs:**
   - Langfuse dashboard (logged in)
   - Settings → Score Configs page
   - Human Annotation page

### Recording Flow

**Part 1: Introduction (2-3 min)**
- Explain transition from Local (Videos 2-4) to Production phase
- Show the workflow diagram from the top of this doc
- Key message: "No custom UI needed - Langfuse handles everything"

**Part 2: Create Score Configs (3-5 min)**
- Navigate: Settings → Score Configs → New Score Config
- Create three configs:

  | Name | Type | Min | Max | Description |
  |------|------|-----|-----|-------------|
  | `accuracy` | Numeric | 1 | 5 | "Rate factual accuracy: 1=wrong, 5=perfect" |
  | `helpfulness` | Numeric | 1 | 5 | "Rate how helpful: 1=useless, 5=excellent" |
  | `safety_passed` | Boolean | - | - | "Does response pass safety checks?" |

- Show how the description becomes the rubric annotators see

**Part 3: Create Annotation Queue (3-5 min)**
- Navigate: Human Annotation → New Queue
- Configure:
  - Name: "Weekly Quality Review"
  - Select all three score configs
  - Optional: Add assignees
- Discuss queue types (show table from doc):
  - Daily Spot Check
  - Failed Responses
  - Safety Review
  - Weekly Deep Dive

**Part 4: Add Traces to Queue (3-5 min)**
- **Method 1: Single trace**
  - Click on any trace in the Traces list
  - Click "Annotate" dropdown → Select queue
- **Method 2: Bulk selection**
  - Use checkboxes in trace list
  - Click "Actions" → "Add to queue"
- Show filtering options (date range, user, tags)

**Part 5: Annotate a Trace (5-7 min)**
- Open Human Annotation → Select "Weekly Quality Review"
- Click first item in queue
- Walk through the annotation interface:
  - **Left panel**: See conversation (user query, agent response)
  - **Right panel**: Score input fields
  - Rate accuracy (1-5)
  - Rate helpfulness (1-5)
  - Toggle safety_passed
  - Add optional comment: "Response was accurate but could be more concise"
- Click "Complete + Next" to proceed

**Part 6: View Results & Export (3-5 min)**
- Return to the annotated trace
- Show scores now attached
- Discuss use cases:
  - "These scores calibrate our LLM Judge in Video 7"
  - "High-quality traces become golden dataset cases"
  - "We'll compare to user feedback in Video 8"
- Optionally show the `annotation_helpers.py` script

### Expected Screen Flow

```
1. Langfuse Dashboard
   └── Settings → Score Configs
       └── Create: accuracy, helpfulness, safety_passed

2. Human Annotation
   └── New Queue: "Weekly Quality Review"

3. Traces List
   └── Select traces → Add to queue

4. Annotation Queue
   └── Annotate first trace
       └── Score all dimensions
       └── Complete + Next

5. Trace Detail
   └── View attached scores
```

### Key Teaching Moments

1. **"Why use Langfuse instead of building our own?"**
   - Would need: annotation UI, queue system, score storage, user management
   - Langfuse handles all of this out of the box
   - Focus your time on improving the agent, not building infrastructure

2. **"When should I manually annotate?"**
   - Not everything! Sample 1-10% of traces
   - Focus on: failures, edge cases, random samples
   - Use LLM Judge (Video 7) for scale, humans for calibration

3. **"How does this connect to other videos?"**
   - Video 2 (Golden Dataset): Annotations become test cases
   - Video 7 (LLM Judge Prod): Annotations calibrate the judge
   - Video 8 (User Feedback): Compare expert vs user ratings

4. **"What makes a good annotator?"**
   - Consistent use of rubric
   - Comments on unusual scores
   - Takes breaks to avoid fatigue

### Troubleshooting During Recording

| Issue | Quick Fix |
|-------|-----------|
| No traces in Langfuse | Check LANGFUSE_* env vars, restart agent API |
| Score configs not showing | Refresh page, check project selection |
| Can't add to queue | Ensure queue has score configs selected |
| Annotations not saving | Check network tab for errors, refresh |

### Post-Recording Git Workflow

```bash
# Ensure you're on the prep branch
git checkout module-8-prep-evals

# Stage the new/modified files
git add evals/annotation_helpers.py
git add ../evals_strategy_plans/05_MANUAL_ANNOTATION.md

# Commit
git commit -m "Implement Video 5: Manual Annotation"

# Tag this state
git tag module-8-04-manual-annotation

# Push commit and tag
git push origin module-8-prep-evals
git push origin module-8-04-manual-annotation
```

### Files Created/Modified in This Video

```
8_Agent_Evals/
├── backend_agent_api/
│   └── evals/
│       └── annotation_helpers.py    # Helper for exporting annotations
└── evals_strategy_plans/
    └── 05_MANUAL_ANNOTATION.md      # This document (updated)
```

### Demo Script: Using annotation_helpers.py

At the end of the video, optionally demonstrate:

```bash
cd 8_Agent_Evals/backend_agent_api

# Export high-quality annotations to golden dataset
python -c "
from evals.annotation_helpers import export_annotations_to_golden_dataset
cases = export_annotations_to_golden_dataset(min_accuracy=4)
print(f'Exported {len(cases)} cases')
"

# Compare LLM judge to expert scores (for calibration)
python -c "
from evals.annotation_helpers import compare_judge_to_expert
result = compare_judge_to_expert()
print(result)
"
```
