# Video Preparation: Manual Annotation (Expert Review)

> Source: `05_MANUAL_ANNOTATION.md` | Phase: Production | Tag: `module-8-04-manual-annotation`

---

## Meta Information

### Why This Video Exists

This video bridges the gap between local evaluation (pydantic-evals) and production-scale quality assurance. Learners have built automated evaluators but need to understand how human expertise validates and calibrates those systems. This establishes manual annotation as the "gold standard" that all other evaluation methods ultimately reference.

### Target Learner Profile

Learners who have:
- Completed Videos 2-4 (local evals with pydantic-evals)
- Understanding of golden datasets, rule-based evaluators, and LLM judges
- Familiarity with Langfuse for tracing (from Module 6)
- Basic comfort navigating web dashboards

### Learning Outcomes

By the end of this video, learners will be able to:
1. Configure score dimensions in Langfuse for structured human evaluation
2. Set up annotation queues to organize team review workflows
3. Annotate production traces and attach quality scores
4. Explain how manual annotations calibrate automated evaluators

### Key Takeaway (One Sentence)

**Human annotation is the gold standard for evaluation quality - use Langfuse's built-in tools instead of building custom infrastructure.**

---

## Narrative Arc

### The Hook (Opening)

*"In the last few videos, we built automated evaluators - rule-based checks and an LLM judge. But here's the question: how do we know if our LLM judge is actually correct? How do we know a score of 0.8 really means 'good quality'? The answer is human expertise - and today we're going to set up a system where domain experts can validate our agent's outputs."*

### The Problem

You've deployed your agent to production. Users are interacting with it hundreds or thousands of times per day. You have automated evaluators running, but:
- How do you know if they're accurate?
- What if there are failure modes they're missing?
- How do you build confidence that your agent is actually helping users?

The only way to truly know quality is to have humans review a sample of interactions. But building a custom annotation system is a huge investment - you'd need a UI, queue management, score storage, user management...

### The Solution (Conceptual)

Instead of building custom infrastructure, leverage Langfuse's built-in annotation features:
- **Score Configs**: Define what dimensions to evaluate (accuracy, helpfulness, safety)
- **Annotation Queues**: Organize which traces need review and by whom
- **Annotation UI**: Browser-based interface for reviewing and scoring traces

This gets you enterprise-grade annotation capabilities without writing any custom code.

### The "Aha" Moment

The realization should come when the learner sees how scores from manual annotation:
1. Appear directly on the trace in Langfuse
2. Can be exported back to the golden dataset (closing the loop from Video 2)
3. Can be compared against LLM judge scores (calibrating Video 4's work)

*"Oh - this isn't just about labeling data. It's about creating a feedback loop that makes all my other evaluators better!"*

---

## Content Outline

### Introduction (3-4 minutes)

- Key talking points:
  - Transition from Local Phase (Videos 2-4) to Production Phase
  - "We've built automated evaluators, but how do we know they're right?"
  - Show the workflow diagram: Traces → Annotation Queue → Scores → Calibration
  - Emphasize: "No custom UI needed - Langfuse handles everything"

- Avoid:
  - Deep-diving into annotation theory or inter-annotator reliability yet
  - Comparing to other annotation platforms
  - Discussing when NOT to use Langfuse (save for later)

### Concept Explanation (2-3 minutes)

- Core concepts to cover:
  - **Score Configs**: Templates that define what you're measuring (think of them as evaluation rubrics)
  - **Annotation Queues**: Buckets where traces wait for human review (like a to-do list for annotators)
  - **Scores**: The actual ratings attached to traces (the output of annotation)

- Analogies or mental models:
  - Score configs are like a grading rubric a teacher creates before students submit assignments
  - Annotation queues are like the "inbox" where papers wait to be graded
  - The annotation interface is like a teacher's desk where they review and mark up work

### Live Demo (12-15 minutes)

This is the core of the video - showing the Langfuse UI in action.

#### Part A: Create Score Configs (3-4 min)

- Setup to show:
  - Langfuse dashboard, logged in
  - Settings → Score Configs page

- Demo flow:
  1. Navigate to Settings → Score Configs
  2. Click "New Score Config"
  3. Create `accuracy` (Numeric 1-5):
     - Name: "accuracy"
     - Type: Numeric
     - Min: 1, Max: 5
     - Description: "Rate the factual accuracy. 1=completely wrong, 3=partially correct, 5=fully accurate"
  4. Create `helpfulness` (Numeric 1-5):
     - Same pattern, emphasize different scoring criteria
  5. Create `safety_passed` (Boolean):
     - Show how binary works differently

- Key moments to highlight:
  - "The description becomes the rubric annotators see - be specific!"
  - "Notice we're NOT writing any code here"

#### Part B: Create Annotation Queue (2-3 min)

- Demo flow:
  1. Navigate to Human Annotation
  2. Click "New Queue"
  3. Configure:
     - Name: "Weekly Quality Review"
     - Select all three score configs
  4. Click Create

- Key moments to highlight:
  - "Different queues for different purposes - spot checks, failure review, deep dives"
  - "You could have a queue per annotator or per review type"

#### Part C: Add Traces to Queue (2-3 min)

- Demo flow:
  1. Go to Traces list
  2. Click on a trace to open detail view
  3. Click "Annotate" dropdown → Select queue
  4. Show bulk selection method:
     - Select multiple traces with checkboxes
     - Actions → Add to queue

- Key moments to highlight:
  - "In production, you might add traces programmatically based on criteria"
  - "Low LLM judge scores could auto-populate a 'Needs Review' queue"

#### Part D: Annotate a Trace (4-5 min)

- Demo flow:
  1. Open Human Annotation → Select queue
  2. Click first item
  3. Walk through interface:
     - Left side: Full conversation (user query, agent response, tool calls)
     - Right side: Score input fields
  4. Score the trace:
     - accuracy: 4 (explain reasoning aloud)
     - helpfulness: 5
     - safety_passed: True
  5. Add comment: "Response was accurate but could be more concise"
  6. Click "Complete + Next"

- Key moments to highlight:
  - "I can see the full conversation context - not just the final response"
  - "Tool calls are visible too - I can see if the agent retrieved the right documents"
  - "Comments are optional but valuable for edge cases"

- Potential issues to address live:
  - If no traces exist: "Make sure your agent is sending traces to Langfuse"
  - If scores don't appear: "Refresh the page - sometimes there's a brief delay"

#### Part E: View Results (2-3 min)

- Demo flow:
  1. Return to the annotated trace in Traces view
  2. Show scores now attached
  3. Navigate to project scores/analytics if available

- Key moments to highlight:
  - "These scores are now queryable - I can filter traces by score"
  - "This creates the data we need for calibrating our LLM judge"

### Code Walkthrough (3-4 minutes)

- Files to show:
  - `evals/annotation_helpers.py` — The helper functions for programmatic access
    - Highlight: `export_annotations_to_golden_dataset()`
    - "High-quality annotations become new golden dataset cases"
    - Highlight: `compare_judge_to_expert()`
    - "Compare your LLM judge to human ratings - are they aligned?"

- Code concepts to explain:
  - Langfuse SDK v3 returns score IDs, not score objects
  - Client-side filtering is required for score-based queries
  - These helpers are optional - the UI workflow is primary

- Skip/minimize:
  - Internal implementation details of `_fetch_scores_for_trace()`
  - The CLI interface at the bottom of the file

### Results & Interpretation (2-3 minutes)

- What to show:
  - Run `python evals/annotation_helpers.py compare-judge --limit 100`
  - Show output (likely "No traces with both scores" for fresh setup)
  - Explain what the metrics would mean if data existed

- How to interpret:
  - average_difference < 0.1: "Your LLM judge is well calibrated"
  - average_difference > 0.3: "Significant misalignment - tune your rubric"

- Connect to real-world value:
  - "Every week, annotate 50-100 traces and run this comparison"
  - "If your judge drifts, you catch it early"

### Wrap-up (2-3 minutes)

- Recap key points:
  - Langfuse provides annotation infrastructure out-of-the-box
  - Score configs define what you measure, queues organize the work
  - Human annotations calibrate automated evaluators

- What comes next:
  - "In upcoming lessons, we'll run automated evaluators in production"
  - "We'll also see how user feedback compares to expert annotations"
  - "The annotations you create here will validate those systems"

- Call to action:
  - "Set up your own score configs for your domain"
  - "Annotate 10 traces from your agent to build intuition"
  - "Think about what dimensions matter most for YOUR use case"

---

## Teaching Tips

### Common Misconceptions

| Misconception | How to Address |
|---------------|----------------|
| "I need to annotate every trace" | "Sample 1-10%. Focus on failures and edge cases. Use automated evals for scale." |
| "I need to build a custom annotation UI" | "Langfuse already has this built in. Your time is better spent improving the agent." |
| "Manual annotation replaces automated evals" | "They complement each other. Humans calibrate the machines; machines scale the humans." |
| "Annotations are only for training data" | "Three uses: calibration, golden dataset expansion, and failure analysis." |

### Questions Learners Might Ask

- Q: "What if different annotators give different scores?"
  - A: "Inter-annotator agreement is important. Start with clear rubrics in your score configs. For critical use cases, have multiple annotators review the same traces and measure agreement."

- Q: "How often should we annotate?"
  - A: "Weekly spot-checks of 50-100 traces is a good baseline. More frequent after major prompt changes. Focus annotation time on edge cases and failures."

- Q: "Can we automate adding traces to queues?"
  - A: "Yes! Use the REST API to programmatically add low-scoring traces. We show this in the strategy document. Great for routing failures to review."

- Q: "What's the difference between this and the LLM Judge from Video 4?"
  - A: "LLM Judge runs automatically on every trace. Manual annotation is for calibrating the judge. Humans annotate 100 traces; you compare their scores to what the judge said. If they disagree, tune the judge's rubric."

### Pacing Notes

- **Slow down** during score config creation - learners need to understand the rubric concept
- **Move quickly** through the annotation queue creation - it's straightforward
- **Slow down** during actual annotation - narrate your thinking as you score
- **Speed up** code walkthrough - helper functions are optional, not critical path

---

## Visual Aids

### Diagrams to Include

1. **Workflow Diagram** (show at start):
   ```
   Production Traces → Annotation Queue → Human Review → Scores
         ↓                                                  ↓
   Automated Evals ←─────── Calibration Data ←─────────────┘
   ```

2. **Feedback Loop Diagram** (show at end):
   ```
   Manual Annotations ─┬─→ Calibrate LLM Judge
                       ├─→ Expand Golden Dataset
                       └─→ Identify Failure Modes
   ```

### Screen Recording Notes

- Terminal font size: 16pt minimum
- Browser zoom: 100-125% for Langfuse UI
- Windows to have open:
  - Langfuse dashboard (logged in, on Traces page)
  - Terminal for running helper script
  - Code editor with annotation_helpers.py (optional)

- Browser tabs to prepare:
  - Langfuse Traces
  - Langfuse Settings → Score Configs
  - Langfuse Human Annotation

### B-Roll Suggestions

- Quick cut to the workflow diagram when explaining concepts
- Zoom on score config description field when emphasizing rubrics
- Side-by-side of annotated trace vs. LLM judge score (if available)

---

## Checklist Before Recording

- [ ] Code is working and tested (run `python evals/annotation_helpers.py export-golden`)
- [ ] Docker services are running and healthy
- [ ] At least 5-10 traces exist in Langfuse (send some test messages first)
- [ ] Langfuse credentials are configured in .env
- [ ] Score configs are NOT pre-created (create them live in the video)
- [ ] Annotation queue is NOT pre-created (create it live)
- [ ] Browser bookmarks for Langfuse pages
- [ ] Terminal font size is readable
- [ ] No sensitive data in trace inputs/outputs
- [ ] Strategy document `05_MANUAL_ANNOTATION.md` is reviewed for accuracy

---

## Post-Recording Checklist

- [ ] Tag moved to include any fixes: `git tag -d module-8-04-manual-annotation && git tag module-8-04-manual-annotation`
- [ ] Push to remote: `git push origin module-8-prep-evals && git push origin module-8-04-manual-annotation`
- [ ] Thumbnail created
- [ ] Video description includes learning outcomes
- [ ] Links to Langfuse documentation in description
