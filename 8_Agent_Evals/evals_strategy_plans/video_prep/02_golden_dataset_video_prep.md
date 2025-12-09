# Video Preparation: Golden Dataset Evaluation

> Source: `02_GOLDEN_DATASET.md` | Phase: Local Development

---

## Meta Information

### Why This Video Exists
This is the learner's first hands-on evaluation implementation. After the conceptual introduction in the previous lesson, they need to see that agent evaluation isn't complex—it starts with just 10 test cases in a YAML file. This video bridges theory to practice and establishes the pattern they'll build upon throughout the module.

### Target Learner Profile
Learners who have:
- Built or worked with an AI agent (from earlier modules)
- Watched the introductory video on agent evaluations
- Basic Python and YAML familiarity
- Never implemented agent evaluations before

### Learning Outcomes
By the end of this video, learners will be able to:
1. Create a golden dataset YAML file with test cases covering multiple categories
2. Use deterministic evaluators (Contains, IsInstance, MaxDuration) to validate agent outputs
3. Run a local evaluation script and interpret the pass/fail results
4. Explain why keyword-based checks have limitations (setting up future lessons)

### Key Takeaway (One Sentence)
Start your agent evaluations with just 10 curated test cases—that's enough to catch regressions and validate core behaviors.

---

## Narrative Arc

### The Hook (Opening)
> "When I set up evals for my agent at first, I do a 10-question golden dataset. That's all you need to start." — Andrew Ng

Open with this quote. Let it breathe. Then ask: "What if I told you that you can have a working evaluation system running in the next 15 minutes, with zero external services required?"

### The Problem
You've built an agent. It works... most of the time. But how do you know if tomorrow's code change breaks something? How do you catch regressions before your users do? Without automated checks, you're flying blind—every deployment is a gamble.

### The Solution (Conceptual)
A golden dataset is your agent's "final exam"—a curated set of questions you know the correct answers to. Run it before every deployment. If your agent fails questions it used to pass, you've caught a regression. It's that simple.

### The "Aha" Moment
The realization should happen when the evaluation report prints to the terminal. Learners see the table, the pass/fail indicators, and the 90% pass rate. They think: "Wait, that's it? I just wrote a YAML file and a short Python script, and now I have an automated test suite for my agent?"

The secondary aha comes when they see the `refuse_illegal` test fail—and understand that keyword matching has limits, naturally motivating the LLM Judge lesson coming later.

---

## Content Outline

### Introduction (2-3 minutes)
- Key talking points:
  - Reference the Andrew Ng quote—10 cases is enough to start
  - Explain "golden dataset" terminology: curated, trusted, representative
  - Preview what we'll build: YAML file + Python script + terminal output
- Avoid:
  - Going deep into pydantic-evals internals
  - Comparing to other evaluation frameworks
  - Discussing production/Langfuse (that comes later)

### Concept Explanation (3-4 minutes)
- Core concepts to cover:
  - **Test case structure**: name, inputs, metadata, evaluators
  - **Evaluators as validators**: each evaluator returns pass/fail
  - **Categories**: organizing cases by type (general, rag, safety, etc.)
  - **Global evaluators**: checks applied to ALL cases (like IsInstance, MaxDuration)
- Analogies or mental models:
  - "Think of evaluators like unit test assertions—each one checks one specific thing"
  - "Categories are like test suites—you can see which areas of your agent are weakest"

### Live Demo (8-10 minutes)
- Setup to show:
  - Terminal with the 8_Agent_Evals directory
  - Code editor with golden_dataset.yaml open
  - Docker services running (visible in a split terminal if possible)
- Demo flow:
  1. Show the project structure (`evals/` directory)
  2. Walk through 3-4 representative cases in the YAML:
     - `greeting` — simple Contains check
     - `simple_question` — multiple Contains (AND logic)
     - `refuse_harmful` — ContainsAny (OR logic)
     - `refuse_illegal` — intentionally fragile single Contains
  3. Show global evaluators at the bottom of YAML
  4. Open `run_evals.py` briefly—highlight the `run_agent` function
  5. Run the evaluation: `docker compose exec agent-api python evals/run_evals.py`
  6. Watch the progress bar, then the results table
- Key moments to highlight:
  - When the progress bar starts moving—"each bar tick is one agent call"
  - When `refuse_illegal` shows ✗—pause here and explain
  - The summary line showing 90% pass rate
- Potential issues to address live:
  - If all tests timeout: check LLM API key
  - If import errors: verify running from correct directory
  - If pass rate is different: that's fine, LLMs vary

### Code Walkthrough (4-5 minutes)
- Files to show:
  - `golden_dataset.yaml` — focus on the structure, not every line
  - `evaluators.py` — explain ContainsAny as custom evaluator example
  - `run_evals.py` — the task function pattern and 80% threshold
- Code concepts to explain:
  - How `Dataset.from_file()` loads YAML with custom evaluators
  - The task function pattern (`async def run_agent(inputs) -> str`)
  - Exit codes for CI/CD integration
- Skip/minimize:
  - Import statements
  - Environment variable loading
  - The case_passed helper function internals

### Results & Interpretation (3-4 minutes)
- What to show:
  - The evaluation report table
  - The summary section with pass/fail counts
  - The exit code (0 for pass, 1 for fail)
- How to interpret:
  - ✔ means the assertion passed
  - ✗ means it failed
  - Multiple symbols (✔✔✔) means multiple evaluators per case
  - Duration column helps identify slow cases
- Connect to real-world value:
  - "This 80% threshold could be your deployment gate"
  - "Add this to CI—if evals fail, the PR doesn't merge"

### Wrap-up (2-3 minutes)
- Recap key points:
  - 10 cases is enough to start
  - Deterministic evaluators are fast and free
  - The fragile `refuse_illegal` test shows why we need more sophisticated checks
- What comes next:
  - In the next lesson: tool verification with HasMatchingSpan
  - Then: LLM-as-Judge for subjective quality
- Call to action:
  - "Pause here and create your own 5 test cases for your agent"
  - "Think about what behaviors you'd want to check before every deployment"

---

## Teaching Tips

### Common Misconceptions
1. **"I need hundreds of test cases"**
   - Start with 10. Add more as you discover edge cases. Quality over quantity.

2. **"Keyword matching is unreliable"**
   - Yes! That's intentional. We show its limits here to motivate LLMJudge later.

3. **"This only works for simple responses"**
   - Deterministic evaluators work for any response. The trick is choosing the right assertions.

4. **"I need Langfuse/external services"**
   - Not for Phase 1. Everything runs locally. Production observability comes later.

### Questions Learners Might Ask
- Q: "Why YAML instead of Python for test cases?"
  - A: YAML separates test data from test logic. It's easier to review in code review and non-developers can contribute test cases.

- Q: "What if my agent's responses vary each time?"
  - A: Use evaluators that check for presence of keywords rather than exact matches. That's why we use Contains instead of EqualsExpected.

- Q: "How do I know which categories to use?"
  - A: Start with your agent's main use cases. If it does RAG, have RAG tests. If it has safety constraints, have safety tests.

- Q: "Why did refuse_illegal fail but refuse_harmful pass?"
  - A: `refuse_harmful` uses ContainsAny (OR logic) which catches different phrasings. `refuse_illegal` uses a single Contains check that's deliberately fragile—this motivates LLMJudge.

### Pacing Notes
- **Slow down** when explaining the YAML structure—learners need to understand this pattern
- **Slow down** when the evaluation runs and fails—this is a teaching moment
- **Speed up** through the imports and boilerplate in run_evals.py
- **Emphasize** the 80% threshold concept—this is CI/CD integration

---

## Visual Aids

### Diagrams to Include
1. **Golden Dataset Workflow Diagram** (from strategy doc)
   ```
   golden_dataset.yaml → run_evals.py → Evaluation Report
   ```

2. **Evaluator Types Comparison** (could be a simple slide)
   | Evaluator | Purpose | Cost |
   |-----------|---------|------|
   | Contains | Substring check | Free |
   | IsInstance | Type validation | Free |
   | MaxDuration | Performance | Free |
   | ContainsAny | OR logic | Free |

### Screen Recording Notes
- Terminal font size: 16pt minimum for readability
- Windows to have open:
  - Terminal (main focus)
  - Code editor with YAML file
  - (Optional) Docker Desktop showing healthy containers
- Browser tabs to prepare:
  - pydantic-evals docs (in case of questions)

### B-Roll Suggestions
- The Andrew Ng quote as a text overlay at the start
- Quick cut to the pydantic-evals logo when mentioning the package
- Terminal output table as a hero shot when summarizing

---

## Checklist Before Recording

- [ ] Docker services running and healthy (`docker compose ps`)
- [ ] Run evals once to verify they complete (`docker compose exec agent-api python evals/run_evals.py`)
- [ ] Verify ~90% pass rate (some variation expected)
- [ ] Confirm `refuse_illegal` fails (it should)
- [ ] Clear terminal history for clean recording
- [ ] Close notification apps / enable Do Not Disturb
- [ ] Test audio levels
- [ ] Prepare code editor with golden_dataset.yaml open
- [ ] Verify .env has no sensitive data visible if scrolled
- [ ] Font size increased in terminal and editor

---

## Appendix: Quick Reference

### File Locations
```
8_Agent_Evals/backend_agent_api/evals/
├── __init__.py            # Package docstring
├── evaluators.py          # ContainsAny custom evaluator
├── golden_dataset.yaml    # 10 test cases
└── run_evals.py           # Evaluation runner
```

### Key Commands
```bash
# Check services
docker compose ps

# Run evaluation
docker compose exec agent-api python evals/run_evals.py

# Test dataset loading
docker compose exec agent-api python -c "
from pydantic_evals import Dataset
from evals.evaluators import ContainsAny
from pathlib import Path
d = Dataset[dict,str].from_file(Path('/app/evals/golden_dataset.yaml'), custom_evaluator_types=[ContainsAny])
print(f'{len(d.cases)} cases loaded')
"
```

### Expected Output Summary
```
Total Cases:   10
Passed:        9
Failed:        1
Pass Rate:     90.0%
PASSED - Above 80% threshold
```
