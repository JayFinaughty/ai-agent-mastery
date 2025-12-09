# Video Preparation: Rule-Based Evaluation (Local)

> Source: `03_RULE_BASED_LOCAL.md` | Phase: Local Development

---

## Meta Information

### Why This Video Exists
Learners have a golden dataset from the previous lesson, but output content checks alone can't verify *how* the agent arrived at its answer. This video fills the gap by teaching tool call verification - ensuring the agent uses the right tools (and avoids dangerous ones) rather than hallucinating or taking shortcuts.

### Target Learner Profile
Someone who has completed the Golden Dataset lesson and understands basic pydantic-evals concepts (Dataset, Case, evaluators). They should be comfortable with YAML syntax and have their evaluation environment running.

### Learning Outcomes
By the end of this video, learners will be able to:
1. Use `HasMatchingSpan` to verify specific tools were called during agent execution
2. Implement negative checks to ensure dangerous tools are NOT called on harmful requests
3. Configure logfire instrumentation for proper span naming
4. Create custom evaluators for domain-specific rule-based checks

### Key Takeaway (One Sentence)
`HasMatchingSpan` lets you verify tool usage deterministically - no AI judgment needed for yes/no questions like "did the agent call the RAG tool?"

---

## Narrative Arc

### The Hook (Opening)
"Your agent just answered a question about documents. The response looks correct. But here's the uncomfortable question: did it actually search your documents, or did it just make something up?"

### The Problem
In the last video, we created a golden dataset with content checks like `Contains: "document"`. But that only verifies what the agent *said*, not what it *did*. An agent could hallucinate a plausible-sounding answer without ever touching your document database. For RAG applications, this is a critical failure mode that content checks can't catch.

Similarly, when a user sends a harmful request, we want to verify the agent didn't call dangerous tools - not just that it said "I can't help with that."

### The Solution (Conceptual)
When your agent runs, pydantic-ai creates OpenTelemetry spans for each tool call. These spans are like a detailed audit log of exactly what the agent did. `HasMatchingSpan` lets us query this audit log to verify:
- "This tool MUST have been called" (positive check)
- "This tool must NOT have been called" (negative check)

This is deterministic verification - no LLM costs, no judgment calls, just yes/no answers.

### The "Aha" Moment
The key insight clicks when learners see a test case that checks both the output AND the tool usage together. For example: "The response mentions documents AND the RAG tool was actually called." They realize that combining content + tool verification catches failure modes that neither approach catches alone.

A secondary "aha" happens with negative checks: "The agent refused the harmful request AND it never called execute_code." This is safety verification you can run on every eval.

---

## Content Outline

### Introduction (2-3 minutes)
- Key talking points:
  - Recap: "In the last video, we built a 10-case golden dataset with content checks"
  - The limitation: "Content checks tell us what the agent said, not what it did"
  - The bridge: "Today we add tool verification - deterministic checks on agent behavior"
- Avoid:
  - Don't re-explain pydantic-evals basics (assume knowledge from previous video)
  - Don't mention Langfuse yet (that's Phase 2)

### Concept Explanation (3-4 minutes)
- Core concepts to cover:
  - **Spans**: When pydantic-ai runs tools, it creates OpenTelemetry spans that record what happened
  - **Span attributes**: Tool names are stored in the `gen_ai.tool.name` attribute, not the span name
  - **HasMatchingSpan**: Queries the span tree to find spans with matching attributes
  - **Positive vs Negative checks**: "must call X" vs "must NOT call Y"
- Analogies or mental models:
  - Spans are like a security camera recording - you can go back and verify what actually happened
  - Think of it as "did you show your work?" verification from school math tests

### Live Demo Part 1: Logfire Configuration (3-4 minutes)
- Setup to show:
  - Terminal with `run_evals.py` open in editor
  - The `.env` file visible (to show no Logfire credentials needed)
- Demo flow:
  1. Show the logfire import and configuration block
  2. Explain `send_to_logfire=False` - "We're running locally, no cloud connection"
  3. Show `logfire.instrument_pydantic_ai()` - "This enables span tracking for tool calls"
  4. Point out the placement: "BEFORE we import the agent - order matters!"
- Key moments to highlight:
  - Span structure: span name is `running tool`, tool name is in `gen_ai.tool.name` attribute
  - Why order matters: instrumentation must happen before agent code loads
- Potential issues to address live:
  - If logfire not installed: `pip install logfire`
  - If spans aren't matching: check that logfire config comes before agent import

### Live Demo Part 2: Adding HasMatchingSpan to Dataset (5-7 minutes)
- Setup to show:
  - `golden_dataset.yaml` open in editor
  - Terminal ready to run evals
- Demo flow:
  1. Start with `document_search` case from Video 2
  2. Add HasMatchingSpan evaluator: show the YAML syntax
  3. Explain `query: { has_attributes: { gen_ai.tool.name: "retrieve_relevant_documents" } }`
  4. Add `evaluation_name` for cleaner reports
  5. Move to `refuse_harmful` - show the negative check with `not_:`
  6. Add a few more cases (list_documents, current_events)
- Code concepts to explain:
  - The nested YAML structure: `query:` → `has_attributes:` → `gen_ai.tool.name:`
  - For negative checks: wrap with `not_:` before `has_attributes:`
  - `evaluation_name` customizes how it appears in the report
  - How to find your agent's tool names (look at `@agent.tool` decorators)
- Skip/minimize:
  - Don't explain every case - pick 3-4 representative ones
  - Don't dwell on YAML indentation details

### Live Demo Part 3: Custom Evaluators (3-4 minutes)
- Files to show:
  - `evals/evaluators.py` - show NoPII and NoForbiddenWords
- Demo flow:
  1. Briefly show the structure: dataclass inheriting from Evaluator
  2. Walk through NoPII: regex patterns for phone, SSN, email
  3. Show how to register in run_evals.py: `custom_evaluator_types=[...]`
  4. Show YAML usage: `- NoPII` or `- NoForbiddenWords: { forbidden: [...] }`
- Key moments to highlight:
  - The `EvaluatorContext` gives you access to output, inputs, metadata, span_tree
  - Custom evaluators are useful when built-ins don't cover your use case
- Skip/minimize:
  - Don't deep-dive into EvaluatorContext internals
  - Keep the regex explanation brief

### Running and Interpretation (4-5 minutes)
- What to show:
  - Run `python evals/run_evals.py`
  - The evaluation report table with HasMatchingSpan columns
  - The summary statistics
- How to interpret:
  - ✅ Green checkmarks mean the span was found (positive) or NOT found (negative)
  - ❌ Red X means the expected condition wasn't met
  - Look at which evaluators failed to understand the issue
- Connect to real-world value:
  - "A HasMatchingSpan failure means your agent might be hallucinating instead of using tools"
  - "A negative check failure means your agent called a dangerous tool on a harmful request"
- Key teaching moment - expected failures:
  - Point out if `refuse_illegal` fails due to keyword mismatch ("cannot" vs "sorry")
  - "This shows why rule-based checks are limited - it sets up what we'll solve in the next video with LLMJudge"

### Wrap-up (2-3 minutes)
- Recap key points:
  - `HasMatchingSpan` verifies tool usage through span inspection
  - Positive checks confirm tools were called; negative checks confirm they weren't
  - Custom evaluators extend the framework for domain-specific needs
- What comes next:
  - "We've covered deterministic checks - but what about subjective quality?"
  - "In the next video, we'll add LLMJudge for nuanced quality assessment"
- Call to action:
  - "Add HasMatchingSpan evaluators to your own golden dataset"
  - "Identify which of your test cases should verify tool usage"
  - "Try creating a custom evaluator for a rule specific to your domain"

---

## Teaching Tips

### Common Misconceptions
1. **"HasMatchingSpan checks the tool's output"** - No, it only verifies the tool was called. Use Contains/LLMJudge to check output content.
2. **"I need a Logfire account"** - No, `send_to_logfire=False` means everything runs locally.
3. **"Negative checks are about checking for negative words"** - No, negative checks verify a tool was NOT called.
4. **"If HasMatchingSpan fails, the evaluator is broken"** - Often it means the agent genuinely didn't call the expected tool.

### Questions Learners Might Ask
- Q: "What if my agent calls a tool with a different name?"
  - A: Check your agent.py for the actual `@agent.tool` function names. Use the exact function name in `has_attributes`.

- Q: "Can I check if a tool was called multiple times?"
  - A: The basic `HasMatchingSpan` checks for presence. For count verification, you'd need a custom evaluator that inspects the span_tree.

- Q: "Why use `has_attributes` instead of `name_contains`?"
  - A: In pydantic-ai, all tool call spans have the same name (`running tool`). The actual tool name is stored in the `gen_ai.tool.name` attribute. Using `has_attributes` lets us match the specific tool.

- Q: "What happens if I forget to configure logfire?"
  - A: All your `HasMatchingSpan` checks will fail because no spans are captured for the evaluator to inspect.

### Pacing Notes
- **Slow down** when explaining the logfire configuration - this is a common stumbling block
- **Speed up** through the YAML syntax after showing 2-3 examples - it's repetitive
- **Pause** after running the evaluation to let learners read the report
- **Emphasize** the expected failures - they're teaching moments, not bugs

---

## Visual Aids

### Diagrams to Include
1. **Span Tree Diagram**: Show how agent execution creates nested spans:
   ```
   Agent Execution
   ├─ [span] agent run
   │   ├─ [span] chat gpt-4o-mini
   │   ├─ [span] running tools
   │   │   └─ [span] running tool  ← attributes: {gen_ai.tool.name: "retrieve_relevant_documents"}
   │   └─ [span] chat gpt-4o-mini
   ```
   Note: Tool name is in the `gen_ai.tool.name` attribute, not the span name!

2. **Positive vs Negative Check Flow**:
   ```
   Positive Check:                    Negative Check:
   ┌─────────────────┐               ┌─────────────────┐
   │ has_attributes: │               │ not_:           │
   │  gen_ai.tool.   │               │   has_attributes│
   │  name: "search" │               │     gen_ai...   │
   └────────┬────────┘               └────────┬────────┘
            │                                 │
            ▼                                 ▼
   Found? → ✅ PASS                  Found? → ❌ FAIL
   Not found? → ❌ FAIL              Not found? → ✅ PASS
   ```

### Screen Recording Notes
- Terminal font size: 16pt minimum (18pt recommended)
- Windows to have open:
  - Editor with `run_evals.py` (left side)
  - Editor with `golden_dataset.yaml` (left side, toggle)
  - Terminal for running commands (right side)
- Browser tabs to prepare:
  - pydantic-evals documentation (for reference if needed)

### B-Roll Suggestions
- Quick cutaway to agent.py showing `@agent.tool` decorators to connect tool names
- Split-screen showing YAML on left, terminal output on right during demo

---

## Checklist Before Recording

- [x] Code is working and tested
- [x] `run_evals.py` has logfire configuration (before agent import)
- [x] `golden_dataset.yaml` has HasMatchingSpan evaluators using `has_attributes`
- [x] `evaluators.py` has NoPII and NoForbiddenWords
- [x] Test data seeded in RAG database (`python evals/seed_test_data.py`)
- [x] Docker services running for agent execution
- [x] Environment variables configured (OpenAI/Anthropic keys)
- [ ] Terminal/editor font size is readable (16pt+)
- [ ] Sample output from eval run saved for reference
- [ ] No sensitive data visible in `.env` file
- [ ] Strategy 1 (Golden Dataset) tag exists: `module-8-01-golden-dataset`
