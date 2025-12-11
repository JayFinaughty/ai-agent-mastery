# Excalidraw Diagram Strategy for Module 8: Agent Evals

> **Purpose:** This document outlines the complete strategy for creating Excalidraw diagrams for all 8 videos in the Agent Evals module of the AI Agent Mastery course.

## Context: What This Module Teaches

Module 8 teaches agent evaluation through **two distinct phases**:

1. **Local Development (Videos 2-4)**: Simple evals with `pydantic-evals`, no external services
2. **Production (Videos 5-8)**: Real user data and observability with Langfuse

**Core Philosophy**: "Start simple. Don't overthink it." — Inspired by Andrew Ng's approach of starting with just a 10-question golden dataset.

## Source Files Reviewed

All files located in: `8_Agent_Evals/evals_strategy_plans/`

### Primary Strategy Documents (8 Videos)

| File | Video | Phase | Key Content |
|------|-------|-------|-------------|
| `01_INTRO_TO_EVALS.md` | 1 | Conceptual | Two-phase framework, lifecycle diagrams |
| `02_GOLDEN_DATASET.md` | 2 | Local | Golden dataset workflow, YAML structure |
| `03_RULE_BASED_LOCAL.md` | 3 | Local | HasMatchingSpan, tool verification, span trees |
| `04_LLM_JUDGE_LOCAL.md` | 4 | Local | LLMJudge evaluator, rubrics, layered approach |
| `05_MANUAL_ANNOTATION.md` | 5 | Production | Langfuse intro, annotation workflow, queues |
| `06_RULE_BASED_PROD.md` | 6 | Production | Reusing evaluators, async execution, score sync |
| `07_LLM_JUDGE_PROD.md` | 7 | Production | Built-in vs custom judges, sampling, cost |
| `08_USER_FEEDBACK.md` | 8 | Production | Two-level feedback, integration pipeline |

### Supporting Documents

| File | Purpose |
|------|---------|
| `00_OVERVIEW.md` | Framework integration matrix, strategy summary |
| `INDEX.md` | Module structure, video order, quick start |
| `META_CONTEXT.md` | Design philosophy, key decisions, codebase context |
| `TAG_STRATEGY.md` | Git workflow, tag naming conventions |

## Key Insights from Analysis

### 1. Two-Phase Architecture

The module follows a clear progression:

```
LOCAL PHASE (No external services)
├── Video 2: Golden Dataset (foundation)
├── Video 3: Rule-Based (deterministic checks)
└── Video 4: LLM Judge (quality assessment)

PRODUCTION PHASE (Langfuse integration)
├── Video 5: Manual Annotation (human-in-the-loop)
├── Video 6: Rule-Based Prod (reuse local evaluators)
├── Video 7: LLM Judge Prod (scale with AI)
└── Video 8: User Feedback (ground truth)
```

### 2. Cumulative Learning Pattern

Each video builds on the previous:
- Video 2: Start with basic evaluators (Contains, IsInstance, MaxDuration)
- Video 3: Add tool verification (HasMatchingSpan)
- Video 4: Add quality assessment (LLMJudge)
- Video 5: Transition to production with Langfuse
- Videos 6-7: Apply local patterns to production
- Video 8: Close the loop with user feedback

### 3. Existing ASCII Diagrams as Blueprints

Each strategy document contains ASCII diagrams that should be translated to visual Excalidraw diagrams:

| Document | ASCII Diagram Location | Content |
|----------|----------------------|---------|
| 01_INTRO | Lines 48-76 | Two-phase framework comparison |
| 01_INTRO | Lines 161-175 | Agent lifecycle: "Does it work?" vs "Is it good?" |
| 02_GOLDEN | Lines 13-28 | Golden dataset workflow |
| 03_RULE_BASED | Lines 14-31 | Rule-based evaluation flow |
| 03_RULE_BASED | Lines 64-70 | OpenTelemetry span tree |
| 04_LLM_JUDGE | Lines 14-33 | LLM judge evaluation process |
| 05_MANUAL | Lines 14-36 | Production annotation workflow |
| 06_RULE_BASED_PROD | Lines 14-29 | Production rule-based flow |
| 07_LLM_JUDGE_PROD | Lines 14-31 | Production LLM judge options |
| 08_USER_FEEDBACK | Lines 12-37 | Two-level feedback system |
| 08_USER_FEEDBACK | Lines 532-559 | Feedback integration pipeline |

### 4. Design Considerations

**User Requirements:**
- Diagrams should NOT be too overbearing or have too much information
- Diagrams SHOULD capture everything in the markdown documents
- Video 01 should have a diagram that connects everything together
- Other videos should focus on their individual strategy

**Visual Consistency:**
- Use color coding: Cool colors (blue/purple) for local, warm colors (green/orange) for production
- Consistent icons: YAML files, terminals, browser windows, tools/wrenches, judge gavels
- Flow direction: Left-to-right for workflows, top-to-bottom for layers
- Keep callout boxes minimal but informative

## Excalidraw Design Strategy: Lessons Learned

> **Context:** This section documents the evolution from initial mistakes to successful design for the Video 01 diagram. These principles apply to all diagrams in this module.

### Initial Mistakes (What NOT to Do)

The first attempt at the Video 01 diagram failed because it was:

1. **Too Basic**: Simple vertical stack of boxes and circles - visually boring
2. **Insufficient Information**: Only titles, no bullet points explaining what each strategy does
3. **No Visual Variety**: Everything looked the same - just boxes in a line
4. **Poor Spacing**: Elements were cramped with no breathing room
5. **Ugly Integration**: Tools listed as gray text at bottom (disconnected from context)
6. **No Flow**: Just a list with no sense of progression or relationship
7. **Not Elegant**: Lacked sophistication - looked like a quick sketch, not a teaching diagram

**User Feedback (verbatim):**
> "This diagram is way too basic, with just boxes and circles; it's very boring. I need you to fundamentally rethink how to make this diagram interesting. Also, there's not enough information here. I want this diagram to have bullet points for each strategy to give substance to what this entails... there's simply not enough information. And there's no flow to this. The ugly gray text at the bottom for the tools doesn't make sense, and the spacing is off... You're not doing a good job here. You need to fundamentally rethink your approach, think deeply, and ensure this diagram is elegant and comprehensive while still being concise."

### What Made the Redesign Successful

#### 1. Information Density: Simple Yet Comprehensive

**Principle:** Each strategy card should explain WHAT it does, not just name it.

**Implementation:**
- 4-6 bullet points per strategy card
- Each bullet point answers: What does this do? When do you use it? What's the benefit?
- Example (Golden Dataset card):
  ```
  • Start with 10 test cases covering core functionality
  • YAML-based, version controlled
  • Fast regression testing (<1 min)
  • Deterministic evaluators: Contains, IsInstance, MaxDuration
  • Foundation for all other eval strategies
  ```

**Balance:** Comprehensive without overwhelming - each point is scannable (5-10 words)

#### 2. Visual Hierarchy: Card-Based Design

**Principle:** Use colored header bars to create visual sections and hierarchy.

**Implementation:**
- **Card structure**: Rounded rectangle with colored header bar + white content area
- **Headers**: Bold white text on colored background (60px height)
- **Content**: Black text on white background with bullets
- **Benefits**:
  - Creates clear visual grouping
  - Color codes phases (blue for local, green for production)
  - Header provides context, content provides detail

**Example:**
```
┌──────────────────────────────┐
│ 🔍 Golden Dataset (blue)     │ ← Colored header with icon
├──────────────────────────────┤
│ • Bullet point 1             │
│ • Bullet point 2             │ ← White content area
│ • Bullet point 3             │
└──────────────────────────────┘
```

#### 3. Color Strategy: Progressive and Purposeful

**Principle:** Colors should create visual variety while maintaining logical grouping.

**Implementation for Video 01:**

**Local Phase (Blues):**
- `#64B5F6` → Light blue (Golden Dataset)
- `#42A5F5` → Medium blue (Rule-Based)
- `#2196F3` → Darker blue (LLM Judge)
- Progression: Gets deeper as strategies get more sophisticated

**Production Phase (Greens):**
- `#66BB6A` → Fresh green (Manual Annotation)
- `#4CAF50` → Standard green (Rule-Based Prod)
- `#81C784` → Sage green (LLM Judge Prod)
- `#A5D6A7` → Soft green (User Feedback)
- Variety: Different shades prevent monotony

**Transition Elements:**
- `#F57C00` → Orange (transition arrow, badges)
- `#9C27B0` → Purple (continuous improvement cycle)

**Benefits:**
- 9 different colors create visual interest
- Blues vs greens instantly show local vs production
- Orange highlights the key transition moment
- Purple shows integration/cycle concept

#### 4. Layout Strategy: Spatial Variety

**Principle:** Use different layout patterns for different sections to create visual interest.

**Implementation:**
- **Left side (Local)**: 3 full-width vertical cards
  - Emphasizes progression: Golden → Rule-Based → LLM Judge
  - Each card same width but flows top to bottom
- **Right side (Production)**: 2×2 grid of smaller cards
  - Shows parallel strategies working together
  - Creates visual balance against left column
- **Center**: Bold transition arrow
  - Connects two phases
  - Orange color draws the eye
- **Bottom**: Full-width cycle box
  - Spans both phases showing integration

**Canvas Size:** 1600px × 830px
- Wide enough for side-by-side phases
- Enough height for all cards plus title and cycle
- Generous spacing between elements

**Spacing:**
- 50px margins on edges
- 30px vertical gap between cards
- 20px padding inside cards
- 350px width for transition arrow (prominent)

#### 5. Question Boxes: Framing the Purpose

**Principle:** Help learners understand the fundamental difference between phases.

**Implementation:**
- Two prominent boxes at top
- **Left box (Blue)**: "Does it work?" with subtitle "Test Core Functionality"
- **Right box (Green)**: "Is it good?" with subtitle "Evaluate Real-World Quality"
- Light background (`#E3F2FD` and `#E8F5E9`) to differentiate from cards
- Positioned above strategy sections to frame the thinking

**Benefits:**
- Immediately communicates the philosophical difference
- Learners understand WHY there are two phases
- Creates narrative structure for the diagram

#### 6. Tool Integration: Badges, Not Text

**Principle:** Show what tools are used without cluttering the diagram.

**Implementation:**
- Small badge elements in phase headers
- "Tools: pydantic-evals" badge in LOCAL DEVELOPMENT header
- "Tools: Langfuse" badge in PRODUCTION header
- Diamond shape (`#FFF3E0` fill, `#F57C00` stroke)
- Compact and integrated, not an afterthought

**Contrast with mistake:**
- ❌ Gray text at bottom listing tools (disconnected)
- ✅ Inline badges showing tool context (integrated)

#### 7. Flow Indicators: Arrows and Progression

**Principle:** Use visual elements to show relationships and movement.

**Implementation:**
- **Bold transition arrow**:
  - 6px stroke width (prominent)
  - Orange color (stands out)
  - Horizontal, pointing right (left to right reading flow)
  - 350px width (substantial presence)
- **"Deploy to Production" badge**:
  - Diamond shape
  - Positioned on arrow
  - Explains the transition
- **Continuous Improvement Cycle box**:
  - Full width at bottom
  - Cross-hatch pattern (different visual texture)
  - 🔄 emoji shows cyclic nature
  - Connects both phases

**Benefits:**
- Shows this isn't just a list - it's a journey
- Transition is explicit and explained
- Cycle shows strategies aren't isolated

#### 8. Shape Variety: Beyond Boxes

**Principle:** Use different shapes to indicate different types of information.

**Shapes used:**
- **Rounded rectangles**: Main strategy cards (headers + content)
- **Rectangles**: Phase headers (LOCAL DEVELOPMENT, PRODUCTION)
- **Diamond**: Tool badges, transition badge
- **Arrow**: Transition between phases
- **Cross-hatched box**: Continuous improvement cycle (different pattern)

**Visual texture:**
- Solid fill: Most elements
- Cross-hatch: Cycle box (indicates integration/different category)
- Stroke weights: 2px for cards, 6px for arrow (hierarchy)

#### 9. Typography: Hierarchy and Readability

**Principle:** Text size and weight should guide the eye.

**Implementation:**
- **Title**: 40px, bold → "Agent Evaluation Framework"
- **Phase headers**: 24px, bold, white on color → "LOCAL DEVELOPMENT"
- **Card headers**: 18px, bold → "🔍 Golden Dataset"
- **Bullet points**: 14px, regular → Content
- **Badges**: 11px → Tool names, transition text
- **Question boxes**: 22px title, 14px subtitle

**Font:** Virgil (Excalidraw default handwritten) - creates approachable, teaching feel

#### 10. Icons/Emojis: Minimal and Purposeful

**Principle:** Use sparingly for visual markers, not decoration.

**Implementation:**
- 🔍 Golden Dataset (search/investigation)
- 📏 Rule-Based (measurement/rules)
- ⚖️ LLM-as-Judge (judgment/evaluation)
- ✍️ Manual Annotation (human input)
- 🤖 Rule-Based (Prod) (automation)
- 🧠 LLM Judge (Prod) (intelligence)
- 💬 User Feedback (communication)
- 🔄 Continuous Improvement Cycle (cycle/iteration)

**Placement:** In card headers, integrated with text, not floating

### Design Process Checklist

Use this checklist when creating any Excalidraw diagram:

**Content:**
- [ ] Does each section have 4-6 bullet points explaining what it does?
- [ ] Is information comprehensive but scannable?
- [ ] Are technical terms explained or contextualized?

**Visual Hierarchy:**
- [ ] Do colored headers create clear sections?
- [ ] Is there a clear title that names the diagram?
- [ ] Do text sizes create hierarchy (title > headers > content)?

**Color:**
- [ ] Do colors group related concepts?
- [ ] Is there enough variety (5+ colors) to avoid monotony?
- [ ] Do colors have semantic meaning (blue = local, green = production)?

**Layout:**
- [ ] Does the layout use multiple patterns (not just boxes in a line)?
- [ ] Is there generous spacing between elements?
- [ ] Does the canvas size accommodate all content without cramping?

**Flow:**
- [ ] Are relationships between elements clear (arrows, positioning)?
- [ ] Is there a sense of progression or narrative?
- [ ] Do elements have clear reading order?

**Details:**
- [ ] Are tools/technologies integrated naturally (badges, not afterthoughts)?
- [ ] Do shapes vary (rectangles, diamonds, arrows, cross-hatch)?
- [ ] Are icons/emojis used purposefully, not decoratively?

**Elegance:**
- [ ] Does the diagram look polished and professional?
- [ ] Is it visually interesting to look at?
- [ ] Would you be proud to show this in a teaching video?

### Key Mantras for Excalidraw Design

1. **"Information-rich, not information-dense"** - Explain everything, but keep it scannable
2. **"Cards, not boxes"** - Use colored headers to create visual hierarchy
3. **"Variety creates interest"** - Mix colors, shapes, layouts, patterns
4. **"Flow, not list"** - Show relationships and progression, not just items
5. **"Integrate, don't append"** - Tool badges, icons, labels should be contextual
6. **"Space is a design element"** - Don't cram; use generous spacing
7. **"Color codes meaning"** - Blues for local, greens for production, orange for transition
8. **"Elegant simplicity"** - Sophisticated but not complex; polished but not overwrought

---

## ⚠️ CRITICAL: The Uniqueness Principle

> **MOST IMPORTANT RULE:** Each of the 8 diagrams MUST be visually distinct and unique. Apply ALL the lessons, but NEVER copy the same structure.

### Why Uniqueness Matters

Each video teaches fundamentally different content:
- **Video 01**: Overview of entire framework (7 strategies)
- **Video 02**: Simple evaluation loop (3 steps)
- **Video 03**: Tool call verification with span trees (hierarchical)
- **Video 04**: LLM judging process with layered approach (stacked layers)
- **Video 05**: Langfuse annotation workflow (UI-focused)
- **Video 06**: Comparison of local vs production (side-by-side)
- **Video 07**: Two paths to LLM judging (decision tree)
- **Video 08**: Two-level feedback system + integration pipeline (dual systems)

**Different content requires different visual structures.** If all diagrams look the same, learners won't distinguish between concepts.

### Apply Lessons, Don't Copy Structure

**✅ DO Apply These Principles to EVERY Diagram:**
1. Information density (4-6 bullets per section)
2. Visual hierarchy (colored headers, clear sections)
3. Color variety (5+ colors with semantic meaning)
4. Generous spacing (don't cram)
5. Flow indicators (arrows, positioning)
6. Shape variety (not just boxes)
7. Typography hierarchy (different sizes)
8. Purposeful icons (integrated, not decorative)
9. Integration (no disconnected elements)
10. Elegance (polished, professional)

**❌ DON'T Copy These Specifics from Video 01:**
1. ❌ Side-by-side "Local vs Production" layout (only Video 01 needs this overview)
2. ❌ 2×2 grid on right side (this was specific to showing 4 production strategies)
3. ❌ 3 full-width cards on left (specific to 3 local strategies)
4. ❌ "Does it work?" vs "Is it good?" question boxes (Video 01 concept)
5. ❌ Big orange transition arrow in center (only relevant for overview)
6. ❌ Continuous improvement cycle box at bottom (Video 01's integration concept)

### How to Create Unique Diagrams

For each video, ask yourself:

1. **What is the PRIMARY concept?**
   - Video 02: Workflow loop (start → execute → result)
   - Video 03: Hierarchy/tree (parent span → child spans)
   - Video 04: Layers/stack (rule → tool → LLM, ordered by cost)
   - Video 05: Interface/dashboard (UI elements, annotation workflow)
   - Video 06: Comparison (local vs production execution)
   - Video 07: Branching paths (built-in vs custom)
   - Video 08: Dual systems (message + conversation) + pipeline

2. **What LAYOUT matches this concept?**
   - Loop → Circular or left-to-right-to-left cycle
   - Hierarchy → Tree structure, vertical with branches
   - Layers → Stacked horizontal bands, pyramid, or funnel
   - Interface → Mockup-style layout with UI components
   - Comparison → Side-by-side panels with differences highlighted
   - Branching → Decision tree, Y-shape, parallel paths
   - Dual systems → Top/bottom split or left/right with integration

3. **What SHAPES best represent this?**
   - Loop: Arrows in a cycle, curved connectors
   - Hierarchy: Tree nodes, nested boxes
   - Layers: Stacked rectangles, funnel shape
   - Interface: Browser window frame, UI cards
   - Comparison: Matching paired boxes with arrows between
   - Branching: Diamond decision points, forked arrows
   - Dual systems: Parallel tracks that merge

4. **What COLORS differentiate sections?**
   - Don't reuse Video 01's exact color scheme
   - Choose colors that make sense for THIS content
   - Example: Video 03 might use green for passing checks, red for violations
   - Example: Video 04 might use gradient from fast/cheap (green) to slow/expensive (red)
   - Example: Video 05 might use Langfuse brand colors

### Examples of Unique Layouts

**Video 01 (Overview):**
```
┌─────────────┐          ┌─────────────┐
│   LOCAL     │  ──────> │ PRODUCTION  │
│ (3 cards)   │          │ (2x2 grid)  │
└─────────────┘          └─────────────┘
```

**Video 02 (Loop) - DIFFERENT:**
```
YAML ──> Execution ──> Results
         ↑                  │
         └──────────────────┘
         (Horizontal loop with curved return)
```

**Video 03 (Hierarchy) - DIFFERENT:**
```
        Agent Span
       /    |    \
   Chat   Tool   Chat
          /  \
    Span1  Span2
    (Tree structure, vertical)
```

**Video 04 (Layers) - DIFFERENT:**
```
╔══════════════════╗ Layer 1: Rules (fast)
╠══════════════════╣ Layer 2: Tools (free)
╠══════════════════╣ Layer 3: LLM ($$)
╚══════════════════╝
(Stacked layers, fail-fast top to bottom)
```

**Video 05 (Dashboard) - DIFFERENT:**
```
┌─────────────────────────────┐
│ [Langfuse UI Frame]         │
│ ┌──────┐ ┌──────┐ ┌──────┐ │
│ │Trace │ │Score │ │Queue │ │
│ └──────┘ └──────┘ └──────┘ │
└─────────────────────────────┘
(UI mockup style)
```

**Video 06 (Comparison) - DIFFERENT:**
```
Local:                Production:
┌──────────┐         ┌──────────┐
│ Sync     │    vs   │ Async    │
│ Batch    │         │ Single   │
│ Terminal │         │ Langfuse │
└──────────┘         └──────────┘
(Mirror comparison, NOT the overview structure)
```

**Video 07 (Branching) - DIFFERENT:**
```
         Production LLM Judging
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Built-in          Custom
   (No code)         (Full control)

(Y-shaped decision tree)
```

**Video 08 (Dual + Pipeline) - DIFFERENT:**
```
┌────────────────┐ Message Feedback
│                │
├────────────────┤ Conversation Rating
│                │
└────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Integration Pipeline (flow)   │
└───────────────────────────────┘
(Two sections stacked, then flow)
```

### Uniqueness Checklist

Before finalizing any diagram, verify:

- [ ] **Layout is different** from all previous diagrams (not just cards in same arrangement)
- [ ] **Primary visual metaphor matches the concept** (loop, tree, layers, UI, comparison, branches, etc.)
- [ ] **Color scheme is appropriate** for THIS video (not copied from Video 01)
- [ ] **Shapes are varied** and serve the content (not using same shapes as Video 01)
- [ ] **Flow direction makes sense** for THIS content (not forced to match Video 01)
- [ ] **All 10 design principles are applied** (information density, hierarchy, colors, spacing, flow, shapes, typography, icons, integration, elegance)
- [ ] **A learner could identify which video this is** just by looking at the diagram structure

### The Balance

**Apply the principles universally:**
- Every diagram should be information-rich
- Every diagram should have visual hierarchy
- Every diagram should use color variety
- Every diagram should have good spacing
- Every diagram should show flow
- Every diagram should mix shapes
- Every diagram should have typography hierarchy
- Every diagram should use icons purposefully
- Every diagram should integrate elements naturally
- Every diagram should look elegant

**But vary the implementation:**
- Video 01: Side-by-side overview with different layouts per side
- Video 02: Horizontal workflow loop
- Video 03: Vertical tree hierarchy
- Video 04: Stacked layers with fail-fast progression
- Video 05: UI mockup with dashboard elements
- Video 06: Side-by-side comparison (different from Video 01's overview)
- Video 07: Branching decision tree
- Video 08: Dual systems + pipeline flow

**Remember:** The lessons are the "how to make it good." The uniqueness is the "what makes it right for THIS content."

---

## Diagram Specifications by Video

### Video 01: Introduction to Agent Evals

**Purpose:** Overview diagram that connects everything together

**Primary Diagram: "The Complete Agent Evaluation Framework"**

**Visual Structure:**
- Split screen: Local Development (left) vs Production (right)
- Local phase (blue/purple):
  - Golden Dataset (10 test cases)
  - Rule-Based (tool verification)
  - LLM Judge (quality scoring)
  - Tools: pydantic-evals, terminal reports
  - Question: "Does it work?"
  - Callout: "No external services needed"
- Production phase (green/orange):
  - Manual Annotation (expert review)
  - Rule-Based Prod (automated monitoring)
  - LLM Judge Prod (scale with AI)
  - User Feedback (real user satisfaction)
  - Tools: Langfuse, dashboards, traces
  - Question: "Is it good?"
  - Callout: "Real user data"
- Center arrow showing progression from local → production
- Small icons for tools (pydantic logo, Langfuse logo)

**Secondary Diagram (Optional): "Agent Lifecycle"**
- Timeline showing when to use each strategy
- Before deployment vs After deployment
- Regression testing vs Quality monitoring

**Source Inspiration:**
- ASCII diagram: `01_INTRO_TO_EVALS.md` lines 48-76
- Lifecycle diagram: `01_INTRO_TO_EVALS.md` lines 161-175
- Framework overview: `00_OVERVIEW.md` lines 13-50

---

### Video 02: Golden Dataset

**Purpose:** Show the foundation - simple local evaluation loop

**Primary Diagram: "Golden Dataset Evaluation Loop"**

**Visual Structure:**
1. **Left box**: YAML file icon
   - Label: "golden_dataset.yaml"
   - Content preview: "10 test cases"
   - Categories listed: General (3), RAG (2), Calculations (2), Safety (2), Web (1)
2. **Center box**: Execution
   - Script: "run_evals.py"
   - Agent icon running
   - Evaluators: Contains, IsInstance, MaxDuration
3. **Right box**: Terminal output
   - Report visualization
   - "✅ 8/10 passed"
   - Pass rate: 80%
4. **Callouts**:
   - "No external services"
   - "Fast, free, reproducible"
   - "Deterministic evaluators only"

**Key Message:** Keep it minimal - this is the starting point

**Source Inspiration:**
- Workflow diagram: `02_GOLDEN_DATASET.md` lines 13-28
- Evaluator table: `02_GOLDEN_DATASET.md` lines 217-227

---

### Video 03: Rule-Based Local Evaluation

**Purpose:** Show how tool call verification works with HasMatchingSpan

**Primary Diagram: "Tool Call Verification with HasMatchingSpan"**

**Visual Structure:**
1. **Top**: User query input
2. **Middle-Left**: Agent execution
   - Show agent running
   - Tool calls being made
   - OpenTelemetry spans generated
3. **Middle-Center**: Span Tree Visualization
   - Agent run span (parent)
   - Chat spans
   - Tool execution spans (children)
   - Highlight: Tool name stored in `gen_ai.tool.name` attribute
4. **Middle-Right**: HasMatchingSpan Evaluator
   - Inspecting span tree
   - Two verification types shown:
     - ✅ Positive: "Tool MUST be called" (green check)
     - ❌ Negative: "Tool must NOT be called" (red cross)
5. **Bottom**: Results
   - Pass/fail with reasons
   - Example: "✅ retrieve_relevant_documents called"
   - Example: "✅ execute_code NOT called (safety check)"

**Callouts:**
- "OpenTelemetry spans capture tool calls"
- "Deterministic: yes/no answers"
- "Verifies behavior, not just output"

**Source Inspiration:**
- Workflow: `03_RULE_BASED_LOCAL.md` lines 14-31
- Span tree: `03_RULE_BASED_LOCAL.md` lines 64-70
- Patterns: `03_RULE_BASED_LOCAL.md` lines 99-144

---

### Video 04: LLM-as-Judge (Local)

**Purpose:** Show AI-powered quality assessment with layered approach

**Primary Diagram: "LLM Judge Evaluation Process"**

**Visual Structure:**
1. **Input Layer** (top):
   - User Query
   - Agent Response
   - Rubric (multi-point criteria)
2. **Judge Processing** (center):
   - GPT-5-mini icon
   - "Thinking" bubble showing chain-of-thought
   - Model considering: Relevance, Helpfulness, Accuracy
3. **Output Layer** (bottom):
   - Score: 0.0-1.0 (numeric display)
   - Reason: Brief explanation (1-2 sentences)
   - Pass/Fail indicator (based on threshold)
4. **Layered Approach** (right side, vertical stack):
   - Layer 1: Rule-based ⚡ (fast, free, instant)
   - Layer 2: Tool verification 🔧 (fast, free, spans)
   - Layer 3: LLM Judge 💰 (slower, costs, quality)
   - Arrow showing "Fail-fast: if rules fail, skip expensive LLM calls"

**Cost Comparison Box** (bottom):
- Table showing gpt-5-nano (~$0.0005), gpt-5-mini (~$0.002), gpt-5 (~$0.01)
- Daily cost estimates for 10-case dataset

**Callouts:**
- "Subjective quality needs AI judgment"
- "Order matters: cheap checks first"
- "Same model as production (gpt-5-mini)"

**Source Inspiration:**
- Workflow: `04_LLM_JUDGE_LOCAL.md` lines 14-33
- Layered approach: `04_LLM_JUDGE_LOCAL.md` lines 507-522
- Cost table: `04_LLM_JUDGE_LOCAL.md` lines 602-614

---

### Video 05: Manual Annotation

**Purpose:** Transition to production - show Langfuse annotation workflow

**Primary Diagram: "Production Annotation Workflow"**

**Visual Structure:**
1. **Left**: Production traces
   - Real user interactions flowing from agent API
   - 1000s of traces per day
   - Icons showing chat messages
2. **Center**: Langfuse Platform (introduce branding)
   - Traces view
   - Annotation Queue
   - Score Configurations (Accuracy 1-5, Helpfulness 1-5, Safety Pass/Fail)
   - Browser UI showing annotator interface
3. **Right**: Scores Attached to Traces
   - Scores synced back to traces
   - Dashboard view with filtering
4. **Bottom**: Use Cases (3 boxes)
   - Calibrate LLM judges (compare expert vs AI scores)
   - Export to golden dataset (high-quality examples)
   - Build training data (fine-tuning)

**Callouts:**
- "First production video - intro to Langfuse for evals"
- "No custom UI needed - Langfuse handles everything"
- "Human experts provide ground truth"
- "Sample 1-10% of traces"

**Visual Style:** Introduce Langfuse colors/branding here

**Source Inspiration:**
- Workflow: `05_MANUAL_ANNOTATION.md` lines 14-36
- Score configs: `05_MANUAL_ANNOTATION.md` lines 75-109
- Use cases: `05_MANUAL_ANNOTATION.md` lines 222-350

---

### Video 06: Rule-Based Production Evaluation

**Purpose:** Show reusing local evaluators in production context

**Primary Diagram: "Same Evaluators, Different Context"**

**Visual Structure:**
- **Side-by-side comparison:**

**Left Panel: "Video 3 - Local"**
- Golden dataset (batch of 10)
- Sync execution (blocking)
- Terminal report
- `dataset.evaluate_sync()`

**Center Panel: "Shared Evaluators"**
- Same `NoPII` evaluator
- Same `NoForbiddenWords` evaluator
- Same `HasMatchingSpan` logic
- Arrow pointing both directions: "Reuse!"

**Right Panel: "Video 6 - Production"**
- Production traces (single request)
- Async execution (fire-and-forget)
- Langfuse scores
- `asyncio.create_task()`

**Bottom Flow:**
1. User request → Agent response (user gets response immediately ⚡)
2. Async eval task (runs in background)
3. Langfuse score sync (rule_no_pii, rule_no_forbidden, rule_check_passed)
4. Dashboard filtering (find violations)

**Callouts:**
- "Same evaluators, different execution model"
- "Non-blocking: users don't wait"
- "Monitor violations in Langfuse dashboard"

**Source Inspiration:**
- Workflow: `06_RULE_BASED_PROD.md` lines 14-29
- Comparison: `06_RULE_BASED_PROD.md` lines 38-46
- Integration: `06_RULE_BASED_PROD.md` lines 183-208

---

### Video 07: LLM-as-Judge (Production)

**Purpose:** Show two paths to production LLM judging with cost control

**Primary Diagram: "Two Paths to Production LLM Judging"**

**Visual Structure:**
- **Split diagram showing two options:**

**Path 1 (Top Half): "Langfuse Built-in Evaluators"**
1. Langfuse UI (no code needed)
2. Template selection: Helpfulness, Relevance, Toxicity, Correctness, Conciseness, Hallucination
3. Configuration: Model (GPT-5-mini), Sample rate (10%), Variable mapping
4. Auto-execution: Langfuse handles everything
5. Scores in dashboard

**Path 2 (Bottom Half): "Custom pydantic-ai Judge"**
1. Custom judge code (`prod_judge.py`)
2. Domain-specific rubric (full control)
3. Multi-dimension scoring (relevance, helpfulness, accuracy)
4. Async execution (same pattern as Video 6)
5. Score sync to Langfuse

**Shared Elements (Center):**
- Sampling: 10% of requests (cost control)
- Model: GPT-5-mini (~$0.002/eval)
- Scores: llm_judge_score (0.0-1.0), llm_judge_passed (0 or 1)
- Dashboard: Trends, filtering, debugging

**Cost Management Box (Right):**
- Daily cost at different sample rates:
  - 100%: ~$20/day (1000 traces)
  - 10%: ~$2/day (recommended)
  - 1%: ~$0.20/day (budget)

**Callouts:**
- "Start with built-in, customize when needed"
- "Smart sampling: evaluate failures & complex traces more"
- "Compare to human annotations (Video 5) for calibration"

**Source Inspiration:**
- Two paths: `07_LLM_JUDGE_PROD.md` lines 55-105 (built-in) and 107-315 (custom)
- Cost management: `07_LLM_JUDGE_PROD.md` lines 359-417
- Sampling: `07_LLM_JUDGE_PROD.md` lines 385-417

---

### Video 08: User Feedback Collection

**Purpose:** Show two-level feedback system and integration with all strategies

**Primary Diagram: "Two-Level Feedback System"**

**Visual Structure:**
- **Left and right split:**

**Left Panel: "Level 1 - Message Feedback"**
1. Agent message bubble (chat UI)
2. Hover interaction (opacity transition)
3. Thumbs up 👍 / Thumbs down 👎 buttons
4. Score: `message_feedback` (1 or 0)
5. Question: "Was this answer good?"
6. Granular: Per-response feedback

**Right Panel: "Level 2 - Conversation Rating"**
1. Periodic popup (after 5, 10, 15 responses)
2. 4-option rating:
   - 😊 Very good (1.0)
   - 🙂 Good (0.67)
   - 😕 Not so good (0.33)
   - 😞 Bad (0.0)
3. Optional comment box (shown if "Bad" selected)
4. Score: `conversation_rating` (0.0-1.0)
5. Question: "Is this agent helping me accomplish my goal?"
6. Holistic: Overall conversation quality

**Bottom**: Both flow to Langfuse scores

**Secondary Diagram: "Feedback Integration Pipeline"**

**Visual Structure:**
1. **Top**: User Feedback collected
2. **Branches**:
   - Negative feedback → LLM Judge triage → Priority score
   - Positive feedback → Export to Golden Dataset (Video 2)
   - Both negative (user + judge) → Manual Annotation Queue (Video 5)
3. **Calibration loop**:
   - Compare User feedback vs LLM Judge scores (Video 7)
   - Disagreements → Tune rubrics
   - Agreement rate tracking
4. **Correlation**:
   - User feedback vs Rule violations (Video 6)
   - Identify which rules predict user dissatisfaction

**Callouts:**
- "Users are the ultimate judges"
- "Two levels: response quality vs conversation quality"
- "Close the loop: feedback improves all eval strategies"
- "Low response rate (1-5%) - quality over quantity"

**Source Inspiration:**
- Two-level system: `08_USER_FEEDBACK.md` lines 12-37
- Integration pipeline: `08_USER_FEEDBACK.md` lines 532-559
- Component details: `08_USER_FEEDBACK.md` lines 154-389

---

## Visual Design Guidelines

### Color Palette

**Local Phase (Cool colors):**
- Primary: #4A90E2 (blue)
- Secondary: #7B68EE (purple)
- Accent: #5DADE2 (light blue)

**Production Phase (Warm colors):**
- Primary: #2ECC71 (green)
- Secondary: #F39C12 (orange)
- Accent: #E67E22 (dark orange)

**Neutral elements:**
- Background: #FFFFFF (white)
- Text: #2C3E50 (dark gray)
- Borders: #BDC3C7 (light gray)

### Icon Library

| Element | Icon/Shape | Description |
|---------|-----------|-------------|
| YAML file | Document with lines | Golden dataset |
| Terminal | Window with `>_` | Local execution |
| Browser | Window with URL bar | Langfuse UI |
| Tool/Wrench | 🔧 | Tool verification |
| Judge gavel | ⚖️ | LLM judge |
| Thumbs | 👍👎 | User feedback |
| Agent | 🤖 | AI agent execution |
| User | 👤 | User interaction |
| Check mark | ✅ | Pass/success |
| Cross | ❌ | Fail/violation |

### Typography

- **Headings**: Bold, 16-18pt
- **Body text**: Regular, 12-14pt
- **Callouts**: Italic or colored background box, 10-12pt
- **Code**: Monospace font for filenames and function names

### Layout Principles

1. **Left-to-right flow** for sequential processes
2. **Top-to-bottom flow** for hierarchies and layers
3. **Side-by-side comparison** for contrasts (local vs prod)
4. **Callout boxes** positioned near relevant elements, not cluttering main flow
5. **White space** liberally - don't cram information
6. **Max 3-4 colors per diagram** to avoid overwhelming
7. **Consistent spacing** between elements (use grid)

---

## Implementation Priority

Suggested order for creating diagrams:

1. **Video 01** (Overview) - Most important, sets the stage for everything
2. **Video 02** (Golden Dataset) - Simplest, establishes the visual pattern
3. **Video 05** (Manual Annotation) - Transition to production, introduces Langfuse
4. **Video 08** (User Feedback) - Shows complete system, integration with all strategies
5. **Videos 03, 04, 06, 07** - Fill in the middle strategies

---

## Success Criteria

A diagram is successful if:

- ✅ A learner can understand the core concept without reading the text
- ✅ It accurately represents the content in the markdown document
- ✅ It's not overwhelming (max 5-7 key elements)
- ✅ It uses consistent visual language with other diagrams in the module
- ✅ It highlights the key insight or "aha moment" for that video
- ✅ It can be explained in 30-60 seconds during video recording

---

## Next Steps

1. Review this strategy document
2. Create diagrams in Excalidraw following the specifications above
3. Export diagrams as PNG/SVG for inclusion in video recordings
4. Link diagram files in the respective strategy documents
5. Iterate based on instructor feedback during video recording

---

## Appendix: Quick Reference

### File Paths
- Strategy docs: `8_Agent_Evals/evals_strategy_plans/[01-08]_*.md`
- Diagrams: `8_Agent_Evals/evals_strategy_plans/diagrams/` (to be created)

### Key Quotes for Diagrams

- "Start simple. Don't overthink it." - Andrew Ng principle
- "Does it work?" (Local) vs "Is it good?" (Production)
- "No external services needed" (Local phase)
- "Reuse, don't rebuild" (Production phase)
- "Users are the ultimate judges" (User feedback)
