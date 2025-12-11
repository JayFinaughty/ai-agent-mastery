# Video Preparation: User Feedback Collection

> Source: `08_USER_FEEDBACK.md` | Phase: Production | Video 8

---

## Meta Information

### Why This Video Exists

Learners have built automated evaluation systems (rules, LLM judge), but these are proxies for what really matters: user satisfaction. This final video closes the loop by showing how to collect direct user feedback at two levels — per-message and per-conversation — and sync it to Langfuse for correlation with automated scores.

### Target Learner Profile

Someone who has:
- Completed Videos 1-7 (the full evaluation journey)
- Implemented Langfuse score sync (Videos 5-7)
- Deployed an agent with real user interactions
- Understands the distinction between automated and human evaluation

### Learning Outcomes

By the end of this video, learners will be able to:
1. Implement message-level feedback (thumbs up/down) with Langfuse integration
2. Build periodic conversation rating popups at configurable intervals
3. Connect trace IDs from backend to frontend for feedback attribution
4. Correlate user feedback with automated evaluation scores

### Key Takeaway (One Sentence)

User feedback is the ground truth that validates your automated evaluations — collect it at multiple granularities and use it to calibrate your LLM judges.

---

## Narrative Arc

### The Hook (Opening)

"You've built rules that catch PII. You've trained an LLM judge to score quality. But there's one question they can't answer: Is the user actually happy? Today we close the loop with the ultimate evaluation signal — direct user feedback."

### The Problem

Automated evaluation is a proxy. Rules catch format violations. LLM judges approximate quality. But neither knows if:
- The user found the response helpful for their actual task
- The conversation as a whole is productive
- There are subtle issues that automated systems miss

### The Solution (Conceptual)

Two-level feedback system:
1. **Message-level**: Quick thumbs up/down on individual responses (captures "was this answer good?")
2. **Conversation-level**: Periodic rating popups (captures "is this agent helping me accomplish my goal?")

Both sync to Langfuse as scores, enabling correlation with automated evaluations.

### The "Aha" Moment

When learners filter Langfuse for traces where `message_feedback = 0` AND `llm_judge_passed = 1` — cases where the automated judge passed but the user disagreed. These are calibration opportunities for improving rubrics.

---

## Content Outline

### Introduction (2-3 minutes)

- Key talking points:
  - "We've automated evaluation. Now let's validate it with the only ground truth that matters — users."
  - Two feedback levels serve different purposes: granular vs. holistic
  - Preview: thumbs widget, rating popup, Langfuse scores

### Concept Explanation (2-3 minutes)

- Core concepts:
  - **Why two levels?** A user might thumbs-up individual responses but rate the overall conversation poorly (or vice versa)
  - **Trace ID plumbing**: Backend must return trace_id so frontend can attach scores to the right Langfuse trace
  - **Configurable intervals**: Rating thresholds are environment-configurable for experimentation

### Live Demo (6-8 minutes)

- Demo flow:
  1. Show the chat UI — hover over an AI message, point out thumbs buttons
  2. Click thumbs up, show the "Thanks!" confirmation
  3. Open Langfuse — find the trace, show `message_feedback = 1` score
  4. Send more messages (5 total) to trigger the rating popup
  5. Select "Good", show it dismisses
  6. Back to Langfuse — show `conversation_rating = 0.67` score
  7. Trigger another popup, select "Bad", show comment box
  8. Submit with comment, show the comment in Langfuse

### Code Walkthrough (4-5 minutes)

- Files to show:
  - `agent_api.py` — trace_id in completion chunk
  - `langfuse.ts` — client singleton and `submitScore()` helper
  - `FeedbackWidget.tsx` — thumbs component
  - `ConversationRating.tsx` — rating popup
  - `useConversationRating.ts` — threshold logic

### Results & Interpretation (2-3 minutes)

- What to show:
  - Langfuse Traces view filtered by user feedback scores
  - Dashboard showing distribution of ratings over time
  - Example trace with both automated and user scores

- How to interpret:
  - High agreement (user positive + judge positive) = system working
  - Disagreement (user negative + judge positive) = calibration opportunity
  - Trending negative ratings = investigate root cause

- Connect to real-world value:
  - User feedback creates a continuous improvement loop
  - Negative feedback prioritizes what to fix next
  - Positive feedback validates your automated systems

### Wrap-up (2 minutes)

- This completes the evaluation toolkit: golden dataset, rules, LLM judge, manual annotation, and now user feedback
- Feedback loop: use negative feedback to improve agent, calibrate judges, build golden dataset
- Call to action: Deploy feedback to your own agent and start collecting ground truth

---

## Teaching Tips

### Common Misconceptions

1. **"User feedback replaces automated evals"**
   - Reality: They complement each other. Automated evals catch issues 24/7; user feedback validates them.

2. **"Low response rate means feedback is useless"**
   - Reality: Even 1-5% response rate provides valuable signal. Extremes (very happy/unhappy) respond more.

3. **"Thumbs up/down is too simple"**
   - Reality: Simplicity increases response rate. Complex surveys get ignored.

4. **"Conversation rating and message feedback measure the same thing"**
   - Reality: A user can thumbs-up individual responses but rate the conversation poorly if it didn't solve their problem.

### Questions Learners Might Ask

- Q: "Why use Langfuse for feedback instead of our own database?"
  - A: Langfuse correlates feedback with traces and other scores automatically. You get dashboards, filtering, and analytics for free.

- Q: "What if users never click the feedback buttons?"
  - A: Low response rate is normal. Focus on the signal you get. Consider A/B testing button placement or prompts.

- Q: "How do I know if my LLM judge is calibrated correctly?"
  - A: Compare judge scores to user feedback. If they disagree often, revisit your rubric.

- Q: "Should I store feedback in both Langfuse and my database?"
  - A: Langfuse is the source of truth for evaluation. You can export if needed, but avoid duplication.

### Pacing Notes

- **Slow down**: When explaining the trace_id plumbing (backend → frontend → Langfuse)
- **Speed up**: Environment variable setup (familiar pattern from earlier videos)
- **Emphasize**: The "aha" moment when filtering for user-judge disagreements

---

## Visual Aids

### Diagrams to Include

1. **Two-Level Feedback Flow** (from strategy doc)
   - Shows message-level vs conversation-level side by side
   - Already in `08_USER_FEEDBACK.md`

2. **Trace ID Flow Diagram**
   ```
   Backend (span) → completion chunk → frontend → Langfuse score
                         ↓
                    trace_id: "abc123"
   ```

3. **Feedback Loop Diagram**
   ```
   User Feedback → Identify Issues → Update Agent → Better Responses → Repeat
        ↓
   Calibrate LLM Judge
        ↓
   Build Golden Dataset
   ```

### Screen Recording Notes

- Terminal/editor font size: 16pt minimum
- Windows to have open:
  - Browser with chat UI (main demo)
  - Langfuse dashboard (Traces view, pre-filtered to show scores)
  - VS Code with relevant files open in tabs
- Browser tabs to prepare:
  - Chat UI (localhost:8082)
  - Langfuse Traces page (filtered by your test user)
- Browser DevTools: Open but minimized (for debugging if needed)

### B-Roll Suggestions

- Close-up of the thumbs up/down hover animation
- Rating popup slide-in animation
- Langfuse scores appearing in the trace detail view
- Dashboard chart showing feedback distribution

---

## Quick Validation Checklist

### Pre-Recording Setup

- [ ] Frontend `.env` has `VITE_LANGFUSE_PUBLIC_KEY` and `VITE_LANGFUSE_HOST` configured
- [ ] Backend is running with Langfuse configured (traces being created)
- [ ] Langfuse dashboard is open and accessible
- [ ] Clear any test conversations for a clean demo

### Verify Message Feedback (Thumbs Up/Down)

1. Send a message and wait for AI response
2. Hover over the AI response bubble
3. Confirm thumbs up/down buttons appear (opacity transition)
4. Click thumbs up
5. Confirm "Thanks!" text replaces the buttons
6. In Langfuse: Navigate to **Traces** → Find the most recent trace → **Scores** tab
7. Confirm `message_feedback` score with value `1` appears

### Verify Conversation Rating Popup

1. Send 5 messages to the agent (to trigger first threshold at 5 AI responses)
2. After 5th AI response, confirm rating popup appears in bottom-right corner
3. Click "Good" option
4. Confirm popup closes immediately
5. In Langfuse: Find the trace → **Scores** tab
6. Confirm `conversation_rating` score with value `0.67` appears

### Verify "Bad" Rating with Comment

1. Continue conversation to 10 messages (or set `VITE_RATING_THRESHOLDS=2,4,6` for faster testing)
2. When rating popup appears, click "Bad"
3. Confirm comment textarea appears with "Sorry to hear that. What went wrong?" prompt
4. Enter a comment (e.g., "Response was too vague")
5. Click "Submit"
6. In Langfuse: Find the trace → Score has the comment attached

### Verify Dismiss Behavior

1. Trigger a rating popup
2. Click the X button to dismiss
3. Confirm popup closes
4. Continue conversation — popup should NOT reappear at the same threshold

### Verify Graceful Degradation

1. Remove `VITE_LANGFUSE_PUBLIC_KEY` from frontend `.env`
2. Restart frontend
3. Confirm chat still works normally
4. Confirm thumbs buttons do NOT appear (graceful degradation when unconfigured)
5. No console errors blocking the UI

### Langfuse Filtering Demo

1. In Langfuse → **Traces**, use filter: `Scores > message_feedback = 0`
   - Shows negatively-rated responses for review
2. Use filter: `Scores > conversation_rating < 0.5`
   - Shows poor conversations (Bad or Not so good)
3. Advanced: Filter for `message_feedback = 0 AND llm_judge_passed = 1`
   - Shows disagreements between user and automated judge (calibration opportunities)

---

## Sample Test Queries for Demo

**Quick wins (users likely happy):**
- "What is Python?"
- "How do I create a list in JavaScript?"

**Potential frustration (might rate poorly):**
- Vague question: "Help me" (agent might not know what to help with)
- Off-topic: "What's the weather?" (if agent isn't equipped for this)

**For "Bad" rating demo:**
- Intentionally ask something complex, then rate it as Bad with comment "Response didn't address my specific situation"

---

## Environment Variables Reference

```bash
# Frontend .env
VITE_LANGFUSE_PUBLIC_KEY=pk-lf-...  # Same as backend's LANGFUSE_PUBLIC_KEY
VITE_LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Custom thresholds (default: 5,10,15,20,25,30)
VITE_RATING_THRESHOLDS=5,10,15,20,25,30
```

---

## Checklist Before Recording

- [ ] `FeedbackWidget.tsx` exists and renders correctly
- [ ] `ConversationRating.tsx` exists and renders correctly
- [ ] `useConversationRating.ts` hook triggers at correct thresholds
- [ ] Backend returns `trace_id` in completion chunks
- [ ] Frontend stores `trace_id` on AI messages
- [ ] Langfuse receives `message_feedback` scores
- [ ] Langfuse receives `conversation_rating` scores with comments
- [ ] Rating popup is dismissable and doesn't reappear at same threshold
- [ ] Prepare 2-3 test conversations of varying quality
- [ ] Clear browser history for clean UI demo
