# Video & Demo Guide: Strategy 1 - User Feedback (Explicit)

## Video Overview

- **Duration**: ~8-10 minutes
- **Key Takeaway**: User feedback is the ground truth for agent quality - automated evals can only approximate what real users actually experience
- **Prerequisites**: Viewers should understand basic agent architecture, have seen the Langfuse/tracing setup from earlier modules

---

## Opening (1-2 min)

### Hook

> "How do you know if your AI agent is actually helping users? You can run automated tests, use LLM judges, and analyze traces... but at the end of the day, the only people who truly know if your agent is useful are the users themselves."

### Problem Statement

- Automated evals can miss subjective quality issues
- Users may have different expectations than what you test for
- You need a feedback loop to continuously improve
- Real user opinions are the ultimate source of truth

### What We'll Build

- Thumbs up/down per-message feedback
- Session-level satisfaction ratings
- Category tagging for negative feedback (to understand *why* things fail)
- Langfuse integration to correlate user feedback with traces

---

## Architecture Overview (1-2 min)

### Show the Diagram

```
User clicks 👍/👎 → Frontend Widget → Backend API → Database
                                         ↓
                                    Langfuse Score
```

### Key Design Decisions

1. **Two feedback levels**: Per-message (specific) vs session-wide (holistic)
2. **Categories for both positive and negative**: Understand what works AND what fails
3. **Non-blocking**: Feedback is optional, doesn't interrupt the conversation
4. **Langfuse sync**: Enables correlation with other metrics

---

## Live Demo (4-5 min)

### Demo Flow

#### 1. Start a Conversation (1 min)

- Open the chat interface
- Ask a question that will get a good response
- Point out: "Notice there's no feedback UI yet - it appears on hover"

#### 2. Thumbs Up Flow (1 min)

- Hover over the AI response
- Show the thumbs up/down icons appearing
- Click thumbs up
- Point out the toast notification
- "That's it - one click feedback. Low friction is key for adoption."

#### 3. Thumbs Down with Category (1.5 min)

- Ask a question that might get an imperfect response
- Hover and click thumbs down
- Show the category selection expanding
- Explain: "When users are unhappy, we want to know WHY"
- Select a category, add a comment
- Submit and show the confirmation

#### 4. Session Feedback (1 min)

- Show a conversation with 5+ messages
- Point out the session feedback banner
- "For longer conversations, we ask about the overall experience"
- Show the positive category options

#### 5. Verify in Database (0.5 min)

- Quick cut to Supabase Table Editor
- Show the feedback records
- Point out the fields: rating_value, category, comment

#### 6. Verify in Langfuse (1 min)

- Open Langfuse dashboard
- Find the trace
- Show the `user_feedback` score attached
- "Now this user feedback is correlated with all our other metrics"

---

## Code Walkthrough (2-3 min)

### Files to Show

#### 1. `feedback_api.py` (1 min)

- Show the `FeedbackRequest` model with categories
- Highlight the validation logic (thumbs: 0/1, category matches sentiment)
- Show the Langfuse sync code:

```python
langfuse_client.score(
    trace_id=feedback.request_id,
    name="user_feedback",
    value=score_value,
    ...
)
```

#### 2. `FeedbackWidget.tsx` (1 min)

- Show the hover-to-reveal pattern
- Highlight the category selection UI
- Point out the request_id correlation

#### 3. `10-user_feedback.sql` (30 sec)

- Show the schema briefly
- Highlight RLS policies: "Users can only see their own feedback, but admins can see all for analytics"

### Key Points to Emphasize

- Request ID is the glue between frontend, backend, and Langfuse
- Feedback is immutable (audit trail)
- Low friction design (one click for positive, details only for negative)

---

## Results & Metrics (1 min)

### What You Can Now Measure

- Overall satisfaction rate (% thumbs up)
- Common issue categories
- Correlation with automated evals
- Trends over time

### Dashboard Ideas

- Show the `/api/feedback/stats` endpoint response
- Explain how to build a feedback dashboard

---

## Wrap-up (1 min)

### Summary

> "User feedback is Strategy 1 because it's the foundation of truth. Every other evaluation strategy we build should ultimately correlate with user satisfaction."

### What's Next

> "But users only provide feedback on 1-5% of interactions. In the next video, we'll look at implicit feedback - behavioral signals that tell us about user satisfaction without asking them directly."

### Call to Action

- Encourage viewers to implement feedback in their own agents
- Low effort, high value

---

## B-Roll Suggestions

1. **Diagram**: User feedback flow (frontend → backend → DB → Langfuse)
2. **Screenshot**: Supabase table with feedback records
3. **Screenshot**: Langfuse trace with user_feedback score
4. **Code highlight**: The Langfuse sync code
5. **UI recording**: Hover-to-reveal animation on feedback widget

---

## Common Questions to Address

### "Why not just use star ratings?"

- Thumbs are faster (lower friction = higher response rate)
- Stars have interpretation variance (what does 3 stars mean?)
- Categories give more actionable info than a number

### "How do you get users to actually leave feedback?"

- Keep it low friction (one click)
- Show appreciation (toast message)
- Don't ask every time (session feedback only after 5+ messages)

### "How do you handle fake/spam feedback?"

- RLS ensures users can only submit their own
- Link to user_id for accountability
- Can add rate limiting if needed

### "Should feedback block the response?"

- No - async by design
- User experience comes first

---

## Technical Notes for Recording

- **Environment**: Use a clean conversation state for demos
- **Langfuse**: Ensure credentials are configured for live demo
- **Browser**: Use incognito/clean profile to avoid cached state
- **Screen Resolution**: 1920x1080 recommended for code readability

---

## Manual Testing Checklist

Use this checklist before recording to ensure everything works:

### Prerequisites

- [ ] Docker services running (`docker compose ps` shows healthy)
- [ ] Frontend accessible at http://localhost:8082
- [ ] Backend accessible at http://localhost:8001
- [ ] Database migration applied (`user_feedback` table exists)
- [ ] Langfuse configured (for score sync verification)
- [ ] Valid user account to log in

### Test Scenarios

#### Scenario 1: Per-Message Thumbs Up

1. Open http://localhost:8082
2. Log in with a valid user account
3. Start a new conversation
4. Send a message and wait for AI response
5. Hover over AI message - thumbs icons appear
6. Click thumbs up
7. ✅ Toast: "Thank you! Your feedback helps us improve."
8. ✅ Buttons replaced with "Thanks for your feedback!"

#### Scenario 2: Per-Message Thumbs Down with Category

1. Hover over a different AI message
2. Click thumbs down
3. ✅ Feedback form expands with categories
4. Select "Incomplete" category
5. Add optional comment
6. Click "Submit Feedback"
7. ✅ Toast appears, form collapses

#### Scenario 3: Session Feedback

1. Have conversation with 5+ messages
2. ✅ Session feedback banner appears
3. Click "Good"
4. ✅ Positive categories appear
5. Select category and submit
6. ✅ Banner shows thanks message

### Verification

- [ ] Check Supabase `user_feedback` table for new records
- [ ] Check Langfuse for `user_feedback` score on trace
- [ ] Verify `user_feedback_category` score if category selected

---

## Implementation Files Reference

| File | Purpose |
|------|---------|
| `backend_agent_api/feedback_api.py` | API endpoints for feedback submission |
| `frontend/src/components/chat/FeedbackWidget.tsx` | Per-message feedback UI |
| `frontend/src/components/chat/SessionFeedback.tsx` | Session-wide feedback banner |
| `sql/10-user_feedback.sql` | Database schema and RLS policies |
| `frontend/src/components/chat/MessageHandling.tsx` | Request ID tracking |
| `frontend/src/components/chat/MessageList.tsx` | SessionFeedback integration |
