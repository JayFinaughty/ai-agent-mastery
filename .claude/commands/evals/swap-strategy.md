---
description: Switch to a branch/tag/strategy and validate environment
argument-hint: <strategy-number|tag|branch> (e.g., "3", "module-8-03-llm-judge-local", "latest")
---

# Swap Strategy - Switch to Branch/Tag/Strategy

Switch to a specific branch, tag, or strategy number and ensure the environment is ready.

The user has provided: **$ARGUMENTS**

Where `<target>` can be:
- Video/strategy number: `1`, `2`, `3`, etc.
- Tag name: `module-8-03-llm-judge-local`
- Branch name: `module-8-prep-evals`, `main`

## Argument Parsing

Parse the user's input to determine the target:

| Input | Interpreted As |
|-------|----------------|
| `1` or `strategy 1` | Tag `module-8-01-golden-dataset` |
| `2` | Tag `module-8-02-rule-based-local` |
| `3` | Tag `module-8-03-llm-judge-local` |
| `4` | Tag `module-8-04-manual-annotation` |
| `5` | Tag `module-8-05-rule-based-prod` |
| `6` | Tag `module-8-06-llm-judge-prod` |
| `7` | Tag `module-8-07-user-feedback` |
| `module-8-*` | Tag (exact) |
| `module-8-prep-evals` | Branch |
| `main` | Branch |
| `latest` or `dev` | Branch `module-8-prep-evals` |

### Video Number to Tag Mapping

```
Video 2  (Strategy 1) → module-8-01-golden-dataset
Video 3  (Strategy 2) → module-8-02-rule-based-local
Video 4  (Strategy 3) → module-8-03-llm-judge-local
Video 5  (Strategy 4) → module-8-04-manual-annotation
Video 6  (Strategy 5) → module-8-05-rule-based-prod
Video 7  (Strategy 6) → module-8-06-llm-judge-prod
Video 8  (Strategy 7) → module-8-07-user-feedback
```

## Swap Workflow

### Phase 1: Pre-Flight Checks

```bash
# Check current state
git status --short
git branch --show-current
git describe --tags --always 2>/dev/null || echo "No tags"

# Check for uncommitted changes
git diff --stat
```

If uncommitted changes exist:
```
⚠️  UNCOMMITTED CHANGES DETECTED

You have uncommitted changes that will be lost if you switch.
Options:
1. Stash them: git stash push -m "WIP before swap"
2. Commit them: git add . && git commit -m "WIP"
3. Discard them: git checkout -- .

What would you like to do?
```

Wait for user confirmation before proceeding.

### Phase 2: Verify Target Exists

```bash
# Check if target is a tag
git tag -l | grep -x "<target>"

# Check if target is a branch
git branch -a | grep "<target>"
```

If target doesn't exist:
```
❌ TARGET NOT FOUND

"<target>" is not a valid tag or branch.

Available tags:
$(git tag -l "module-8-*")

Available branches:
$(git branch -a | grep module-8)

Did you mean one of these?
```

### Phase 3: Perform Checkout

```bash
# Fetch latest from remote
git fetch --all --tags

# Checkout the target
git checkout <target>

# Verify checkout
git log --oneline -1
```

Report whether we're on:
- A branch (can commit)
- A tag/detached HEAD (read-only, for recording)

### Phase 4: Rebuild Docker Environment

```bash
cd /Users/rasmus/Projects/cole/ai-agent-mastery/8_Agent_Evals

# Stop current containers
docker compose down

# Rebuild with new code (no cache to ensure fresh)
docker compose build --no-cache

# Start containers
docker compose up -d

# Wait for services to be healthy
sleep 10

# Check status
docker compose ps
```

### Phase 5: Validate Environment

Run validation checks to ensure everything works:

#### 5.1 Health Check

```bash
# Backend API health
curl -s http://localhost:8001/health || curl -s http://localhost:8001/api/health || echo "No health endpoint"

# Frontend (if applicable)
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 || echo "Frontend not running"
```

#### 5.2 API Smoke Test

```bash
# Test the main agent endpoint is responding
curl -s -X POST http://localhost:8001/api/pydantic-agent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token" \
  -d '{"query": "hello", "user_id": "test", "session_id": "test", "request_id": "test-swap-validation"}' \
  --max-time 30 \
  | head -c 500
```

#### 5.3 Strategy-Specific Validation

Based on the target strategy, run relevant checks:

**Phase 1 Strategies (Local - pydantic-evals)**:

```bash
# Check if evals directory exists
ls -la 8_Agent_Evals/backend_agent_api/evals/

# Check if golden dataset exists (after Video 2)
ls -la 8_Agent_Evals/backend_agent_api/evals/golden_dataset.yaml 2>/dev/null || echo "No golden dataset yet"

# Check if run_evals.py exists
ls -la 8_Agent_Evals/backend_agent_api/evals/run_evals.py 2>/dev/null || echo "No eval runner yet"
```

**Phase 2 Strategies (Production - Langfuse)**:

```bash
# Verify Langfuse connection (after Video 5)
docker compose exec backend_agent_api python -c "
from langfuse import Langfuse
import os
if os.getenv('LANGFUSE_PUBLIC_KEY'):
    lf = Langfuse()
    print('Langfuse: Connected')
else:
    print('Langfuse: Not configured')
"
```

#### 5.4 Database Connectivity

```bash
# Verify Supabase connection
docker compose exec backend_agent_api python -c "
from supabase import create_client
import os
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
print('Supabase: Connected')
"
```

### Phase 6: Generate Status Report

Compile all results into a summary:

```
═══════════════════════════════════════════════════════════════════
 STRATEGY SWAP COMPLETE
═══════════════════════════════════════════════════════════════════

 FROM → TO
 ─────────────────────────────────────────────────────────────────
 Branch: module-8-prep-evals  →  Tag: module-8-03-llm-judge-local

 CURRENT STATE
 ─────────────────────────────────────────────────────────────────
 Target:     module-8-03-llm-judge-local
 Type:       Tag (detached HEAD - read only)
 Phase:      Local Development
 Commit:     a1b2c3d "Implement Video 4: LLM Judge (Local)"

 DOCKER STATUS
 ─────────────────────────────────────────────────────────────────
 backend_agent_api    ✅ Running (healthy)
 frontend             ✅ Running
 postgres             ✅ Running (if local)

 VALIDATION RESULTS
 ─────────────────────────────────────────────────────────────────
 Health endpoint      ✅ Responding (200)
 Agent endpoint       ✅ Responding (got response)
 Supabase             ✅ Connected
 Langfuse             ⚠️ Not yet configured (Phase 2)

 AVAILABLE FEATURES AT THIS STATE
 ─────────────────────────────────────────────────────────────────
 PHASE 1: LOCAL DEVELOPMENT
 ✅ Video 2: Golden Dataset - 10-case YAML, run_evals.py
 ✅ Video 3: Rule-Based (Local) - Contains, HasMatchingSpan evaluators
 ✅ Video 4: LLM Judge (Local) - LLMJudge with rubrics

 PHASE 2: PRODUCTION
 ❌ Video 5: Manual Annotation - not yet implemented
 ❌ Video 6: Rule-Based (Prod) - not yet implemented
 ❌ Video 7: LLM Judge (Prod) - not yet implemented
 ❌ Video 8: User Feedback - not yet implemented

 READY FOR
 ─────────────────────────────────────────────────────────────────
 ✅ Recording video for Video 4 (LLM Judge Local)
 ✅ Demo of local evals (Videos 2-4)
 ⚠️  Read-only state - checkout branch to make changes

═══════════════════════════════════════════════════════════════════
```

### Error Handling

If any validation fails, report clearly:

```
═══════════════════════════════════════════════════════════════════
 STRATEGY SWAP - ISSUES DETECTED
═══════════════════════════════════════════════════════════════════

 Target: module-8-03-llm-judge-local
 Status: ⚠️  PARTIAL SUCCESS

 ISSUES
 ─────────────────────────────────────────────────────────────────
 ❌ Agent endpoint not responding
    Error: Connection refused on port 8001
    Fix: Check docker compose logs backend_agent_api

 ❌ Langfuse not configured
    Error: LANGFUSE_PUBLIC_KEY not set
    Fix: Add to .env file (only needed for Phase 2)

 DOCKER LOGS (last 20 lines)
 ─────────────────────────────────────────────────────────────────
 [logs here]

 RECOMMENDED ACTIONS
 ─────────────────────────────────────────────────────────────────
 1. Fix the issues above
 2. Run: docker compose restart backend_agent_api
 3. Re-run validation: curl http://localhost:8001/health

═══════════════════════════════════════════════════════════════════
```

## Quick Reference

Common swap commands:

```bash
# Swap to specific video state for recording
/swap-strategy 2   # Golden Dataset
/swap-strategy 3   # Rule-Based Local
/swap-strategy 4   # LLM Judge Local

# Swap to latest development
/swap-strategy latest

# Swap to specific tag
/swap-strategy module-8-05-manual-annotation

# Swap to main branch
/swap-strategy main
```

## Notes

- Always stash or commit changes before swapping
- Tags are read-only (detached HEAD) - checkout branch to make changes
- Docker rebuild ensures code matches git state
- Validation confirms environment is actually working
- Phase 1 strategies (1-3) don't require Langfuse
- Phase 2 strategies (4-7) require Langfuse configuration
