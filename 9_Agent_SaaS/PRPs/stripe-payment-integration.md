name: "Stripe Payment Integration for Agent Tokens"
description: |
  Complete implementation of Stripe payment system for purchasing and consuming agent interaction tokens.
  This PRP provides all necessary context for one-pass implementation success.

---

## Goal

**Feature Goal**: Implement a complete Stripe-based payment system that allows users to purchase agent interaction tokens, track token usage, and enforce token consumption for agent interactions.

**Deliverable**:
- Three-tier token purchase system (100/$5, 250/$10, 600/$20)
- Webhook-based token fulfillment with idempotency
- Token balance enforcement in agent endpoint
- Complete transaction audit trail
- User-facing purchase flow, success/failure pages, and usage history

**Success Criteria**:
- [ ] Users can purchase tokens through Stripe with 3 pricing tiers
- [ ] Tokens are granted automatically via webhook after successful payment
- [ ] Agent endpoint rejects requests when user has insufficient tokens
- [ ] Agent endpoint deducts 1 token per successful interaction
- [ ] Duplicate webhook events don't grant duplicate tokens (idempotency)
- [ ] Users can view their token balance and purchase history
- [ ] All token transactions are audited in transactions table
- [ ] Webhook signatures are properly verified for security
- [ ] Environment setup documentation is complete

## Why

**Business Value**:
- Enables monetization of AI agent interactions
- Creates sustainable business model for compute-intensive operations
- Provides clear pricing tiers for different usage levels

**Integration with Existing Features**:
- Builds on existing Supabase authentication system
- Extends current user profile structure
- Integrates into existing agent endpoint flow (after rate limiting)
- Uses existing frontend routing and component patterns

**Problems This Solves**:
- Prevents unlimited free usage of expensive AI operations
- Provides transparent pricing for users
- Creates audit trail for billing disputes
- Enables future feature: automatic token refills, enterprise plans

## What

### User-Visible Behavior

**Purchase Flow**:
1. User clicks "Purchase Tokens" from chat interface or profile
2. User selects from 3 token packages
3. User completes Stripe payment (card details, confirmation)
4. User is redirected to success page showing new token balance
5. Tokens appear immediately in user's balance

**Token Consumption**:
1. User sends message to agent
2. If insufficient tokens: Error message "You need more tokens to use the agent"
3. If sufficient tokens: Message is processed, 1 token deducted, balance updated in UI

**Token Management**:
1. User can view current token balance in chat header
2. User can view purchase history with dates, amounts, and Stripe receipt links
3. User can see token consumption history (when tokens were used)

### Technical Requirements

**Backend** (Python/FastAPI):
- User table extended with `tokens` column (INTEGER, default 0)
- Transactions table for audit trail
- Two new endpoints: `/api/create-payment-intent` and `/api/webhook/stripe`
- Modified `/api/pydantic-agent` endpoint to check/deduct tokens
- PostgreSQL function for atomic token operations
- Stripe webhook signature verification
- Idempotency handling for webhooks

**Frontend** (React/TypeScript):
- Token purchase page with 3 pricing tiers
- Stripe Elements integration for payment form
- Success and failure pages
- Token balance display in chat header
- Token usage history page
- Error handling for insufficient tokens

**Database** (PostgreSQL/Supabase):
- Migration: Add tokens column to user_profiles
- New table: transactions (purchases and consumption)
- RLS policies for transactions table
- PostgreSQL function: deduct_token_if_sufficient

**Environment Configuration**:
- Backend: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
- Frontend: VITE_STRIPE_PUBLISHABLE_KEY
- Updated .env.example files

## All Needed Context

### Documentation & References

```yaml
# CRITICAL: Read these before implementation

# Stripe Python SDK
- url: https://docs.stripe.com/api/payment_intents/create?lang=python
  why: Payment intent creation pattern - amount, currency, metadata
  critical: Amount is in cents (2000 = $20.00)

- url: https://docs.stripe.com/webhooks/signature
  why: Webhook signature verification with stripe.Webhook.construct_event()
  critical: MUST verify signatures to prevent fraud

- url: https://docs.stripe.com/api/idempotent_requests
  why: Idempotency key patterns for safe retries
  critical: Use Stripe event ID to prevent duplicate token grants

- url: https://docs.stripe.com/error-handling?lang=python
  why: Complete exception hierarchy and retry patterns
  critical: Different exceptions require different handling

# Stripe React
- url: https://docs.stripe.com/sdks/stripejs-react
  why: React integration with Elements provider and loadStripe
  critical: Initialize outside component to avoid recreation

- url: https://docs.stripe.com/payments/payment-element
  why: PaymentElement usage for embedded payment forms
  critical: Always check if stripe/elements loaded before processing

# Supabase Atomic Operations
- url: https://supabase.com/docs/reference/python/rpc
  why: RPC pattern for calling PostgreSQL functions
  critical: Client has no native transaction support - use functions

- url: https://supabase.com/docs/guides/database/functions
  why: PostgreSQL function creation patterns
  critical: Use FOR UPDATE locking to prevent race conditions

- url: https://supabase.com/docs/guides/database/postgres/row-level-security
  why: RLS policy patterns with functions
  critical: Functions run with SECURITY INVOKER by default (caller's permissions)

# Codebase Examples (MUST READ)
- file: backend_agent_api/agent_api.py
  lines: 104-151, 190-206
  why: Authentication pattern (verify_token), rate limit checking, error handling
  critical: Token check goes after rate limit check (line 206)

- file: backend_agent_api/db_utils.py
  lines: 190-243
  why: Database operation patterns, async functions, error handling
  critical: All DB operations are async and use global supabase client

- file: sql/1-user_profiles_requests.sql
  why: Table creation pattern, triggers, CASCADE handling
  critical: UUID primary keys, timestamp defaults, foreign key patterns

- file: sql/2-user_profiles_requests_rls.sql
  why: RLS policy structure, is_admin() function, policy naming
  critical: Separate policies for users vs admins, deny delete pattern

- file: frontend/src/App.tsx
  lines: 43-68
  why: Route structure with ProtectedRoute wrapper
  critical: All authenticated routes use ProtectedRoute component

- file: frontend/src/lib/api.ts
  why: API client patterns, fetch with auth headers, error handling
  critical: Bearer token from Supabase auth context

# AI Documentation (local research files)
- docfile: PRPs/planning/stripe-python-research.md
  why: Complete Stripe Python SDK patterns with code examples
  critical: Webhook handling, error types, retry patterns

- docfile: PRPs/planning/stripe-react-research.md
  why: Complete Stripe React integration with TypeScript examples
  critical: Elements provider setup, PaymentElement usage, error handling

- docfile: PRPs/planning/supabase-atomic-research.md
  why: Atomic operation patterns for token operations
  critical: FOR UPDATE locking pattern, RPC usage, race condition prevention

- docfile: PRPs/planning/stripe-integration-codebase-analysis.md
  why: Detailed analysis of existing codebase patterns
  critical: Integration points, file references, naming conventions
```

### Current Codebase Structure

```
8_Agent_SaaS/
├── backend_agent_api/
│   ├── agent_api.py              # FastAPI app, auth, endpoints
│   ├── db_utils.py               # Database operations
│   ├── clients.py                # Client initialization
│   ├── agent.py                  # Pydantic AI agent
│   ├── tools.py                  # Agent tools
│   ├── .env.example              # Environment template
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Routes and providers
│   │   ├── pages/
│   │   │   ├── Chat.tsx          # Main chat interface
│   │   │   ├── Login.tsx         # Authentication
│   │   │   └── Admin.tsx         # Admin panel
│   │   ├── components/
│   │   │   ├── auth/
│   │   │   ├── chat/
│   │   │   └── ui/               # Radix UI components
│   │   ├── hooks/
│   │   │   └── useAuth.tsx       # Auth context
│   │   └── lib/
│   │       ├── api.ts            # API client
│   │       └── supabase.ts       # Supabase client
│   ├── .env.example
│   └── package.json
├── sql/
│   ├── 1-user_profiles_requests.sql
│   ├── 2-user_profiles_requests_rls.sql
│   ├── 3-conversations_messages.sql
│   └── 4-conversations_messages_rls.sql
└── PRPs/
    ├── INITIAL.md
    ├── templates/
    └── planning/                  # Research documents
```

### Desired Codebase Structure (Files to Add)

```
backend_agent_api/
├── stripe_utils.py               # NEW: Stripe client, payment intent creation
└── token_utils.py                # NEW: Token operations (check, deduct, grant)

frontend/src/
├── pages/
│   ├── PurchaseTokens.tsx        # NEW: Token purchase page
│   ├── PaymentSuccess.tsx        # NEW: Success page after payment
│   ├── PaymentFailure.tsx        # NEW: Failure page
│   └── TokenHistory.tsx          # NEW: Purchase and usage history
├── components/
│   ├── tokens/
│   │   ├── TokenBalance.tsx      # NEW: Balance display component
│   │   ├── PricingCard.tsx       # NEW: Pricing tier card
│   │   └── CheckoutForm.tsx      # NEW: Stripe payment form
│   └── chat/
│       └── ChatHeader.tsx        # MODIFY: Add token balance display

sql/
├── 10-stripe-tokens.sql          # NEW: Add tokens column to user_profiles
├── 11-stripe-transactions.sql    # NEW: Create transactions table
├── 12-stripe-tokens-rls.sql      # NEW: RLS policies for transactions
├── 13-stripe-functions.sql       # NEW: PostgreSQL functions for atomic operations
└── 14-stripe-migration.sql       # NEW: Migration for existing users

PRPs/planning/
└── stripe-setup.md               # NEW: Post-implementation setup guide
```

### Known Gotchas & Library Quirks

```python
# CRITICAL GOTCHAS

# 1. Pydantic v2 (this codebase uses v2)
# - Use model_dump() not dict()
# - Use ConfigDict instead of class Config
# - from_attributes=True instead of orm_mode=True

# 2. FastAPI with async
# - All database operations MUST be async functions
# - Use await for supabase client calls
# - Request.body() is async: await request.body()

# 3. Stripe webhook payload
# - MUST use raw body string for signature verification
# - Do NOT use request.json() before verification
# - FastAPI: payload = await request.body(); payload.decode('utf-8')

# 4. Stripe amounts
# - Amounts are in CENTS not dollars
# - $5.00 = 500, $10.00 = 1000, $20.00 = 2000
# - Currency must be lowercase: 'usd' not 'USD'

# 5. Supabase Python client
# - NO native transaction support
# - Use PostgreSQL functions via RPC for atomic operations
# - supabase.rpc('function_name', {'param': value}).execute()

# 6. PostgreSQL FOR UPDATE
# - Required to prevent race conditions in token operations
# - SELECT tokens FROM users WHERE id = user_id FOR UPDATE;
# - Locks the row until transaction completes

# 7. Webhook idempotency
# - Stripe may retry webhook delivery
# - MUST check if event already processed
# - Use Stripe event ID as unique key in transactions table

# 8. Environment variables
# - Backend uses python-dotenv
# - Frontend uses Vite: VITE_ prefix required
# - VITE_STRIPE_PUBLISHABLE_KEY not STRIPE_PUBLISHABLE_KEY

# 9. Line length limit
# - Max 100 characters per line (ruff enforced)
# - Break long lines with proper indentation

# 10. Existing patterns
# - All DB async functions in db_utils.py
# - Global clients initialized in lifespan
# - HTTPBearer security for endpoints
# - verify_token dependency for authentication
```

## Implementation Blueprint

### Data Models and Structure

```python
# backend_agent_api/models.py (NEW FILE)

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from uuid import UUID


class TokenPurchaseRequest(BaseModel):
    """Request to create payment intent for token purchase."""

    tier: Literal["tier_1", "tier_2", "tier_3"]
    user_id: str


class PaymentIntentResponse(BaseModel):
    """Response containing Stripe client secret."""

    client_secret: str
    amount: int
    currency: str = "usd"


class TokenBalance(BaseModel):
    """User's current token balance."""

    user_id: str
    tokens: int


class TransactionRecord(BaseModel):
    """Transaction history record."""

    id: UUID
    user_id: UUID
    transaction_type: Literal["purchase", "consumption"]
    amount: int
    tokens: int
    stripe_payment_intent_id: Optional[str] = None
    created_at: datetime


# Database schema (SQL)
"""
-- user_profiles table modification
ALTER TABLE user_profiles
ADD COLUMN tokens INTEGER DEFAULT 0 NOT NULL CHECK (tokens >= 0);

-- transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('purchase', 'consumption')),
    amount INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    stripe_payment_intent_id TEXT,
    stripe_event_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX idx_transactions_stripe_event_id
ON transactions(stripe_event_id)
WHERE stripe_event_id IS NOT NULL;

CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
"""
```

### Implementation Tasks (Dependency-Ordered)

```yaml
# PHASE 1: Database Schema (Foundation)

Task 1: Create database migration scripts
  Files:
    CREATE: sql/10-stripe-tokens.sql
      - ALTER TABLE user_profiles ADD COLUMN tokens
      - Add CHECK constraint (tokens >= 0)
      - Set DEFAULT 0 NOT NULL
      - MIRROR pattern from: sql/1-user_profiles_requests.sql (lines 6-13)

    CREATE: sql/11-stripe-transactions.sql
      - CREATE TABLE transactions with UUID primary key
      - Foreign key to user_profiles with CASCADE
      - Unique index on stripe_event_id for idempotency
      - Indexes on user_id and transaction_type
      - MIRROR pattern from: sql/3-conversations_messages.sql (table structure)

    CREATE: sql/12-stripe-tokens-rls.sql
      - Enable RLS on transactions table
      - Policy: Users can SELECT their own transactions
      - Policy: Admins can SELECT all transactions
      - Policy: Deny DELETE on transactions (audit trail)
      - MIRROR pattern from: sql/2-user_profiles_requests_rls.sql

    CREATE: sql/13-stripe-functions.sql
      - Function: deduct_token_if_sufficient(user_id UUID)
      - Function: grant_tokens_for_purchase(user_id UUID, tokens INT, event_id TEXT, payment_intent_id TEXT)
      - Use FOR UPDATE locking pattern
      - Return JSON with success/error
      - CRITICAL: Reference PRPs/planning/supabase-atomic-research.md (lines 26-68, 239-293)

    CREATE: sql/14-stripe-migration.sql
      - UPDATE user_profiles SET tokens = 0 WHERE tokens IS NULL
      - For existing deployments

# PHASE 2: Backend Implementation

Task 3: Create Stripe utility module
  CREATE: backend_agent_api/stripe_utils.py
    Responsibilities:
      - Initialize Stripe client with secret key
      - create_payment_intent(tier, user_id) -> PaymentIntentResponse
      - verify_webhook_signature(payload, signature) -> stripe.Event
      - PRICING_TIERS constant = {tier_1: (500, 100), tier_2: (1000, 250), tier_3: (2000, 600)}
    Patterns:
      - Import stripe library
      - Use os.getenv("STRIPE_SECRET_KEY")
      - stripe.api_key = secret_key
      - Error handling with try/except stripe.error.*
      - REFERENCE: PRPs/planning/stripe-python-research.md (lines 6-60, 66-102, 174-240)
    Critical:
      - Amounts in cents (500 = $5.00)
      - Include user_id and tier in metadata
      - Idempotency key for payment intent creation

Task 4: Create token utility module
  CREATE: backend_agent_api/token_utils.py
    Responsibilities:
      - check_user_tokens(supabase, user_id) -> int
      - deduct_token_atomic(supabase, user_id) -> bool
      - grant_tokens_atomic(supabase, user_id, tokens, event_id, payment_intent_id) -> bool
    Patterns:
      - Async functions (all DB operations)
      - Use supabase.rpc() for atomic operations
      - Error handling with try/except
      - Logging for debugging
      - REFERENCE: PRPs/planning/supabase-atomic-research.md (lines 70-86, 403-444)
    Critical:
      - Call PostgreSQL functions via RPC
      - Check success field in response
      - Return bool for simple control flow

Task 5: Add payment intent endpoint to agent_api.py
  MODIFY: backend_agent_api/agent_api.py
    Location: After line 189 (before @app.post("/api/pydantic-agent"))
    Add:
      @app.post("/api/create-payment-intent")
      async def create_payment_intent_endpoint(
          request: TokenPurchaseRequest,
          user: Dict[str, Any] = Depends(verify_token)
      ) -> PaymentIntentResponse
    Logic:
      - Verify request.user_id == user["id"]
      - Call stripe_utils.create_payment_intent(request.tier, request.user_id)
      - Return PaymentIntentResponse with client_secret
    Error Handling:
      - HTTPException(401) for user mismatch
      - HTTPException(500) for Stripe errors
    REFERENCE: agent_api.py lines 190-206 (authentication pattern)

Task 6: Add Stripe webhook endpoint to agent_api.py
  MODIFY: backend_agent_api/agent_api.py
    Location: After payment intent endpoint
    Add:
      @app.post("/api/webhook/stripe")
      async def stripe_webhook_endpoint(request: Request)
    Logic:
      - Get raw body: payload = (await request.body()).decode('utf-8')
      - Get signature: sig_header = request.headers.get("stripe-signature")
      - Verify: event = stripe_utils.verify_webhook_signature(payload, sig_header)
      - Handle event.type == "payment_intent.succeeded"
      - Extract user_id and tier from event.data.object.metadata
      - Calculate tokens from tier
      - Call token_utils.grant_tokens_atomic(user_id, tokens, event.id, payment_intent_id)
      - Return {"success": True}
    Error Handling:
      - Return 400 for invalid payload
      - Return 400 for invalid signature
      - Log errors for debugging
      - ALWAYS return 200 after processing to acknowledge webhook
    REFERENCE: PRPs/planning/stripe-python-research.md (lines 66-102)
    Critical:
      - Use RAW body string for signature verification
      - Check for duplicate event_id (handled by grant_tokens_atomic)
      - Return 200 even on errors (after logging)

Task 7: Modify pydantic-agent endpoint for token checking
  MODIFY: backend_agent_api/agent_api.py
    Location: Line 206 (after rate limit check)
    Add:
      # Check user has sufficient tokens
      user_tokens = await token_utils.check_user_tokens(supabase, request.user_id)
      if user_tokens < 1:
          return StreamingResponse(
              stream_error_response(
                  "Insufficient tokens. Please purchase more tokens to continue.",
                  request.session_id
              ),
              media_type='text/plain'
          )

      # Deduct token atomically
      token_deducted = await token_utils.deduct_token_atomic(supabase, request.user_id)
      if not token_deducted:
          return StreamingResponse(
              stream_error_response(
                  "Failed to deduct token. Please try again.",
                  request.session_id
              ),
              media_type='text/plain'
          )
    REFERENCE: agent_api.py lines 200-206 (rate limit pattern)
    Critical:
      - Check AFTER rate limit (line 206)
      - Deduct BEFORE agent processing
      - Use stream_error_response for user feedback

Task 8: Add Stripe to requirements.txt
  MODIFY: backend_agent_api/requirements.txt
    Add: stripe>=10.0.0

Task 9: Update backend .env.example
  MODIFY: backend_agent_api/.env.example
    Add:
      # Stripe Configuration
      STRIPE_SECRET_KEY=sk_test_your_key_here
      STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here

# PHASE 3: Frontend Implementation

Task 10: Install Stripe packages
  Action:
    cd frontend
    npm install @stripe/stripe-js @stripe/react-stripe-js

Task 11: Create token purchase page
  CREATE: frontend/src/pages/PurchaseTokens.tsx
    Responsibilities:
      - Display 3 pricing cards (tier_1, tier_2, tier_3)
      - Handle tier selection
      - Create payment intent via /api/create-payment-intent
      - Initialize Stripe Elements with client_secret
      - Show CheckoutForm component
    Components:
      - Use PricingCard for each tier
      - Use CheckoutForm for payment
    Patterns:
      - useState for loading, selected tier, client secret
      - useAuth hook for user_id and token
      - Fetch with Bearer token authorization
      - REFERENCE: PRPs/planning/stripe-react-research.md (lines 25-70, 103-175)
    Critical:
      - loadStripe outside component (memoized)
      - Elements provider wraps CheckoutForm
      - Loading states for UX

Task 12: Create checkout form component
  CREATE: frontend/src/components/tokens/CheckoutForm.tsx
    Responsibilities:
      - Render PaymentElement
      - Handle form submission
      - Confirm payment with Stripe
      - Redirect to success/failure page
    Patterns:
      - useStripe and useElements hooks
      - useState for loading and errors
      - confirmPayment with return_url
      - REFERENCE: PRPs/planning/stripe-react-research.md (lines 106-175)
    Critical:
      - Check if stripe/elements loaded before processing
      - Disable button while loading
      - Show user-friendly error messages

Task 13: Create pricing card component
  CREATE: frontend/src/components/tokens/PricingCard.tsx
    Responsibilities:
      - Display tier name, price, token amount
      - Highlight selected tier
      - Handle click to select
    Props:
      - tier: string
      - price: number
      - tokens: number
      - selected: boolean
      - onClick: () => void
    Patterns:
      - Use Radix UI Card component
      - Tailwind CSS for styling
      - REFERENCE: frontend/src/components/ui/ (UI patterns)

Task 14: Create success page
  CREATE: frontend/src/pages/PaymentSuccess.tsx
    Responsibilities:
      - Retrieve payment intent from URL params
      - Display success message
      - Show new token balance
      - Link back to chat
    Patterns:
      - useEffect to retrieve payment intent
      - useStripe().retrievePaymentIntent()
      - useNavigate for routing
      - REFERENCE: PRPs/planning/stripe-react-research.md (lines 299-338)

Task 15: Create failure page
  CREATE: frontend/src/pages/PaymentFailure.tsx
    Responsibilities:
      - Display failure message
      - Show error details if available
      - Link to try again
    Patterns:
      - URLSearchParams to get error from URL
      - Error message display
      - Retry button

Task 16: Create token balance component
  CREATE: frontend/src/components/tokens/TokenBalance.tsx
    Responsibilities:
      - Fetch current user token balance from Supabase
      - Display token count with icon
      - Link to purchase page
      - Real-time updates via Supabase subscriptions
    Patterns:
      - useEffect to fetch balance
      - supabase.from('user_profiles').select('tokens')
      - Subscription to user_profiles changes
      - REFERENCE: frontend/src/hooks/useAuth.tsx (Supabase patterns)

Task 17: Create token history page
  CREATE: frontend/src/pages/TokenHistory.tsx
    Responsibilities:
      - Fetch transactions from Supabase
      - Display purchase and consumption history
      - Show dates, amounts, token counts
      - Paginate results
    Patterns:
      - useEffect to fetch data
      - supabase.from('transactions').select()
      - Table component from Radix UI
      - REFERENCE: frontend/src/pages/Admin.tsx (table patterns if exists)

Task 18: Add routes to App.tsx
  MODIFY: frontend/src/App.tsx
    Location: Inside <Routes> (around line 55)
    Add:
      <Route
        path="/purchase-tokens"
        element={
          <ProtectedRoute>
            <PurchaseTokens />
          </ProtectedRoute>
        }
      />
      <Route path="/payment-success" element={<PaymentSuccess />} />
      <Route path="/payment-failure" element={<PaymentFailure />} />
      <Route
        path="/token-history"
        element={
          <ProtectedRoute>
            <TokenHistory />
          </ProtectedRoute>
        }
      />
    REFERENCE: App.tsx lines 43-68 (route pattern)

Task 19: Modify Chat component to show token balance
  MODIFY: frontend/src/pages/Chat.tsx
    Location: Top of page (header area)
    Add:
      import TokenBalance from '@/components/tokens/TokenBalance'
      // In render, add <TokenBalance /> to header
    Pattern:
      - Place in visible location
      - Style to match existing design

Task 20: Update frontend .env.example
  MODIFY: frontend/.env.example
    Add:
      # Stripe Configuration
      VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here

# PHASE 4: Testing & Documentation

Task 21: Test database functions
  Action:
    # In Supabase SQL editor
    SELECT deduct_token_if_sufficient(auth.uid());
    # Should return {success: false, error: "Insufficient tokens"}

    # Grant test tokens
    UPDATE user_profiles SET tokens = 10 WHERE id = auth.uid();

    # Test deduction
    SELECT deduct_token_if_sufficient(auth.uid());
    # Should return {success: true, remaining_tokens: 9}

    # Test grant
    SELECT grant_tokens_for_purchase(
      auth.uid(),
      100,
      'test_event_123',
      'pi_test_123'
    );
    # Should return {success: true, new_balance: 109}

    # Test idempotency (run again)
    # Should return {success: false, error: "Event already processed"}

Task 22: Create post-implementation setup guide
  CREATE: PRPs/planning/stripe-setup.md
    Content:
      # Stripe Setup Guide

      ## Environment Variables
      [List all required variables for backend and frontend]

      ## Stripe Dashboard Setup
      1. Create Stripe account (or use existing)
      2. Get API keys from Dashboard > Developers > API keys
      3. Create webhook endpoint: Dashboard > Developers > Webhooks
         - URL: https://your-domain.com/api/webhook/stripe
         - Events: payment_intent.succeeded
      4. Get webhook signing secret

      ## Testing
      [Provide test scenarios and commands]

      ## Production Checklist
      - [ ] Replace test API keys with live keys
      - [ ] Update webhook URL to production domain
      - [ ] Test webhook delivery
      - [ ] Set up Stripe webhook monitoring
```

### Critical Pseudocode for Complex Operations

```python
# backend_agent_api/stripe_utils.py

import stripe
import os
from typing import Dict, Tuple

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

PRICING_TIERS: Dict[str, Tuple[int, int]] = {
    "tier_1": (500, 100),    # $5.00 for 100 tokens
    "tier_2": (1000, 250),   # $10.00 for 250 tokens
    "tier_3": (2000, 600),   # $20.00 for 600 tokens
}

async def create_payment_intent(tier: str, user_id: str) -> Dict[str, Any]:
    """
    Create Stripe payment intent for token purchase.

    CRITICAL:
    - Amount must be in cents
    - Include user_id and tier in metadata
    - Use idempotency key for safety
    """
    if tier not in PRICING_TIERS:
        raise ValueError(f"Invalid tier: {tier}")

    amount_cents, token_count = PRICING_TIERS[tier]

    try:
        # Create payment intent with metadata
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            metadata={
                "user_id": user_id,
                "tier": tier,
                "tokens": token_count,
            },
            # PATTERN: Idempotency key prevents duplicate charges
            idempotency_key=f"purchase_{user_id}_{int(time.time())}"
        )

        return {
            "client_secret": payment_intent.client_secret,
            "amount": amount_cents,
            "currency": "usd"
        }

    except stripe.error.StripeError as e:
        # PATTERN: Log and re-raise for FastAPI error handling
        logging.error(f"Stripe error creating payment intent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Payment error: {str(e)}")


def verify_webhook_signature(payload: str, sig_header: str) -> stripe.Event:
    """
    Verify Stripe webhook signature and parse event.

    CRITICAL:
    - Must use RAW body string, not parsed JSON
    - Raises exception on invalid signature
    - This prevents webhook spoofing attacks
    """
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        return event

    except ValueError as e:
        # Invalid payload
        logging.error(f"Invalid webhook payload: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid payload")

    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logging.error(f"Invalid webhook signature: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid signature")


# backend_agent_api/token_utils.py

async def check_user_tokens(supabase: Client, user_id: str) -> int:
    """
    Get current token balance for user.

    PATTERN: Simple SELECT query, no locking needed
    """
    try:
        response = supabase.table("user_profiles").select("tokens").eq(
            "id", user_id
        ).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]["tokens"]
        else:
            return 0

    except Exception as e:
        logging.error(f"Error checking user tokens: {str(e)}")
        return 0


async def deduct_token_atomic(supabase: Client, user_id: str) -> bool:
    """
    Atomically deduct one token from user balance.

    CRITICAL:
    - Uses PostgreSQL function for atomicity
    - Function uses FOR UPDATE to prevent race conditions
    - Returns False if insufficient tokens
    """
    try:
        response = supabase.rpc(
            "deduct_token_if_sufficient",
            {"p_user_id": user_id}
        ).execute()

        if response.data and len(response.data) > 0:
            result = response.data[0]
            return result.get("success", False)

        return False

    except Exception as e:
        logging.error(f"Error deducting token: {str(e)}")
        return False


async def grant_tokens_atomic(
    supabase: Client,
    user_id: str,
    tokens: int,
    event_id: str,
    payment_intent_id: str
) -> bool:
    """
    Atomically grant tokens from successful payment.

    CRITICAL:
    - Uses event_id for idempotency
    - Function checks if event already processed
    - Prevents duplicate token grants on webhook retries
    """
    try:
        response = supabase.rpc(
            "grant_tokens_for_purchase",
            {
                "p_user_id": user_id,
                "p_tokens": tokens,
                "p_event_id": event_id,
                "p_payment_intent_id": payment_intent_id
            }
        ).execute()

        if response.data and len(response.data) > 0:
            result = response.data[0]
            return result.get("success", False)

        return False

    except Exception as e:
        logging.error(f"Error granting tokens: {str(e)}")
        return False


# sql/13-stripe-functions.sql

CREATE OR REPLACE FUNCTION deduct_token_if_sufficient(p_user_id UUID)
RETURNS TABLE(success BOOLEAN, remaining_tokens INTEGER, error_message TEXT)
LANGUAGE plpgsql
AS $$
DECLARE
    current_tokens INTEGER;
BEGIN
    -- CRITICAL: FOR UPDATE locks row to prevent race conditions
    SELECT tokens INTO current_tokens
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    IF current_tokens < 1 THEN
        RETURN QUERY SELECT FALSE, current_tokens, 'Insufficient tokens';
        RETURN;
    END IF;

    -- Atomically deduct token
    UPDATE user_profiles
    SET tokens = tokens - 1,
        updated_at = NOW()
    WHERE id = p_user_id;

    -- Record consumption in transactions table
    INSERT INTO transactions (user_id, transaction_type, amount, tokens)
    VALUES (p_user_id, 'consumption', -1, 1);

    -- Return success with new balance
    RETURN QUERY SELECT TRUE, current_tokens - 1, NULL::TEXT;
END;
$$;


CREATE OR REPLACE FUNCTION grant_tokens_for_purchase(
    p_user_id UUID,
    p_tokens INTEGER,
    p_event_id TEXT,
    p_payment_intent_id TEXT
)
RETURNS TABLE(success BOOLEAN, new_balance INTEGER, error_message TEXT)
LANGUAGE plpgsql
AS $$
DECLARE
    current_balance INTEGER;
    event_exists BOOLEAN;
BEGIN
    -- CRITICAL: Check if event already processed (idempotency)
    SELECT EXISTS(
        SELECT 1 FROM transactions
        WHERE stripe_event_id = p_event_id
    ) INTO event_exists;

    IF event_exists THEN
        RETURN QUERY SELECT FALSE, 0, 'Event already processed';
        RETURN;
    END IF;

    -- Lock user row
    SELECT tokens INTO current_balance
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    -- Grant tokens
    UPDATE user_profiles
    SET tokens = tokens + p_tokens,
        updated_at = NOW()
    WHERE id = p_user_id;

    -- Record purchase in transactions table
    INSERT INTO transactions (
        user_id,
        transaction_type,
        amount,
        tokens,
        stripe_payment_intent_id,
        stripe_event_id
    ) VALUES (
        p_user_id,
        'purchase',
        p_tokens,
        p_tokens,
        p_payment_intent_id,
        p_event_id
    );

    -- Return success with new balance
    RETURN QUERY SELECT TRUE, current_balance + p_tokens, NULL::TEXT;
END;
$$;


// frontend/src/pages/PurchaseTokens.tsx

import { useState, useMemo } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import { Elements } from '@stripe/react-stripe-js';
import { useAuth } from '@/hooks/useAuth';
import CheckoutForm from '@/components/tokens/CheckoutForm';
import PricingCard from '@/components/tokens/PricingCard';

// CRITICAL: Initialize outside component to prevent recreation
const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY);

const PRICING_TIERS = [
  { id: 'tier_1', name: '100 Tokens', price: 5, tokens: 100 },
  { id: 'tier_2', name: '250 Tokens', price: 10, tokens: 250 },
  { id: 'tier_3', name: '600 Tokens', price: 20, tokens: 600 },
];

export default function PurchaseTokens() {
  const { user, session } = useAuth();
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTierSelect = async (tierId: string) => {
    setSelectedTier(tierId);
    setLoading(true);
    setError(null);

    try {
      // PATTERN: Fetch with Bearer token from Supabase auth
      const response = await fetch(
        `${import.meta.env.VITE_AGENT_ENDPOINT.replace('/api/pydantic-agent', '')}/api/create-payment-intent`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${session?.access_token}`,
          },
          body: JSON.stringify({
            tier: tierId,
            user_id: user?.id,
          }),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to create payment intent');
      }

      const data = await response.json();
      setClientSecret(data.client_secret);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Purchase Tokens</h1>

      {!clientSecret && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PRICING_TIERS.map((tier) => (
            <PricingCard
              key={tier.id}
              tier={tier.id}
              name={tier.name}
              price={tier.price}
              tokens={tier.tokens}
              selected={selectedTier === tier.id}
              onClick={() => handleTierSelect(tier.id)}
              loading={loading && selectedTier === tier.id}
            />
          ))}
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      {clientSecret && (
        // PATTERN: Elements provider wraps payment form
        <Elements stripe={stripePromise} options={{ clientSecret }}>
          <CheckoutForm />
        </Elements>
      )}
    </div>
  );
}
```

### Integration Points

```yaml
Backend Integration:
  NEW endpoints in agent_api.py:
    - POST /api/create-payment-intent (after line 189)
    - POST /api/webhook/stripe (after payment intent endpoint)

  MODIFY endpoint in agent_api.py:
    - POST /api/pydantic-agent (add token check at line 206)

  NEW modules:
    - stripe_utils.py (Stripe operations)
    - token_utils.py (Token operations)

  MODIFY files:
    - requirements.txt (add stripe)
    - .env.example (add Stripe keys)

Database Integration:
  NEW tables:
    - transactions (with RLS policies)

  MODIFY tables:
    - user_profiles (add tokens column)

  NEW functions:
    - deduct_token_if_sufficient()
    - grant_tokens_for_purchase()

  Migration strategy:
    - Scripts 10-14 in sql/ directory
    - Execute in order
    - Test each before proceeding

Frontend Integration:
  NEW routes in App.tsx:
    - /purchase-tokens
    - /payment-success
    - /payment-failure
    - /token-history

  NEW pages:
    - PurchaseTokens.tsx
    - PaymentSuccess.tsx
    - PaymentFailure.tsx
    - TokenHistory.tsx

  NEW components:
    - TokenBalance.tsx (display in Chat header)
    - CheckoutForm.tsx (Stripe payment form)
    - PricingCard.tsx (pricing tier display)

  MODIFY files:
    - package.json (add Stripe packages)
    - .env.example (add Stripe publishable key)
    - Chat.tsx (add TokenBalance display)
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Backend validation
cd backend_agent_api
source venv_linux/bin/activate  # CRITICAL: Use venv_linux

# Type checking
mypy stripe_utils.py token_utils.py agent_api.py

# Linting (auto-fix)
ruff check . --fix
ruff format .

# Expected: No errors. If errors, read message and fix.

# Frontend validation
cd frontend

# Type checking
npm run build  # Will fail on TypeScript errors

# Linting
npm run lint

# Expected: No errors. If errors, read message and fix.
```

### Level 2: Unit Tests

```python
# backend_agent_api/tests/test_token_utils.py (CREATE)

import pytest
from unittest.mock import Mock, AsyncMock, patch
from token_utils import (
    check_user_tokens,
    deduct_token_atomic,
    grant_tokens_atomic
)

@pytest.mark.asyncio
async def test_check_user_tokens_success():
    """User has tokens"""
    mock_supabase = Mock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
        data=[{"tokens": 50}]
    )

    result = await check_user_tokens(mock_supabase, "user-123")
    assert result == 50

@pytest.mark.asyncio
async def test_check_user_tokens_no_user():
    """User not found returns 0"""
    mock_supabase = Mock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
        data=[]
    )

    result = await check_user_tokens(mock_supabase, "user-123")
    assert result == 0

@pytest.mark.asyncio
async def test_deduct_token_success():
    """Token deduction succeeds"""
    mock_supabase = Mock()
    mock_supabase.rpc.return_value.execute.return_value = Mock(
        data=[{"success": True, "remaining_tokens": 49}]
    )

    result = await deduct_token_atomic(mock_supabase, "user-123")
    assert result is True

@pytest.mark.asyncio
async def test_deduct_token_insufficient():
    """Token deduction fails - insufficient balance"""
    mock_supabase = Mock()
    mock_supabase.rpc.return_value.execute.return_value = Mock(
        data=[{"success": False, "error_message": "Insufficient tokens"}]
    )

    result = await deduct_token_atomic(mock_supabase, "user-123")
    assert result is False

@pytest.mark.asyncio
async def test_grant_tokens_success():
    """Tokens granted successfully"""
    mock_supabase = Mock()
    mock_supabase.rpc.return_value.execute.return_value = Mock(
        data=[{"success": True, "new_balance": 150}]
    )

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )
    assert result is True

@pytest.mark.asyncio
async def test_grant_tokens_duplicate_event():
    """Duplicate event returns false (idempotency)"""
    mock_supabase = Mock()
    mock_supabase.rpc.return_value.execute.return_value = Mock(
        data=[{"success": False, "error_message": "Event already processed"}]
    )

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )
    assert result is False


# Run tests
# cd backend_agent_api
# source venv_linux/bin/activate
# pytest tests/test_token_utils.py -v
```

## Final Validation Checklist

Backend:
- [ ] All tests pass: source venv_linux/bin/activate && pytest tests/ -v
- [ ] No linting errors: ruff check backend_agent_api/
- [ ] No type errors: mypy backend_agent_api/
- [ ] Token deduction is atomic
- [ ] Idempotency prevents duplicate token grants
- [ ] All error cases handled gracefully

Frontend:
- [ ] No linting errors: npm run lint
- [ ] Payment flow works with test card
- [ ] Success/failure redirects work
- [ ] Token balance displays correctly
- [ ] Token balance updates in real-time
- [ ] Error messages are user-friendly

Documentation:
- [ ] stripe-setup.md created with all setup steps
- [ ] .env.example files updated
- [ ] Environment variables documented
- [ ] Test cards documented
- [ ] Webhook setup instructions complete

Security:
- [ ] Webhook signatures verified
- [ ] No API keys in code or version control
- [ ] RLS policies prevent unauthorized access
- [ ] Admin-only operations protected
- [ ] HTTPS required for production webhook

## Anti-Patterns to Avoid

❌ **Don't** use `SELECT then UPDATE` for token operations - race conditions
✅ **Do** use PostgreSQL functions with `FOR UPDATE` locking

❌ **Don't** skip webhook signature verification - security vulnerability
✅ **Do** always verify with `stripe.Webhook.construct_event()`

❌ **Don't** use parsed JSON for webhook signature - verification will fail
✅ **Do** use raw body string: `(await request.body()).decode('utf-8')`

❌ **Don't** forget idempotency checks - duplicate token grants
✅ **Do** check `stripe_event_id` uniqueness in grant function

❌ **Don't** hardcode Stripe amounts in dollars - will charge wrong amount
✅ **Do** use cents: $5.00 = 500 cents

❌ **Don't** initialize `loadStripe()` inside component - performance issue
✅ **Do** initialize outside component or with `useMemo`

❌ **Don't** assume payment succeeded without checking - user frustration
✅ **Do** check payment intent status in success page

❌ **Don't** expose secret keys in frontend - security breach
✅ **Do** use publishable key in frontend, secret key only in backend

❌ **Don't** commit API keys to git - credential leak
✅ **Do** use .env files and add to .gitignore

❌ **Don't** skip error handling in webhook - failed token grants
✅ **Do** handle all exceptions and return 200 to Stripe (after logging)

---

## Confidence Score: 9/10

**Why 9/10**:
- ✅ Complete context provided (codebase analysis, Stripe docs, Supabase patterns)
- ✅ Exact file references with line numbers
- ✅ Pseudocode for all complex operations
- ✅ Comprehensive testing strategy
- ✅ Security considerations addressed
- ✅ Idempotency and race conditions handled
- ✅ All integration points identified
- ⚠️ Minor uncertainty: Exact UI component styling (but Radix UI patterns provided)

**Validation**: This PRP enables an AI agent unfamiliar with the codebase to implement the feature successfully using only the PRP content, the referenced documentation, and codebase file access.

---

## Additional Notes

### Testing with Stripe

**Test Cards**:
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`
- Requires authentication: `4000 0025 0000 3155`
- Insufficient funds: `4000 0000 0000 9995`

**Test Mode**:
- All test API keys start with `sk_test_` or `pk_test_`
- Test mode charges are not real
- Stripe dashboard has separate test and live modes

### Deployment Considerations

**Environment Variables** (Production):
- Replace all `sk_test_` keys with `sk_live_` keys
- Replace `pk_test_` with `pk_live_`
- Generate new webhook secret for production URL
- Set `ENVIRONMENT=production` in backend

**Webhook Setup** (Production):
- Create webhook in Stripe dashboard
- URL: `https://your-domain.com/api/webhook/stripe`
- Select event: `payment_intent.succeeded`
- Copy webhook signing secret to `STRIPE_WEBHOOK_SECRET`
- Test webhook delivery with Stripe dashboard test tool

**Database Migration** (Production):
- Backup database before running migrations
- Run migrations during low-traffic period
- Execute scripts 10-14 in order
- Verify each script success before next
- Monitor error logs during first hour

### Monitoring and Maintenance

**Key Metrics to Monitor**:
- Webhook delivery success rate
- Failed token deductions (could indicate bugs)
- Average payment conversion rate
- Token consumption rate per user

**Common Issues**:
- Webhook signature verification failures: Check webhook secret
- Duplicate token grants: Verify idempotency working
- Race conditions: Check FOR UPDATE usage in functions
- Failed payments: Check Stripe dashboard for decline reasons