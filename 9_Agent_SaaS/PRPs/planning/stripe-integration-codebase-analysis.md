# Stripe Payment Integration - Codebase Analysis

## Feature Context

Based on INITIAL.md, this analysis covers the implementation of Stripe payment integration for purchasing agent tokens. The feature includes:

- **Frontend**: Purchase options page, success/failure pages, user profile token display, token balance in chat interface, token usage history page
- **Backend**: User table tokens column, transactions table, Stripe webhook endpoint, payment intent creation, token validation in pydantic-agent endpoint
- **Database**: User table modifications, transactions table creation, RLS policies
- **Pricing**: Three tiers - 100 tokens ($5), 250 tokens ($10), 600 tokens ($20)

## Current Codebase Structure Analysis

### Backend (backend_agent_api/)

**FastAPI Application Structure**:
- Main application: `agent_api.py` - Contains FastAPI app with CORS, lifespan management, and single pydantic-agent endpoint
- Authentication: JWT token verification via `verify_token()` function using Supabase auth API
- Request/Response Models: Pydantic models like `AgentRequest`, `FileAttachment`
- Error Handling: Streaming error responses and HTTP exceptions

**Key Patterns**:
- **Authentication Flow**: `verify_token()` function validates JWT tokens and returns user data
- **Rate Limiting**: `check_rate_limit()` function in `db_utils.py` checks requests table
- **Database Operations**: All DB operations are in `db_utils.py` with async functions
- **Request Processing**: `/api/pydantic-agent` endpoint handles all agent interactions
- **Response Streaming**: Uses `StreamingResponse` for real-time token streaming

**Environment Configuration**: 
- Uses python-dotenv for environment variable loading
- Production vs development environment handling
- Client initialization in lifespan manager

**Current Validation Flow in pydantic-agent endpoint**:
1. Token authentication via `verify_token()`
2. User ID validation against token
3. Rate limit check via `check_rate_limit()`
4. Request processing

### Database (sql/)

**Schema Patterns**:
- **File Naming Convention**: Numbered sequence (0-all-tables.sql, 1-user_profiles_requests.sql, etc.)
- **Table Structure**: Uses UUID primary keys, proper foreign key relationships
- **User Table**: `user_profiles` table with id (UUID), email, full_name, is_admin, timestamps
- **RLS Policies**: Comprehensive Row Level Security in separate files (e.g., 2-user_profiles_requests_rls.sql)
- **Admin Functions**: `is_admin()` function for privilege checks

**Current user_profiles table structure**:
```sql
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
```

**RLS Policy Patterns**:
- Users can view/update their own records: `USING (auth.uid() = id)`
- Admins can access all records: `USING (is_admin())`
- Computed columns for session parsing: `GENERATED ALWAYS AS (...) STORED`
- Deny delete policies: `FOR DELETE USING (false)`

### Frontend (frontend/)

**Architecture**:
- **React Router**: BrowserRouter with protected routes using `ProtectedRoute` component
- **Authentication**: `useAuth` hook with AuthProvider context for global auth state
- **State Management**: React Query for server state, local useState for component state
- **UI Framework**: Radix UI components with Tailwind CSS styling
- **File Structure**: Feature-based organization (components/chat/, components/auth/, etc.)

**Key Components**:
- **App.tsx**: Root component with providers and routing
- **Chat.tsx**: Main chat page using composition pattern with custom hooks
- **useAuth.tsx**: Authentication context with Supabase auth integration
- **api.ts**: API client functions for backend communication

**API Integration Patterns**:
- **Authentication**: Supabase client for auth, passes access tokens to backend
- **API Calls**: Uses fetch with Bearer token authorization
- **Streaming**: Supports streaming responses with `onStreamChunk` callbacks
- **Error Handling**: Toast notifications for user feedback

**Current API Endpoints**:
- `VITE_AGENT_ENDPOINT`: Points to `/api/pydantic-agent`
- Supabase direct queries for conversations and messages

**Authentication Flow**:
1. Supabase auth (email/password, Google OAuth)
2. JWT token stored in auth context
3. Token passed to backend API calls
4. User profile synchronization with Google metadata

## Files to Reference in PRP

### Backend Patterns

**C:\Users\colem\dynamous\8_Agent_SaaS\backend_agent_api\agent_api.py**:
- **FastAPI app structure** - Shows how to add new endpoints alongside existing pydantic-agent
- **Authentication pattern** - `verify_token()` function for JWT validation
- **Request models** - Pydantic model structure (lines 153-163)
- **Endpoint structure** - POST endpoint with dependency injection (lines 190-400)
- **Error handling** - Streaming error responses (lines 167-188)
- **Environment variable usage** - Pattern for accessing env vars

**C:\Users\colem\dynamous\8_Agent_SaaS\backend_agent_api\db_utils.py**:
- **Database operation patterns** - Async functions with proper error handling
- **Rate limiting implementation** - `check_rate_limit()` function (lines 190-221)
- **Request tracking** - `store_request()` function (lines 224-243)
- **Supabase client usage** - Query patterns and error handling

**C:\Users\colem\dynamous\8_Agent_SaaS\backend_agent_api\clients.py**:
- **Client initialization patterns** - How to add new service clients
- **Environment configuration** - Pattern for configuring external services

### Database Patterns

**C:\Users\colem\dynamous\8_Agent_SaaS\sql\1-user_profiles_requests.sql**:
- **Table creation pattern** - Shows user_profiles table structure to extend
- **Foreign key relationships** - Proper CASCADE handling
- **Auto-generated fields** - Timestamp and trigger patterns

**C:\Users\colem\dynamous\8_Agent_SaaS\sql\2-user_profiles_requests_rls.sql**:
- **RLS policy structure** - User-level and admin-level access patterns
- **Policy naming convention** - Descriptive policy names
- **Admin function usage** - `is_admin()` function integration

**C:\Users\colem\dynamous\8_Agent_SaaS\sql\3-conversations_messages.sql**:
- **Complex table structure** - Shows patterns for transaction-like tables
- **Computed columns** - Generated fields for data parsing
- **Indexing strategy** - Performance optimization patterns

**C:\Users\colem\dynamous\8_Agent_SaaS\sql\4-conversations_messages_rls.sql**:
- **Advanced RLS patterns** - Computed column usage in policies
- **Multi-table access control** - User and admin access separation

### Frontend Patterns

**C:\Users\colem\dynamous\8_Agent_SaaS\frontend\src\App.tsx**:
- **Route structure** - How to add new protected routes (lines 43-68)
- **Provider hierarchy** - Order of context providers
- **Authentication integration** - ProtectedRoute component usage

**C:\Users\colem\dynamous\8_Agent_SaaS\frontend\src\lib\api.ts**:
- **API client patterns** - Fetch with authentication headers
- **Streaming response handling** - Complex streaming logic (lines 57-180)
- **Error handling** - Comprehensive error handling patterns
- **Request structure** - Payload formatting

**C:\Users\colem\dynamous\8_Agent_SaaS\frontend\src\hooks\useAuth.tsx**:
- **Authentication context pattern** - Context setup and provider
- **Supabase integration** - Auth state management
- **User profile synchronization** - Automatic profile updates

**C:\Users\colem\dynamous\8_Agent_SaaS\frontend\src\pages\Chat.tsx**:
- **Page component structure** - Hook composition pattern
- **State management** - Multiple useState hooks coordination
- **Effect handling** - useEffect patterns for data loading

**C:\Users\colem\dynamous\8_Agent_SaaS\frontend\src\types\database.types.ts**:
- **TypeScript type definitions** - Database type patterns
- **Supabase type integration** - Generated type usage

## Existing Patterns to Follow

### Naming Conventions

**Backend**:
- **Files**: snake_case (agent_api.py, db_utils.py)
- **Functions**: snake_case (verify_token, store_message)
- **Variables**: snake_case (user_id, session_id)
- **Classes**: PascalCase (AgentRequest, FileAttachment)

**Frontend**:
- **Files**: PascalCase for components (Chat.tsx), camelCase for utilities (api.ts)
- **Components**: PascalCase (ProtectedRoute, ChatLayout)
- **Hooks**: camelCase with 'use' prefix (useAuth, useMessageHandling)
- **Variables**: camelCase (selectedConversation, isLoading)

**Database**:
- **Tables**: snake_case (user_profiles, conversations)
- **Columns**: snake_case (created_at, user_id)
- **Policies**: Descriptive names ("Users can view their own profile")

### Import Patterns

**Backend**:
```python
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from supabase import Client
```

**Frontend**:
```typescript
import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/ui/button';
```

### Error Handling Patterns

**Backend**:
- HTTP exceptions with descriptive messages
- Try-catch blocks with logging
- Streaming error responses for user feedback

**Frontend**:
- Toast notifications for user feedback
- Loading states and error states
- Graceful fallbacks

### Database Query Patterns

**Supabase Queries**:
```python
response = supabase.table("table_name") \
    .select("*") \
    .eq("user_id", user_id) \
    .execute()
```

### API Response Patterns

**Backend**:
- Streaming JSON responses for real-time updates
- Structured error messages
- Consistent data formats

**Frontend**:
- Fetch with Bearer token authentication
- Streaming response handling
- Type-safe response parsing

## Integration Points

### Backend Integration Points

**New API Endpoints**:
- Add to `agent_api.py` alongside existing `/api/pydantic-agent` endpoint
- Follow same authentication pattern with `verify_token()` dependency
- Use same error handling and response patterns

**Pydantic-Agent Endpoint Modification**:
- Add token check after line 206 (rate limiting check)
- Add token deduction before agent processing
- Update error handling to include token-related errors

**Database Integration**:
- Add token operations to `db_utils.py`
- Follow existing async function patterns
- Use same Supabase client instance

### Database Integration Points

**User Table Extension**:
- Add tokens column to existing `user_profiles` table
- Create migration script following numbered naming convention
- Update RLS policies to include new column

**Transactions Table**:
- Create new table following existing patterns in `sql/` directory
- Use UUID foreign keys to `user_profiles`
- Implement RLS policies similar to `conversations` table

### Frontend Integration Points

**New Routes**:
- Add to `App.tsx` routing structure (around line 50-60)
- Use `ProtectedRoute` wrapper for authentication
- Follow existing route pattern

**API Integration**:
- Extend `api.ts` with new functions for payment and token operations
- Follow existing authentication and error handling patterns
- Use same fetch patterns with Bearer tokens

**User Interface**:
- Add token display to existing chat interface
- Create new pages following existing page component patterns
- Use existing UI components from `components/ui/`

**Supabase Queries**:
- Add direct Supabase queries for token history (following `fetchConversations` pattern)
- Use existing Supabase client from `lib/supabase.ts`

## Implementation Strategy (STEP-BY-STEP)

### Phase 1: Database Schema Updates
1. Create migration script `sql/10-stripe-tokens.sql` with user_profiles tokens column
2. Create `sql/11-stripe-transactions.sql` with transactions table
3. Create `sql/12-stripe-tokens-rls.sql` with RLS policies

### Phase 2: Backend Implementation
1. Add Stripe client configuration to `clients.py`
2. Add token utility functions to `db_utils.py`
3. Create new endpoints in `agent_api.py` (payment intent, webhook)
4. Modify pydantic-agent endpoint to check/deduct tokens
5. Add Stripe webhook signature verification

### Phase 3: Frontend Implementation
1. Install Stripe packages (@stripe/stripe-js, @stripe/react-stripe-js)
2. Add Stripe provider to App.tsx
3. Create purchase page components
4. Add success/failure pages
5. Update user profile with token display
6. Add token balance to chat interface
7. Create token usage history page

### Phase 4: Environment Configuration
1. Update .env.example with Stripe variables
2. Document Stripe setup requirements

## Potential Challenges

### Technical Challenges

**Stripe Webhook Idempotency**:
- Need to prevent duplicate token grants on webhook retries
- Use Stripe event ID as idempotency key in transactions table
- Handle race conditions in concurrent webhook processing

**Token Validation Race Conditions**:
- Multiple simultaneous requests could cause token count issues
- Implement atomic token deduction using database transactions
- Consider using Supabase RPC functions for atomic operations

**Frontend Payment Flow**:
- Handle payment redirect flows properly
- Manage loading states during payment processing
- Implement proper error recovery for failed payments

### Database Migration Challenges**:
- Existing users need default token values
- Ensure migration is reversible
- Handle production deployment without downtime

### Authentication Integration**:
- Stripe webhook authentication vs Supabase JWT
- Separate authentication for webhook endpoint
- Ensure webhook endpoint bypasses normal auth middleware

### Testing Complexity**:
- Stripe webhook testing requires proper signature verification
- Payment flow testing with Stripe test mode
- Integration testing with multiple external services

This analysis provides the foundation for implementing Stripe integration while following the established patterns and conventions of the codebase.
