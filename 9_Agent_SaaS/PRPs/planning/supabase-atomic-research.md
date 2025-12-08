# Supabase Python Client: Atomic Operations and Transaction Handling Research

## Executive Summary

Supabase Python client does not provide native transaction support for client-side operations. The recommended approach for atomic operations is to implement business logic within PostgreSQL functions and invoke them via RPC (Remote Procedure Calls). This ensures proper concurrency handling and atomic operations at the database level.

## 1. Atomic Updates

### Key Findings

**Native Client Limitations:**
- Supabase Python client does not support multi-query transactions
- No BEGIN/COMMIT/ROLLBACK transaction control from client
- Individual UPDATE operations are atomic by nature in PostgreSQL

**Atomic Nature of UPDATE Operations:**
According to Supabase community discussions, individual `UPDATE...WHERE` operations are atomic:
- SQL UPDATE reads and writes back the row without allowing other changes until complete
- The WHERE clause is part of the UPDATE operation
- No other process can select a row that's already selected for update

**Source:** [Supabase Discussion #26271](https://github.com/orgs/supabase/discussions/26271)

### PostgreSQL Function Approach

**Recommended Pattern:**
```sql
-- Example atomic token deduction function
CREATE OR REPLACE FUNCTION deduct_tokens(
    user_id UUID,
    amount INTEGER
) RETURNS TABLE(
    success BOOLEAN,
    remaining_tokens INTEGER,
    error_message TEXT
) AS $$
DECLARE
    current_tokens INTEGER;
BEGIN
    -- Lock the user row for update
    SELECT tokens INTO current_tokens
    FROM users
    WHERE id = user_id
    FOR UPDATE;

    -- Check if user exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    -- Check sufficient balance
    IF current_tokens < amount THEN
        RETURN QUERY SELECT FALSE, current_tokens, 'Insufficient tokens';
        RETURN;
    END IF;

    -- Perform atomic deduction
    UPDATE users
    SET tokens = tokens - amount,
        updated_at = NOW()
    WHERE id = user_id;

    -- Return success with new balance
    RETURN QUERY SELECT TRUE, current_tokens - amount, NULL::TEXT;
END;
$$ LANGUAGE plpgsql;
```

**Python Client Usage:**
```python
# Call atomic function via RPC
response = supabase.rpc(
    'deduct_tokens',
    {
        'user_id': str(user_uuid),
        'amount': 10
    }
).execute()

result = response.data[0]
if result['success']:
    print(f"Tokens deducted. Remaining: {result['remaining_tokens']}")
else:
    print(f"Error: {result['error_message']}")
```

### Race Condition Prevention

**FOR UPDATE Locking:**
```sql
-- Prevents concurrent modifications
SELECT * FROM table_name WHERE condition FOR UPDATE;
```

**Common Pitfalls to Avoid:**
- Don't use SELECT followed by UPDATE (introduces race conditions)
- Avoid client-side balance checking before updates
- Always handle insufficient balance at database level

## 2. Transaction Patterns

### Current State
**No Native Support:** Supabase client libraries don't support long-running transactions as confirmed in [Supabase Discussion #526](https://github.com/orgs/supabase/discussions/526).

**Recommended Workaround:** Encapsulate transaction logic in PostgreSQL functions and call via RPC.

### PostgreSQL Function Patterns

**Simple Transaction Function:**
```sql
CREATE OR REPLACE FUNCTION transfer_tokens(
    from_user_id UUID,
    to_user_id UUID,
    amount INTEGER
) RETURNS TABLE(
    success BOOLEAN,
    message TEXT
) AS $$
BEGIN
    -- Start implicit transaction
    -- Lock both user rows in consistent order to prevent deadlocks
    PERFORM 1 FROM users
    WHERE id IN (from_user_id, to_user_id)
    ORDER BY id
    FOR UPDATE;

    -- Validate sender has sufficient tokens
    IF (SELECT tokens FROM users WHERE id = from_user_id) < amount THEN
        RETURN QUERY SELECT FALSE, 'Insufficient tokens';
        RETURN;
    END IF;

    -- Validate recipient exists
    IF NOT EXISTS (SELECT 1 FROM users WHERE id = to_user_id) THEN
        RETURN QUERY SELECT FALSE, 'Recipient not found';
        RETURN;
    END IF;

    -- Perform atomic transfer
    UPDATE users SET tokens = tokens - amount WHERE id = from_user_id;
    UPDATE users SET tokens = tokens + amount WHERE id = to_user_id;

    RETURN QUERY SELECT TRUE, 'Transfer completed';
EXCEPTION
    WHEN OTHERS THEN
        -- Automatic rollback on any error
        RETURN QUERY SELECT FALSE, SQLERRM;
END;
$$ LANGUAGE plpgsql;
```

### Rollback Handling

**Automatic Rollback:** PostgreSQL functions automatically rollback on exceptions:
```sql
-- Function with error handling
CREATE OR REPLACE FUNCTION safe_operation()
RETURNS BOOLEAN AS $$
BEGIN
    -- Multiple operations here
    INSERT INTO table1 VALUES (...);
    UPDATE table2 SET ...;

    RETURN TRUE;
EXCEPTION
    WHEN unique_violation THEN
        -- Specific error handling
        RAISE NOTICE 'Duplicate key error';
        RETURN FALSE;
    WHEN OTHERS THEN
        -- General error handling
        RAISE NOTICE 'Error: %', SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
```

**Source:** [Supabase Database Functions Guide](https://supabase.com/docs/guides/database/functions)

## 3. RPC Function Creation

### Creating PostgreSQL Functions

**Basic Syntax:**
```sql
CREATE OR REPLACE FUNCTION function_name(param1 TYPE, param2 TYPE)
RETURNS return_type
LANGUAGE plpgsql
AS $$
BEGIN
    -- Function body
    RETURN result;
END;
$$;
```

**Source:** [Supabase Database Functions Documentation](https://supabase.com/docs/guides/database/functions)

### Python Client RPC Usage

**Method Signature:**
```python
supabase.rpc(fn, params=None, get=False, head=False, count=None)
```

**Parameters:**
- `fn` (Required): Name of the stored procedure
- `params` (Optional): Dictionary of parameters
- `get` (Optional): Prevents returning data when True
- `head` (Optional): Read-only access mode
- `count` (Optional): Row counting algorithm

**Source:** [Python RPC Reference](https://supabase.com/docs/reference/python/rpc)

### Practical Examples

**1. Function without arguments:**
```python
response = supabase.rpc("hello_world").execute()
```

**2. Function with arguments:**
```python
response = supabase.rpc("echo", {"say": "👋"}).execute()
```

**3. Function with filters:**
```python
response = (
    supabase.rpc("list_stored_planets")
    .eq("id", 1)
    .single()
    .execute()
)
```

### Error Handling in Database Functions

**Exception Handling Pattern:**
```sql
CREATE OR REPLACE FUNCTION validate_and_update_tokens(
    user_id UUID,
    required_tokens INTEGER
) RETURNS JSON AS $$
DECLARE
    current_balance INTEGER;
    result JSON;
BEGIN
    -- Validate input
    IF user_id IS NULL OR required_tokens <= 0 THEN
        RAISE EXCEPTION 'Invalid parameters: user_id=%, tokens=%', user_id, required_tokens;
    END IF;

    -- Get current balance with lock
    SELECT tokens INTO current_balance
    FROM users
    WHERE id = user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'User not found: %', user_id;
    END IF;

    IF current_balance < required_tokens THEN
        RAISE EXCEPTION 'Insufficient tokens: required=%, available=%', required_tokens, current_balance;
    END IF;

    -- Update tokens
    UPDATE users
    SET tokens = tokens - required_tokens,
        updated_at = NOW()
    WHERE id = user_id;

    result := json_build_object(
        'success', true,
        'remaining_tokens', current_balance - required_tokens,
        'deducted', required_tokens
    );

    RETURN result;

EXCEPTION
    WHEN OTHERS THEN
        result := json_build_object(
            'success', false,
            'error', SQLERRM,
            'error_code', SQLSTATE
        );
        RETURN result;
END;
$$ LANGUAGE plpgsql;
```

## 4. Row Level Security with Transactions

### RLS with PostgreSQL Functions

**Key Considerations:**
- RLS policies are evaluated for each query within functions
- Functions run with caller's permissions by default (SECURITY INVOKER)
- `auth.uid()` and `auth.jwt()` work within functions

**Source:** [Row Level Security Documentation](https://supabase.com/docs/guides/database/postgres/row-level-security)

### Security Models

**1. Security Invoker (Default - Recommended):**
```sql
CREATE OR REPLACE FUNCTION user_update_tokens(amount INTEGER)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY INVOKER  -- Uses caller's permissions
AS $$
BEGIN
    -- RLS policies apply here using caller's context
    UPDATE users
    SET tokens = tokens + amount
    WHERE id = auth.uid();  -- Current user's ID

    RETURN FOUND;
END;
$$;
```

**2. Security Definer (Use with Caution):**
```sql
CREATE OR REPLACE FUNCTION admin_reset_tokens(user_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER  -- Uses function owner's permissions
SET search_path = public  -- Always set search_path for security
AS $$
BEGIN
    -- Bypasses RLS - function runs with elevated permissions
    UPDATE users SET tokens = 0 WHERE id = user_id;
    RETURN FOUND;
END;
$$;
```

### RLS Policy Examples

**User-specific token access:**
```sql
-- Enable RLS on users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policy for token operations
CREATE POLICY "Users can modify their own tokens"
ON users
FOR UPDATE
USING (id = auth.uid());

-- Policy for reading token balance
CREATE POLICY "Users can view their own tokens"
ON users
FOR SELECT
USING (id = auth.uid());
```

### Security Best Practices

**1. Always validate user context:**
```sql
-- Validate user in function
IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'Authentication required';
END IF;
```

**2. Use helper functions for common checks:**
```sql
-- Helper function for user validation
CREATE OR REPLACE FUNCTION is_authenticated()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN auth.uid() IS NOT NULL;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER;
```

**3. Performance optimization for RLS:**
- Add indexes on columns used in policies
- Use `SELECT` when calling functions in policies
- Minimize joins in policy logic

## 5. Error Handling

### Supabase Python Client Error Patterns

**Error Categories:**
Based on research, Supabase RPCs can fail due to:
- **Network Errors**: Connection issues, timeouts
- **Resource Errors**: Over-utilization, 5xx responses
- **Usage Errors**: Incorrect parameters, function not found

**Source:** [Supabase RPC Python Integration Guide](https://www.restack.io/docs/supabase-knowledge-supabase-rpc-python-integration)

### Python Error Handling Patterns

**Basic Exception Handling:**
```python
from supabase import create_client
import logging

def deduct_user_tokens(user_id: str, amount: int) -> dict:
    """
    Safely deduct tokens with comprehensive error handling.

    Returns:
        dict: {'success': bool, 'data': any, 'error': str}
    """
    try:
        response = supabase.rpc(
            'deduct_tokens',
            {
                'user_id': user_id,
                'amount': amount
            }
        ).execute()

        if response.data and len(response.data) > 0:
            result = response.data[0]
            return {
                'success': result.get('success', False),
                'data': result,
                'error': result.get('error_message')
            }
        else:
            return {
                'success': False,
                'data': None,
                'error': 'No data returned from function'
            }

    except Exception as e:
        logging.error(f"RPC call failed: {str(e)}")
        return {
            'success': False,
            'data': None,
            'error': f'Network or system error: {str(e)}'
        }
```

### Database Constraint Violations

**Common PostgreSQL Error Codes:**
```python
CONSTRAINT_ERRORS = {
    '23505': 'Duplicate key violation',
    '23503': 'Foreign key violation',
    '23514': 'Check constraint violation',
    '23502': 'Not null violation'
}

def handle_database_error(error_code: str, error_message: str) -> str:
    """Map database error codes to user-friendly messages."""
    if error_code in CONSTRAINT_ERRORS:
        return CONSTRAINT_ERRORS[error_code]
    return f"Database error: {error_message}"
```

**Function-level Constraint Handling:**
```sql
CREATE OR REPLACE FUNCTION safe_token_deduction(
    user_id UUID,
    amount INTEGER
) RETURNS JSON AS $$
BEGIN
    -- Input validation
    IF amount <= 0 THEN
        RETURN json_build_object('success', false, 'error', 'Amount must be positive');
    END IF;

    -- Business logic with constraint checking
    UPDATE users
    SET tokens = tokens - amount
    WHERE id = user_id AND tokens >= amount;

    IF NOT FOUND THEN
        RETURN json_build_object('success', false, 'error', 'Insufficient tokens or user not found');
    END IF;

    RETURN json_build_object('success', true, 'message', 'Tokens deducted successfully');

EXCEPTION
    WHEN check_violation THEN
        RETURN json_build_object('success', false, 'error', 'Token balance cannot be negative');
    WHEN foreign_key_violation THEN
        RETURN json_build_object('success', false, 'error', 'Invalid user reference');
END;
$$ LANGUAGE plpgsql;
```

### Network Errors and Retries

**Important Note:** Supabase will not rollback write operations once they complete, even if network fails afterward. Always query the database to verify state.

**Retry Pattern with Exponential Backoff:**
```python
import time
import random
from typing import Optional

def rpc_with_retry(
    function_name: str,
    params: dict,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[dict]:
    """
    Execute RPC with exponential backoff retry logic.

    Args:
        function_name: PostgreSQL function name
        params: Function parameters
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries

    Returns:
        Response data or None if all retries failed
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = supabase.rpc(function_name, params).execute()
            return response.data

        except Exception as e:
            last_error = e

            # Don't retry on client errors (4xx)
            if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                break

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"RPC attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
            else:
                logging.error(f"RPC failed after {max_retries + 1} attempts: {e}")

    raise last_error
```

### Verification Pattern for Write Operations

**Post-Write Verification:**
```python
def verified_token_deduction(user_id: str, amount: int) -> dict:
    """
    Deduct tokens with verification to handle network issues.
    """
    # Get initial balance
    initial_response = supabase.table('users').select('tokens').eq('id', user_id).execute()
    if not initial_response.data:
        return {'success': False, 'error': 'User not found'}

    initial_balance = initial_response.data[0]['tokens']
    expected_balance = initial_balance - amount

    # Attempt deduction
    try:
        deduction_response = supabase.rpc(
            'deduct_tokens',
            {'user_id': user_id, 'amount': amount}
        ).execute()

        # Network might fail here, so verify the operation
        verification_response = supabase.table('users').select('tokens').eq('id', user_id).execute()

        if verification_response.data:
            actual_balance = verification_response.data[0]['tokens']
            if actual_balance == expected_balance:
                return {'success': True, 'balance': actual_balance}
            elif actual_balance == initial_balance:
                return {'success': False, 'error': 'Deduction failed - balance unchanged'}
            else:
                return {'success': False, 'error': 'Unexpected balance state'}

    except Exception as e:
        # Verify if operation succeeded despite network error
        verification_response = supabase.table('users').select('tokens').eq('id', user_id).execute()
        if verification_response.data:
            actual_balance = verification_response.data[0]['tokens']
            if actual_balance == expected_balance:
                return {'success': True, 'balance': actual_balance, 'warning': 'Operation succeeded despite network error'}

        return {'success': False, 'error': f'Network error: {str(e)}'}
```

## Common Pitfalls to Avoid

### 1. Race Conditions
- **Don't:** Use SELECT then UPDATE pattern
- **Do:** Use atomic functions with FOR UPDATE locking

### 2. Security Issues
- **Don't:** Use SECURITY DEFINER without setting search_path
- **Do:** Use SECURITY INVOKER and proper RLS policies

### 3. Error Handling
- **Don't:** Assume network success means database success
- **Do:** Implement verification patterns for critical operations

### 4. Performance
- **Don't:** Create policies without proper indexes
- **Do:** Add indexes on RLS policy columns

## Best Practices Summary

1. **Use PostgreSQL Functions for Atomic Operations**
2. **Implement Proper Error Handling at Function Level**
3. **Use RLS with SECURITY INVOKER for User Context**
4. **Add Retry Logic with Exponential Backoff**
5. **Verify Critical Operations Post-Network-Failure**
6. **Index Columns Used in RLS Policies**
7. **Use FOR UPDATE Locking to Prevent Race Conditions**

## References

- [Supabase Python RPC Reference](https://supabase.com/docs/reference/python/rpc)
- [Database Functions Guide](https://supabase.com/docs/guides/database/functions)
- [Row Level Security Documentation](https://supabase.com/docs/guides/database/postgres/row-level-security)
- [Client-side Transactions Discussion](https://github.com/orgs/supabase/discussions/526)
- [Atomic UPDATE Discussion](https://github.com/orgs/supabase/discussions/26271)
- [One-Time Token Tutorial](https://makerkit.dev/blog/tutorials/one-time-tokens-supabase-postgres)
- [Supabase RPC Integration Guide](https://www.restack.io/docs/supabase-knowledge-supabase-rpc-python-integration)