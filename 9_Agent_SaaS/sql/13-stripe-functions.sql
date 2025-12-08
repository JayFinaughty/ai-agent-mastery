-- PostgreSQL functions for atomic token operations

-- Function to atomically deduct one token if user has sufficient balance
CREATE OR REPLACE FUNCTION deduct_token_if_sufficient(p_user_id UUID)
RETURNS TABLE(success BOOLEAN, remaining_tokens INTEGER, error_message TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_tokens INTEGER;
BEGIN
    -- Lock the user row to prevent race conditions
    SELECT tokens INTO current_tokens
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    -- Check if user exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    -- Check if user has sufficient tokens
    IF current_tokens < 1 THEN
        RETURN QUERY SELECT FALSE, current_tokens, 'Insufficient tokens';
        RETURN;
    END IF;

    -- Atomically deduct one token
    UPDATE user_profiles
    SET tokens = tokens - 1,
        updated_at = NOW()
    WHERE id = p_user_id;

    -- Record consumption in transactions table
    INSERT INTO transactions (
        user_id,
        transaction_type,
        amount,
        tokens
    ) VALUES (
        p_user_id,
        'consumption',
        -1,
        1
    );

    -- Return success with new balance
    RETURN QUERY SELECT TRUE, current_tokens - 1, NULL::TEXT;
END;
$$;

-- Function to atomically grant tokens from successful Stripe payment
CREATE OR REPLACE FUNCTION grant_tokens_for_purchase(
    p_user_id UUID,
    p_tokens INTEGER,
    p_event_id TEXT,
    p_payment_intent_id TEXT
)
RETURNS TABLE(success BOOLEAN, new_balance INTEGER, error_message TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_balance INTEGER;
    event_exists BOOLEAN;
BEGIN
    -- Check if event has already been processed (idempotency)
    SELECT EXISTS(
        SELECT 1 FROM transactions
        WHERE stripe_event_id = p_event_id
    ) INTO event_exists;

    IF event_exists THEN
        RETURN QUERY SELECT FALSE, 0, 'Event already processed';
        RETURN;
    END IF;

    -- Lock user row to prevent race conditions
    SELECT tokens INTO current_balance
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    -- Check if user exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    -- Atomically grant tokens
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

-- Add comments for documentation
COMMENT ON FUNCTION deduct_token_if_sufficient IS
'Atomically deduct one token from user balance with FOR UPDATE locking';

COMMENT ON FUNCTION grant_tokens_for_purchase IS
'Atomically grant tokens from Stripe payment with idempotency checking';

-- Grant execute permissions to service role
GRANT EXECUTE ON FUNCTION deduct_token_if_sufficient(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION grant_tokens_for_purchase(UUID, INTEGER, TEXT, TEXT) TO service_role;