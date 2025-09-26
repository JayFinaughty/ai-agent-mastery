-- Create transactions table for audit trail of token purchases and consumption

-- Drop existing table if it exists
DROP TABLE IF EXISTS transactions CASCADE;

-- Create transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    transaction_type TEXT NOT NULL CHECK (
        transaction_type IN ('purchase', 'consumption')
    ),
    amount INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    stripe_payment_intent_id TEXT,
    stripe_event_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create unique index for idempotency (prevent duplicate webhook processing)
CREATE UNIQUE INDEX idx_transactions_stripe_event_id
ON transactions(stripe_event_id)
WHERE stripe_event_id IS NOT NULL;

-- Create indexes for query performance
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE transactions IS
'Audit trail for all token purchases and consumption events';

COMMENT ON COLUMN transactions.transaction_type IS
'Type of transaction: purchase (via Stripe) or consumption (agent usage)';

COMMENT ON COLUMN transactions.stripe_event_id IS
'Stripe webhook event ID for idempotency checking';