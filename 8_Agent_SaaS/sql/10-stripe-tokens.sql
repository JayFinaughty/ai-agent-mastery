-- Add tokens column to user_profiles table for Stripe payment integration

-- Add tokens column with constraints
ALTER TABLE user_profiles
ADD COLUMN tokens INTEGER DEFAULT 0 NOT NULL CHECK (tokens >= 0);

-- Add comment for documentation
COMMENT ON COLUMN user_profiles.tokens IS
'Number of tokens available for AI agent interactions. Purchased via Stripe.';