-- Enable Row Level Security on transactions table
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

-- Users can view their own transaction history
CREATE POLICY "Users can view their own transactions"
ON transactions
FOR SELECT
USING (auth.uid() = user_id);

-- Admins can view all transactions
CREATE POLICY "Admins can view all transactions"
ON transactions
FOR SELECT
USING (is_admin());

-- Deny DELETE on transactions (audit trail must be preserved)
CREATE POLICY "Deny delete for transactions"
ON transactions
FOR DELETE
USING (false);

-- Only backend service can insert transactions (via PostgreSQL functions)
-- No direct INSERT policy needed - functions run with elevated permissions