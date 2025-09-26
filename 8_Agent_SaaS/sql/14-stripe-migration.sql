-- Migration script for existing users
-- Ensures all existing user_profiles have the tokens column set to 0

-- Update any NULL tokens values to 0 (should not happen with NOT NULL constraint)
UPDATE user_profiles
SET tokens = 0
WHERE tokens IS NULL;

-- Verify migration
DO $$
DECLARE
    null_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO null_count
    FROM user_profiles
    WHERE tokens IS NULL;

    IF null_count > 0 THEN
        RAISE EXCEPTION 'Migration failed: % users still have NULL tokens', null_count;
    ELSE
        RAISE NOTICE 'Migration successful: All users have valid token balances';
    END IF;
END $$;