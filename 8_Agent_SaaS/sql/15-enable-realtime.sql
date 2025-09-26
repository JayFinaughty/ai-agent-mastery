-- ============================================================================
-- Enable Realtime for Token Balance Updates
-- ============================================================================
-- This script enables Supabase Realtime on the user_profiles table
-- to support real-time token balance updates in the frontend UI

-- Enable realtime replication for user_profiles table
ALTER PUBLICATION supabase_realtime ADD TABLE user_profiles;

-- Verify realtime is enabled
SELECT
    schemaname,
    tablename,
    'Realtime enabled' as status
FROM pg_publication_tables
WHERE pubname = 'supabase_realtime'
AND tablename = 'user_profiles';

-- Expected result: One row showing user_profiles with status "Realtime enabled"
-- If no rows returned, realtime is NOT enabled

-- ============================================================================
-- What This Enables
-- ============================================================================
-- ✅ Real-time token balance updates when user sends messages to AI agent
-- ✅ Instant token balance refresh after successful Stripe purchases
-- ✅ No page refresh needed - updates via WebSocket push notifications
-- ✅ Filtered subscriptions - users only see their own token changes
-- ✅ Secure - respects Row Level Security (RLS) policies

-- ============================================================================
-- How It Works
-- ============================================================================
-- Frontend TokenBalance component subscribes to changes:
--
-- supabase
--   .channel('token-updates')
--   .on('postgres_changes', {
--     event: '*',
--     schema: 'public',
--     table: 'user_profiles',
--     filter: `id=eq.${user.id}`
--   }, (payload) => {
--     setTokens(payload.new.tokens);
--   })
--   .subscribe();
--
-- When backend updates tokens via deduct_token_if_sufficient() or
-- grant_tokens_for_purchase(), Supabase broadcasts the change to all
-- subscribed clients, and the UI updates automatically.

-- ============================================================================
-- Testing Realtime
-- ============================================================================
-- After running this script:
--
-- 1. Login to your frontend application
-- 2. Open browser DevTools → Network → WS (WebSocket) tab
-- 3. You should see a WebSocket connection to realtime-v1.supabase.co
-- 4. Send a message to the AI agent
-- 5. Watch the token balance decrease in real-time (no refresh)
--
-- To test manually via SQL:
-- UPDATE user_profiles SET tokens = tokens + 10 WHERE id = auth.uid();
-- (Your frontend should update immediately)

-- ============================================================================
-- Troubleshooting
-- ============================================================================
-- If realtime doesn't work after running this:
--
-- 1. Check RLS policies allow SELECT on user_profiles for authenticated users
-- 2. Verify WebSocket connection in browser DevTools
-- 3. Check Supabase anon key is correct in frontend .env
-- 4. Ensure user is authenticated (realtime requires valid session)
-- 5. See detailed troubleshooting: PRPs/planning/enable-realtime.md