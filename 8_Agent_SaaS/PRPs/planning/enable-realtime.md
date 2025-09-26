# Enable Supabase Realtime for Token Balance Updates

## Problem

Token balance doesn't update in real-time when:
1. Tokens are deducted after sending a message to the AI agent
2. Tokens are granted after a successful purchase

Users have to refresh the page to see their updated token balance.

## Solution

Enable Supabase Realtime subscriptions on the `user_profiles` table.

## Steps to Enable Realtime

### Option 1: Via Supabase Dashboard (Recommended)

1. **Go to your Supabase project dashboard**
   - Navigate to https://supabase.com/dashboard/project/YOUR_PROJECT_ID

2. **Open Database settings**
   - Click on "Database" in the left sidebar
   - Click on "Replication" tab

3. **Enable replication for user_profiles table**
   - Find `user_profiles` in the list of tables
   - Toggle the switch to enable replication
   - Click "Save"

4. **Verify replication is enabled**
   - The `user_profiles` row should show "Enabled" under the replication column

### Option 2: Via SQL (Alternative)

Run this SQL in the Supabase SQL Editor:

```sql
-- Enable realtime for user_profiles table
ALTER PUBLICATION supabase_realtime ADD TABLE user_profiles;

-- Verify it was added
SELECT * FROM pg_publication_tables
WHERE pubname = 'supabase_realtime'
AND tablename = 'user_profiles';
```

If you see a row returned, realtime is enabled.

## How It Works

The `TokenBalance` component already has the subscription code:

```typescript
const channel = supabase
  .channel('token-updates')
  .on(
    'postgres_changes',
    {
      event: '*',
      schema: 'public',
      table: 'user_profiles',
      filter: `id=eq.${user.id}`,
    },
    (payload) => {
      if (payload.new && 'tokens' in payload.new) {
        setTokens((payload.new as { tokens: number }).tokens);
      }
    }
  )
  .subscribe();
```

This listens for ANY change (`event: '*'`) to the `user_profiles` table where `id` matches the current user.

## Testing Real-Time Updates

1. **Enable realtime** using one of the options above

2. **Open your frontend** and login

3. **Open browser DevTools** → Network tab → Filter by "WS" (WebSocket)
   - You should see a WebSocket connection to Supabase realtime

4. **Test token deduction**:
   - Send a message to the AI agent
   - Watch your token balance decrease in real-time (no refresh needed)

5. **Test token purchase**:
   - Purchase tokens via Stripe
   - After webhook processes, balance should update automatically
   - Payment success page should also show new balance

## Troubleshooting

### Realtime not working after enabling

1. **Check RLS policies**:
   ```sql
   -- User must be able to SELECT their own profile
   SELECT * FROM user_profiles WHERE id = auth.uid();
   ```
   If this returns no rows, check your RLS policies in `sql/2-user_profiles_requests_rls.sql`

2. **Check browser console** for errors:
   - Look for Supabase realtime connection errors
   - Check that the subscription was successful

3. **Verify WebSocket connection**:
   - DevTools → Network → WS tab
   - Should see `realtime-v1.supabase.co` connection
   - Status should be "101 Switching Protocols"

4. **Test with SQL update**:
   ```sql
   -- Update your own tokens manually
   UPDATE user_profiles
   SET tokens = tokens + 10
   WHERE id = 'YOUR_USER_ID';
   ```

   The UI should update immediately without refresh.

### Still not working?

Check these common issues:

1. **Supabase client configuration**:
   - Ensure `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` are correct
   - Rebuild frontend if you changed env vars

2. **Row Level Security**:
   - Realtime respects RLS policies
   - User must have SELECT permission on their own row

3. **Network issues**:
   - WebSocket may be blocked by firewall/proxy
   - Try disabling VPN or browser extensions

## Performance Considerations

Realtime subscriptions are lightweight:
- Each user only subscribes to their own row: `filter: id=eq.${user.id}`
- Subscription is cleaned up when component unmounts
- No polling - uses WebSocket push notifications

## Security

✅ **Safe**: Realtime respects Row Level Security policies
✅ **Filtered**: Users only receive updates for their own data
✅ **Authenticated**: Requires valid Supabase anon key

The subscription filter `id=eq.${user.id}` ensures users only receive updates for their own token balance, not other users' data.