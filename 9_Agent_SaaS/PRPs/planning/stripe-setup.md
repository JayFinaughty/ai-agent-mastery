# Stripe Payment Integration Setup Guide

Complete guide for setting up Stripe payment integration for token-based agent access.

## Prerequisites

- Stripe account (create at https://stripe.com)
- Supabase project with database access
- Backend and frontend deployed and accessible

## Part 1: Database Setup

### 1. Run SQL Migrations

Execute the following SQL scripts in order in your Supabase SQL Editor:

```sql
-- 1. Add tokens column to user_profiles
-- Run: sql/10-stripe-tokens.sql

-- 2. Create transactions table
-- Run: sql/11-stripe-transactions.sql

-- 3. Enable RLS on transactions
-- Run: sql/12-stripe-tokens-rls.sql

-- 4. Create atomic functions
-- Run: sql/13-stripe-functions.sql

-- 5. Migrate existing users
-- Run: sql/14-stripe-migration.sql

-- 6. Enable realtime updates
-- Run: sql/15-enable-realtime.sql
```

### 2. Verify Database Setup

```sql
-- Check tokens column exists
SELECT id, email, tokens FROM user_profiles LIMIT 5;

-- Check transactions table
SELECT * FROM transactions LIMIT 1;

-- Test atomic function
SELECT * FROM deduct_token_if_sufficient(auth.uid());
```

### 3. Enable Realtime Updates

**CRITICAL**: Enable Supabase Realtime on `user_profiles` table for real-time token balance updates.

See detailed instructions in: `PRPs/planning/enable-realtime.md`

**Quick steps**:
1. Go to Supabase Dashboard → Database → Replication
2. Find `user_profiles` table
3. Toggle replication to "Enabled"
4. Save

This enables:
- Real-time token balance updates when sending messages
- Instant balance refresh after purchases
- No page refresh needed to see new token count

## Part 2: Stripe CLI Installation (Required for Local Development)

### Install Stripe CLI

**IMPORTANT**: Stripe no longer allows `localhost` URLs in the dashboard for webhooks. You MUST use the Stripe CLI for local development.

#### macOS
```bash
brew install stripe/stripe-cli/stripe
```

#### Windows
Download from: https://github.com/stripe/stripe-cli/releases/latest
Or use Scoop:
```bash
scoop bucket add stripe https://github.com/stripe/scoop-stripe-cli.git
scoop install stripe
```

#### Linux
```bash
# Debian/Ubuntu
curl -s https://packages.stripe.dev/api/security/keypair/stripe-cli-gpg/public | gpg --dearmor | sudo tee /usr/share/keyrings/stripe.gpg
echo "deb [signed-by=/usr/share/keyrings/stripe.gpg] https://packages.stripe.dev/stripe-cli-debian-local stable main" | sudo tee -a /etc/apt/sources.list.d/stripe.list
sudo apt update
sudo apt install stripe
```

### Login to Stripe CLI

```bash
stripe login
```

This opens your browser to authorize the CLI with your Stripe account.

## Part 3: Get API Keys from Dashboard

1. Navigate to https://dashboard.stripe.com/apikeys
2. Copy your **Publishable key** (starts with `pk_test_` for test mode)
3. Copy your **Secret key** (starts with `sk_test_` for test mode)

## Part 4: Environment Variables

### Backend (.env)

Add to `backend_agent_api/.env`:

```env
# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your_actual_secret_key_here
# IMPORTANT: Leave STRIPE_WEBHOOK_SECRET empty for now - you'll get it from CLI
STRIPE_WEBHOOK_SECRET=
```

### Frontend (.env)

Add to `frontend/.env`:

```env
# Stripe Configuration
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your_actual_publishable_key_here
```

**CRITICAL**: Never commit actual API keys to version control!

## Part 5: Start Stripe CLI Webhook Forwarding

### CRITICAL: Start the webhook listener BEFORE testing

**This step replaces dashboard webhook configuration for local development.**

1. **Start your backend API** (must be running on port 8001):
   ```bash
   cd backend_agent_api
   python agent_api.py
   ```

2. **In a separate terminal, start Stripe CLI listener**:
   ```bash
   stripe listen --forward-to localhost:8001/api/webhook/stripe
   ```

3. **Copy the webhook signing secret** from the CLI output:
   ```
   > Ready! Your webhook signing secret is whsec_xxxxxxxxxxxxx (^C to quit)
   ```

4. **Update your backend .env** with the webhook secret:
   ```env
   STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx
   ```

5. **Restart your backend** to load the new webhook secret

**IMPORTANT**:
- Keep the `stripe listen` command running while testing
- The webhook secret changes each time you run `stripe listen`
- You'll need to update `.env` and restart backend if you restart the CLI

## Part 6: Testing

### Test Card Numbers

Stripe provides test cards for different scenarios:

| Scenario | Card Number | Details |
|----------|-------------|---------|
| Success | 4242424242424242 | Any future date, any CVC |
| Decline | 4000000000000002 | Any future date, any CVC |
| Insufficient funds | 4000000000009995 | Any future date, any CVC |
| Expired card | 4000000000000069 | Any future date, any CVC |
| Incorrect CVC | 4000000000000127 | Any future date, any CVC |
| Authentication required | 4000002500003155 | Triggers 3D Secure |

### Test Flow

1. **Create test user**:
   ```sql
   -- Grant 10 tokens to test user
   UPDATE user_profiles
   SET tokens = 10
   WHERE email = 'test@example.com';
   ```

2. **Test token consumption**:
   - Login to frontend
   - Send message to agent
   - Verify token count decrements in database

3. **Test token purchase flow**:
   - Make sure `stripe listen` is running in a terminal
   - Navigate to `/purchase-tokens` in browser
   - Select a tier (e.g., Tier 1 - $5.00 / 100 tokens)
   - Use test card `4242424242424242`
   - Fill in any future expiry date and any CVC
   - Complete purchase
   - Watch the `stripe listen` terminal for webhook event
   - Verify tokens added to account
   - Check transaction in database

4. **Test webhook delivery in Stripe CLI**:

   You should see output like:
   ```
   2024-01-15 10:30:45   --> payment_intent.succeeded [evt_xxx]
   2024-01-15 10:30:45  <--  [200] POST http://localhost:8001/api/webhook/stripe [evt_xxx]
   ```

   The `[200]` indicates successful webhook processing.

5. **Test webhook idempotency** (optional):
   ```bash
   # Trigger test webhook event
   stripe trigger payment_intent.succeeded
   ```

### Verify Webhook Delivery

**For Local Development:**
- Check the `stripe listen` terminal output
- Look for `[200]` status codes
- Check your backend logs for token grant messages

**For Production:**
1. Go to https://dashboard.stripe.com/webhooks
2. Click on your webhook endpoint
3. Check **Recent events** for successful deliveries
4. Response should be `200 OK`

## Part 5: Production Deployment Checklist

### Pre-Deployment

- [ ] All test scenarios pass
- [ ] Database migrations completed
- [ ] Webhook tested with Stripe CLI
- [ ] Frontend build succeeds without errors
- [ ] Backend linting passes

### Switch to Live Mode

1. **Get live API keys**:
   - Navigate to https://dashboard.stripe.com/apikeys
   - Toggle to **Live mode** in top-left
   - Copy live keys (start with `sk_live_` and `pk_live_`)

2. **Create live webhook**:
   - Go to https://dashboard.stripe.com/webhooks (live mode)
   - Add endpoint with production URL
   - Select `payment_intent.succeeded` event
   - Copy live webhook signing secret

3. **Update environment variables**:
   ```env
   # Backend
   STRIPE_SECRET_KEY=sk_live_your_live_secret_key
   STRIPE_WEBHOOK_SECRET=whsec_your_live_webhook_secret

   # Frontend
   VITE_STRIPE_PUBLISHABLE_KEY=pk_live_your_live_publishable_key
   ```

4. **Deploy and verify**:
   - Deploy backend with new env vars
   - Deploy frontend with new env vars
   - Make test purchase with real card
   - Verify webhook delivery in dashboard
   - Check tokens granted correctly

### Post-Deployment Monitoring

- [ ] Monitor webhook delivery rate (should be 100%)
- [ ] Check for failed token deductions in logs
- [ ] Review transaction records for anomalies
- [ ] Set up alerts for webhook failures
- [ ] Monitor Stripe dashboard for disputes

## Part 6: Troubleshooting

### Webhook signature verification fails

**Symptom**: `Invalid signature` errors in logs

**Solutions**:
1. Verify `STRIPE_WEBHOOK_SECRET` matches dashboard
2. Ensure using raw request body, not parsed JSON
3. Check webhook secret corresponds to correct endpoint
4. Regenerate webhook secret in dashboard if needed

### Tokens not granted after payment

**Symptom**: Payment succeeds but tokens don't appear

**Solutions**:
1. Check webhook delivery in Stripe dashboard
2. Verify backend logs for webhook processing errors
3. Check database for idempotency (duplicate `stripe_event_id`)
4. Ensure `payment_intent.succeeded` event is configured
5. Verify metadata includes `user_id` and `tier`

### Race condition on token deduction

**Symptom**: User charged tokens but no response

**Solutions**:
1. Verify `FOR UPDATE` locking in SQL function
2. Check database connection pool settings
3. Review transaction logs for conflicts
4. Ensure atomic operations via RPC

### Frontend build errors

**Symptom**: TypeScript errors in token components

**Solutions**:
1. Run `npm install` in frontend directory
2. Verify Stripe packages installed correctly
3. Check `VITE_STRIPE_PUBLISHABLE_KEY` is set
4. Clear build cache: `rm -rf node_modules/.vite`

## Part 7: Security Best Practices

### API Key Management

- ✅ Use environment variables for all keys
- ✅ Different keys for test and production
- ✅ Rotate keys periodically (every 90 days)
- ✅ Restrict key permissions in Stripe dashboard
- ❌ Never hardcode keys in source code
- ❌ Never commit keys to version control
- ❌ Never expose secret keys in client code

### Webhook Security

- ✅ Always verify webhook signatures
- ✅ Use HTTPS for webhook endpoints
- ✅ Implement idempotency checking
- ✅ Log webhook events for auditing
- ❌ Never skip signature verification
- ❌ Never process unverified events

### Database Security

- ✅ Use RLS policies for data access
- ✅ Audit trail for all transactions
- ✅ Atomic operations for token changes
- ✅ Regular database backups
- ❌ Never allow direct token manipulation
- ❌ Never delete transaction records

## Part 8: Support Resources

### Stripe Documentation

- API Reference: https://docs.stripe.com/api
- Webhooks Guide: https://docs.stripe.com/webhooks
- Testing Guide: https://docs.stripe.com/testing
- Error Codes: https://docs.stripe.com/error-codes

### Supabase Documentation

- RPC Functions: https://supabase.com/docs/reference/python/rpc
- Row Level Security: https://supabase.com/docs/guides/database/postgres/row-level-security
- PostgreSQL Functions: https://supabase.com/docs/guides/database/functions

### Stripe Dashboard Links

- API Keys: https://dashboard.stripe.com/apikeys
- Webhooks: https://dashboard.stripe.com/webhooks
- Events & Logs: https://dashboard.stripe.com/events
- Payment Intents: https://dashboard.stripe.com/payments

## Success Criteria

✅ All database migrations applied successfully
✅ Webhook endpoint receiving events with 200 responses
✅ Test purchases complete end-to-end
✅ Tokens granted automatically after payment
✅ Token consumption working in agent endpoint
✅ Idempotency preventing duplicate grants
✅ Transaction audit trail complete
✅ Frontend displaying token balance correctly
✅ Real-time token updates working

---

**Setup complete!** You now have a fully functional token-based payment system integrated with Stripe.