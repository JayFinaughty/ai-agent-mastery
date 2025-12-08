# Stripe Python SDK Research for Payment Integration

## Overview
This document provides comprehensive research on the Stripe Python SDK for payment integration, covering payment intent creation, webhook handling, idempotency, error handling, and environment configuration.

## 1. Payment Intent Creation

### Official Documentation
- **Main Reference**: https://docs.stripe.com/api/payment_intents/create?lang=python
- **Payment Intents Overview**: https://docs.stripe.com/payments/payment-intents
- **PaymentIntent Object**: https://docs.stripe.com/api/payment_intents/object

### Required Parameters
- **amount**: A positive integer representing how much to charge in the smallest currency unit (e.g., 100 cents = $1.00)
- **currency**: Three-letter ISO currency code in lowercase (e.g., 'usd', 'eur')

### Code Example
```python
import stripe

stripe.api_key = "sk_test_..."

# Create a PaymentIntent
payment_intent = stripe.PaymentIntent.create(
    amount=2000,  # $20.00
    currency='usd',
    metadata={
        'order_id': '12345',
        'customer_name': 'John Doe'
    }
)
```

### Metadata Best Practices
- **Documentation**: https://docs.stripe.com/api/metadata
- **Limits**: Up to 50 key-value pairs per object
- **Key names**: Maximum 40 characters
- **Values**: Maximum 500 characters
- **Use cases**: Store order IDs, customer references, internal tracking data
- **Security**: Never store PII, card details, or sensitive information in metadata

### Advanced Parameters
```python
payment_intent = stripe.PaymentIntent.create(
    amount=2000,
    currency='usd',
    payment_method_types=['card'],
    confirmation_method='manual',
    metadata={
        'order_id': '12345',
        'integration_check': 'accept_a_payment'
    }
)
```

### Important Notes
- Minimum amount is $0.50 USD or equivalent
- Amount supports up to 8 digits (max $999,999.99)
- When PaymentIntent creates a charge, metadata is copied to the charge

## 2. Webhook Handling

### Official Documentation
- **Webhook Guide**: https://docs.stripe.com/webhooks
- **Signature Verification**: https://docs.stripe.com/webhooks/signature
- **Python Examples**: https://github.com/stripe/stripe-python/blob/master/examples/webhooks.py

### Signature Verification with construct_event()
```python
import stripe
from flask import Flask, request
import os

app = Flask(__name__)
webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

@app.route("/webhooks", methods=["POST"])
def handle_webhook():
    payload = request.data.decode("utf-8")
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        # Invalid payload
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return "Invalid signature", 400

    # Handle the event
    if event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']
        handle_payment_success(payment_intent)
    elif event['type'] == 'payment_intent.payment_failed':
        payment_intent = event['data']['object']
        handle_payment_failure(payment_intent)

    return "Success", 200
```

### Key Event Types to Handle
- **payment_intent.succeeded**: Payment completed successfully
- **payment_intent.payment_failed**: Payment failed
- **payment_intent.requires_action**: Additional authentication required
- **payment_intent.canceled**: Payment was canceled

### Framework-Specific Considerations
- **Flask**: Use `request.data.decode("utf-8")` for raw payload
- **Django**: Use `request.body.decode("utf-8")`
- **FastAPI/AIOHTTP**: Use `await request.text()` (not `await request.json()`)

### Security Best Practices
- Always verify webhook signatures before processing
- Use HTTPS endpoints only
- Implement idempotency to handle duplicate events
- Validate event structure before processing

## 3. Idempotency Handling

### Official Documentation
- **Idempotent Requests**: https://docs.stripe.com/api/idempotent_requests
- **Blog Post on Idempotency**: https://stripe.com/blog/idempotency

### Using Idempotency Keys
```python
import stripe
import uuid

# Generate a unique idempotency key
idempotency_key = str(uuid.uuid4())

payment_intent = stripe.PaymentIntent.create(
    amount=2000,
    currency='usd',
    idempotency_key=idempotency_key
)
```

### Best Practices for Idempotency Keys
- **Generation**: Use UUID v4 for sufficient entropy
- **Scope**: One key per logical operation
- **Lifetime**: Keys expire after 24 hours
- **Retry Strategy**: Use same key for retries of the same operation
- **Parameter Consistency**: Same idempotency key must use identical parameters

### Deduplication with Event IDs
```python
def handle_webhook_event(event):
    event_id = event['id']

    # Check if we've already processed this event
    if already_processed(event_id):
        return "Already processed", 200

    # Process the event
    process_event(event)

    # Mark as processed
    mark_as_processed(event_id)

    return "Success", 200
```

### When to Use New Keys
- Always generate new keys for 4xx errors
- Use exponential backoff for retries
- Don't use keys for GET or DELETE requests

## 4. Error Handling

### Official Documentation
- **Error Handling**: https://docs.stripe.com/error-handling?lang=python
- **Error Codes**: https://docs.stripe.com/error-codes
- **API Errors**: https://docs.stripe.com/api/errors/handling

### Exception Hierarchy
```python
import stripe

try:
    payment_intent = stripe.PaymentIntent.create(
        amount=2000,
        currency='usd'
    )
except stripe.error.CardError as e:
    # Payment was declined
    print(f"Payment declined: {e.user_message}")
    print(f"Decline code: {e.decline_code}")
    print(f"Error code: {e.code}")
except stripe.error.RateLimitError as e:
    # Too many requests made to the API too quickly
    print("Rate limit exceeded")
except stripe.error.InvalidRequestError as e:
    # Invalid parameters were supplied to Stripe's API
    print(f"Invalid request: {e.user_message}")
except stripe.error.AuthenticationError as e:
    # Authentication with Stripe's API failed
    print("Authentication failed")
except stripe.error.APIConnectionError as e:
    # Network communication with Stripe failed
    print("Network error")
except stripe.error.StripeError as e:
    # Generic Stripe error
    print(f"Stripe error: {e}")
except Exception as e:
    # Something else happened
    print(f"Unexpected error: {e}")
```

### Common Exception Types
- **stripe.error.CardError**: Payment/card declines and issues
- **stripe.error.RateLimitError**: Too many API requests
- **stripe.error.InvalidRequestError**: Invalid parameters
- **stripe.error.AuthenticationError**: API authentication failures
- **stripe.error.APIConnectionError**: Network communication problems
- **stripe.error.StripeError**: Generic Stripe errors

### Retry Patterns
```python
import time
import random
from stripe.error import RateLimitError, APIConnectionError

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except (RateLimitError, APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise e

            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

    raise Exception("Max retries exceeded")
```

### Best Practices
- Use idempotency keys for safe retries
- Implement exponential backoff for network errors
- Log errors with context for debugging
- Handle specific exceptions rather than catching all

## 5. Environment Configuration

### Official Documentation
- **API Keys**: https://docs.stripe.com/keys
- **Development Environment**: https://docs.stripe.com/get-started/development-environment?lang=python
- **Authentication**: https://docs.stripe.com/api/authentication

### Required API Keys
- **Secret Key**: `sk_test_...` (test) or `sk_live_...` (live)
- **Publishable Key**: `pk_test_...` (test) or `pk_live_...` (live)
- **Webhook Secret**: `whsec_...` (unique per endpoint)

### SDK Initialization Patterns
```python
import stripe
import os

# Method 1: Set globally
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

# Method 2: Per-request (recommended for multi-tenant apps)
payment_intent = stripe.PaymentIntent.create(
    amount=2000,
    currency='usd',
    api_key=os.environ.get("STRIPE_SECRET_KEY")
)
```

### Environment Configuration
```python
import os
from typing import Optional

class StripeConfig:
    def __init__(self):
        self.secret_key = os.environ.get("STRIPE_SECRET_KEY")
        self.publishable_key = os.environ.get("STRIPE_PUBLISHABLE_KEY")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
        self.is_live = not self.secret_key.startswith("sk_test_")

    def validate(self):
        if not self.secret_key:
            raise ValueError("STRIPE_SECRET_KEY environment variable is required")
        if not self.webhook_secret:
            raise ValueError("STRIPE_WEBHOOK_SECRET environment variable is required")

# Usage
config = StripeConfig()
config.validate()
stripe.api_key = config.secret_key
```

### Test vs Production Mode
```python
def is_test_mode(api_key: str) -> bool:
    """Check if API key is for test mode."""
    return api_key.startswith("sk_test_")

def get_dashboard_url(api_key: str, payment_intent_id: str) -> str:
    """Generate dashboard URL for payment intent."""
    base_url = "https://dashboard.stripe.com"
    if is_test_mode(api_key):
        base_url += "/test"
    return f"{base_url}/payments/{payment_intent_id}"
```

### Security Considerations
- Never hardcode API keys in source code
- Use different keys for test and production environments
- Rotate webhook secrets regularly
- Restrict API key permissions in Stripe Dashboard
- Use HTTPS for all webhook endpoints
- Store keys in secure environment variables or secret management systems

## Common Pitfalls to Avoid

### Payment Intent Creation
- Don't store sensitive data in metadata
- Validate amount is within Stripe's limits (minimum $0.50)
- Use proper currency codes (lowercase, 3 letters)

### Webhook Handling
- Don't skip signature verification
- Handle webhook events idempotently
- Don't rely on event order
- Test webhook endpoints with Stripe CLI

### Error Handling
- Don't catch all exceptions with generic handlers
- Log error details for debugging
- Implement proper retry logic for transient errors
- Return appropriate HTTP status codes

### Security
- Never log or expose API keys
- Use HTTPS for all Stripe communications
- Validate all user inputs before API calls
- Implement rate limiting on your endpoints

## Testing Resources

### Stripe CLI
- Install: https://stripe.com/docs/stripe-cli
- Listen to webhooks: `stripe listen --forward-to localhost:4000/webhooks`
- Trigger test events: `stripe trigger payment_intent.succeeded`

### Test Cards
- Successful payment: `4242424242424242`
- Declined payment: `4000000000000002`
- Authentication required: `4000002500003155`

This research provides a comprehensive foundation for implementing Stripe payment integration using the Python SDK with proper error handling, security, and best practices.