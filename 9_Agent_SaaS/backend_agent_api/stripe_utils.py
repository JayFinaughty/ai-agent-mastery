"""
Stripe utility functions for payment processing.

This module handles Stripe payment intent creation and webhook signature
verification for the token purchase system.
"""

import os
import time
import logging
from typing import Dict, Tuple, Any

import stripe
from fastapi import HTTPException

# Initialize Stripe API key from environment
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Pricing tiers: (amount_in_cents, token_count)
PRICING_TIERS: Dict[str, Tuple[int, int]] = {
    "tier_1": (500, 100),
    "tier_2": (1000, 250),
    "tier_3": (2000, 600),
}


async def create_payment_intent(tier: str, user_id: str) -> Dict[str, Any]:
    """
    Create Stripe payment intent for token purchase.

    Args:
        tier: Pricing tier ID (tier_1, tier_2, or tier_3)
        user_id: User UUID purchasing tokens

    Returns:
        Dict containing client_secret, amount, and currency

    Raises:
        ValueError: If tier is invalid
        HTTPException: If Stripe API call fails

    Example:
        >>> result = await create_payment_intent("tier_1", "user-uuid")
        >>> print(result["client_secret"])
    """
    if tier not in PRICING_TIERS:
        raise ValueError(f"Invalid tier: {tier}")

    amount_cents, token_count = PRICING_TIERS[tier]

    try:
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            metadata={
                "user_id": user_id,
                "tier": tier,
                "tokens": token_count,
            },
            idempotency_key=f"purchase_{user_id}_{tier}_{int(time.time())}"
        )

        return {
            "client_secret": payment_intent.client_secret,
            "amount": amount_cents,
            "currency": "usd"
        }

    except stripe.error.StripeError as e:
        logging.error(f"Stripe error creating payment intent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Payment error: {str(e)}"
        )


def verify_webhook_signature(payload: str, sig_header: str) -> stripe.Event:
    """
    Verify Stripe webhook signature and parse event.

    Args:
        payload: Raw request body as string
        sig_header: Stripe-Signature header value

    Returns:
        Verified Stripe Event object

    Raises:
        HTTPException: If signature verification fails or payload is invalid

    Note:
        MUST use raw body string, not parsed JSON, for signature verification
    """
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        logging.error("STRIPE_WEBHOOK_SECRET not configured")
        raise HTTPException(
            status_code=500,
            detail="Webhook secret not configured"
        )

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        return event

    except ValueError as e:
        logging.error(f"Invalid webhook payload: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid payload")

    except stripe.error.SignatureVerificationError as e:
        logging.error(f"Invalid webhook signature: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid signature")