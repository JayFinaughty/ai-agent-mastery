"""
Unit tests for Stripe utility functions.

Tests payment intent creation and webhook signature verification.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
import stripe
from stripe_utils import (
    create_payment_intent,
    verify_webhook_signature,
    PRICING_TIERS
)


@pytest.mark.asyncio
async def test_pricing_tiers_structure():
    """Test that pricing tiers are properly configured."""
    assert "tier_1" in PRICING_TIERS
    assert "tier_2" in PRICING_TIERS
    assert "tier_3" in PRICING_TIERS

    assert PRICING_TIERS["tier_1"] == (500, 100)
    assert PRICING_TIERS["tier_2"] == (1000, 250)
    assert PRICING_TIERS["tier_3"] == (2000, 600)


@pytest.mark.asyncio
@patch('stripe_utils.stripe.PaymentIntent.create')
async def test_create_payment_intent_tier_1(mock_create):
    """Test creating payment intent for tier 1."""
    mock_payment_intent = Mock()
    mock_payment_intent.client_secret = "pi_test_secret_123"
    mock_create.return_value = mock_payment_intent

    result = await create_payment_intent("tier_1", "user-123")

    assert result["client_secret"] == "pi_test_secret_123"
    assert result["amount"] == 500
    assert result["currency"] == "usd"

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["amount"] == 500
    assert call_kwargs["currency"] == "usd"
    assert call_kwargs["metadata"]["user_id"] == "user-123"
    assert call_kwargs["metadata"]["tier"] == "tier_1"
    assert call_kwargs["metadata"]["tokens"] == 100


@pytest.mark.asyncio
@patch('stripe_utils.stripe.PaymentIntent.create')
async def test_create_payment_intent_tier_2(mock_create):
    """Test creating payment intent for tier 2."""
    mock_payment_intent = Mock()
    mock_payment_intent.client_secret = "pi_test_secret_456"
    mock_create.return_value = mock_payment_intent

    result = await create_payment_intent("tier_2", "user-456")

    assert result["client_secret"] == "pi_test_secret_456"
    assert result["amount"] == 1000
    assert result["currency"] == "usd"


@pytest.mark.asyncio
@patch('stripe_utils.stripe.PaymentIntent.create')
async def test_create_payment_intent_tier_3(mock_create):
    """Test creating payment intent for tier 3."""
    mock_payment_intent = Mock()
    mock_payment_intent.client_secret = "pi_test_secret_789"
    mock_create.return_value = mock_payment_intent

    result = await create_payment_intent("tier_3", "user-789")

    assert result["client_secret"] == "pi_test_secret_789"
    assert result["amount"] == 2000
    assert result["currency"] == "usd"


@pytest.mark.asyncio
async def test_create_payment_intent_invalid_tier():
    """Test creating payment intent with invalid tier raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tier"):
        await create_payment_intent("tier_invalid", "user-123")


@pytest.mark.asyncio
@patch('stripe_utils.stripe.PaymentIntent.create')
async def test_create_payment_intent_stripe_error(mock_create):
    """Test creating payment intent handles Stripe errors."""
    mock_create.side_effect = stripe.error.StripeError("API Error")

    with pytest.raises(HTTPException) as exc_info:
        await create_payment_intent("tier_1", "user-123")

    assert exc_info.value.status_code == 500
    assert "Payment error" in exc_info.value.detail


@pytest.mark.asyncio
@patch('stripe_utils.stripe.PaymentIntent.create')
async def test_create_payment_intent_idempotency_key(mock_create):
    """Test that payment intent uses idempotency key."""
    mock_payment_intent = Mock()
    mock_payment_intent.client_secret = "pi_test_secret"
    mock_create.return_value = mock_payment_intent

    await create_payment_intent("tier_1", "user-123")

    call_kwargs = mock_create.call_args[1]
    assert "idempotency_key" in call_kwargs
    assert call_kwargs["idempotency_key"].startswith("purchase_user-123_tier_1_")


@patch('stripe_utils.stripe.Webhook.construct_event')
@patch('stripe_utils.os.getenv')
def test_verify_webhook_signature_success(mock_getenv, mock_construct):
    """Test successful webhook signature verification."""
    mock_getenv.return_value = "whsec_test_secret"
    mock_event = {"type": "payment_intent.succeeded", "id": "evt_123"}
    mock_construct.return_value = mock_event

    payload = '{"type": "payment_intent.succeeded"}'
    sig_header = "t=123,v1=abc"

    result = verify_webhook_signature(payload, sig_header)

    assert result == mock_event
    mock_construct.assert_called_once_with(payload, sig_header, "whsec_test_secret")


@patch('stripe_utils.os.getenv')
def test_verify_webhook_signature_no_secret(mock_getenv):
    """Test webhook verification fails when secret not configured."""
    mock_getenv.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        verify_webhook_signature("payload", "signature")

    assert exc_info.value.status_code == 500
    assert "Webhook secret not configured" in exc_info.value.detail


@patch('stripe_utils.stripe.Webhook.construct_event')
@patch('stripe_utils.os.getenv')
def test_verify_webhook_signature_invalid_payload(mock_getenv, mock_construct):
    """Test webhook verification handles invalid payload."""
    mock_getenv.return_value = "whsec_test_secret"
    mock_construct.side_effect = ValueError("Invalid payload")

    with pytest.raises(HTTPException) as exc_info:
        verify_webhook_signature("invalid", "signature")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid payload"


@patch('stripe_utils.stripe.Webhook.construct_event')
@patch('stripe_utils.os.getenv')
def test_verify_webhook_signature_invalid_signature(mock_getenv, mock_construct):
    """Test webhook verification handles invalid signature."""
    mock_getenv.return_value = "whsec_test_secret"
    mock_construct.side_effect = stripe.error.SignatureVerificationError(
        "Invalid signature", "sig_header"
    )

    with pytest.raises(HTTPException) as exc_info:
        verify_webhook_signature("payload", "bad_signature")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid signature"