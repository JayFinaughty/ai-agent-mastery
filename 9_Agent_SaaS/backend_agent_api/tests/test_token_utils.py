"""
Unit tests for token utility functions.

Tests atomic token operations including checking, deducting, and granting tokens.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from token_utils import (
    check_user_tokens,
    deduct_token_atomic,
    grant_tokens_atomic
)


@pytest.mark.asyncio
async def test_check_user_tokens_success():
    """Test checking user tokens when user has tokens."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = [{"tokens": 50}]

    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

    result = await check_user_tokens(mock_supabase, "user-123")

    assert result == 50
    mock_supabase.table.assert_called_once_with("user_profiles")


@pytest.mark.asyncio
async def test_check_user_tokens_no_user():
    """Test checking tokens when user not found returns 0."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = []

    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

    result = await check_user_tokens(mock_supabase, "user-123")

    assert result == 0


@pytest.mark.asyncio
async def test_check_user_tokens_none_data():
    """Test checking tokens when response data is None."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = None

    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

    result = await check_user_tokens(mock_supabase, "user-123")

    assert result == 0


@pytest.mark.asyncio
async def test_check_user_tokens_exception():
    """Test checking tokens handles exceptions gracefully."""
    mock_supabase = Mock()
    mock_supabase.table.side_effect = Exception("Database error")

    result = await check_user_tokens(mock_supabase, "user-123")

    assert result == 0


@pytest.mark.asyncio
async def test_deduct_token_success():
    """Test token deduction succeeds when user has tokens."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = [{"success": True, "remaining_tokens": 49}]

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await deduct_token_atomic(mock_supabase, "user-123")

    assert result is True
    mock_supabase.rpc.assert_called_once_with(
        "deduct_token_if_sufficient",
        {"p_user_id": "user-123"}
    )


@pytest.mark.asyncio
async def test_deduct_token_insufficient():
    """Test token deduction fails when user has insufficient tokens."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = [{"success": False, "error_message": "Insufficient tokens"}]

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await deduct_token_atomic(mock_supabase, "user-123")

    assert result is False


@pytest.mark.asyncio
async def test_deduct_token_no_data():
    """Test token deduction handles empty response data."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = []

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await deduct_token_atomic(mock_supabase, "user-123")

    assert result is False


@pytest.mark.asyncio
async def test_deduct_token_exception():
    """Test token deduction handles exceptions gracefully."""
    mock_supabase = Mock()
    mock_supabase.rpc.side_effect = Exception("RPC error")

    result = await deduct_token_atomic(mock_supabase, "user-123")

    assert result is False


@pytest.mark.asyncio
async def test_grant_tokens_success():
    """Test granting tokens succeeds."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = [{"success": True, "new_balance": 150}]

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )

    assert result is True
    mock_supabase.rpc.assert_called_once_with(
        "grant_tokens_for_purchase",
        {
            "p_user_id": "user-123",
            "p_tokens": 100,
            "p_event_id": "evt_test123",
            "p_payment_intent_id": "pi_test123"
        }
    )


@pytest.mark.asyncio
async def test_grant_tokens_duplicate_event():
    """Test granting tokens fails for duplicate event (idempotency)."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = [{
        "success": False,
        "error_message": "Event already processed"
    }]

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )

    assert result is False


@pytest.mark.asyncio
async def test_grant_tokens_no_data():
    """Test granting tokens handles empty response data."""
    mock_supabase = Mock()
    mock_response = Mock()
    mock_response.data = []

    mock_supabase.rpc.return_value.execute.return_value = mock_response

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )

    assert result is False


@pytest.mark.asyncio
async def test_grant_tokens_exception():
    """Test granting tokens handles exceptions gracefully."""
    mock_supabase = Mock()
    mock_supabase.rpc.side_effect = Exception("RPC error")

    result = await grant_tokens_atomic(
        mock_supabase,
        "user-123",
        100,
        "evt_test123",
        "pi_test123"
    )

    assert result is False


@pytest.mark.asyncio
async def test_grant_tokens_various_amounts():
    """Test granting different token amounts."""
    mock_supabase = Mock()

    test_cases = [
        (100, 100),
        (250, 250),
        (600, 600),
    ]

    for tokens, expected_balance in test_cases:
        mock_response = Mock()
        mock_response.data = [{"success": True, "new_balance": expected_balance}]
        mock_supabase.rpc.return_value.execute.return_value = mock_response

        result = await grant_tokens_atomic(
            mock_supabase,
            "user-123",
            tokens,
            f"evt_{tokens}",
            f"pi_{tokens}"
        )

        assert result is True