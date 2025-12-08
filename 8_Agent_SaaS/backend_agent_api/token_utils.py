"""
Token utility functions for managing user token balances.

This module provides atomic operations for checking, deducting, and granting
tokens using PostgreSQL functions via Supabase RPC.
"""

import logging
from supabase import Client


async def check_user_tokens(supabase: Client, user_id: str) -> int:
    """
    Get current token balance for user.

    Args:
        supabase: Supabase client instance
        user_id: User UUID

    Returns:
        Current token balance (0 if user not found or error)

    Example:
        >>> balance = await check_user_tokens(supabase, "user-uuid")
        >>> print(f"User has {balance} tokens")
    """
    try:
        response = supabase.table("user_profiles").select("tokens").eq(
            "id", user_id
        ).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]["tokens"]
        else:
            logging.warning(f"User {user_id} not found in user_profiles")
            return 0

    except Exception as e:
        logging.error(f"Error checking user tokens: {str(e)}")
        return 0


async def deduct_token_atomic(supabase: Client, user_id: str) -> bool:
    """
    Atomically deduct one token from user balance.

    Uses PostgreSQL function with FOR UPDATE locking to prevent race
    conditions. The function checks sufficient balance and deducts atomically.

    Args:
        supabase: Supabase client instance
        user_id: User UUID

    Returns:
        True if token was deducted successfully, False if insufficient balance

    Example:
        >>> success = await deduct_token_atomic(supabase, "user-uuid")
        >>> if success:
        >>>     print("Token deducted, proceeding with request")
    """
    try:
        response = supabase.rpc(
            "deduct_token_if_sufficient",
            {"p_user_id": user_id}
        ).execute()

        if response.data and len(response.data) > 0:
            result = response.data[0]
            success = result.get("success", False)

            if not success:
                error_msg = result.get("error_message", "Unknown error")
                logging.info(
                    f"Token deduction failed for user {user_id}: {error_msg}"
                )

            return success

        logging.error("No data returned from deduct_token_if_sufficient")
        return False

    except Exception as e:
        logging.error(f"Error deducting token: {str(e)}")
        return False


async def grant_tokens_atomic(
    supabase: Client,
    user_id: str,
    tokens: int,
    event_id: str,
    payment_intent_id: str
) -> bool:
    """
    Atomically grant tokens from successful Stripe payment.

    Uses PostgreSQL function with idempotency checking via event_id.
    Prevents duplicate token grants if webhook is retried.

    Args:
        supabase: Supabase client instance
        user_id: User UUID receiving tokens
        tokens: Number of tokens to grant
        event_id: Stripe event ID for idempotency
        payment_intent_id: Stripe payment intent ID for audit trail

    Returns:
        True if tokens were granted successfully, False if already processed
        or error

    Example:
        >>> success = await grant_tokens_atomic(
        ...     supabase, "user-uuid", 100, "evt_123", "pi_123"
        ... )
        >>> if success:
        >>>     print("Tokens granted successfully")
    """
    try:
        print(f"[grant_tokens_atomic] Calling RPC with user_id={user_id}, tokens={tokens}, event_id={event_id}")
        response = supabase.rpc(
            "grant_tokens_for_purchase",
            {
                "p_user_id": user_id,
                "p_tokens": tokens,
                "p_event_id": event_id,
                "p_payment_intent_id": payment_intent_id
            }
        ).execute()

        print(f"[grant_tokens_atomic] RPC response data: {response.data}")
        print(f"[grant_tokens_atomic] RPC response count: {response.count}")

        if response.data and len(response.data) > 0:
            result = response.data[0]
            success = result.get("success", False)

            if not success:
                error_msg = result.get("error_message", "Unknown error")
                logging.info(
                    f"Token grant failed for user {user_id}: {error_msg}"
                )
            else:
                new_balance = result.get("new_balance", 0)
                logging.info(
                    f"Granted {tokens} tokens to user {user_id}. "
                    f"New balance: {new_balance}"
                )

            return success

        print("[grant_tokens_atomic] ERROR: No data returned from grant_tokens_for_purchase")
        logging.error("No data returned from grant_tokens_for_purchase")
        return False

    except Exception as e:
        print(f"[grant_tokens_atomic] EXCEPTION: {str(e)}")
        import traceback
        print(f"[grant_tokens_atomic] Traceback: {traceback.format_exc()}")
        logging.error(f"Error granting tokens: {str(e)}")
        return False