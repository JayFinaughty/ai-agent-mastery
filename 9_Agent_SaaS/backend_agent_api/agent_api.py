from typing import List, Optional, Dict, Any, AsyncIterator, Union, Tuple
from fastapi import FastAPI, HTTPException, Security, Depends, Request, Form
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager, nullcontext
from supabase import create_client, Client
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from dotenv import load_dotenv
from httpx import AsyncClient
from pathlib import Path
from mem0 import Memory
import asyncio
import base64
import time
import json
import sys
import os

# Import Langfuse configuration
from configure_langfuse import configure_langfuse

# Import database utility functions
from db_utils import (
    fetch_conversation_history,
    create_conversation,
    update_conversation_title,
    generate_session_id,
    generate_conversation_title,
    store_message,
    convert_history_to_pydantic_format,
    check_rate_limit,
    store_request
)

# Import Stripe and token utilities
from stripe_utils import (
    create_payment_intent as stripe_create_payment_intent,
    verify_webhook_signature,
    PRICING_TIERS
)
from token_utils import (
    check_user_tokens,
    deduct_token_atomic,
    grant_tokens_atomic
)

from pydantic_ai import Agent, BinaryContent
# Import all the message part classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, TextPart, ModelMessagesTypeAdapter,
    UserPromptPart, PartDeltaEvent, PartStartEvent, TextPartDelta
)

from agent import agent, AgentDeps, get_model
from clients import get_agent_clients, get_mem0_client_async

# Check if we're in production
is_production = os.getenv("ENVIRONMENT") == "production"

if not is_production:
    # Development: prioritize .env file
    project_root = Path(__file__).resolve().parent
    dotenv_path = project_root / '.env'
    load_dotenv(dotenv_path, override=True)
else:
    # Production: use cloud platform env vars only
    load_dotenv()

# Define clients as None initially
embedding_client = None
supabase = None
http_client = None
title_agent = None
mem0_client = None
tracer = None

# Define the lifespan context manager for the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application.
    
    Handles initialization and cleanup of resources.
    """
    global embedding_client, supabase, http_client, title_agent, mem0_client, tracer

    # Initialize Langfuse tracer (returns None if not configured)
    tracer = configure_langfuse()    
    
    # Startup: Initialize all clients
    embedding_client, supabase = get_agent_clients()
    http_client = AsyncClient()
    title_agent = Agent(model=get_model())
    mem0_client = await get_mem0_client_async()
    
    yield  # This is where the app runs
    
    # Shutdown: Clean up resources
    if http_client:
        await http_client.aclose()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()        

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Verify the JWT token from Supabase and return the user information.
    
    Args:
        credentials: The HTTP Authorization credentials containing the bearer token
        
    Returns:
        Dict[str, Any]: The user information from Supabase
        
    Raises:
        HTTPException: If the token is invalid or the user cannot be verified
    """
    try:
        # Get the token from the Authorization header
        token = credentials.credentials
        
        # Access the global HTTP client
        global http_client # noqa: F824
        if not http_client:
            raise HTTPException(status_code=500, detail="HTTP client not initialized")
        
        # Get the Supabase URL and anon key from environment variables
        # These should match the environment variable names used in your project
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        # Make request to Supabase auth API to get user info using the global HTTP client
        response = await http_client.get(
            f"{supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": supabase_key
            }
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Auth response error: {response.text}")
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Return the user information
        user_data = response.json()
        return user_data
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Request/Response Models
class FileAttachment(BaseModel):
    fileName: str
    content: str  # Base64 encoded content
    mimeType: str

class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str
    files: Optional[List[FileAttachment]] = None

class TokenPurchaseRequest(BaseModel):
    tier: str
    user_id: str

class PaymentIntentResponse(BaseModel):
    client_secret: str
    amount: int
    currency: str = "usd"


# Add this helper function to your backend code
async def stream_error_response(error_message: str, session_id: str):
    """
    Creates a streaming response for error messages.

    Args:
        error_message: The error message to display to the user
        session_id: The current session ID

    Yields:
        Encoded JSON chunks for the streaming response
    """
    # First yield the error message as text
    yield json.dumps({"text": error_message}).encode('utf-8') + b'\n'

    # Then yield a final chunk with complete flag
    final_data = {
        "text": error_message,
        "session_id": session_id,
        "error": error_message,
        "complete": True
    }
    yield json.dumps(final_data).encode('utf-8') + b'\n'

async def refund_token(supabase: Client, user_id: str, reason: str):
    """
    Refund a token to the user when request fails.

    Args:
        supabase: Supabase client instance
        user_id: User UUID to refund
        reason: Reason for refund (for logging)
    """
    try:
        # Use grant_tokens_atomic with special IDs for refunds
        event_id = f"refund_{int(time.time() * 1000)}"
        payment_intent_id = f"error_refund_{reason}"

        await grant_tokens_atomic(
            supabase,
            user_id,
            1,  # Refund 1 token
            event_id,
            payment_intent_id
        )
        print(f"Refunded 1 token to user {user_id} due to: {reason}")
    except Exception as e:
        print(f"Error refunding token: {str(e)}")

@app.post("/api/create-payment-intent")
async def create_payment_intent(
    request: TokenPurchaseRequest,
    user: Dict[str, Any] = Depends(verify_token)
) -> PaymentIntentResponse:
    """
    Create a Stripe payment intent for token purchase.

    Args:
        request: Token purchase request with tier and user_id
        user: Authenticated user from verify_token dependency

    Returns:
        PaymentIntentResponse with client_secret for frontend

    Raises:
        HTTPException: If user mismatch or Stripe error
    """
    if request.user_id != user.get("id"):
        raise HTTPException(
            status_code=401,
            detail="User ID mismatch"
        )

    try:
        result = await stripe_create_payment_intent(request.tier, request.user_id)
        return PaymentIntentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create payment intent: {str(e)}"
        )

@app.post("/api/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events for payment processing.

    Args:
        request: FastAPI Request object with raw body

    Returns:
        Success response (always returns 200 to acknowledge webhook)

    Note:
        MUST use raw body for signature verification
        Returns 200 even on errors (after logging) to prevent retries
    """
    try:
        print("=== WEBHOOK CALLED ===")
        payload = (await request.body()).decode('utf-8')
        sig_header = request.headers.get("stripe-signature")

        print(f"Signature header present: {sig_header is not None}")

        if not sig_header:
            print("Warning: No Stripe signature header found")
            return {"success": False, "error": "No signature"}

        event = verify_webhook_signature(payload, sig_header)
        print(f"Event type received: {event['type']}")

        if event['type'] == 'payment_intent.succeeded':
            print("Processing payment_intent.succeeded event")
            payment_intent = event['data']['object']
            metadata = payment_intent.get('metadata', {})

            user_id = metadata.get('user_id')
            tier = metadata.get('tier')
            event_id = event['id']
            payment_intent_id = payment_intent['id']

            print(f"Metadata - user_id: {user_id}, tier: {tier}")
            print(f"Event ID: {event_id}, Payment Intent ID: {payment_intent_id}")

            if not user_id or not tier:
                print(
                    f"Warning: Missing metadata in payment intent "
                    f"{payment_intent_id}"
                )
                return {"success": False, "error": "Missing metadata"}

            _, token_count = PRICING_TIERS.get(tier, (0, 0))
            print(f"Token count for tier {tier}: {token_count}")

            if token_count == 0:
                print(f"Warning: Invalid tier {tier}")
                return {"success": False, "error": "Invalid tier"}

            print(f"Calling grant_tokens_atomic for user {user_id} with {token_count} tokens")
            success = await grant_tokens_atomic(
                supabase,
                user_id,
                token_count,
                event_id,
                payment_intent_id
            )
            print(f"grant_tokens_atomic returned: {success}")

            if success:
                print(
                    f"Successfully granted {token_count} tokens to user "
                    f"{user_id}"
                )
            else:
                print(
                    f"Failed to grant tokens to user {user_id} "
                    f"(may be duplicate event)"
                )
        else:
            print(f"Event type {event['type']} - not processing (only handling payment_intent.succeeded)")

        print("=== WEBHOOK COMPLETE ===")
        return {"success": True}

    except HTTPException as e:
        print(f"Webhook validation error: {e.detail}")
        print(f"HTTPException details: {e}")
        return {"success": False, "error": e.detail}

    except Exception as e:
        print(f"Webhook processing error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}

@app.post("/api/pydantic-agent")
async def pydantic_agent(request: AgentRequest, user: Dict[str, Any] = Depends(verify_token)):
    # Verify that the user ID in the request matches the user ID from the token
    if request.user_id != user.get("id"):
        return StreamingResponse(
            stream_error_response("User ID in request does not match authenticated user", request.session_id),
            media_type='text/plain'
        )

    token_was_deducted = False  # Track if we need to refund on error

    try:
        # Check rate limit
        rate_limit_ok = await check_rate_limit(supabase, request.user_id)
        if not rate_limit_ok:
            return StreamingResponse(
                stream_error_response("Rate limit exceeded. Please try again later.", request.session_id),
                media_type='text/plain'
            )

        # Check user has sufficient tokens
        user_tokens = await check_user_tokens(supabase, request.user_id)
        if user_tokens < 1:
            return StreamingResponse(
                stream_error_response(
                    "Insufficient tokens. Please purchase more tokens to continue.",
                    request.session_id
                ),
                media_type='text/plain'
            )

        # Deduct token atomically
        token_deducted = await deduct_token_atomic(supabase, request.user_id)
        if not token_deducted:
            return StreamingResponse(
                stream_error_response(
                    "Failed to deduct token. Please try again.",
                    request.session_id
                ),
                media_type='text/plain'
            )

        token_was_deducted = True  # Mark that we deducted a token

        # Start request tracking in parallel
        request_tracking_task = asyncio.create_task(
            store_request(supabase, request.request_id, request.user_id, request.query)
        )
        
        session_id = request.session_id
        conversation_record = None
        conversation_title = None
        
        # Check if session_id is empty, create a new conversation if needed
        if not session_id:
            session_id = generate_session_id(request.user_id)
            # Create a new conversation record
            conversation_record = await create_conversation(supabase, request.user_id, session_id)
        
        # Store user's query immediately with any file attachments
        file_attachments = None
        if request.files:
            # Convert Pydantic models to dictionaries for storage
            file_attachments = [{
                "fileName": file.fileName,
                "content": file.content,
                "mimeType": file.mimeType
            } for file in request.files]
            
        await store_message(
            supabase=supabase,
            session_id=session_id,
            message_type="human",
            content=request.query,
            files=file_attachments
        )
        
        # Fetch conversation history from the DB
        conversation_history = await fetch_conversation_history(supabase, session_id)
        
        # Convert conversation history to Pydantic AI format
        pydantic_messages = await convert_history_to_pydantic_format(conversation_history)
        
        # Retrieve relevant memories with Mem0
        relevant_memories = {"results": []}
        try:
            relevant_memories = await mem0_client.search(query=request.query, user_id=request.user_id, limit=3)
        except:
            # Slight hack - retry again with a new connection pool
            time.sleep(1)
            relevant_memories = await mem0_client.search(query=request.query, user_id=request.user_id, limit=3)

        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        
        # Create memory task to run in parallel
        memory_messages = [{"role": "user", "content": request.query}]
        memory_task = asyncio.create_task(mem0_client.add(memory_messages, user_id=request.user_id))
        
        # Start title generation in parallel if this is a new conversation
        title_task = None
        if conversation_record:
            title_task = asyncio.create_task(generate_conversation_title(title_agent, request.query))
        
        async def stream_response():
            # Process title result if it exists (in the background)
            nonlocal conversation_title
            nonlocal token_was_deducted

            try:
                # Use the global HTTP client
                agent_deps = AgentDeps(
                    embedding_client=embedding_client,
                    supabase=supabase,
                    http_client=http_client,
                    brave_api_key=os.getenv("BRAVE_API_KEY", ""),
                    searxng_base_url=os.getenv("SEARXNG_BASE_URL", ""),
                    memories=memories_str
                )

                # Process any file attachments for the agent
                binary_contents = []
                if request.files:
                    for file in request.files:
                        try:
                            # Decode the base64 content
                            binary_data = base64.b64decode(file.content)
                            # Create a BinaryContent object
                            fileMimeType = "application/pdf" if file.mimeType == "text/plain" else file.mimeType
                            binary_content = BinaryContent(
                                data=binary_data,
                                media_type=fileMimeType
                            )
                            binary_contents.append(binary_content)
                        except Exception as e:
                            print(f"Error processing file {file.fileName}: {str(e)}")

                # Create input for the agent with the query and any binary contents
                agent_input = [request.query]
                if binary_contents:
                    agent_input.extend(binary_contents)

                full_response = ""

                # Use tracer context if available, otherwise use nullcontext
                span_context = tracer.start_as_current_span("Pydantic-Ai-Trace") if tracer else nullcontext()

                with span_context as span:
                    if tracer and span:
                        # Set user and session attributes for Langfuse
                        span.set_attribute("langfuse.user.id", request.user_id)
                        span.set_attribute("langfuse.session.id", session_id)
                        span.set_attribute("input.value", request.query)

                    # Run the agent with the user prompt, binary contents, and the chat history
                    async with agent.iter(agent_input, deps=agent_deps, message_history=pydantic_messages) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # A model request node => We can stream tokens from the model's request
                                async with node.stream(run.ctx) as request_stream:
                                    async for event in request_stream:
                                        if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                            yield json.dumps({"text": event.part.content}).encode('utf-8') + b'\n'
                                            full_response += event.part.content
                                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                            delta = event.delta.content_delta
                                            yield json.dumps({"text": full_response}).encode('utf-8') + b'\n'
                                            full_response += delta

                    # Set the output value after completion if tracing
                    if tracer and span:
                        span.set_attribute("output.value", full_response)

                # After streaming is complete, store the full response in the database
                message_data = run.result.new_messages_json()

                # Store agent's response
                await store_message(
                    supabase=supabase,
                    session_id=session_id,
                    message_type="ai",
                    content=full_response,
                    message_data=message_data,
                    data={"request_id": request.request_id}
                )

                # Wait for title generation to complete if it's running
                if title_task:
                    try:
                        title_result = await title_task
                        conversation_title = title_result
                        # Update the conversation title in the database
                        await update_conversation_title(supabase, session_id, conversation_title)

                        # Send the final title in the last chunk
                        final_data = {
                            "text": full_response,
                            "session_id": session_id,
                            "conversation_title": conversation_title,
                            "complete": True
                        }
                        yield json.dumps(final_data).encode('utf-8') + b'\n'
                    except Exception as e:
                        print(f"Error processing title: {str(e)}")
                else:
                    yield json.dumps({"text": full_response, "complete": True}).encode('utf-8') + b'\n'

                # Wait for the memory task to complete if needed
                try:
                    await memory_task
                except Exception as e:
                    print(f"Error updating memories: {str(e)}")

                # Wait for request tracking task to complete
                try:
                    await request_tracking_task
                except Exception as e:
                    print(f"Error tracking request: {str(e)}")
                except asyncio.CancelledError:
                    # This is expected if the task was cancelled
                    pass

            except Exception as e:
                # Refund token if streaming fails
                print(f"Error in stream_response: {str(e)}")
                if token_was_deducted:
                    await refund_token(supabase, request.user_id, f"stream_error_{request.request_id}")
                    token_was_deducted = False  # Prevent double refund

                # Yield error message to user
                error_msg = "I apologize, but I encountered an error processing your request. Your token has been refunded."
                yield json.dumps({"text": error_msg, "error": str(e), "complete": True}).encode('utf-8') + b'\n'

        return StreamingResponse(stream_response(), media_type='text/plain')

    except Exception as e:
        print(f"Error processing request: {str(e)}")

        # Refund token if we deducted one
        if token_was_deducted:
            await refund_token(supabase, request.user_id, f"error_{request.request_id}")

        # Store error message in conversation if session_id exists
        if request.session_id:
            await store_message(
                supabase=supabase,
                session_id=request.session_id,
                message_type="ai",
                content="I apologize, but I encountered an error processing your request.",
                data={"error": str(e), "request_id": request.request_id}
            )
        # Return a streaming response with the error
        return StreamingResponse(
            stream_error_response(f"Error: {str(e)}", request.session_id),
            media_type='text/plain'
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration and monitoring.
    
    Returns:
        Dict with status and service health information
    """
    # Check if critical dependencies are initialized
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "embedding_client": embedding_client is not None,
            "supabase": supabase is not None,
            "http_client": http_client is not None,
            "title_agent": title_agent is not None,
            "mem0_client": mem0_client is not None
        }
    }
    
    # If any critical service is not initialized, mark as unhealthy
    if not all(health_status["services"].values()):
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
