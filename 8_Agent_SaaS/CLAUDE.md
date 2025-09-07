# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a modular AI agent deployment system with three independently deployable components:

1. **backend_agent_api**: FastAPI server serving a Pydantic AI agent with RAG, web search, image analysis, and code execution
2. **backend_rag_pipeline**: Document processing pipeline that watches local files or Google Drive for changes
3. **frontend**: React/TypeScript application with real-time streaming chat interface

All components share a Supabase database for data persistence and vector storage.

## Development Commands

### Backend Agent API
```bash
cd backend_agent_api
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
uvicorn agent_api:app --reload --port 8001
```

### Backend RAG Pipeline
```bash
cd backend_rag_pipeline
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# Local files pipeline
python Local_Files/main.py --directory "./data"

# Google Drive pipeline
python Google_Drive/main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev          # Development server (port 8081)
npm run build        # Production build
npm run lint         # ESLint check
npm run preview      # Preview production build
```

### Docker Compose (Full Stack)
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Rebuild and restart
docker compose up -d --build

# Stop all services
docker compose down
```

## Testing

### Backend Agent API
```bash
cd backend_agent_api
pytest                    # Run all tests
pytest tests/test_tools.py   # Specific test file
```

### Backend RAG Pipeline
```bash
cd backend_rag_pipeline
pytest                    # Run all tests
pytest Local_Files/tests/   # Local files tests only
pytest Google_Drive/tests/  # Google Drive tests only
```

## Database Setup

Execute SQL scripts in order from the `sql/` directory:
1. `1-user_profiles_requests.sql`
2. `2-user_profiles_requests_rls.sql`
3. `3-conversations_messages.sql`
4. `4-conversations_messages_rls.sql`
5. `5-documents.sql`
6. `6-document_metadata.sql`
7. `7-document_rows.sql`
8. `8-execute_sql_rpc.sql`

**Important**: For local Ollama with nomic-embed-text, change vector dimensions from 1536 to 768 in documents table schema.

## Environment Configuration

Each component requires its own `.env` file copied from `.env.example`.

### Backend Agent API (.env)
```env
# LLM Configuration
LLM_PROVIDER=openai                           # openai, openrouter, or ollama
LLM_BASE_URL=https://api.openai.com/v1        # API endpoint
LLM_API_KEY=your_api_key_here                 # API key for LLM provider
LLM_CHOICE=gpt-4o-mini                        # Model name
VISION_LLM_CHOICE=gpt-4o-mini                 # Vision model for image analysis

# Embedding Configuration
EMBEDDING_PROVIDER=openai                     # openai or ollama
EMBEDDING_BASE_URL=https://api.openai.com/v1  # Embedding API endpoint
EMBEDDING_API_KEY=your_api_key_here           # Usually same as LLM_API_KEY
EMBEDDING_MODEL_CHOICE=text-embedding-3-small # Embedding model

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:port/db  # For mem0 (long-term memory)
SUPABASE_URL=https://your-project.supabase.co     # Supabase project URL
SUPABASE_SERVICE_KEY=your_service_key_here        # Supabase service key (not anon key)

# Web Search Configuration
BRAVE_API_KEY=your_brave_key                  # Leave empty if using SearXNG
SEARXNG_BASE_URL=http://localhost:8080        # Leave empty if using Brave
```

### Backend RAG Pipeline (.env)
```env
# Database Configuration (same as agent API)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here

# Embedding Configuration (must match agent API)
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_api_key_here
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Environment
ENVIRONMENT=development                       # or production
```

### Frontend (.env)
```env
# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co  # Same as backend
VITE_SUPABASE_ANON_KEY=your_anon_key_here           # Anon key (NOT service key)

# Agent API Configuration
VITE_AGENT_ENDPOINT=http://localhost:8001/api/pydantic-agent  # Local development
# VITE_AGENT_ENDPOINT=https://your-api-url/api/pydantic-agent  # Production

# Features
VITE_ENABLE_STREAMING=true                    # false for n8n agents
```

### Google Drive RAG Pipeline Setup

The Google Drive pipeline requires OAuth2 credentials from Google Cloud Console:

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create new project or select existing one
   - Enable Google Drive API

2. **Create OAuth2 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Desktop application"
   - Download the JSON file

3. **Place Credentials**:
   ```bash
   # Default location (expected by pipeline)
   backend_rag_pipeline/Google_Drive/credentials.json
   
   # Or specify custom path with --credentials flag
   python Google_Drive/main.py --credentials "./custom/path/credentials.json"
   ```

4. **First Run Authorization**:
   - First time running Google Drive pipeline opens browser for OAuth
   - Authorize access to Google Drive
   - Token saved to `Google_Drive/token.json` automatically
   - Subsequent runs use stored token

5. **Google Drive Configuration**:
   ```bash
   # Watch entire Google Drive
   python Google_Drive/main.py
   
   # Watch specific folder (get folder ID from Google Drive URL)
   python Google_Drive/main.py --folder-id "1ABC123XYZ789"
   ```

## Code Architecture Notes

### Agent Implementation
- **agent.py**: Main Pydantic AI agent with system prompt and dependencies
- **tools.py**: Tool implementations (RAG, web search, image analysis, code execution)
- **clients.py**: Client configurations for LLMs, databases, and services
- **agent_api.py**: FastAPI wrapper with streaming support

### RAG Pipeline Architecture
- **Single-run vs Continuous modes**: Pipeline supports both scheduled jobs and continuous monitoring
- **Dual source support**: Can watch local files or Google Drive
- **docker_entrypoint.py**: Handles mode selection and pipeline initialization
- **common/**: Shared utilities for text processing and database operations

### Frontend Architecture
- **React 18 + TypeScript + Vite**: Modern frontend stack
- **Shadcn UI**: Component library built on Radix UI
- **Real-time streaming**: Uses Server-Sent Events for live AI responses
- **Supabase integration**: Authentication and real-time database updates

## Key Integration Points

1. **Agent ↔ Database**: Agent queries vector embeddings via `retrieve_relevant_documents_tool`
2. **RAG Pipeline ↔ Database**: Pipeline stores document chunks and embeddings in `documents` table
3. **Frontend ↔ Agent**: POST requests to `/api/pydantic-agent` with streaming responses
4. **Frontend ↔ Database**: Direct Supabase client for conversation management

## Deployment Patterns

- **Development**: Run each component separately with live reload
- **Docker Compose**: Single-machine deployment with all services
- **Microservices**: Deploy each component to different cloud services (Render, GCP, etc.)

## Common Issues

- **Vector dimension mismatches**: Ensure embedding model dimensions match database schema
- **CORS errors**: Check `VITE_AGENT_ENDPOINT` configuration in frontend
- **Missing function calling**: Not all models support tools - verify model capabilities
- **Port conflicts**: Default ports are 8001 (agent), 8081 (frontend dev), 8082 (frontend prod)

## Core Development Philosophy

### KISS (Keep It Simple, Stupid)

Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)

Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles

- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and module should have one clear purpose.
- **Fail Fast**: Check for potential errors early and raise exceptions immediately when issues occur.

## 🧱 Code Structure & Modularity

### File and Function Limits

- **Never create a file longer than 500 lines of code**. If approaching this limit, refactor by splitting into modules.
- **Functions should be under 50 lines** with a single, clear responsibility.
- **Classes should be under 100 lines** and represent a single concept or entity.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Line lenght should be max 100 characters** ruff rule in pyproject.toml
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests.

## 📋 Style & Conventions

### Python Style Guide

- **Follow PEP8** with these specific choices:
  - Line length: 100 characters (set by Ruff in pyproject.toml)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures
- **Always use type hints** for function signatures and class attributes
- **Format with `ruff format`** (faster alternative to Black)
- **Use `pydantic` v2** for data validation and settings management

### Docstring Standards

Use Google-style docstrings for all public functions, classes, and modules:

```python
def calculate_discount(
    price: Decimal,
    discount_percent: float,
    min_amount: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate the discounted price for a product.

    Args:
        price: Original price of the product
        discount_percent: Discount percentage (0-100)
        min_amount: Minimum allowed final price

    Returns:
        Final price after applying discount

    Raises:
        ValueError: If discount_percent is not between 0 and 100
        ValueError: If final price would be below min_amount

    Example:
        >>> calculate_discount(Decimal("100"), 20)
        Decimal('80.00')
    """
```

### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes/methods**: `_leading_underscore`
- **Type aliases**: `PascalCase`
- **Enum values**: `UPPER_SNAKE_CASE`

## 🚨 Error Handling

### Exception Best Practices

```python
# Create custom exceptions for your domain
class PaymentError(Exception):
    """Base exception for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account has insufficient funds."""
    def __init__(self, required: Decimal, available: Decimal):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )

# Use specific exception handling
try:
    process_payment(amount)
except InsufficientFundsError as e:
    logger.warning(f"Payment failed: {e}")
    return PaymentResult(success=False, reason="insufficient_funds")
except PaymentError as e:
    logger.error(f"Payment error: {e}")
    return PaymentResult(success=False, reason="payment_error")

# Use context managers for resource management
from contextlib import contextmanager

@contextmanager
def database_transaction():
    """Provide a transactional scope for database operations."""
    conn = get_connection()
    trans = conn.begin_transaction()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()
```

## 🏗️ Data Models and Validation

### Example Pydantic Models strict with pydantic v2

```python
from pydantic import BaseModel, Field, validator, EmailStr
from datetime import datetime
from typing import Optional, List
from decimal import Decimal

class ProductBase(BaseModel):
    """Base product model with common fields."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    price: Decimal = Field(..., gt=0, decimal_places=2)
    category: str
    tags: List[str] = []

    @validator('price')
    def validate_price(cls, v):
        if v > Decimal('1000000'):
            raise ValueError('Price cannot exceed 1,000,000')
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

class ProductCreate(ProductBase):
    """Model for creating new products."""
    pass

class ProductUpdate(BaseModel):
    """Model for updating products - all fields optional."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    category: Optional[str] = None
    tags: Optional[List[str]] = None

class Product(ProductBase):
    """Complete product model with database fields."""
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True  # Enable ORM mode
```

## 🛡️ Security Best Practices

### Security Guidelines

- Never commit secrets - use environment variables
- Validate all user input with Pydantic
- Use parameterized queries for database operations
- Implement rate limiting for APIs
- Use HTTPS for all external communications
- Implement proper authentication and authorization

## 📚 Useful Resources

### Essential Tools

- Ruff: https://github.com/astral-sh/ruff
- Pytest: https://docs.pytest.org/
- Pydantic: https://docs.pydantic.dev/
- FastAPI: https://fastapi.tiangolo.com/

### Python Best Practices

- PEP 8: https://pep8.org/
- PEP 484 (Type Hints): https://www.python.org/dev/peps/pep-0484/
- The Hitchhiker's Guide to Python: https://docs.python-guide.org/

## ⚠️ Important Notes

- **NEVER ASSUME OR GUESS** - When in doubt, ask for clarification
- **Always verify file paths and module names** before use
- **Keep CLAUDE.md updated** when adding new patterns or dependencies
- **Test your code** - No feature is complete without tests
- **Document your decisions** - Future developers (including yourself) will thank you

## 🔍 Search Command Requirements

**CRITICAL**: Always use `rg` (ripgrep) instead of traditional `grep` and `find` commands:

```bash
# ❌ Don't use grep
grep -r "pattern" .

# ✅ Use rg instead
rg "pattern"

# ❌ Don't use find with name
find . -name "*.py"

# ✅ Use rg with file filtering
rg --files | rg "\.py$"
# or
rg --files -g "*.py"
```