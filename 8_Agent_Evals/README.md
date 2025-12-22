# Dynamous AI Agent Mastery - Complete Agent

The complete, production-ready AI agent from the Dynamous AI Agent Mastery course. This is the final version of the agent with all features implemented, packages fully updated, and comprehensive evaluation capabilities built in.

**This module includes everything from previous modules plus:**
- Agent evaluations framework for testing and monitoring quality (see [`backend_agent_api/evals/`](./backend_agent_api/evals/))
- Production evaluation integration with Langfuse scoring
- User feedback collection (thumbs up/down, conversation ratings)
- All dependencies updated to latest stable versions

Whether you're learning, experimenting, or deploying to production, this is the recommended starting point. For monetization with Stripe integration, see [Module 9: Agent SaaS](../9_Agent_SaaS/).

## Modular Architecture

The deployment structure has been designed for maximum flexibility and scalability:

```
8_Agent_Evals/
├── backend_agent_api/      # AI Agent with FastAPI - The brain of the system
│   └── evals/              # Evaluation framework (golden datasets, LLM judges, production evals)
├── backend_rag_pipeline/   # Document processing pipeline - Handles knowledge ingestion
├── frontend/               # React application - User interface
├── sql/                    # Database schemas - Foundation for all components
└── ~deployment_guides~/    # Platform-specific deployment instructions
```

Each component is self-contained with its own:
- Dependencies and virtual environment
- Environment configuration
- README with specific instructions
- Deployment capabilities

This modular approach allows you to:
- Deploy components to different services (e.g., agent on GCP Cloud Run, RAG on DigitalOcean, frontend on Render)
- Scale components independently based on load
- Update and maintain each component without affecting others
- Choose different deployment strategies for each component

## Prerequisites

- Docker/Docker Desktop (recommended) OR Python 3.11+ and Node.js 18+ with npm
- Supabase account (or self-hosted instance)
- LLM provider account (OpenAI, OpenRouter, or local Ollama)
- Optional: Brave API key for web search (or local SearXNG)
- Optional: Google Drive API credentials for Google Drive RAG
- Optional: Langfuse account for observability and evaluations

## Database Setup

The database is the foundation for all components. Set it up first:

1. **Create a Supabase project:**
   - **Cloud**: Create a project at [https://supabase.com](https://supabase.com)
   - **Local**: Navigate to http://localhost:8000 (default Supabase dashboard)

2. **Navigate to the SQL Editor** in your Supabase dashboard

3. **Run the complete database setup:**
   ```sql
   -- Copy and paste the contents of sql/0-all-tables.sql
   -- This creates all tables, functions, triggers, and security policies
   ```

   **Warning**: The `0-all-tables.sql` script will DROP and recreate the agent tables (user_profiles, conversations, messages, documents, etc.). This resets the agent data to a blank slate - existing agent data will be lost, but other tables in your Supabase project remain untouched.

**Alternative**: You can run the individual scripts (`1-user_profiles_requests.sql` through `9-rag_pipeline_state.sql`) if you prefer granular control.

**Ollama Configuration**: For local Ollama implementations using models like nomic-embed-text, modify the vector dimensions from 1536 to 768 in `0-all-tables.sql` (lines 133 and 149).

## Deployment Methods

### Method 1: Development Mode (Manual Components)

For development without Docker or to run individual containers separately, see the component-specific READMEs:

- [Backend Agent API](./backend_agent_api/README.md) - Python agent with FastAPI
- [Backend RAG Pipeline](./backend_rag_pipeline/README.md) - Document processing pipeline
- [Frontend](./frontend/README.md) - React application

### Method 2: Smart Deployment Script (Recommended)

The easiest way to deploy the stack is using the included Python deployment script, which automatically handles both local and cloud deployment scenarios.

#### Configure Environment Variables

First, configure your environment variables:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your configuration (see [Environment Variables Reference](#environment-variables-reference) for details).

#### Deploy with Script

##### Cloud Deployment (Standalone with Caddy)
Deploy as a self-contained stack with built-in reverse proxy:

```bash
# Deploy to cloud (includes Caddy reverse proxy)
python deploy.py --type cloud

# Stop cloud deployment
python deploy.py --down --type cloud
```

##### Local Deployment (Integrate with the Local AI Package)
Deploy to work alongside your existing Local AI Package with shared Caddy:

```bash
# Deploy alongside the Local AI Package (uses existing Caddy)
python deploy.py --type local --project localai

# Stop local deployment
python deploy.py --down --type local --project localai
```

**To enable reverse proxy routes in your Local AI Package**:

1. **Copy and configure** the addon file:
   ```bash
   # Copy caddy-addon.conf to your Local AI Package's caddy-addon folder
   cp caddy-addon.conf /path/to/local-ai-package/caddy-addon/

   # Edit lines 2 and 21 to set your desired subdomains:
   # Line 2: subdomain.yourdomain.com (for agent API)
   # Line 21: subdomain2.yourdomain.com (for frontend)
   ```

2. **Restart Caddy in the Local AI Package** to load the new configuration:
   ```bash
   docker compose -p localai restart caddy
   ```

### Method 3: Direct Docker Compose (Advanced Users)

For advanced users who prefer direct Docker Compose control:

#### 1. Configure Environment Variables

Use the same environment variable configuration shown in Method 2 above.

#### 2. Start All Services

**Cloud Deployment (with Caddy):**
```bash
# Build and start all services including Caddy
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d --build

# Restart services without rebuilding
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d

# Stop all services
docker compose -f docker-compose.yml -f docker-compose.caddy.yml down
```

**Local Deployment (for AI stack integration):**
```bash
# Build and start services with local overrides
docker compose -f docker-compose.yml -p localai up -d --build

# Restart services without rebuilding
docker compose -f docker-compose.yml -p localai up -d

# Stop all services
docker compose -f docker-compose.yml -p localai down
```

**Base Services Only (no reverse proxy):**
```bash
# Build and start base services only
docker compose up -d --build

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

#### 3. Access the Application

**Cloud Deployment Access:**
- **Frontend**: https://your-frontend-hostname (configured in .env)
- **Agent API**: https://your-agent-api-hostname (configured in .env)
- **Health Check**: https://your-agent-api-hostname/health

**Local/Base Deployment Access:**
- **Frontend**: http://localhost:8082
- **Agent API**: http://localhost:8001
- **Health Check**: http://localhost:8001/health

#### 4. Add Documents to RAG Pipeline

For local files:
```bash
# Copy documents to the RAG pipeline directory
cp your-documents/* ./rag-documents/
```

For Google Drive:
```bash
# Place your Google Drive credentials (if using OAuth and not a service account)
cp credentials.json ./google-credentials/
```

#### Docker Compose Management Commands

```bash
# View logs for specific service
docker compose logs -f agent-api
docker compose logs -f rag-pipeline
docker compose logs -f frontend

# Rebuild specific service
docker compose build agent-api
docker compose up -d agent-api

# Check service health
docker compose ps
```

## Development Mode

For development with live reloading, run each component separately. You'll need 3-4 terminal windows:

### Quick Setup for Each Component

1. **Terminal 1 - Agent API:**
   ```bash
   cd backend_agent_api
   python -m venv venv
   venv\Scripts\activate  # or source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Then edit with your config
   uvicorn agent_api:app --reload --port 8001
   ```

2. **Terminal 2 - RAG Pipeline:**
   ```bash
   cd backend_rag_pipeline
   python -m venv venv
   venv\Scripts\activate  # or source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Then edit with your config
   python docker_entrypoint.py --pipeline local --mode continuous
   ```

3. **Terminal 3 - Frontend:**
   ```bash
   cd frontend
   npm install
   cp .env.example .env  # Then edit with your config
   npm run dev
   ```

4. **Terminal 4 (Optional) - Code Execution MCP server:**
   ```bash
   deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python sse
   ```

**Note:** Don't forget to run the SQL scripts first (see Database Setup above) and configure each `.env` file with your credentials.

## Deployment Options

We provide three deployment strategies, from simple to enterprise-grade:

### Option 1: DigitalOcean with Docker Compose (Simplest)
Deploy the entire stack on a single DigitalOcean Droplet using Docker Compose:
- **Pros**: Simple setup, everything in one place, easy to manage
- **Cons**: All components scale together, single point of failure
- **Best for**: Small teams, prototypes, and getting started quickly
- **Setup**: Use the provided `docker-compose.yml` to deploy all services together

### Option 2: Render Platform (Recommended)
Deploy each component separately on Render for better scalability:
- **Agent API**: Deploy as a Docker container with autoscaling
- **RAG Pipeline**: Set up as a scheduled job (cron)
- **Frontend**: Deploy as a static site from the build output
- **Pros**: Automatic scaling, managed infrastructure, good free tier
- **Cons**: Requires configuring each service separately
- **Best for**: Production applications with moderate traffic

### Option 3: Google Cloud Platform (Enterprise)
For enterprise deployments with maximum flexibility:
- **Agent API**: Cloud Run for serverless, auto-scaling containers
- **RAG Pipeline**: Cloud Scheduler + Cloud Run for scheduled processing
- **Frontend**: Cloud Storage + Cloud CDN for global static hosting
- **Database**: Consider Cloud SQL for Postgres instead of Supabase
- **Pros**: Enterprise features, global scale, fine-grained control
- **Cons**: More complex setup, requires GCP knowledge
- **Best for**: Large-scale production deployments

### Deployment Decision Matrix

| Feature | DigitalOcean | Render | Google Cloud |
|---------|--------------|---------|--------------|
| Setup Complexity | Low | Medium | Medium-High |
| Cost for Small Apps | $$ | $ (Free tier) | $ (Free tier) |
| Scalability | Manual | Automatic | Automatic |
| Geographic Distribution | Single region | Multi-region | Global |
| Best For | Quick start or Local AI | Most cloud projects | Enterprise |

## Environment Variables Reference

### Agent API & RAG Pipeline

```env
# LLM Configuration
LLM_PROVIDER=openai                          # openai, openrouter, or ollama
LLM_BASE_URL=https://api.openai.com/v1       # API endpoint
LLM_API_KEY=your_api_key                     # Your API key
LLM_CHOICE=gpt-4o-mini                       # Model for agent
VISION_LLM_CHOICE=gpt-4o-mini                # Model for image analysis

# Embedding Configuration
EMBEDDING_PROVIDER=openai                    # openai or ollama
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Database
DATABASE_URL=postgresql://user:pass@host:port/db  # For Mem0
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# Web Search (one or the other)
BRAVE_API_KEY=your_brave_key                 # Leave empty if using SearXNG
SEARXNG_BASE_URL=http://localhost:8080       # Leave empty if using Brave

# Agent Observability & Evaluations (optional)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# RAG Pipeline Configuration
RAG_PIPELINE_TYPE=local                      # local or google_drive
RUN_MODE=continuous                          # continuous or single
RAG_PIPELINE_ID=my-pipeline                  # Required for single-run mode
CHECK_INTERVAL=60                            # Seconds between checks

# Google Drive (for RAG Pipeline)
GOOGLE_DRIVE_CREDENTIALS_JSON=               # Service account JSON string
RAG_WATCH_FOLDER_ID=                         # Specific folder ID to watch

# Local Files (for RAG Pipeline)
RAG_WATCH_DIRECTORY=/app/Local_Files/data    # Container path
```

### Frontend

```env
# Supabase (required)
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_AGENT_ENDPOINT=http://localhost:8001/api/pydantic-agent
VITE_ENABLE_STREAMING=true

# Langfuse Admin Integration (optional)
VITE_LANGFUSE_HOST_WITH_PROJECT=https://cloud.langfuse.com/project/your-project-id

# User Feedback (optional - requires Langfuse)
VITE_LANGFUSE_PUBLIC_KEY=your_public_key     # Same as backend public key
VITE_LANGFUSE_HOST=https://cloud.langfuse.com
VITE_RATING_THRESHOLDS=5,10,15,20,25,30      # When to show conversation rating popup

# Reverse Proxy (for Caddy deployments)
AGENT_API_HOSTNAME=agent.yourdomain.com
FRONTEND_HOSTNAME=chat.yourdomain.com
```

## Agent Observability with Langfuse (Optional)

This deployment includes optional Langfuse integration for comprehensive agent observability and evaluation tracking. Langfuse provides detailed insights into agent conversations, performance metrics, and quality scoring.

### What Langfuse Provides

- **Conversation Tracking**: Complete agent interaction histories with user and session context
- **Performance Metrics**: Response times, token usage, and cost tracking
- **Evaluation Scores**: Automated rule-based and LLM judge scores synced per trace
- **User Feedback**: Thumbs up/down and conversation ratings from end users
- **Debugging Tools**: Detailed execution traces for troubleshooting agent behavior

### Setup (Completely Optional)

**To enable Langfuse observability and evaluations:**

1. **Create a Langfuse account** at [https://cloud.langfuse.com/](https://cloud.langfuse.com/) (free tier available)

2. **Create a new project** and obtain your credentials

3. **Add Langfuse environment variables** to your `.env` file:
   ```env
   # Backend: Agent observability and production evals
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com

   # Frontend: User feedback collection
   VITE_LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   VITE_LANGFUSE_HOST=https://cloud.langfuse.com
   ```

**To disable Langfuse (default behavior):**
- Simply leave the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` empty
- The agent runs normally with no observability overhead

### Benefits for Different Use Cases

- **Development**: Debug agent behavior and optimize conversation flows
- **Production**: Monitor performance, track quality scores, and identify issues
- **Evaluation**: View automated scores from rule-based checks and LLM judges
- **User Insights**: Collect and analyze thumbs up/down feedback from users

The Langfuse integration is designed to be zero-impact when disabled, making it perfect for development environments where observability isn't needed.

## Agent Evaluations

The agent includes a complete evaluation framework in [`backend_agent_api/evals/`](./backend_agent_api/evals/). This enables you to:

- **Test locally** with golden datasets before deploying changes
- **Monitor production** with automated rule-based checks and LLM judge scoring
- **Collect user feedback** through thumbs up/down ratings and conversation ratings
- **Export annotations** from Langfuse for training data or calibration

To run local evaluations:
```bash
cd backend_agent_api
python evals/run_evals.py                  # Run general dataset
python evals/run_evals.py --dataset rag    # Run RAG/web search dataset
python evals/run_evals.py --dataset all    # Run both datasets
```

Production evaluations run automatically when Langfuse is configured, syncing scores to your Langfuse dashboard.

## Troubleshooting

### Docker Compose Issues

1. **Services won't start**:
   ```bash
   # Check service logs
   docker compose logs -f

   # Rebuild images
   docker compose build --no-cache
   ```

2. **Port conflicts**:
   ```bash
   # Check what's using ports
   netstat -tlnp | grep :8001

   # Stop conflicting services or change ports in docker-compose.yml
   ```

3. **Environment variables not loading**:
   ```bash
   # Verify .env file exists and has correct format
   cat .env

   # Check environment in container
   docker compose exec agent-api env | grep LLM_
   ```

### Common Issues

1. **Database connection**: Verify Supabase credentials and network access
2. **Vector dimensions**: Ensure embedding model dimensions match database schema (1536 for OpenAI, 768 for nomic-embed-text)
3. **CORS errors**: Check API endpoint configuration in frontend `.env`
4. **Memory issues**: Increase Docker memory limits for large models

### Verification Steps

1. **Database**: Check Supabase dashboard for table creation
2. **Agent API Health**: Visit http://localhost:8001/health
3. **API Documentation**: Visit http://localhost:8001/docs
4. **RAG Pipeline**: Check logs with `docker compose logs rag-pipeline`
5. **Frontend**: Open browser console for any errors

### Health Checks

Monitor service health:
```bash
# Check all service health
docker compose ps

# Check specific service logs
docker compose logs -f agent-api

# Test API health endpoint
curl http://localhost:8001/health
```

## Testing

### Agent Evaluations

Run the evaluation suite to test agent behavior:

```bash
cd backend_agent_api

# Run general golden dataset (10 test cases)
python evals/run_evals.py

# Run RAG/web search dataset (15 test cases)
python evals/run_evals.py --dataset rag

# Run all datasets
python evals/run_evals.py --dataset all
```

### Frontend Testing with Playwright

The frontend includes Playwright tests for end-to-end testing with mocked Supabase and agent API calls.

```bash
cd frontend

# Make sure Playwright is installed
npx playwright install --with-deps

# Run all tests
npm run test

# Run tests with interactive UI
npm run test:ui

# Run tests in headed browser mode (see the browser)
npm run test:headed

# Debug tests
npx playwright test --debug
```

**Test Features:**
- Complete mocking - No database or API calls
- Authentication flow - Login, signup, logout
- Chat interface - Send messages, receive responses
- Conversation management - New chats, conversation history
- Loading states - UI feedback during operations

## Support

For detailed instructions on each component, refer to their individual README files:
- `backend_agent_api/README.md` - Agent API specifics
- `backend_rag_pipeline/README.md` - RAG pipeline details
- `frontend/README.md` - Frontend development guide

Remember: The modular structure allows you to start with local deployment and gradually move components to the cloud as needed!
