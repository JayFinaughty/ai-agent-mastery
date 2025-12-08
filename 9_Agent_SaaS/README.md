# Dynamous AI Agent Mastery - Module 9: AI Agent SaaS

A production-ready AI agent application with Stripe payment integration for token-based billing. This module takes the agent deployment from previous modules and adds monetization capabilities, turning it into a complete SaaS product. Users purchase tokens via Stripe and consume them when interacting with the AI agent.

The system includes independently deployable components: the AI agent API (with payment endpoints), RAG pipeline, and frontend application (with purchase flows). Each component can be deployed, scaled, and maintained separately while working together as a cohesive system.

## Modular Architecture

The deployment structure has been designed for maximum flexibility and scalability:

```
9_Agent_SaaS/
├── backend_agent_api/      # AI Agent with FastAPI + Stripe payment endpoints
├── backend_rag_pipeline/   # Document processing pipeline - Handles knowledge ingestion
├── frontend/               # React application with token purchase flows
└── sql/                    # Database schemas including token billing tables
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
- Stripe account for payment processing (https://stripe.com)
- Optional: Brave API key for web search (or local SearXNG)
- Optional: Google Drive API credentials for Google Drive RAG

## Database Setup

The database is the foundation for all components. Set it up first:

1. **Create a Supabase project:**
   - **Cloud**: Create a project at [https://supabase.com](https://supabase.com)
   - **Local**: Navigate to http://localhost:8000 (default Supabase dashboard)

2. **Navigate to the SQL Editor** in your Supabase dashboard

3. **Run the database setup:**

   **Option A - Fresh install (recommended):**
   ```sql
   -- Copy and paste the contents of sql/0-all-tables.sql
   -- This creates all tables INCLUDING token billing for Stripe
   ```

   **⚠️ Important**: The `0-all-tables.sql` script will DROP and recreate the agent tables. This resets the agent data to a blank slate.

   **Option B - Upgrading from a previous module:**

   If you already have the base agent tables from a previous module, just run the Stripe migrations:
   ```sql
   -- Run these scripts in order:
   -- sql/10-stripe-tokens.sql      (adds tokens column to user_profiles)
   -- sql/11-stripe-transactions.sql (transaction audit trail)
   -- sql/12-stripe-tokens-rls.sql   (row level security)
   -- sql/13-stripe-functions.sql    (atomic token operations)
   -- sql/14-stripe-migration.sql    (grant tokens to existing users)
   -- sql/15-enable-realtime.sql     (live balance updates in UI)
   ```

**Ollama Configuration**: For local Ollama with nomic-embed-text, modify the vector dimensions from 1536 to 768 in `0-all-tables.sql` (lines 133 and 149).

## Stripe Setup

Stripe powers the token billing system. Users purchase token packages and consume them when using the AI agent.

<details>
<summary><strong>Click to expand Stripe setup instructions</strong></summary>

### How Token Billing Works

- Users purchase token packages ($5/100, $10/250, $20/600 tokens)
- Each AI agent request consumes 1 token
- Real-time balance updates in the UI
- Transaction history for audit trail

### Install Stripe CLI (Required for Local Development)

Stripe no longer allows `localhost` URLs in the dashboard for webhooks. You must use the Stripe CLI for local development.

**macOS:**
```bash
brew install stripe/stripe-cli/stripe
```

**Windows:**
```bash
scoop bucket add stripe https://github.com/stripe/scoop-stripe-cli.git
scoop install stripe
```

**Linux (Debian/Ubuntu):**
```bash
curl -s https://packages.stripe.dev/api/security/keypair/stripe-cli-gpg/public | gpg --dearmor | sudo tee /usr/share/keyrings/stripe.gpg
echo "deb [signed-by=/usr/share/keyrings/stripe.gpg] https://packages.stripe.dev/stripe-cli-debian-local stable main" | sudo tee -a /etc/apt/sources.list.d/stripe.list
sudo apt update && sudo apt install stripe
```

**Login to Stripe:**
```bash
stripe login
```

### Get API Keys

1. Go to https://dashboard.stripe.com/apikeys
2. Copy your **Publishable key** (`pk_test_...`) → Frontend `.env`
3. Copy your **Secret key** (`sk_test_...`) → Backend `.env`

### Start Webhook Forwarding (Local Development)

1. **Start your backend API** on port 8001

2. **In a separate terminal, start Stripe CLI:**
   ```bash
   stripe listen --forward-to localhost:8001/api/webhook/stripe
   ```

3. **Copy the webhook secret** from CLI output:
   ```
   > Ready! Your webhook signing secret is whsec_xxxxx
   ```

4. **Update backend `.env`** with the webhook secret and restart

**Note:** Keep `stripe listen` running while testing. The secret changes each time you restart the CLI.

### Test Cards

| Scenario | Card Number |
|----------|-------------|
| Success | `4242 4242 4242 4242` |
| Decline | `4000 0000 0000 0002` |
| Auth Required | `4000 0025 0000 3155` |

Use any future expiry date and any 3-digit CVC.

### Test the Flow

1. Login to frontend
2. Navigate to `/purchase-tokens`
3. Select a tier and use test card `4242424242424242`
4. Watch the `stripe listen` terminal for webhook events
5. Verify tokens added to account

### Production Webhook Setup

1. **Get live API keys** from Stripe Dashboard (toggle to Live mode)
2. **Create webhook endpoint** in Stripe Dashboard → Developers → Webhooks:
   - URL: `https://your-domain.com/api/webhook/stripe`
   - Events: `payment_intent.succeeded`
3. **Copy the webhook signing secret** (`whsec_...`)
4. **Update environment variables** with live keys

### Troubleshooting

**Webhook signature fails:**
- Verify `STRIPE_WEBHOOK_SECRET` matches CLI output (or dashboard secret for production)
- Ensure backend is using raw request body for verification

**Tokens not granted after payment:**
- Check `stripe listen` terminal for `[200]` responses
- Verify webhook is receiving `payment_intent.succeeded` events
- Check backend logs for errors

**Real-time balance updates not working:**
- Ensure `sql/15-enable-realtime.sql` was run in Supabase
- Check Supabase Dashboard → Database → Replication → `user_profiles` is enabled

For complete setup documentation, see `PRPs/planning/stripe-setup.md`.

</details>

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

Edit `.env` with your configuration:

```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key_here
LLM_CHOICE=gpt-4o-mini

# Embedding Configuration
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Database Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# Frontend Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here
VITE_AGENT_ENDPOINT=http://localhost:8001/api/pydantic-agent

# Optional: LangFuse integration
VITE_LANGFUSE_HOST_WITH_PROJECT=http://localhost:3000/project/your-project-id

# RAG Pipeline Configuration
RAG_PIPELINE_TYPE=local  # or google_drive
RUN_MODE=continuous      # or single for scheduled runs
RAG_PIPELINE_ID=dev-local-pipeline  # Required for single-run mode

# Optional: Google Drive Configuration
GOOGLE_DRIVE_CREDENTIALS_JSON=  # Service account JSON for Google Drive if using a Service Account
RAG_WATCH_FOLDER_ID=           # Specific Google Drive folder ID

# Optional: Local Files Configuration  
RAG_WATCH_DIRECTORY=           # Override container path (default: /app/Local_Files/data)

# Stripe Configuration (Required for payments)
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here  # Get from Stripe CLI

# Optional Langfuse agent monitoring configuration
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Hostnames for Caddy reverse proxy routes
# Leave these commented if you aren't deploying to production
AGENT_API_HOSTNAME=agent.yourdomain.com
FRONTEND_HOSTNAME=chat.yourdomain.com
```

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
docker compose -p dynamous-agent logs -f

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

**For Cloud Deployment:**
```bash
# View logs for specific service
docker compose -p dynamous-agent logs -f agent-api
docker compose -p dynamous-agent logs -f rag-pipeline
docker compose -p dynamous-agent logs -f frontend
docker compose -p dynamous-agent logs -f caddy

# Rebuild specific service
docker compose -p dynamous-agent build agent-api
docker compose -p dynamous-agent up -d agent-api

# Check service health
docker compose -p dynamous-agent ps
```

**For Local Deployment:**
```bash
# View logs for specific service  
docker compose -p localai logs -f agent-api
docker compose -p localai logs -f rag-pipeline
docker compose -p localai logs -f frontend

# Rebuild specific service
docker compose -p localai build agent-api
docker compose -p localai up -d agent-api

# Check service health
docker compose -p localai ps
```

**For Base Services Only:**
```bash
# View logs for specific service
docker compose -p dynamous-agent logs -f agent-api
docker compose -p dynamous-agent logs -f rag-pipeline
docker compose -p dynamous-agent logs -f frontend

# Rebuild and restart services
docker compose up -d --build

# Scale RAG pipeline (if using single-run mode)
docker compose up -d --scale rag-pipeline=0  # Stop for scheduled runs

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
- **Setup**: Use the provided `docker compose.yml` to deploy all services together

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
| Setup Complexity | ⭐ (Easiest) | ⭐⭐ (Still Pretty Easy) | ⭐⭐⭐ (Moderate) |
| Cost for Small Apps | $$ | $ (Free tier) | $ (Free tier) |
| Scalability | Manual | Automatic | Automatic |
| Geographic Distribution | Single region | Multi-region | Global |
| Best For | Quick start or Local AI | Most cloud based projects | Enterprise |

## Environment Variables Reference

### Agent API & RAG Pipeline
```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_CHOICE=gpt-4o-mini
VISION_LLM_CHOICE=gpt-4o-mini

# Embedding Configuration  
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Database
DATABASE_URL=postgresql://user:pass@host:port/db
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# Web Search
BRAVE_API_KEY=your_brave_key
SEARXNG_BASE_URL=http://localhost:8080

# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# RAG Pipeline Configuration
RAG_PIPELINE_TYPE=local          # local or google_drive
RUN_MODE=continuous              # continuous or single
RAG_PIPELINE_ID=my-pipeline      # Required for single-run mode

# Google Drive (for RAG Pipeline)
GOOGLE_DRIVE_CREDENTIALS_JSON=   # Service account JSON string
RAG_WATCH_FOLDER_ID=            # Specific folder ID to watch

# Local Files (for RAG Pipeline)
RAG_WATCH_DIRECTORY=            # Container path override
```

### Frontend
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_AGENT_ENDPOINT=http://localhost:8001/api/pydantic-agent
VITE_ENABLE_STREAMING=true

# Stripe Configuration
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key

# Optional: LangFuse integration for admin dashboard
VITE_LANGFUSE_HOST_WITH_PROJECT=http://localhost:3000/project/your-project-id

# Reverse Proxy Configuration (for Caddy deployments)
AGENT_API_HOSTNAME=agent.yourdomain.com
FRONTEND_HOSTNAME=chat.yourdomain.com
```

## Agent Observability with LangFuse (Optional)

This deployment includes optional LangFuse integration for comprehensive agent observability. LangFuse provides detailed insights into agent conversations, performance metrics, and debugging capabilities - particularly valuable for production deployments.

### What LangFuse Provides
- **Conversation Tracking**: Complete agent interaction histories with user and session context
- **Performance Metrics**: Response times, token usage, and cost tracking
- **Debugging Tools**: Detailed execution traces for troubleshooting agent behavior
- **User Analytics**: Insights into user patterns and agent effectiveness

### Setup (Completely Optional)

**To enable LangFuse observability:**

1. **Create a LangFuse account** at [https://cloud.langfuse.com/](https://cloud.langfuse.com/) (free tier available)

2. **Create a new project** and obtain your credentials

3. **Add LangFuse environment variables** to your agent API `.env` file:
   ```env
   # Agent observability (optional - leave empty to disable)
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

4. **Optional: Enable frontend integration** by setting in your frontend `.env`:
   ```env
   # Add clickable LangFuse links in the admin dashboard
   VITE_LANGFUSE_HOST_WITH_PROJECT=https://cloud.langfuse.com/project/your-project-id
   ```

**To disable LangFuse (default behavior):**
- Simply leave the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` empty
- The agent runs normally with no observability overhead

### Benefits for Different Use Cases

- **Development**: Debug agent behavior and optimize conversation flows
- **Production**: Monitor performance, track usage costs, and identify issues
- **Analytics**: Understand user interactions and improve agent effectiveness
- **Team Collaboration**: Share conversation traces and debugging information

The LangFuse integration is designed to be zero-impact when disabled, making it perfect for development environments where observability isn't needed.

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
   
   # Stop conflicting services or change ports in docker compose.yml
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
2. **Vector dimensions**: Ensure embedding model dimensions match database schema
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

# Test frontend
curl http://localhost:8082/health
```

## Testing

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

# Run specific test file
npx playwright test auth.spec.ts

# Debug tests
npx playwright test --debug
```

**Test Features:**
- ✅ **Complete mocking** - No database or API calls
- ✅ **Authentication flow** - Login, signup, logout
- ✅ **Chat interface** - Send messages, receive responses
- ✅ **Conversation management** - New chats, conversation history
- ✅ **Loading states** - UI feedback during operations

The tests use comprehensive mocks for:
- Supabase authentication and database
- Agent API streaming responses
- User sessions and conversation data

## Support

For detailed instructions on each component, refer to their individual README files:
- `backend_agent_api/README.md` - Agent API specifics
- `backend_rag_pipeline/README.md` - RAG pipeline details  
- `frontend/README.md` - Frontend development guide

Remember: The modular structure allows you to start with local deployment and gradually move components to the cloud as needed!