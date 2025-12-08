-- ==============================================================================
-- Dynamous AI Agent Database Schema - Complete Setup
-- ==============================================================================
-- This file contains all tables, functions, triggers, and policies in correct order
-- Run this script to set up the complete database schema for the Dynamous AI Agent

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS vector;

DO $$ 
DECLARE
    rec RECORD;
BEGIN
    -- Drop policies safely
    FOR rec IN 
        SELECT schemaname, tablename, policyname 
        FROM pg_policies 
        WHERE schemaname = 'public' 
        AND policyname IN (
            'Deny delete for messages',
            'Admins can insert messages',
            'Admins can view all messages',
            'Users can insert messages in their conversations',
            'Users can view their own messages',
            'Deny delete for conversations',
            'Admins can insert conversations',
            'Admins can update all conversations',
            'Admins can view all conversations',
            'Users can update their own conversations',
            'Users can insert their own conversations',
            'Users can view their own conversations',
            'Deny delete for requests',
            'Admins can insert requests',
            'Admins can view all requests',
            'Users can view their own requests',
            'Admins can update all profiles',
            'Admins can view all profiles',
            'Only admins can change admin status',
            'Users can update their own profile',
            'Users can view their own profile',
            'Deny delete for user_profiles',
            'Users can view their own transactions',
            'Admins can view all transactions',
            'Deny delete for transactions'
        )
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON %I.%I', 
                      rec.policyname, rec.schemaname, rec.tablename);
    END LOOP;

    -- Drop triggers safely
    FOR rec IN 
        SELECT t.tgname, c.relname 
        FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = 'public'
        AND t.tgname IN ('on_auth_user_created', 'update_rag_pipeline_state_updated_at')
        AND NOT t.tgisinternal
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', rec.tgname, rec.relname);
    END LOOP;

    -- Drop triggers from auth schema if they exist
    BEGIN
        DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
    EXCEPTION 
        WHEN undefined_table THEN 
            NULL; -- auth.users table doesn't exist, ignore
        WHEN undefined_object THEN 
            NULL; -- trigger doesn't exist, ignore
    END;

END $$;

-- Drop functions
DROP FUNCTION IF EXISTS public.handle_new_user();
DROP FUNCTION IF EXISTS public.is_admin();
DROP FUNCTION IF EXISTS match_documents(vector, int, jsonb);
DROP FUNCTION IF EXISTS execute_custom_sql(text);
DROP FUNCTION IF EXISTS update_rag_pipeline_state_updated_at();
DROP FUNCTION IF EXISTS deduct_token_if_sufficient(UUID);
DROP FUNCTION IF EXISTS grant_tokens_for_purchase(UUID, INTEGER, TEXT, TEXT);

-- Drop tables (in reverse dependency order) - CASCADE will handle dependencies
DROP TABLE IF EXISTS document_rows CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS document_metadata CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS requests CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS rag_pipeline_state CASCADE;

-- ==============================================================================
-- CREATE TABLES
-- ==============================================================================

-- 1. User Profiles Table
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    tokens INTEGER DEFAULT 0 NOT NULL CHECK (tokens >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 2. Requests Table
CREATE TABLE requests (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_query TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(id) ON DELETE CASCADE
);

-- 3. Conversations Table
CREATE TABLE conversations (
    session_id VARCHAR PRIMARY KEY NOT NULL,
    user_id UUID NOT NULL,
    title VARCHAR,  -- Auto-generated from first message
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ DEFAULT NOW(),
    is_archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
   
    UNIQUE(session_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles(id) ON DELETE CASCADE
);

-- 4. Messages Table
CREATE TABLE messages (
    id INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    computed_session_user_id UUID GENERATED ALWAYS AS (
        CAST(SPLIT_PART(session_id, '~', 1) AS UUID)
    ) STORED,
    session_id VARCHAR NOT NULL,
    message JSONB NOT NULL,
    message_data TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
   
    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
);

-- 5. Document Metadata Table
CREATE TABLE document_metadata (
    id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    schema TEXT
);

-- 6. Document Rows Table
CREATE TABLE document_rows (
    id SERIAL PRIMARY KEY,
    dataset_id TEXT REFERENCES document_metadata(id),
    row_data JSONB  -- Store the actual row data
);

-- 7. Documents Table (with vector embeddings)
CREATE TABLE documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed like 768 for nomic-embed-text (Ollama)
);

-- 8. RAG Pipeline State Table
CREATE TABLE rag_pipeline_state (
    pipeline_id TEXT PRIMARY KEY,     -- User-defined pipeline ID (from RAG_PIPELINE_ID env var)
    pipeline_type TEXT NOT NULL,      -- 'google_drive' or 'local_files'
    last_check_time TIMESTAMP,        -- Last successful check for changes
    known_files JSONB,                -- File metadata for change detection (file_id -> timestamp mapping)
    last_run TIMESTAMP,               -- Last successful run timestamp
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 9. Transactions Table (Token Purchase Audit Trail)
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('purchase', 'consumption')),
    amount INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    stripe_payment_intent_id TEXT,
    stripe_event_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE UNIQUE INDEX idx_transactions_stripe_event_id
ON transactions(stripe_event_id)
WHERE stripe_event_id IS NOT NULL;

-- ==============================================================================
-- CREATE INDEXES
-- ==============================================================================

-- Conversation and message indexes
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_computed_session ON messages(computed_session_user_id);

-- RAG pipeline state indexes
CREATE INDEX idx_rag_pipeline_state_pipeline_type ON rag_pipeline_state(pipeline_type);
CREATE INDEX idx_rag_pipeline_state_last_run ON rag_pipeline_state(last_run);

-- Transaction indexes
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);

-- ==============================================================================
-- CREATE FUNCTIONS
-- ==============================================================================

-- 1. Handle New User Function
CREATE OR REPLACE FUNCTION public.handle_new_user() 
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email)
    VALUES (new.id, new.email);
    RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 2. Admin Check Function
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN AS $$
DECLARE
  is_admin_user BOOLEAN;
BEGIN
  SELECT COALESCE(up.is_admin, FALSE) INTO is_admin_user
  FROM user_profiles up
  WHERE up.id = auth.uid();
  
  RETURN is_admin_user;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 3. Document Search Function
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536), -- 1536 works for OpenAI embeddings, change if needed like 768 for nomic-embed-text (Ollama)
  match_count int default null,
  filter jsonb DEFAULT '{}'
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 4. Execute Custom SQL Function
CREATE OR REPLACE FUNCTION execute_custom_sql(sql_query text)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER -- This makes the function run with the privileges of the creator
AS $$
DECLARE
  result JSONB;
BEGIN
  -- Execute the SQL and capture the result
  EXECUTE 'SELECT jsonb_agg(t) FROM (' || sql_query || ') t' INTO result;
  RETURN COALESCE(result, '[]'::jsonb);
EXCEPTION
  WHEN OTHERS THEN
    RETURN jsonb_build_object(
      'error', SQLERRM,
      'detail', SQLSTATE
    );
END;
$$;

-- 5. RAG Pipeline State Update Function
CREATE OR REPLACE FUNCTION update_rag_pipeline_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 6. Atomic Token Deduction Function
CREATE OR REPLACE FUNCTION deduct_token_if_sufficient(p_user_id UUID)
RETURNS TABLE(success BOOLEAN, remaining_tokens INTEGER, error_message TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_tokens INTEGER;
BEGIN
    SELECT tokens INTO current_tokens
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    IF current_tokens IS NULL THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found'::TEXT;
        RETURN;
    END IF;

    IF current_tokens < 1 THEN
        RETURN QUERY SELECT FALSE, current_tokens, 'Insufficient tokens'::TEXT;
        RETURN;
    END IF;

    UPDATE user_profiles
    SET tokens = tokens - 1,
        updated_at = NOW()
    WHERE id = p_user_id;

    INSERT INTO transactions (
        user_id,
        transaction_type,
        amount,
        tokens
    ) VALUES (
        p_user_id,
        'consumption',
        -1,
        1
    );

    RETURN QUERY SELECT TRUE, current_tokens - 1, NULL::TEXT;
END;
$$;

-- 7. Atomic Token Grant Function
CREATE OR REPLACE FUNCTION grant_tokens_for_purchase(
    p_user_id UUID,
    p_tokens INTEGER,
    p_event_id TEXT,
    p_payment_intent_id TEXT
)
RETURNS TABLE(success BOOLEAN, new_balance INTEGER, error_message TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_balance INTEGER;
    event_exists BOOLEAN;
BEGIN
    -- Check if event has already been processed (idempotency)
    SELECT EXISTS(
        SELECT 1 FROM transactions
        WHERE stripe_event_id = p_event_id
    ) INTO event_exists;

    IF event_exists THEN
        RETURN QUERY SELECT FALSE, 0, 'Event already processed';
        RETURN;
    END IF;

    -- Lock user row to prevent race conditions
    SELECT tokens INTO current_balance
    FROM user_profiles
    WHERE id = p_user_id
    FOR UPDATE;

    -- Check if user exists
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'User not found';
        RETURN;
    END IF;

    -- Atomically grant tokens
    UPDATE user_profiles
    SET tokens = tokens + p_tokens,
        updated_at = NOW()
    WHERE id = p_user_id;

    -- Record purchase in transactions table
    INSERT INTO transactions (
        user_id,
        transaction_type,
        amount,
        tokens,
        stripe_payment_intent_id,
        stripe_event_id
    ) VALUES (
        p_user_id,
        'purchase',
        p_tokens,
        p_tokens,
        p_payment_intent_id,
        p_event_id
    );

    -- Return success with new balance
    RETURN QUERY SELECT TRUE, current_balance + p_tokens, NULL::TEXT;
END;
$$;

-- Grant execute permissions to service role
GRANT EXECUTE ON FUNCTION deduct_token_if_sufficient(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION grant_tokens_for_purchase(UUID, INTEGER, TEXT, TEXT) TO service_role;

-- ==============================================================================
-- CREATE TRIGGERS
-- ==============================================================================

-- 1. Auto-create user profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- 2. Auto-update RAG pipeline state timestamp
CREATE TRIGGER update_rag_pipeline_state_updated_at
    BEFORE UPDATE ON rag_pipeline_state
    FOR EACH ROW
    EXECUTE FUNCTION update_rag_pipeline_state_updated_at();

-- ==============================================================================
-- ENABLE ROW LEVEL SECURITY
-- ==============================================================================

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_pipeline_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

-- ==============================================================================
-- CREATE ROW LEVEL SECURITY POLICIES
-- ==============================================================================

-- User Profiles Policies
CREATE POLICY "Users can view their own profile"
ON user_profiles 
FOR SELECT
USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
ON user_profiles 
FOR UPDATE
USING (auth.uid() = id)
WITH CHECK (auth.uid() = id AND is_admin IS NOT DISTINCT FROM FALSE);

CREATE POLICY "Only admins can change admin status"
ON user_profiles 
FOR UPDATE 
TO authenticated
USING (is_admin())
WITH CHECK (is_admin());

CREATE POLICY "Admins can view all profiles"
ON user_profiles 
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can update all profiles"
ON user_profiles 
FOR UPDATE
USING (is_admin());

CREATE POLICY "Deny delete for user_profiles" ON user_profiles FOR DELETE USING (false);

-- Requests Policies
CREATE POLICY "Users can view their own requests"
ON requests
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all requests"
ON requests
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can insert requests"
ON requests
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for requests" ON requests FOR DELETE USING (false);

-- Conversations Policies
CREATE POLICY "Users can view their own conversations"
ON conversations
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own conversations"
ON conversations
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own conversations"
ON conversations
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all conversations"
ON conversations
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can update all conversations"
ON conversations
FOR UPDATE
USING (is_admin());

CREATE POLICY "Admins can insert conversations"
ON conversations
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for conversations" ON conversations FOR DELETE USING (false);

-- Messages Policies
CREATE POLICY "Users can view their own messages"
ON messages
FOR SELECT
USING (
  auth.uid() = computed_session_user_id
);

CREATE POLICY "Users can insert messages in their conversations"
ON messages
FOR INSERT
WITH CHECK (
  auth.uid() = computed_session_user_id
);

CREATE POLICY "Admins can view all messages"
ON messages
FOR SELECT
USING (is_admin());

CREATE POLICY "Admins can insert messages"
ON messages
FOR INSERT
WITH CHECK (is_admin());

CREATE POLICY "Deny delete for messages" ON messages FOR DELETE USING (false);

ALTER TABLE document_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_rows ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Document tables are locked down - just backend can access
CREATE POLICY "Deny all access to document_metadata" ON document_metadata FOR ALL USING (false);
CREATE POLICY "Deny all access to document_rows" ON document_rows FOR ALL USING (false);
CREATE POLICY "Deny all access to documents" ON documents FOR ALL USING (false);

-- Transaction Policies
CREATE POLICY "Users can view their own transactions"
ON transactions
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all transactions"
ON transactions
FOR SELECT
USING (is_admin());

CREATE POLICY "Deny delete for transactions" ON transactions FOR DELETE USING (false);

-- ==============================================================================
-- ENABLE REALTIME
-- ==============================================================================

-- Enable realtime replication for user_profiles table (for token balance updates)
ALTER PUBLICATION supabase_realtime ADD TABLE user_profiles;

-- ==============================================================================
-- REVOKE PERMISSIONS
-- ==============================================================================

-- By default, revoke execute permission from public and authenticated users for security-sensitive functions
REVOKE EXECUTE ON FUNCTION execute_custom_sql(text) FROM PUBLIC, authenticated;

-- ==============================================================================
-- SETUP COMPLETE
-- ==============================================================================

-- The database schema is now fully configured with:
-- ✅ All tables created with proper relationships
-- ✅ Indexes for performance optimization
-- ✅ Functions for user management, document search, and SQL execution
-- ✅ Triggers for automated user profile creation and timestamp updates
-- ✅ Row Level Security enabled with comprehensive policies
-- ✅ Proper security permissions configured
-- ✅ Safe cleanup that handles non-existent tables/policies/triggers

-- Next steps:
-- 1. Configure your application environment variables
-- 2. Run your AI agent and RAG pipeline
-- 3. Test the frontend application

-- Note: For Ollama with nomic-embed-text, change vector dimensions from 1536 to 768 in:
-- - documents table definition
-- - match_documents function definition