#!/usr/bin/env python3
"""
Seed Test Data for Agent Evaluations

This script populates the RAG database with test documents so that
evaluation cases can properly test document retrieval functionality.

Usage:
    cd backend_agent_api
    python evals/seed_test_data.py

What it creates:
    - 3 test documents in document_metadata
    - Multiple chunks per document in documents table (with embeddings)
    - Sample structured data in document_rows (for SQL query testing)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from clients import get_agent_clients
from tools import get_embedding

# Test documents to seed
TEST_DOCUMENTS = [
    {
        "id": "sales-report-2024",
        "title": "Q4 2024 Sales Report",
        "url": "https://example.com/reports/sales-2024-q4",
        "mime_type": "text/plain",
        "chunks": [
            "Q4 2024 Sales Report Executive Summary. This document contains the quarterly sales analysis for Q4 2024. Total revenue reached $2.5 million, representing a 15% increase over Q3.",
            "Regional Sales Breakdown. North America led with $1.2 million in sales, followed by Europe at $800,000 and Asia Pacific at $500,000. The sales team exceeded targets in all regions.",
            "Product Performance Analysis. Our flagship product line accounted for 60% of total sales. New product launches in October contributed an additional $400,000 in revenue.",
            "Sales Forecast for 2025. Based on current trends, we project Q1 2025 sales to reach $2.8 million. Key growth drivers include expansion into new markets and product line extensions.",
        ],
    },
    {
        "id": "company-policies",
        "title": "Employee Handbook - Company Policies",
        "url": "https://example.com/docs/employee-handbook",
        "mime_type": "text/plain",
        "chunks": [
            "Welcome to the Company. This employee handbook contains important policies and procedures. All employees are expected to familiarize themselves with these guidelines.",
            "Remote Work Policy. Employees may work remotely up to 3 days per week with manager approval. Core hours are 10am-3pm for team collaboration regardless of location.",
            "Time Off and Leave Policy. Full-time employees receive 20 days of paid time off annually. Sick leave is provided separately with unlimited days for genuine illness.",
            "Professional Development. The company provides $2,000 annually for professional development including courses, conferences, and certifications relevant to your role.",
        ],
    },
    {
        "id": "technical-documentation",
        "title": "API Integration Guide",
        "url": "https://example.com/docs/api-guide",
        "mime_type": "text/plain",
        "chunks": [
            "API Integration Overview. This document describes how to integrate with our REST API. Authentication uses OAuth 2.0 with JWT tokens for secure access.",
            "Authentication Endpoints. POST /auth/token to obtain access tokens. Include client_id and client_secret in the request body. Tokens expire after 1 hour.",
            "Data Endpoints. GET /api/v1/users returns user list. GET /api/v1/documents returns available documents. All responses are JSON formatted with pagination support.",
            "Error Handling. The API returns standard HTTP status codes. 400 for bad requests, 401 for unauthorized, 404 for not found, 500 for server errors. Error responses include detailed messages.",
        ],
    },
]

# Sample structured data for SQL query testing
SAMPLE_SALES_DATA = [
    {"category": "Electronics", "product": "Laptop Pro", "sales": 150000, "units": 500, "region": "North America"},
    {"category": "Electronics", "product": "Tablet X", "sales": 80000, "units": 800, "region": "Europe"},
    {"category": "Software", "product": "CloudSuite", "sales": 200000, "units": 1000, "region": "North America"},
    {"category": "Software", "product": "DataAnalyzer", "sales": 120000, "units": 600, "region": "Asia Pacific"},
    {"category": "Services", "product": "Consulting", "sales": 300000, "units": 150, "region": "Europe"},
    {"category": "Services", "product": "Training", "sales": 50000, "units": 200, "region": "North America"},
]


async def seed_documents(embedding_client, supabase) -> None:
    """Seed test documents into the database."""
    print("\n📚 Seeding test documents...")

    for doc in TEST_DOCUMENTS:
        print(f"\n  Processing: {doc['title']}")

        # Insert document metadata
        try:
            supabase.table("document_metadata").upsert(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "url": doc["url"],
                }
            ).execute()
            print(f"    ✓ Metadata inserted")
        except Exception as e:
            print(f"    ✗ Metadata error: {e}")
            continue

        # Insert document chunks with embeddings
        for i, chunk in enumerate(doc["chunks"]):
            try:
                # Generate embedding
                embedding = await get_embedding(chunk, embedding_client)

                # Insert chunk
                supabase.table("documents").insert(
                    {
                        "content": chunk,
                        "metadata": {
                            "file_id": doc["id"],
                            "file_title": f"{doc['title']} - Section {i + 1}",
                            "file_url": doc["url"],
                            "mime_type": doc["mime_type"],
                            "chunk_index": i,
                        },
                        "embedding": embedding,
                    }
                ).execute()
                print(f"    ✓ Chunk {i + 1}/{len(doc['chunks'])} embedded and inserted")
            except Exception as e:
                print(f"    ✗ Chunk {i + 1} error: {e}")


async def seed_structured_data(supabase) -> None:
    """Seed structured sales data for SQL query testing."""
    print("\n📊 Seeding structured sales data...")

    dataset_id = "sales-data-2024"

    # Insert dataset metadata with schema
    try:
        schema = {
            "category": "string",
            "product": "string",
            "sales": "number",
            "units": "number",
            "region": "string",
        }
        supabase.table("document_metadata").upsert(
            {
                "id": dataset_id,
                "title": "2024 Sales Data",
                "url": "internal://sales-database",
                "schema": str(schema),
            }
        ).execute()
        print("  ✓ Dataset metadata inserted")
    except Exception as e:
        print(f"  ✗ Dataset metadata error: {e}")
        return

    # Insert rows
    for i, row in enumerate(SAMPLE_SALES_DATA):
        try:
            supabase.table("document_rows").insert(
                {
                    "dataset_id": dataset_id,
                    "row_data": row,
                }
            ).execute()
            print(f"  ✓ Row {i + 1}/{len(SAMPLE_SALES_DATA)} inserted")
        except Exception as e:
            print(f"  ✗ Row {i + 1} error: {e}")


async def clear_test_data(supabase) -> None:
    """Clear existing test data before seeding."""
    print("\n🗑️  Clearing existing test data...")

    test_doc_ids = [doc["id"] for doc in TEST_DOCUMENTS] + ["sales-data-2024"]

    # Clear documents
    for doc_id in test_doc_ids:
        try:
            supabase.table("documents").delete().eq(
                "metadata->>file_id", doc_id
            ).execute()
        except Exception:
            pass

    # Clear document_rows
    try:
        supabase.table("document_rows").delete().eq(
            "dataset_id", "sales-data-2024"
        ).execute()
    except Exception:
        pass

    # Clear document_metadata
    for doc_id in test_doc_ids:
        try:
            supabase.table("document_metadata").delete().eq("id", doc_id).execute()
        except Exception:
            pass

    print("  ✓ Cleared")


async def verify_data(supabase) -> None:
    """Verify the seeded data."""
    print("\n✅ Verifying seeded data...")

    # Check document_metadata
    result = supabase.table("document_metadata").select("id, title").execute()
    print(f"  Documents in metadata: {len(result.data)}")
    for doc in result.data:
        print(f"    - {doc['id']}: {doc['title']}")

    # Check documents (chunks)
    result = supabase.table("documents").select("id, metadata").execute()
    print(f"  Total chunks: {len(result.data)}")

    # Check document_rows
    result = supabase.table("document_rows").select("id, dataset_id").execute()
    print(f"  Structured data rows: {len(result.data)}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("SEED TEST DATA FOR AGENT EVALUATIONS")
    print("=" * 60)

    # Get clients
    embedding_client, supabase = get_agent_clients()

    if not supabase:
        print("\n❌ Error: Could not connect to Supabase")
        print("   Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    # Clear existing test data
    await clear_test_data(supabase)

    # Seed documents
    await seed_documents(embedding_client, supabase)

    # Seed structured data
    await seed_structured_data(supabase)

    # Verify
    await verify_data(supabase)

    print("\n" + "=" * 60)
    print("✅ Test data seeding complete!")
    print("=" * 60)
    print("\nYou can now run: python evals/run_evals.py")


if __name__ == "__main__":
    asyncio.run(main())
