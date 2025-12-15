#!/usr/bin/env python3
"""
Seed NeuroVerse RAG Mock Data for Agent Evaluations

This script populates the RAG database with the NeuroVerse Studios mock documents
so that evaluation cases can properly test document retrieval and web search functionality.

Usage:
    cd backend_agent_api
    python evals/seed_rag_mock_data.py

What it creates:
    - 9 NeuroVerse documents in document_metadata
    - Multiple chunks per document in documents table (with embeddings)
"""

import asyncio
import os
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from clients import get_agent_clients
from tools import get_embedding

# Path to mock data
MOCK_DATA_DIR = Path(__file__).parent.parent.parent / "mock_data"

# Document configurations with IDs and titles
DOCUMENT_CONFIGS = [
    {
        "id": "neuroverse-company-overview",
        "filename": "NeuroVerse Studios_ Company Overview.md",
        "title": "NeuroVerse Studios: Company Overview",
        "url": "https://neuroverse.internal/docs/company-overview",
    },
    {
        "id": "neuroverse-q1-2024-report",
        "filename": "NeuroVerse Studios - Q1 2024 Quarterly Report.md",
        "title": "NeuroVerse Studios Q1 2024 Quarterly Report",
        "url": "https://neuroverse.internal/reports/q1-2024",
    },
    {
        "id": "neuroverse-strategic-plan",
        "filename": "NeuroVerse Studios_ 2024-2026 Strategic Plan.md",
        "title": "NeuroVerse Studios 2024-2026 Strategic Plan",
        "url": "https://neuroverse.internal/docs/strategic-plan-2024-2026",
    },
    {
        "id": "nae-technical-spec",
        "filename": "Neural Adaptation Engine (NAE).md",
        "title": "Neural Adaptation Engine (NAE) Technical Specification",
        "url": "https://neuroverse.internal/docs/nae-tech-spec",
    },
    {
        "id": "nae-technical-spec-full",
        "filename": "Neural Adaptation Engine (NAE) - Technical Specification.md",
        "title": "Neural Adaptation Engine (NAE) Technical Specification v2.3",
        "url": "https://neuroverse.internal/docs/nae-tech-spec-v2.3",
    },
    {
        "id": "neural-privacy-framework",
        "filename": "Neural Data Privacy Framework.md",
        "title": "Neural Data Privacy Framework",
        "url": "https://neuroverse.internal/legal/privacy-framework",
    },
    {
        "id": "marketing-strategy-meeting",
        "filename": "Marketing Strategy Meeting_ Mindweaver Launch.md",
        "title": "Marketing Strategy Meeting: Mindweaver Launch",
        "url": "https://neuroverse.internal/meetings/marketing-strategy-2024-03",
    },
    {
        "id": "product-dev-meeting-nae",
        "filename": "Product Development Meeting_ Neural Adaptation Engine (NAE).md",
        "title": "Product Development Meeting: NAE",
        "url": "https://neuroverse.internal/meetings/product-dev-nae-2024-02",
    },
    {
        "id": "neural-sync-research-brief",
        "filename": "Research Brief_ Neural Synchronization in Multiplayer Gaming.md",
        "title": "Research Brief: Neural Synchronization in Multiplayer Gaming",
        "url": "https://neuroverse.internal/research/neural-sync-multiplayer",
    },
]


def chunk_markdown(content: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split markdown content into chunks, trying to respect section boundaries.

    Args:
        content: The markdown content to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of content chunks
    """
    # Split by markdown headers (## or ###)
    sections = re.split(r'(?=^##+ )', content, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If adding this section would exceed chunk_size, save current and start new
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + section
            else:
                current_chunk = section
        else:
            if current_chunk:
                current_chunk += "\n\n" + section
            else:
                current_chunk = section

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If we ended up with just one huge chunk, split it more aggressively
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            # Split by paragraphs
            paragraphs = chunk.split('\n\n')
            sub_chunk = ""
            for para in paragraphs:
                if len(sub_chunk) + len(para) > chunk_size and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = para
                else:
                    sub_chunk += "\n\n" + para if sub_chunk else para
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks


async def seed_documents(embedding_client, supabase) -> None:
    """Seed NeuroVerse mock documents into the database."""
    print("\n📚 Seeding NeuroVerse mock documents...")

    for doc_config in DOCUMENT_CONFIGS:
        file_path = MOCK_DATA_DIR / doc_config["filename"]

        if not file_path.exists():
            print(f"\n  ⚠️  File not found: {doc_config['filename']}")
            continue

        print(f"\n  Processing: {doc_config['title']}")

        # Read file content
        content = file_path.read_text(encoding='utf-8')

        # Chunk the content
        chunks = chunk_markdown(content)
        print(f"    Split into {len(chunks)} chunks")

        # Insert document metadata
        try:
            supabase.table("document_metadata").upsert(
                {
                    "id": doc_config["id"],
                    "title": doc_config["title"],
                    "url": doc_config["url"],
                }
            ).execute()
            print(f"    ✓ Metadata inserted")
        except Exception as e:
            print(f"    ✗ Metadata error: {e}")
            continue

        # Insert document chunks with embeddings
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = await get_embedding(chunk, embedding_client)

                # Insert chunk
                supabase.table("documents").insert(
                    {
                        "content": chunk,
                        "metadata": {
                            "file_id": doc_config["id"],
                            "file_title": f"{doc_config['title']} - Section {i + 1}",
                            "file_url": doc_config["url"],
                            "mime_type": "text/markdown",
                            "chunk_index": i,
                        },
                        "embedding": embedding,
                    }
                ).execute()
                print(f"    ✓ Chunk {i + 1}/{len(chunks)} embedded and inserted")
            except Exception as e:
                print(f"    ✗ Chunk {i + 1} error: {e}")


async def clear_neuroverse_data(supabase) -> None:
    """Clear existing NeuroVerse test data before seeding."""
    print("\n🗑️  Clearing existing NeuroVerse data...")

    doc_ids = [doc["id"] for doc in DOCUMENT_CONFIGS]

    # Clear documents
    for doc_id in doc_ids:
        try:
            supabase.table("documents").delete().eq(
                "metadata->>file_id", doc_id
            ).execute()
        except Exception:
            pass

    # Clear document_metadata
    for doc_id in doc_ids:
        try:
            supabase.table("document_metadata").delete().eq("id", doc_id).execute()
        except Exception:
            pass

    print("  ✓ Cleared")


async def verify_data(supabase) -> None:
    """Verify the seeded data."""
    print("\n✅ Verifying seeded data...")

    # Check document_metadata for NeuroVerse docs
    result = supabase.table("document_metadata").select("id, title").execute()
    neuroverse_docs = [d for d in result.data if d['id'].startswith('neuroverse') or d['id'].startswith('nae') or d['id'].startswith('neural') or d['id'].startswith('marketing') or d['id'].startswith('product')]
    print(f"  NeuroVerse documents in metadata: {len(neuroverse_docs)}")
    for doc in neuroverse_docs:
        print(f"    - {doc['id']}: {doc['title'][:50]}...")

    # Check documents (chunks)
    result = supabase.table("documents").select("id, metadata").execute()
    neuroverse_chunks = [d for d in result.data if d.get('metadata', {}).get('file_id', '').startswith(('neuroverse', 'nae', 'neural', 'marketing', 'product'))]
    print(f"  NeuroVerse total chunks: {len(neuroverse_chunks)}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("SEED NEUROVERSE RAG MOCK DATA FOR EVALUATIONS")
    print("=" * 60)

    # Check mock data directory exists
    if not MOCK_DATA_DIR.exists():
        print(f"\n❌ Error: Mock data directory not found at {MOCK_DATA_DIR}")
        print("   Run: unzip Mock_Data_For_RAG.zip -d 8_Agent_Evals/mock_data/")
        sys.exit(1)

    # Get clients
    embedding_client, supabase = get_agent_clients()

    if not supabase:
        print("\n❌ Error: Could not connect to Supabase")
        print("   Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    # Clear existing NeuroVerse data
    await clear_neuroverse_data(supabase)

    # Seed documents
    await seed_documents(embedding_client, supabase)

    # Verify
    await verify_data(supabase)

    print("\n" + "=" * 60)
    print("✅ NeuroVerse mock data seeding complete!")
    print("=" * 60)
    print("\nYou can now run: python evals/run_evals.py --dataset rag")


if __name__ == "__main__":
    asyncio.run(main())
