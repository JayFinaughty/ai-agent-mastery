#!/usr/bin/env python3
"""
Golden Dataset Evaluation Runner with Rule-Based Tool Verification

Video 3: Rule-Based Evals (Local) with HasMatchingSpan

This script evaluates the agent against a curated set of test cases,
including tool call verification using OpenTelemetry spans.

Usage:
    cd backend_agent_api
    python evals/run_evals.py

Exit Codes:
    0 - Pass rate >= 80%
    1 - Pass rate < 80%
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from httpx import AsyncClient
from pydantic_evals import Dataset

# Load environment variables from parent directory (8_Agent_Evals/.env)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

# ============================================================
# LOGFIRE CONFIGURATION FOR TOOL SPAN TRACKING (Video 3)
# ============================================================
# Configure logfire BEFORE importing the agent to ensure spans are captured.
# This enables HasMatchingSpan to check tool calls via OpenTelemetry spans.
#
# Note: Tool names are stored in the gen_ai.tool.name span ATTRIBUTE,
# not in the span name itself. Use has_attributes in HasMatchingSpan queries.

import logfire

logfire.configure(
    service_name='agent_evals',
    send_to_logfire=False,  # Local only - no cloud connection needed
)

# Instrument pydantic-ai to generate spans for tool calls
logfire.instrument_pydantic_ai()

# ============================================================

from agent import agent, AgentDeps
from clients import get_agent_clients

# Import custom evaluators for YAML deserialization
from evals.evaluators import ContainsAny, NoPII, NoForbiddenWords

# Configuration
PASS_THRESHOLD = 0.8  # 80% of cases must pass


async def run_agent(inputs: dict) -> str:
    """
    Task function that runs our agent on a single input.

    This function is called once per test case in the dataset.
    The pydantic-evals framework handles timing and evaluation.

    Args:
        inputs: Dictionary containing the 'query' key from the test case

    Returns:
        The agent's response as a string
    """
    # Get existing clients (embedding_client, supabase)
    embedding_client, supabase = get_agent_clients()

    # Create HTTP client for this evaluation
    async with AsyncClient() as http_client:
        # Build agent dependencies
        deps = AgentDeps(
            supabase=supabase,
            embedding_client=embedding_client,
            http_client=http_client,
            brave_api_key=os.getenv("BRAVE_API_KEY"),
            searxng_base_url=os.getenv("SEARXNG_BASE_URL"),
            memories="",  # No memories for evaluation - isolated runs
        )

        # Run the agent
        result = await agent.run(inputs["query"], deps=deps)

        # Return the output as string
        return str(result.output)


async def main() -> int:
    """
    Load the golden dataset and run evaluation.

    Returns:
        Exit code (0 for pass, 1 for fail)
    """
    # Load dataset from YAML
    dataset_path = Path(__file__).parent / "golden_dataset.yaml"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return 1

    # Load dataset, registering our custom evaluators
    dataset = Dataset[dict, str].from_file(
        dataset_path, custom_evaluator_types=[ContainsAny, NoPII, NoForbiddenWords]
    )

    print("\n" + "=" * 60)
    print("GOLDEN DATASET EVALUATION")
    print("=" * 60)
    print(f"\nDataset: {dataset_path.name}")
    print(f"Cases: {len(dataset.cases)}")
    print(f"Pass Threshold: {PASS_THRESHOLD:.0%}")
    print("\n" + "-" * 60 + "\n")

    # Run evaluation
    report = await dataset.evaluate(run_agent)

    # Print detailed report
    report.print(include_input=True, include_output=True)

    # Calculate results
    # A case passes if all its assertions pass (no False values)
    def case_passed(case) -> bool:
        if case.evaluator_failures:
            return False
        return all(result.value for result in case.assertions.values())

    passed = sum(1 for case in report.cases if case_passed(case))
    total = len(report.cases)
    pass_rate = passed / total if total > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Cases:   {total}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {total - passed}")
    print(f"Pass Rate:     {pass_rate:.1%}")
    print("=" * 60)

    # Determine exit code
    if pass_rate >= PASS_THRESHOLD:
        print(f"\nPASSED - Above {PASS_THRESHOLD:.0%} threshold")
        return 0
    else:
        print(f"\nFAILED - Below {PASS_THRESHOLD:.0%} threshold")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
