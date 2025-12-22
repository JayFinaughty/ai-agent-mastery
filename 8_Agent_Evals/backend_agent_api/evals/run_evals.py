#!/usr/bin/env python3
"""
Golden Dataset Evaluation Runner with LLM Judge

Video 4: LLM Judge (Local) with LLMJudge evaluators

This script evaluates the agent against a curated set of test cases,
including tool call verification and AI-powered quality assessment.

Usage:
    cd backend_agent_api
    python evals/run_evals.py                  # Run general dataset
    python evals/run_evals.py --dataset rag    # Run RAG/web search dataset
    python evals/run_evals.py --dataset all    # Run both datasets

Datasets:
    general - Original golden dataset (general agent behavior)
    rag     - RAG and web search focused tests (NeuroVerse documents)
    all     - Run both datasets

Exit Codes:
    0 - Pass rate >= 80%
    1 - Pass rate < 80%
"""

import argparse
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
# VIDEO 4: Configure LLMJudge to use the same model as the agent
# ============================================================
# LLMJudge needs to use the configured LLM_API_KEY, not OPENAI_API_KEY.
# We configure the default judge model to use the same provider as our agent.

from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Configure the judge model to use the same API key as the agent
judge_model = OpenAIChatModel(
    os.getenv('LLM_CHOICE', 'gpt-4o-mini'),
    provider=OpenAIProvider(
        base_url=os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'),
        api_key=os.getenv('LLM_API_KEY'),
    )
)
set_default_judge_model(judge_model)

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


async def run_dataset(dataset_path: Path, dataset_name: str) -> tuple[int, int, float]:
    """
    Run evaluation on a single dataset.

    Args:
        dataset_path: Path to the dataset YAML file
        dataset_name: Human-readable name for the dataset

    Returns:
        Tuple of (passed, total, pass_rate)
    """
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return 0, 0, 0.0

    # Load dataset, registering our custom evaluators
    dataset = Dataset[dict, str].from_file(
        dataset_path, custom_evaluator_types=[ContainsAny, NoPII, NoForbiddenWords]
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION: {dataset_name}")
    print("=" * 60)
    print(f"\nDataset: {dataset_path.name}")
    print(f"Cases: {len(dataset.cases)}")
    print(f"Pass Threshold: {PASS_THRESHOLD:.0%}")
    print("\n" + "-" * 60 + "\n")

    # Run evaluation
    report = await dataset.evaluate(run_agent)

    # Print detailed report
    report.print(include_input=True, include_output=True, include_reasons=True)

    # Calculate results
    def case_passed(case) -> bool:
        if case.evaluator_failures:
            return False
        return all(result.value for result in case.assertions.values())

    passed = sum(1 for case in report.cases if case_passed(case))
    total = len(report.cases)
    pass_rate = passed / total if total > 0 else 0

    return passed, total, pass_rate


async def main() -> int:
    """
    Load the golden dataset(s) and run evaluation.

    Returns:
        Exit code (0 for pass, 1 for fail)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run agent evaluations")
    parser.add_argument(
        "--dataset",
        choices=["general", "rag", "all"],
        default="general",
        help="Which dataset to run (default: general)"
    )
    args = parser.parse_args()

    # Define dataset paths
    evals_dir = Path(__file__).parent
    datasets = {
        "general": (evals_dir / "golden_dataset.yaml", "General Golden Dataset"),
        "rag": (evals_dir / "golden_dataset_rag.yaml", "RAG & Web Search Dataset"),
    }

    # Determine which datasets to run
    if args.dataset == "all":
        datasets_to_run = list(datasets.items())
    else:
        datasets_to_run = [(args.dataset, datasets[args.dataset])]

    # Run evaluations
    total_passed = 0
    total_cases = 0

    for dataset_key, (dataset_path, dataset_name) in datasets_to_run:
        passed, total, pass_rate = await run_dataset(dataset_path, dataset_name)
        total_passed += passed
        total_cases += total

        # Print individual dataset summary
        print("\n" + "-" * 60)
        print(f"{dataset_name} Results:")
        print(f"  Passed: {passed}/{total} ({pass_rate:.1%})")
        print("-" * 60)

    # Overall summary
    overall_pass_rate = total_passed / total_cases if total_cases > 0 else 0

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total Cases:   {total_cases}")
    print(f"Passed:        {total_passed}")
    print(f"Failed:        {total_cases - total_passed}")
    print(f"Pass Rate:     {overall_pass_rate:.1%}")
    print("=" * 60)

    # Determine exit code
    if overall_pass_rate >= PASS_THRESHOLD:
        print(f"\nPASSED - Above {PASS_THRESHOLD:.0%} threshold")
        return 0
    else:
        print(f"\nFAILED - Below {PASS_THRESHOLD:.0%} threshold")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
