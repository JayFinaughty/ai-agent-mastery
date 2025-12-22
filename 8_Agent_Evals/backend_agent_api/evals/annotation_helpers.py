"""
Annotation Helpers for Langfuse

Video 5: Manual Annotation

Helper functions for working with Langfuse annotations.
Used for exporting annotated data and calibrating LLM judges.

Usage:
    from evals.annotation_helpers import (
        export_annotations_to_golden_dataset,
        compare_judge_to_expert
    )

    # Export high-quality annotations as golden dataset
    cases = export_annotations_to_golden_dataset(min_accuracy=4)

    # Compare LLM judge to expert annotations
    metrics = compare_judge_to_expert()
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)


# In-memory cache for score data to avoid duplicate API calls
# Maps score_id -> {"name": str, "value": any, "id": str}
_score_cache: dict[str, dict] = {}


def get_langfuse_client():
    """
    Get configured Langfuse client.

    Requires environment variables:
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_HOST (optional, defaults to cloud.langfuse.com)
    """
    try:
        from langfuse import get_client
        return get_client()
    except ImportError:
        raise ImportError(
            "langfuse package not installed. Install with: pip install langfuse"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Langfuse client. Check your credentials: {e}"
        )


def _fetch_scores_for_trace(langfuse, score_ids: list[str]) -> list[dict]:
    """
    Fetch score details for a list of score IDs.

    In Langfuse SDK v3, trace.scores is a list of score IDs (strings),
    not score objects. This helper fetches the actual score details.

    Uses an in-memory cache to avoid duplicate API calls for the same score ID.

    Args:
        langfuse: Langfuse client
        score_ids: List of score ID strings

    Returns:
        List of score dictionaries with 'name' and 'value' keys
    """
    scores = []
    for score_id in score_ids:
        # Check cache first
        if score_id in _score_cache:
            scores.append(_score_cache[score_id])
            continue

        try:
            # Use score_v_2.get to fetch individual score by ID
            score = langfuse.api.score_v_2.get(score_ids=score_id).data[0]

            # Sleep to avoid rate limits
            time.sleep(5)

            score_data = {
                "name": score.name,
                "value": score.value,
                "id": score.id
            }

            # Store in cache for future lookups
            _score_cache[score_id] = score_data
            scores.append(score_data)
        except Exception as e:
            print(e)
            # Skip scores that can't be fetched
            continue
    return scores


def export_annotations_to_golden_dataset(
    min_accuracy: int = 4,
    output_path: Optional[Path] = None,
    limit: int = 100
) -> list[dict]:
    """
    Export high-quality annotated traces as golden dataset cases.

    Fetches traces from Langfuse that have expert accuracy annotations
    meeting the minimum threshold, and exports them as test cases
    compatible with pydantic-evals.

    Args:
        min_accuracy: Minimum accuracy score (1-5) to include.
                     Default 4 means "mostly correct with minor inaccuracies"
        output_path: Where to save YAML file. Defaults to
                    golden_dataset_from_annotations.yaml in evals directory
        limit: Maximum number of traces to fetch from Langfuse

    Returns:
        List of case dictionaries suitable for pydantic-evals Dataset

    Example:
        >>> cases = export_annotations_to_golden_dataset(min_accuracy=4)
        >>> print(f"Exported {len(cases)} high-quality cases")
        Exported 23 high-quality cases

    Output format:
        cases:
          - name: annotated_a1b2c3d4
            inputs:
              query: "What documents do you have?"
            metadata:
              source: expert_annotation
              expert_accuracy: 5
              trace_id: "a1b2c3d4-..."
    """
    langfuse = get_langfuse_client()

    # Fetch recent traces
    print(f"Fetching up to {limit} traces from Langfuse...")
    traces = langfuse.api.trace.list(limit=limit)

    golden_cases = []
    for trace in traces.data:
        # Check if trace has accuracy score meeting threshold
        # In Langfuse SDK v3, trace.scores is a list of score IDs (strings)
        accuracy_score = None
        if trace.scores:
            scores = _fetch_scores_for_trace(langfuse, trace.scores)
            for score in scores:
                if score["name"] == "accuracy":
                    accuracy_score = score["value"]
                    break

        if accuracy_score is not None and accuracy_score >= min_accuracy:
            # Extract input from trace
            case = {
                "name": f"annotated_{trace.id[:8]}",
                "inputs": {"query": trace.input or ""},
                "metadata": {
                    "source": "expert_annotation",
                    "expert_accuracy": accuracy_score,
                    "trace_id": trace.id,
                }
            }
            golden_cases.append(case)

    # Determine output path
    if output_path is None:
        output_path = Path(__file__).parent / "golden_dataset_from_annotations.yaml"

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.dump({"cases": golden_cases}, f, default_flow_style=False, sort_keys=False)

    print(f"Exported {len(golden_cases)} cases to {output_path}")
    return golden_cases


def compare_judge_to_expert(limit: int = 500) -> dict:
    """
    Compare LLM judge scores to expert annotations.

    Fetches traces that have both automated LLM judge scores and
    expert accuracy annotations, then calculates correlation metrics.
    Use this to calibrate your LLM judge rubrics.

    Args:
        limit: Maximum number of traces to fetch from Langfuse

    Returns:
        Dictionary with calibration metrics:
        - sample_size: Number of traces with both scores
        - average_difference: Mean absolute difference (0-1 scale)
        - comparisons: Sample of individual comparisons

    Example:
        >>> metrics = compare_judge_to_expert()
        >>> print(f"Average difference: {metrics['average_difference']}")
        Average difference: 0.15

    Interpretation:
        - avg_diff < 0.1: Excellent calibration
        - avg_diff 0.1-0.2: Good calibration
        - avg_diff 0.2-0.3: Needs tuning
        - avg_diff > 0.3: Significant misalignment
    """
    langfuse = get_langfuse_client()

    print(f"Fetching up to {limit} traces from Langfuse...")
    traces = langfuse.api.trace.list(limit=limit)

    comparisons = []
    for trace in traces.data:
        judge_score = None
        expert_score = None

        # In Langfuse SDK v3, trace.scores is a list of score IDs (strings)
        if trace.scores:
            scores = _fetch_scores_for_trace(langfuse, trace.scores)
            for score in scores:
                if score["name"] == "llm_judge_score":
                    judge_score = score["value"]
                elif score["name"] == "accuracy":
                    # Normalize 1-5 scale to 0-1 for comparison
                    expert_score = score["value"] / 5.0

        if judge_score is not None and expert_score is not None:
            comparisons.append({
                "trace_id": trace.id,
                "judge": round(judge_score, 3),
                "expert": round(expert_score, 3),
                "diff": round(abs(judge_score - expert_score), 3)
            })

    if not comparisons:
        return {
            "error": "No traces found with both llm_judge_score and accuracy scores",
            "sample_size": 0,
            "suggestion": "Run LLM judge on annotated traces first"
        }

    avg_diff = sum(c["diff"] for c in comparisons) / len(comparisons)

    return {
        "sample_size": len(comparisons),
        "average_difference": round(avg_diff, 3),
        "interpretation": _interpret_calibration(avg_diff),
        "comparisons": comparisons[:10]  # Sample for review
    }


def _interpret_calibration(avg_diff: float) -> str:
    """Interpret the average difference for calibration quality."""
    if avg_diff < 0.1:
        return "Excellent - Judge is well calibrated with expert annotations"
    elif avg_diff < 0.2:
        return "Good - Minor tuning may improve alignment"
    elif avg_diff < 0.3:
        return "Fair - Consider adjusting judge rubric"
    else:
        return "Poor - Significant misalignment, review rubric and examples"


def export_training_data(
    min_accuracy: int = 4,
    output_path: Optional[Path] = None,
    limit: int = 500
) -> list[dict]:
    """
    Export annotations as training data in JSONL format.

    Useful for fine-tuning models on high-quality agent interactions.

    Args:
        min_accuracy: Minimum accuracy score (1-5) to include
        output_path: Where to save JSONL file. Defaults to training_data.jsonl
        limit: Maximum number of traces to fetch

    Returns:
        List of training examples in chat format

    Output format (JSONL):
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    langfuse = get_langfuse_client()

    print(f"Fetching up to {limit} traces from Langfuse...")
    traces = langfuse.api.trace.list(limit=limit)

    training_data = []
    for trace in traces.data:
        # Check if trace has high accuracy score
        # In Langfuse SDK v3, trace.scores is a list of score IDs (strings)
        accuracy_score = None
        if trace.scores:
            scores = _fetch_scores_for_trace(langfuse, trace.scores)
            for score in scores:
                if score["name"] == "accuracy":
                    accuracy_score = score["value"]
                    break

        if accuracy_score is not None and accuracy_score >= min_accuracy:
            if trace.input and trace.output:
                training_data.append({
                    "messages": [
                        {"role": "user", "content": trace.input},
                        {"role": "assistant", "content": trace.output}
                    ]
                })

    # Determine output path
    if output_path is None:
        output_path = Path(__file__).parent / "training_data.jsonl"

    # Save as JSONL
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"Exported {len(training_data)} examples to {output_path}")
    return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Langfuse annotation helpers")
    parser.add_argument(
        "command",
        choices=["export-golden", "compare-judge", "export-training"],
        help="Command to run"
    )
    parser.add_argument(
        "--min-accuracy",
        type=int,
        default=4,
        help="Minimum accuracy score (1-5) for export commands"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum traces to fetch"
    )

    args = parser.parse_args()

    if args.command == "export-golden":
        print("\n=== Exporting Annotations to Golden Dataset ===\n")
        cases = export_annotations_to_golden_dataset(
            min_accuracy=args.min_accuracy,
            limit=args.limit
        )
        print(f"\nFound {len(cases)} high-quality annotated traces")

    elif args.command == "compare-judge":
        print("\n=== Comparing LLM Judge to Expert Annotations ===\n")
        result = compare_judge_to_expert(limit=args.limit)
        if "error" in result:
            print(f"Error: {result['error']}")
            print(f"Suggestion: {result.get('suggestion', '')}")
        else:
            print(f"Sample size: {result['sample_size']}")
            print(f"Average difference: {result['average_difference']}")
            print(f"Interpretation: {result['interpretation']}")

    elif args.command == "export-training":
        print("\n=== Exporting Training Data ===\n")
        data = export_training_data(
            min_accuracy=args.min_accuracy,
            limit=args.limit
        )
        print(f"\nExported {len(data)} training examples")
