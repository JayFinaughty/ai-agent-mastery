"""
Agent Evaluation Module

This module provides evaluation capabilities for the AI Agent.

Modules:
    - evaluators: Custom evaluators (NoPII, NoForbiddenWords, ContainsAny)
    - run_evals: Golden dataset evaluation runner (local development)
    - prod_rules: Production rule-based evaluation with Langfuse sync (Video 6)
    - annotation_helpers: Langfuse annotation utilities (Video 5)

Local Evaluations (Phase 1):
    cd backend_agent_api
    python evals/run_evals.py

Production Evaluations (Phase 2):
    Automatically run on each API request when Langfuse is configured.
    See prod_rules.py for details.
"""
