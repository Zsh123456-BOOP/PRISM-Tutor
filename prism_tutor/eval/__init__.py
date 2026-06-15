"""Evaluation utilities for PRISM-Tutor.

The package is intentionally dependency-light so that metrics, judge dry-runs,
statistics, and audit sampling can run before the full experiment runner exists.
"""

from .aggregate import compute_auto_metrics
from .judge_client import JudgeClientConfig, MockJudgeClient, make_judge_client

__all__ = [
    "JudgeClientConfig",
    "MockJudgeClient",
    "compute_auto_metrics",
    "make_judge_client",
]
