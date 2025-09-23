"""
API module for the eventlog package.
Provides unified external interfaces for event log candidate suggestion, validation, statistics, and conversion.
"""

from .core import (
    convert_to_event_log
)

from .scoring import (
    suggest_mandatory_column_candidates,
)
from .statistics import (
    get_event_log_statistic,
    get_event_log_quality,
    validate_event_log,
)

__all__ = [
    "suggest_mandatory_column_candidates",
    "validate_event_log",
    "get_event_log_statistic",
    "get_event_log_quality",
    "convert_to_event_log"
]