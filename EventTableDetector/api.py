"""
API module for the eventlog package.
Provides unified external interfaces for event log candidate suggestion, validation, statistics, and conversion.
"""

from .core import (
    convert_to_event_log,
    search_data_platform
)

from .scoring import (
    suggest_mandatory_column_candidates,
)
from .statistics import (
    get_event_log_statistic,
    validate_event_log,
)

__all__ = [
    "search_data_platform"
    "suggest_mandatory_column_candidates",
    "validate_event_log",
    "get_event_log_statistic",
    "convert_to_event_log"
]

