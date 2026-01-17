"""
Rate sheet parsers for different complexity levels.

Each parser takes an Excel file and returns NormalizedRates.
"""

from .easy import parse_easy
from .medium import parse_medium
from .hard import parse_hard

__all__ = ["parse_easy", "parse_medium", "parse_hard"]
