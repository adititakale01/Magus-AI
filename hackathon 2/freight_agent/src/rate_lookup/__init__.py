"""
Rate Lookup Module

Handles loading and querying rate sheets of varying complexity:
- Easy: Clean flat tables
- Medium: Multi-sheet with port codes and aliases
- Hard: Messy real-world data with ditto marks, merged cells, etc.

Usage:
    from rate_lookup import RateLookupService

    service = RateLookupService(Path("rate_sheets/01_rates_easy.xlsx"))
    match = service.lookup(origin="Shanghai", destination="Rotterdam", mode="sea", container_size_ft=40)
"""

from .service import RateLookupService
from .detector import detect_format
from .models import NormalizedRates

__all__ = ["RateLookupService", "detect_format", "NormalizedRates"]
