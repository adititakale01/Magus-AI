"""
Internal models for rate lookup.

NormalizedRates is the unified format that all rate sheets are parsed into.
This allows the lookup logic to be simple regardless of source format.
"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class NormalizedRates:
    """
    Unified internal format for rate data.

    All rate sheets (easy, medium, hard) are normalized to this format.
    This decouples parsing complexity from lookup logic.

    Attributes:
        sea_rates: DataFrame with columns:
            - origin (str, lowercase)
            - destination (str, lowercase)
            - rate_20ft (float)
            - rate_40ft (float)
            - transit_days (int or None)

        air_rates: DataFrame with columns:
            - origin (str, lowercase)
            - destination (str, lowercase)
            - rate_per_kg (float)
            - min_charge (float)
            - transit_days (int or None)

        aliases: Dict mapping canonical names to list of aliases.
            Example: {"ho chi minh city": ["hcmc", "saigon", "sgn"]}

        source_format: Which format this was parsed from ("easy", "medium", "hard")
    """
    sea_rates: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    air_rates: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    aliases: dict[str, list[str]] = field(default_factory=dict)
    source_format: str = "unknown"

    def get_all_names(self, canonical: str) -> list[str]:
        """
        Get all possible names for a location (canonical + aliases).

        Args:
            canonical: The canonical/normalized location name

        Returns:
            List of all names including the canonical name and aliases
        """
        canonical_lower = canonical.lower()
        names = [canonical_lower]

        if canonical_lower in self.aliases:
            names.extend(self.aliases[canonical_lower])

        return names

    def find_canonical(self, name: str) -> str | None:
        """
        Find the canonical name for a given alias.

        Args:
            name: A location name (could be canonical or alias)

        Returns:
            The canonical name if found, None otherwise
        """
        name_lower = name.lower()

        # Check if it's already canonical
        if name_lower in self.aliases:
            return name_lower

        # Search through aliases
        for canonical, alias_list in self.aliases.items():
            if name_lower in [a.lower() for a in alias_list]:
                return canonical

        # Not in alias map - return as-is (might be in rate table directly)
        return name_lower
