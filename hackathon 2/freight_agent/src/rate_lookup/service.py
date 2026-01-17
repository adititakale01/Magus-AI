"""
RateLookupService - Main interface for rate lookups.

Auto-detects rate sheet format, parses into unified format,
and provides fuzzy-matching lookup capabilities.
"""

from pathlib import Path
from typing import Literal
import re
import pandas as pd

from models import RateMatch
from rate_lookup.models import NormalizedRates
from .detector import detect_format
from .parsers import parse_easy, parse_medium, parse_hard


# Built-in aliases for common port name variations
# These supplement any aliases found in the rate sheets
BUILTIN_ALIASES: dict[str, list[str]] = {
    "ho chi minh city": ["hcmc", "saigon", "sgn", "hcm", "ho chi minh"],
    "shanghai": ["sha", "pvg", "pudong", "cnsha"],
    "los angeles": ["la", "lax", "long beach"],
    "san francisco": ["sfo", "sf"],
    "rotterdam": ["rtm"],
    "hamburg": ["ham"],
    "felixstowe": ["fxt"],
    "yokohama": ["yok", "tokyo", "tokyo/yokohama"],
    "tokyo": ["nrt", "narita", "yokohama", "tokyo narita", "tokyo/yokohama"],
    "shenzhen": ["szx", "shekou"],
    "ningbo": ["ngb", "cnnbg", "ningpo"],
    "busan": ["pus", "pusan"],
    "qingdao": ["tao", "tsingtao"],
    "melbourne": ["mel"],
    "gdansk": ["gdn", "gdynia"],
    "frankfurt": ["fra"],
    "paris": ["cdg", "paris cdg"],
    "chicago": ["ord"],
    "new york": ["jfk", "nyc"],
    "mumbai": ["bom", "bombay"],
    "singapore": ["sin"],
    "amsterdam": ["ams"],
    "london": ["lhr"],
    "manzanillo": ["manzanillo mx", "mzt"],
}


def _clean_location(raw: str) -> str:
    """
    Clean a location string for better matching.

    Handles:
    - "San Francisco (SFO)" -> "san francisco"
    - "HCMC (Saigon)" -> "hcmc"
    - "Busan, South Korea" -> "busan"
    - "Tokyo/Yokohama area" -> "tokyo/yokohama"
    - "Manzanillo MX" -> "manzanillo"
    """
    if not raw:
        return ""

    name = raw.strip().lower()

    # Remove parenthetical info: "San Francisco (SFO)" -> "San Francisco"
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()

    # Remove country suffixes: "Busan, South Korea" -> "Busan"
    name = re.sub(r',\s*(south korea|china|japan|mexico|usa|uk|germany|netherlands|vietnam|poland|australia|france|india).*$', '', name, flags=re.IGNORECASE).strip()

    # Remove trailing "area": "Tokyo/Yokohama area" -> "Tokyo/Yokohama"
    name = re.sub(r'\s+area\s*$', '', name, flags=re.IGNORECASE).strip()

    # Remove country codes at end: "Manzanillo MX" -> "Manzanillo"
    name = re.sub(r'\s+(mx|cn|jp|us|uk|de|nl|vn|pl|au|fr|in|kr)\s*$', '', name, flags=re.IGNORECASE).strip()

    return name


class RateLookupService:
    """
    Service for looking up freight rates from Excel rate sheets.

    Handles:
    - Auto-detection of rate sheet format (easy/medium/hard)
    - Parsing into unified internal format
    - Fuzzy matching on port names via aliases
    - Air freight chargeable weight calculation

    Usage:
        service = RateLookupService(Path("rates.xlsx"))
        match = service.lookup(
            origin="HCMC",
            destination="Los Angeles",
            mode="sea",
            container_size_ft=40
        )
    """

    def __init__(self, rate_sheet_path: Path):
        """
        Initialize the service with a rate sheet.

        Args:
            rate_sheet_path: Path to the Excel rate sheet file
        """
        self._path = rate_sheet_path
        self._format = detect_format(rate_sheet_path)
        self._rates = self._parse_rates()

        # Merge built-in aliases with sheet aliases
        self._aliases = {**BUILTIN_ALIASES}
        for canonical, sheet_aliases in self._rates.aliases.items():
            if canonical in self._aliases:
                # Extend existing list
                self._aliases[canonical] = list(set(
                    self._aliases[canonical] + sheet_aliases
                ))
            else:
                self._aliases[canonical] = sheet_aliases

    def _parse_rates(self) -> NormalizedRates:
        """Parse the rate sheet using the appropriate parser."""
        if self._format == "easy":
            return parse_easy(self._path)
        elif self._format == "medium":
            return parse_medium(self._path)
        else:
            return parse_hard(self._path)

    @property
    def format(self) -> str:
        """Return the detected format of this rate sheet."""
        return self._format

    def lookup(
        self,
        origin: str,
        destination: str,
        mode: Literal["sea", "air"],
        container_size_ft: Literal[20, 40] | None = None,
        actual_weight_kg: float | None = None,
        volume_cbm: float | None = None,
    ) -> RateMatch | None:
        """
        Look up a rate for the given route and parameters.

        Args:
            origin: Origin port/city name (will be fuzzy matched)
            destination: Destination port/city name (will be fuzzy matched)
            mode: "sea" or "air"
            container_size_ft: For sea freight - 20 or 40
            actual_weight_kg: For air freight - actual weight in kg
            volume_cbm: For air freight - volume in cubic meters

        Returns:
            RateMatch if found, None otherwise
        """
        if mode == "sea":
            return self._lookup_sea(origin, destination, container_size_ft)
        else:
            return self._lookup_air(origin, destination, actual_weight_kg, volume_cbm)

    def _lookup_sea(
        self,
        origin: str,
        destination: str,
        container_size_ft: Literal[20, 40] | None,
    ) -> RateMatch | None:
        """Look up a sea freight rate."""
        df = self._rates.sea_rates

        if df.empty:
            return None

        # Clean the location names first
        origin = _clean_location(origin)
        destination = _clean_location(destination)

        # Try to find a match with fuzzy matching
        match_result = self._find_match(df, origin, destination)

        if match_result is None:
            return None

        row, matched_origin, matched_dest = match_result

        # Get the appropriate rate based on container size
        rate = None
        if container_size_ft == 20:
            rate = row.get("rate_20ft")
        elif container_size_ft == 40:
            rate = row.get("rate_40ft")
        else:
            # Default to 40ft if not specified
            rate = row.get("rate_40ft")
            container_size_ft = 40

        if rate is None or pd.isna(rate):
            return None

        transit = row.get("transit_days")
        if pd.isna(transit):
            transit = None

        return RateMatch(
            origin=str(row["origin"]),
            destination=str(row["destination"]),
            mode="sea",
            rate_per_container=float(rate),
            container_size_ft=container_size_ft,
            transit_days=int(transit) if transit else None,
            source_sheet=self._format,
            matched_origin_alias=matched_origin if matched_origin != row["origin"] else None,
            matched_dest_alias=matched_dest if matched_dest != row["destination"] else None,
        )

    def _lookup_air(
        self,
        origin: str,
        destination: str,
        actual_weight_kg: float | None,
        volume_cbm: float | None,
    ) -> RateMatch | None:
        """Look up an air freight rate."""
        df = self._rates.air_rates

        if df.empty:
            return None

        # Clean the location names first
        origin = _clean_location(origin)
        destination = _clean_location(destination)

        # Try to find a match with fuzzy matching
        match_result = self._find_match(df, origin, destination)

        if match_result is None:
            return None

        row, matched_origin, matched_dest = match_result

        rate_per_kg = row.get("rate_per_kg")
        if rate_per_kg is None or pd.isna(rate_per_kg):
            return None

        min_charge = row.get("min_charge")
        if pd.isna(min_charge):
            min_charge = None

        # Calculate chargeable weight
        chargeable_weight = self._calculate_chargeable_weight(
            actual_weight_kg, volume_cbm
        )

        transit = row.get("transit_days")
        if pd.isna(transit):
            transit = None

        return RateMatch(
            origin=str(row["origin"]),
            destination=str(row["destination"]),
            mode="air",
            rate_per_kg=float(rate_per_kg),
            min_charge=float(min_charge) if min_charge else None,
            chargeable_weight_kg=chargeable_weight,
            transit_days=int(transit) if transit else None,
            source_sheet=self._format,
            matched_origin_alias=matched_origin if matched_origin != row["origin"] else None,
            matched_dest_alias=matched_dest if matched_dest != row["destination"] else None,
        )

    def _find_match(
        self,
        df: pd.DataFrame,
        origin: str,
        destination: str,
    ) -> tuple[pd.Series, str, str] | None:
        """
        Find a matching row in the dataframe using fuzzy matching.

        Returns tuple of (row, matched_origin, matched_dest) or None.
        """
        origin_lower = origin.lower().strip()
        dest_lower = destination.lower().strip()

        # Get all possible names for origin and destination
        origin_names = self._get_all_names(origin_lower)
        dest_names = self._get_all_names(dest_lower)

        # Try each combination
        for o_name in origin_names:
            for d_name in dest_names:
                mask = (df["origin"] == o_name) & (df["destination"] == d_name)
                matches = df[mask]
                if not matches.empty:
                    return matches.iloc[0], o_name, d_name

        return None

    def _get_all_names(self, name: str) -> list[str]:
        """Get all possible names for a location (including aliases)."""
        name_lower = name.lower()
        names = [name_lower]

        # Check if this name is a canonical name with aliases
        if name_lower in self._aliases:
            names.extend(self._aliases[name_lower])

        # Check if this name is an alias pointing to a canonical name
        for canonical, alias_list in self._aliases.items():
            if name_lower in [a.lower() for a in alias_list]:
                names.append(canonical)
                names.extend(alias_list)
                break

        return list(set(names))

    def _calculate_chargeable_weight(
        self,
        actual_weight_kg: float | None,
        volume_cbm: float | None,
    ) -> float | None:
        """
        Calculate chargeable weight for air freight.

        Formula: max(actual_kg, volume_cbm * 167)
        """
        if actual_weight_kg is None and volume_cbm is None:
            return None

        actual = actual_weight_kg or 0
        volumetric = (volume_cbm or 0) * 167

        return max(actual, volumetric)
