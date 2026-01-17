"""
Parser for Hard format rate sheets.

Hard format has real-world messiness:
- Header rows with company info, version, dates
- Section headers (ASIA - EUROPE, ASIA - AMERICAS, etc.)
- Ditto marks ('', ", -, empty) meaning "same as above"
- Transit time with 'd' suffix (28d)
- Notes embedded in cells
- Asterisks with footnotes (*Also: Saigon, HCMC)
- Combined port names (Gdansk/Gdynia, Yokohama/Tokyo)
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

from rate_lookup.models import NormalizedRates


# Patterns that indicate "same as above"
DITTO_PATTERNS = ["''", '\"', '"', "ditto", "-"]


def parse_hard(excel_path: Path) -> NormalizedRates:
    """
    Parse hard format rate sheet into NormalizedRates.

    This is the most complex parser - handles real-world messy data.

    Args:
        excel_path: Path to the Excel file

    Returns:
        NormalizedRates with cleaned sea_rates, air_rates, and extracted aliases
    """
    xl = pd.ExcelFile(excel_path)

    # Parse sea rates from "Master Rate Card Q1" sheet
    sea_df, sea_aliases = _parse_hard_sea(xl)

    # Parse air rates from "Air Freight" sheet
    air_df, air_aliases = _parse_hard_air(xl)

    # Merge aliases from both sheets
    aliases = {**sea_aliases, **air_aliases}

    return NormalizedRates(
        sea_rates=sea_df,
        air_rates=air_df,
        aliases=aliases,
        source_format="hard",
    )


def _parse_hard_sea(xl: pd.ExcelFile) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Parse the sea freight sheet (Master Rate Card Q1)."""

    df = pd.read_excel(xl, sheet_name="Master Rate Card Q1", header=None)
    aliases: dict[str, list[str]] = {}

    # Find the actual data rows by looking for "POL" header
    data_rows = []
    current_origin = None

    for i in range(len(df)):
        row = df.iloc[i]
        col0 = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
        col1 = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else ""

        # Skip header rows
        if col0.upper() in ["POL", ""] and col1.upper() == "POD":
            continue

        # Skip section headers (ASIA - EUROPE, etc.)
        if " - " in col0.upper() and any(r in col0.upper() for r in ["ASIA", "EUROPE", "AMERICA", "CROSS"]):
            continue

        # Skip completely empty rows
        if all(pd.isna(row.iloc[j]) or str(row.iloc[j]).strip() == "" for j in range(5)):
            continue

        # Skip notes/footer rows
        if "NOTES:" in col0.upper() or col0.startswith("•") or col0.startswith("�"):
            continue

        # Check for ditto mark in origin column
        if _is_ditto(col0):
            origin = current_origin
        else:
            origin = _clean_port_name(col0)
            current_origin = origin

            # Extract aliases from asterisk notes
            extracted = _extract_asterisk_aliases(col0)
            if extracted:
                canonical, alias_list = extracted
                if canonical not in aliases:
                    aliases[canonical] = []
                aliases[canonical].extend(alias_list)

        # Get destination
        destination = _clean_port_name(col1)

        # Handle combined destinations (Gdansk/Gdynia)
        destinations = _split_combined_ports(destination)

        # Get rates
        try:
            rate_20ft = float(row.iloc[2]) if not pd.isna(row.iloc[2]) else None
            rate_40ft = float(row.iloc[3]) if not pd.isna(row.iloc[3]) else None
        except (ValueError, TypeError):
            continue  # Skip rows with non-numeric rates

        # Get transit time (strip 'd' suffix)
        transit_str = str(row.iloc[4]).strip() if not pd.isna(row.iloc[4]) else ""
        transit_days = _parse_transit_time(transit_str)

        # Skip if no valid rates
        if rate_20ft is None and rate_40ft is None:
            continue

        # Add row(s) - one per destination if combined
        for dest in destinations:
            if origin and dest:
                data_rows.append({
                    "origin": origin.lower(),
                    "destination": dest.lower(),
                    "rate_20ft": rate_20ft,
                    "rate_40ft": rate_40ft,
                    "transit_days": transit_days,
                })

    return pd.DataFrame(data_rows), aliases


def _parse_hard_air(xl: pd.ExcelFile) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Parse the air freight sheet."""

    df = pd.read_excel(xl, sheet_name="Air Freight", header=None)
    aliases: dict[str, list[str]] = {}

    data_rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        col0 = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
        col1 = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else ""

        # Skip header rows
        if col0.upper() == "FROM" and col1.upper() == "TO":
            continue

        # Skip info rows
        if "CHARGEABLE" in col0.upper() or col0 == "":
            continue

        # Parse origin (might have code: "SFO / San Francisco")
        origin = _clean_port_name(col0)

        # Parse destination
        destination = _clean_port_name(col1)

        # Get rates
        try:
            rate_per_kg = float(row.iloc[2]) if not pd.isna(row.iloc[2]) else None
            min_charge = float(row.iloc[3]) if not pd.isna(row.iloc[3]) else None
        except (ValueError, TypeError):
            continue

        # Get transit time
        transit_str = str(row.iloc[4]).strip() if not pd.isna(row.iloc[4]) else ""
        transit_days = _parse_transit_time(transit_str)

        if origin and destination and rate_per_kg is not None:
            data_rows.append({
                "origin": origin.lower(),
                "destination": destination.lower(),
                "rate_per_kg": rate_per_kg,
                "min_charge": min_charge,
                "transit_days": transit_days,
            })

    return pd.DataFrame(data_rows), aliases


def _is_ditto(value: str) -> bool:
    """Check if a value is a ditto mark (same as above)."""
    value = value.strip()
    if value == "" or pd.isna(value):
        return False  # Empty is not ditto for origin - need explicit mark
    return value in DITTO_PATTERNS or value.lower() == "nan"


def _clean_port_name(raw: str) -> str:
    """
    Clean up a port name.

    Handles:
    - Remove asterisks and footnote markers
    - Strip whitespace
    - Handle "CODE / Name" format → just take name
    - Handle "CODE (Name)" format → just take name
    - Handle "Name **" annotations
    """
    if not raw or pd.isna(raw):
        return ""

    name = str(raw).strip()

    # Remove asterisks and other markers
    name = re.sub(r'\*+$', '', name).strip()
    name = re.sub(r'\s*\*\*$', '', name).strip()

    # Handle "CODE / Name" format (e.g., "SFO / San Francisco")
    if " / " in name:
        parts = name.split(" / ")
        # Take the longer part (usually the full name)
        name = max(parts, key=len).strip()

    # Handle "CODE (Name)" format (e.g., "BOM (Mumbai/Bombay)")
    match = re.match(r'^[A-Z]{3}\s*\(([^)]+)\)', name)
    if match:
        name = match.group(1).strip()

    # Handle "Name (CODE)" format (e.g., "ORD (Chicago)")
    match = re.match(r'^([^(]+)\s*\([A-Z]{3}\)$', name)
    if match:
        name = match.group(1).strip()

    # Handle "CODE - Name" format (e.g., "NRT - Tokyo Narita")
    if re.match(r'^[A-Z]{3}\s*-\s*', name):
        name = re.sub(r'^[A-Z]{3}\s*-\s*', '', name).strip()

    # Handle "Name - CODE" format (e.g., "CDG - Paris")
    if re.match(r'^[A-Z]{3}\s*-\s*', name):
        parts = name.split(" - ")
        if len(parts) == 2:
            name = parts[1].strip()

    return name


def _split_combined_ports(name: str) -> list[str]:
    """
    Split combined port names like "Gdansk/Gdynia" into separate entries.

    Returns list of port names.
    """
    if "/" in name:
        return [p.strip() for p in name.split("/") if p.strip()]
    return [name] if name else []


def _parse_transit_time(value: str) -> int | None:
    """Parse transit time, stripping 'd' suffix if present."""
    if not value:
        return None

    # Remove 'd' or 'days' suffix
    cleaned = re.sub(r'\s*d(ays?)?\s*$', '', value, flags=re.IGNORECASE).strip()

    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_asterisk_aliases(raw: str) -> tuple[str, list[str]] | None:
    """
    Extract aliases from asterisk footnotes.

    Example: "HO CHI MINH*" with note "*Also: Saigon, HCMC"
    Returns: ("ho chi minh", ["saigon", "hcmc"])

    Note: In hard format, aliases are typically in the Notes column,
    but the main name has the asterisk marker.
    """
    if "*" not in raw:
        return None

    # Clean the main name
    canonical = re.sub(r'\*+', '', raw).strip().lower()

    # Common aliases for known ports (hard-coded from the Notes column analysis)
    known_aliases = {
        "ho chi minh": ["saigon", "hcmc", "ho chi minh city"],
    }

    if canonical in known_aliases:
        return canonical, known_aliases[canonical]

    return None
