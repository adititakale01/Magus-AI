"""
Parser for Easy format rate sheets.

Easy format has:
- Clean flat tables
- Direct column names (Origin, Destination, etc.)
- Two sheets: "Sea Freight Rates" and "Air Freight Rates"
- No data cleaning needed
"""

from pathlib import Path
import pandas as pd

from rate_lookup.models import NormalizedRates


# Column name mappings for standardization
SEA_COLUMN_MAP = {
    "origin": "origin",
    "destination": "destination",
    "20ft price (usd)": "rate_20ft",
    "40ft price (usd)": "rate_40ft",
    "transit (days)": "transit_days",
}

AIR_COLUMN_MAP = {
    "origin": "origin",
    "destination": "destination",
    "rate per kg (usd)": "rate_per_kg",
    "min charge (usd)": "min_charge",
    "transit (days)": "transit_days",
}


def parse_easy(excel_path: Path) -> NormalizedRates:
    """
    Parse easy format rate sheet into NormalizedRates.

    Easy format is straightforward:
    - Load each sheet
    - Rename columns to standard names
    - Lowercase all location names

    Args:
        excel_path: Path to the Excel file

    Returns:
        NormalizedRates with sea_rates and air_rates DataFrames
    """
    xl = pd.ExcelFile(excel_path)

    # Parse sea freight rates
    sea_df = pd.read_excel(xl, sheet_name="Sea Freight Rates")
    sea_df.columns = [c.lower() for c in sea_df.columns]
    sea_df = sea_df.rename(columns=SEA_COLUMN_MAP)
    sea_df["origin"] = sea_df["origin"].str.lower().str.strip()
    sea_df["destination"] = sea_df["destination"].str.lower().str.strip()

    # Parse air freight rates
    air_df = pd.read_excel(xl, sheet_name="Air Freight Rates")
    air_df.columns = [c.lower() for c in air_df.columns]
    air_df = air_df.rename(columns=AIR_COLUMN_MAP)
    air_df["origin"] = air_df["origin"].str.lower().str.strip()
    air_df["destination"] = air_df["destination"].str.lower().str.strip()

    # Easy format has no alias table - we'll use built-in aliases
    return NormalizedRates(
        sea_rates=sea_df,
        air_rates=air_df,
        aliases={},  # No aliases in easy format
        source_format="easy",
    )
