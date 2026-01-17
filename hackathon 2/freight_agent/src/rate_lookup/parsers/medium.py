"""
Parser for Medium format rate sheets.

Medium format has:
- Three sheets: "Port Codes", "Sea Rates", "Air Rates"
- Port Codes sheet contains code-to-name mapping and aliases
- Rate sheets use codes instead of full names
- Requires JOIN to resolve codes to names
"""

from pathlib import Path
import pandas as pd

from rate_lookup.models import NormalizedRates


def parse_medium(excel_path: Path) -> NormalizedRates:
    """
    Parse medium format rate sheet into NormalizedRates.

    Medium format requires:
    1. Load port codes and build code→name lookup
    2. Extract aliases from the Aliases column
    3. Load rate sheets and JOIN with port codes
    4. Return unified format with aliases

    Args:
        excel_path: Path to the Excel file

    Returns:
        NormalizedRates with sea_rates, air_rates, and aliases
    """
    xl = pd.ExcelFile(excel_path)

    # =========================================================================
    # Step 1: Parse port codes and build lookups
    # =========================================================================
    codes_df = pd.read_excel(xl, sheet_name="Port Codes")

    # Build code → port name mapping
    code_to_name: dict[str, str] = {}
    for _, row in codes_df.iterrows():
        code = str(row["Code"]).strip().upper()
        port_name = str(row["Port Name"]).strip().lower()
        code_to_name[code] = port_name

    # =========================================================================
    # Step 2: Extract aliases from Aliases column
    # =========================================================================
    aliases: dict[str, list[str]] = {}
    for _, row in codes_df.iterrows():
        port_name = str(row["Port Name"]).strip().lower()
        code = str(row["Code"]).strip().lower()

        alias_str = str(row.get("Aliases", ""))
        if alias_str and alias_str != "nan":
            alias_list = [a.strip().lower() for a in alias_str.split(",")]
            # Include the code as an alias too
            alias_list.append(code)
            aliases[port_name] = alias_list
        else:
            # At minimum, the code is an alias
            aliases[port_name] = [code]

    # =========================================================================
    # Step 3: Parse and JOIN sea rates
    # =========================================================================
    sea_df = pd.read_excel(xl, sheet_name="Sea Rates")

    # Resolve codes to names
    sea_df["origin"] = sea_df["Origin Code"].apply(
        lambda x: code_to_name.get(str(x).strip().upper(), str(x).lower())
    )
    sea_df["destination"] = sea_df["Dest Code"].apply(
        lambda x: code_to_name.get(str(x).strip().upper(), str(x).lower())
    )

    # Rename rate columns
    sea_df = sea_df.rename(columns={
        "20ft": "rate_20ft",
        "40ft": "rate_40ft",
        "Days": "transit_days",
    })

    # Keep only needed columns
    sea_df = sea_df[["origin", "destination", "rate_20ft", "rate_40ft", "transit_days"]]

    # =========================================================================
    # Step 4: Parse and JOIN air rates
    # =========================================================================
    air_df = pd.read_excel(xl, sheet_name="Air Rates")

    # Resolve codes to names
    air_df["origin"] = air_df["Origin Code"].apply(
        lambda x: code_to_name.get(str(x).strip().upper(), str(x).lower())
    )
    air_df["destination"] = air_df["Dest Code"].apply(
        lambda x: code_to_name.get(str(x).strip().upper(), str(x).lower())
    )

    # Rename rate columns
    air_df = air_df.rename(columns={
        "Per KG": "rate_per_kg",
        "Minimum": "min_charge",
        "Days": "transit_days",
    })

    # Keep only needed columns
    air_df = air_df[["origin", "destination", "rate_per_kg", "min_charge", "transit_days"]]

    return NormalizedRates(
        sea_rates=sea_df,
        air_rates=air_df,
        aliases=aliases,
        source_format="medium",
    )
