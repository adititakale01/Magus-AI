"""
Auto-detect rate sheet format.

Analyzes the Excel file structure to determine which parser to use.
"""

from pathlib import Path
from typing import Literal
import pandas as pd


def detect_format(excel_path: Path) -> Literal["easy", "medium", "hard"]:
    """
    Analyze Excel structure to determine rate sheet format.

    Detection rules:
    1. If sheet names include "Port Codes" or similar → Medium
    2. If first sheet has ditto marks or messy headers → Hard
    3. Otherwise → Easy (clean flat table)

    Args:
        excel_path: Path to the Excel file

    Returns:
        "easy", "medium", or "hard"
    """
    xl = pd.ExcelFile(excel_path)
    sheet_names_lower = [s.lower() for s in xl.sheet_names]

    # Check for Medium format: has port codes sheet
    if any("port" in name and "code" in name for name in sheet_names_lower):
        return "medium"

    if "port codes" in sheet_names_lower or "codes" in sheet_names_lower:
        return "medium"

    # Check for Hard format: look at first sheet content
    first_sheet = xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=first_sheet, header=None, nrows=30)

    # Check for messy header (company name, version info, etc.)
    first_cell = str(df.iloc[0, 0]).upper() if not pd.isna(df.iloc[0, 0]) else ""
    if any(keyword in first_cell for keyword in ["FREIGHT", "RATE CARD", "GLOBAL", "SOLUTIONS"]):
        return "hard"

    # Check for ditto marks in the data
    ditto_patterns = ["''", '\"', "ditto"]
    df_str = df.astype(str)
    for pattern in ditto_patterns:
        if df_str.apply(lambda col: col.str.contains(pattern, case=False, na=False)).any().any():
            return "hard"

    # Check for section headers (ASIA - EUROPE, etc.)
    for i in range(min(20, len(df))):
        cell_val = str(df.iloc[i, 0]).upper() if not pd.isna(df.iloc[i, 0]) else ""
        if " - " in cell_val and any(region in cell_val for region in ["ASIA", "EUROPE", "AMERICA"]):
            return "hard"

    # Default: Easy format
    return "easy"
