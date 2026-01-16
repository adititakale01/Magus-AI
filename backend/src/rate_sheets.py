from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import pandas as pd

from src.config import AppConfig


SEA_COLUMNS = ["Origin", "Destination", "20ft Price (USD)", "40ft Price (USD)", "Transit (Days)"]
AIR_COLUMNS = ["Origin", "Destination", "Rate per kg (USD)", "Min Charge (USD)", "Transit (Days)"]


@dataclass(frozen=True)
class NormalizedRateSheets:
    difficulty: str
    source_path: Path
    source_mtime: float
    sheet_names: list[str]
    sea: pd.DataFrame
    air: pd.DataFrame
    aliases: dict[str, list[str]] = field(default_factory=dict)
    codes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def normalize_rate_sheets(*, config: AppConfig, difficulty: str) -> NormalizedRateSheets:
    source_path = _resolve_source_path(config=config, difficulty=difficulty)
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))

    sheet_names = pd.ExcelFile(source_path).sheet_names
    source_mtime = source_path.stat().st_mtime

    if difficulty == "easy":
        sea = pd.read_excel(source_path, sheet_name="Sea Freight Rates")
        air = pd.read_excel(source_path, sheet_name="Air Freight Rates")
        sea = _standardize_sea_df(sea)
        air = _standardize_air_df(air)
        return NormalizedRateSheets(
            difficulty=difficulty,
            source_path=source_path,
            source_mtime=source_mtime,
            sheet_names=sheet_names,
            sea=sea,
            air=air,
        )

    if difficulty == "medium":
        return _normalize_medium(source_path=source_path, source_mtime=source_mtime, sheet_names=sheet_names)

    if difficulty == "hard":
        return _normalize_hard(source_path=source_path, source_mtime=source_mtime, sheet_names=sheet_names, config=config)

    raise ValueError(f"Unsupported difficulty: {difficulty}")


def _resolve_source_path(*, config: AppConfig, difficulty: str) -> Path:
    rel = None
    if difficulty == "easy":
        rel = config.rates.easy_file
    elif difficulty == "medium":
        rel = config.rates.medium_file
    elif difficulty == "hard":
        rel = config.rates.hard_file
    else:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return (config.data.data_dir / rel).resolve()


def _standardize_sea_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "20ft": "20ft Price (USD)",
            "40ft": "40ft Price (USD)",
            "Days": "Transit (Days)",
            "T/T": "Transit (Days)",
        }
    )
    for col in SEA_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing sea column: {col}")
    out["Origin"] = out["Origin"].map(_canonicalize_location)
    out["Destination"] = out["Destination"].map(_canonicalize_location)
    out["20ft Price (USD)"] = pd.to_numeric(out["20ft Price (USD)"], errors="coerce")
    out["40ft Price (USD)"] = pd.to_numeric(out["40ft Price (USD)"], errors="coerce")
    out["Transit (Days)"] = out["Transit (Days)"].map(_parse_days).astype("Int64")
    out = out.dropna(subset=["Origin", "Destination"])
    return out[SEA_COLUMNS].reset_index(drop=True)


def _standardize_air_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "Per KG": "Rate per kg (USD)",
            "Minimum": "Min Charge (USD)",
            "Days": "Transit (Days)",
            "Rate/kg": "Rate per kg (USD)",
        }
    )
    for col in AIR_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing air column: {col}")
    out["Origin"] = out["Origin"].map(_canonicalize_location)
    out["Destination"] = out["Destination"].map(_canonicalize_location)
    out["Rate per kg (USD)"] = pd.to_numeric(out["Rate per kg (USD)"], errors="coerce")
    out["Min Charge (USD)"] = pd.to_numeric(out["Min Charge (USD)"], errors="coerce")
    out["Transit (Days)"] = out["Transit (Days)"].map(_parse_days).astype("Int64")
    out = out.dropna(subset=["Origin", "Destination"])
    return out[AIR_COLUMNS].reset_index(drop=True)


def _normalize_medium(*, source_path: Path, source_mtime: float, sheet_names: list[str]) -> NormalizedRateSheets:
    port_codes = pd.read_excel(source_path, sheet_name="Port Codes")
    sea_rates = pd.read_excel(source_path, sheet_name="Sea Rates")
    air_rates = pd.read_excel(source_path, sheet_name="Air Rates")

    warnings: list[str] = []

    code_to_port: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    codes: dict[str, str] = {}

    for _, row in port_codes.iterrows():
        code = str(row.get("Code", "")).strip().upper()
        name = _canonicalize_location(row.get("Port Name"))
        if not code or not name:
            continue
        code_to_port[code] = name
        codes[name] = code

        raw_aliases = str(row.get("Aliases", "") or "")
        alias_list = [a.strip() for a in raw_aliases.split(",") if a.strip()]
        alias_list.append(code)
        aliases[name] = _unique_preserve(alias_list)

    def map_code(code: object) -> str | None:
        c = str(code or "").strip().upper()
        return code_to_port.get(c)

    sea = pd.DataFrame(
        {
            "Origin": sea_rates["Origin Code"].map(map_code),
            "Destination": sea_rates["Dest Code"].map(map_code),
            "20ft Price (USD)": sea_rates["20ft"],
            "40ft Price (USD)": sea_rates["40ft"],
            "Transit (Days)": sea_rates["Days"],
        }
    )
    if sea["Origin"].isna().any() or sea["Destination"].isna().any():
        warnings.append("Some sea rate rows reference unknown port codes.")
    sea = _standardize_sea_df(sea)

    air = pd.DataFrame(
        {
            "Origin": air_rates["Origin Code"].map(map_code),
            "Destination": air_rates["Dest Code"].map(map_code),
            "Rate per kg (USD)": air_rates["Per KG"],
            "Min Charge (USD)": air_rates["Minimum"],
            "Transit (Days)": air_rates["Days"],
        }
    )
    if air["Origin"].isna().any() or air["Destination"].isna().any():
        warnings.append("Some air rate rows reference unknown port codes.")
    air = _standardize_air_df(air)

    return NormalizedRateSheets(
        difficulty="medium",
        source_path=source_path,
        source_mtime=source_mtime,
        sheet_names=sheet_names,
        sea=sea,
        air=air,
        aliases=aliases,
        codes=codes,
        warnings=warnings,
    )


def _normalize_hard(
    *,
    source_path: Path,
    source_mtime: float,
    sheet_names: list[str],
    config: AppConfig,
) -> NormalizedRateSheets:
    warnings: list[str] = []
    codes: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}

    sea_raw = pd.read_excel(source_path, sheet_name="Master Rate Card Q1", header=None)
    sea_rows: list[dict] = []
    header_rows = [i for i in range(len(sea_raw)) if str(sea_raw.iloc[i, 0]).strip().upper() == "POL"]
    last_origin: str | None = None

    for idx in range(len(sea_raw)):
        if idx in header_rows:
            last_origin = None
            continue

        row = sea_raw.iloc[idx].tolist()
        if not any(x == x for x in row):  # all-NaN
            continue

        first = row[0]
        if isinstance(first, str) and first.strip().upper() in {"ASIA - EUROPE", "ASIA - AMERICAS", "CROSS-TRADE", "NOTES:"}:
            continue

        if isinstance(first, str) and first.strip().upper() == "POL":
            continue

        origin_cell = row[0] if len(row) > 0 else None
        dest_cell = row[1] if len(row) > 1 else None
        p20 = row[2] if len(row) > 2 else None
        p40 = row[3] if len(row) > 3 else None
        transit = row[4] if len(row) > 4 else None

        if dest_cell != dest_cell:  # NaN destination
            continue

        if _is_ditto(origin_cell):
            origin = last_origin
        else:
            origin = _canonicalize_location(origin_cell)
            last_origin = origin

        destination = _canonicalize_location(dest_cell)
        if not origin or not destination:
            continue

        sea_rows.append(
            {
                "Origin": origin,
                "Destination": destination,
                "20ft Price (USD)": p20,
                "40ft Price (USD)": p40,
                "Transit (Days)": transit,
            }
        )

    if not sea_rows:
        warnings.append("No sea rates were parsed from the hard sheet.")
    sea = _standardize_sea_df(pd.DataFrame(sea_rows))

    air_raw = pd.read_excel(source_path, sheet_name="Air Freight", header=None)
    header_idx = _find_row_index(air_raw, "FROM")
    if header_idx is None:
        raise ValueError("Could not find Air Freight header row (FROM/TO).")

    headers = [str(x).strip() for x in air_raw.iloc[header_idx].tolist()]
    col_from = _find_col(headers, {"FROM"})
    col_to = _find_col(headers, {"TO"})
    col_rate = _find_col(headers, {"USD/KG", "USD PER KG", "PER KG", "$/KG", "$ / KG"})
    col_min = _find_col(headers, {"MIN $", "MIN", "MINIMUM"})
    col_days = _find_col(headers, {"DAYS", "DAY"})

    if None in (col_from, col_to, col_rate, col_min, col_days):
        raise ValueError(f"Unrecognized Air Freight columns: {headers}")

    air_rows: list[dict] = []
    for idx in range(header_idx + 1, len(air_raw)):
        from_cell = air_raw.iloc[idx, col_from]
        to_cell = air_raw.iloc[idx, col_to]
        if from_cell != from_cell and to_cell != to_cell:  # both NaN
            continue
        if from_cell != from_cell or to_cell != to_cell:
            continue

        origin, o_code = _parse_air_cell(str(from_cell), config=config)
        dest, d_code = _parse_air_cell(str(to_cell), config=config)
        if o_code and origin and origin not in codes:
            codes[origin] = o_code
            aliases.setdefault(origin, []).append(o_code)
        if d_code and dest and dest not in codes:
            codes[dest] = d_code
            aliases.setdefault(dest, []).append(d_code)

        air_rows.append(
            {
                "Origin": origin,
                "Destination": dest,
                "Rate per kg (USD)": air_raw.iloc[idx, col_rate],
                "Min Charge (USD)": air_raw.iloc[idx, col_min],
                "Transit (Days)": air_raw.iloc[idx, col_days],
            }
        )

    if not air_rows:
        warnings.append("No air rates were parsed from the hard sheet.")
    air = _standardize_air_df(pd.DataFrame(air_rows))

    aliases = {k: _unique_preserve(v) for k, v in aliases.items()}

    return NormalizedRateSheets(
        difficulty="hard",
        source_path=source_path,
        source_mtime=source_mtime,
        sheet_names=sheet_names,
        sea=sea,
        air=air,
        aliases=aliases,
        codes=codes,
        warnings=warnings,
    )


def _canonicalize_location(value: object) -> str | None:
    if value is None or value != value:  # NaN
        return None
    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"\*+", "", s).strip()
    s = re.sub(r"\([^)]*\)", "", s).strip()
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    s = re.sub(r"\s+", " ", s).strip(" -")
    if not s:
        return None

    sl = s.casefold()
    if "ho chi minh" in sl:
        return "Ho Chi Minh City"
    if sl in {"la", "l.a."}:
        return "Los Angeles"
    if sl.startswith("ho chi minh"):
        return "Ho Chi Minh City"

    return s.title()


def _parse_days(value: object) -> int | None:
    if value is None or value != value:
        return None
    if isinstance(value, (int, float)) and value == value:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _is_ditto(value: object) -> bool:
    if value is None or value != value:
        return True
    s = str(value).strip()
    if not s:
        return True
    s = s.replace(" ", "")
    return s in {"''", '"', "-", "–", "—"}


def _find_row_index(df: pd.DataFrame, first_cell_value: str) -> int | None:
    target = first_cell_value.strip().casefold()
    for i in range(len(df)):
        v = df.iloc[i, 0]
        if isinstance(v, str) and v.strip().casefold() == target:
            return i
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        for cell in row:
            if isinstance(cell, str) and cell.strip().casefold() == target:
                return i
    return None


def _find_col(headers: list[str], candidates: set[str]) -> int | None:
    cand = {c.strip().casefold() for c in candidates}
    for i, h in enumerate(headers):
        if h.strip().casefold() in cand:
            return i
    return None


def _parse_air_cell(text: str, *, config: AppConfig) -> tuple[str | None, str | None]:
    s = text.strip()
    code = None
    name_part = s

    m = re.match(r"^([A-Z]{3})\s*[/\\-]\s*(.+)$", s)
    if not m:
        m = re.match(r"^([A-Z]{3})\s+(.+)$", s)
    if m:
        code = m.group(1).strip().upper()
        name_part = m.group(2).strip()

    name_part = re.sub(r"\*+", "", name_part).strip()
    name_part = name_part.strip(" -/")
    name_part = name_part.strip()
    name_part = name_part.strip("()")
    if "/" in name_part:
        name_part = name_part.split("/", 1)[0].strip()
    name = _canonicalize_location(name_part) if name_part else _canonicalize_location(s)

    if name and config.aliases:
        for canonical, alias_list in config.aliases.items():
            for alias in [canonical, *alias_list]:
                a = str(alias).strip()
                if not a:
                    continue
                if a.casefold() in str(text).casefold():
                    name = canonical
                    break

    return name, code


def _unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out
