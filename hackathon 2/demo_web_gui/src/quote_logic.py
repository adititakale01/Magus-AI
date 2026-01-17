from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from src.agent import ShipmentRequest
from src.config import AppConfig


@dataclass(frozen=True)
class RateMatch:
    mode: str
    origin: str
    destination: str
    currency: str
    transit_days: int | None
    base_rate: float
    base_unit: str
    min_charge: float | None = None
    source: dict | None = None


def normalize_location(raw: str | None, *, aliases: dict[str, list[str]]) -> str | None:
    if not raw:
        return None
    cleaned = _clean(raw)
    tokens = set(cleaned.split())

    for canonical, alias_list in aliases.items():
        for alias in [canonical, *alias_list]:
            a = _clean(alias)
            if not a:
                continue
            if len(a) <= 3 and a in tokens:
                return canonical
            if a in cleaned:
                return canonical
    return raw.strip()


def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def format_location(name: str | None, *, codes: dict[str, str]) -> str:
    if not name:
        return ""
    code = codes.get(name)
    return f"{name} ({code})" if code else name


def air_chargeable_weight_kg(*, actual_kg: float, volume_cbm: float, factor: float) -> float:
    return max(actual_kg, volume_cbm * factor)


def lookup_easy_air_rate(
    *,
    df_air: pd.DataFrame,
    origin: str,
    destination: str,
    currency: str,
) -> RateMatch | None:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().casefold())

    target_o = norm(origin)
    target_d = norm(destination)
    df = df_air.copy()
    df["_o"] = df["Origin"].map(norm)
    df["_d"] = df["Destination"].map(norm)
    hits = df[(df["_o"] == target_o) & (df["_d"] == target_d)]
    if hits.empty:
        return None

    row = hits.iloc[0]
    return RateMatch(
        mode="air",
        origin=origin,
        destination=destination,
        currency=currency,
        transit_days=int(row.get("Transit (Days)")) if pd.notna(row.get("Transit (Days)")) else None,
        base_rate=float(row["Rate per kg (USD)"]),
        base_unit="kg",
        min_charge=float(row["Min Charge (USD)"]) if pd.notna(row.get("Min Charge (USD)")) else None,
        source={"sheet": "Air Freight Rates"},
    )


def lookup_easy_sea_rate(
    *,
    df_sea: pd.DataFrame,
    origin: str,
    destination: str,
    container_size_ft: int,
    currency: str,
) -> RateMatch | None:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().casefold())

    col = "20ft Price (USD)" if container_size_ft == 20 else "40ft Price (USD)"
    if col not in df_sea.columns:
        return None

    target_o = norm(origin)
    target_d = norm(destination)
    df = df_sea.copy()
    df["_o"] = df["Origin"].map(norm)
    df["_d"] = df["Destination"].map(norm)
    hits = df[(df["_o"] == target_o) & (df["_d"] == target_d)]
    if hits.empty:
        return None

    row = hits.iloc[0]
    return RateMatch(
        mode="sea",
        origin=origin,
        destination=destination,
        currency=currency,
        transit_days=int(row.get("Transit (Days)")) if pd.notna(row.get("Transit (Days)")) else None,
        base_rate=float(row[col]),
        base_unit="container",
        min_charge=None,
        source={"sheet": "Sea Freight Rates", "column": col},
    )


def compute_quote_amounts(
    *,
    request: ShipmentRequest,
    rate: RateMatch,
    config: AppConfig,
    margin_override: float | None = None,
    discount_pct: float = 0.0,
    surcharge: float = 0.0,
) -> dict:
    margin = float(margin_override) if margin_override is not None else float(config.pricing.margin)
    discount_pct = float(discount_pct or 0.0)
    surcharge = float(surcharge or 0.0)

    if rate.mode == "air":
        if request.actual_weight_kg is None or request.volume_cbm is None:
            raise ValueError("Air freight requires actual_weight_kg and volume_cbm.")
        chargeable = air_chargeable_weight_kg(
            actual_kg=float(request.actual_weight_kg),
            volume_cbm=float(request.volume_cbm),
            factor=config.pricing.air_volume_factor,
        )
        base = chargeable * rate.base_rate
        if rate.min_charge is not None:
            base = max(base, float(rate.min_charge))
        base_before_discount = float(base)
        base_after_discount = base_before_discount * (1.0 - discount_pct)
        final_before_surcharge = base_after_discount * (1.0 + margin)
        final = final_before_surcharge + surcharge
        return {
            "chargeable_weight_kg": chargeable,
            "base_amount": base_before_discount,
            "base_amount_after_discount": base_after_discount,
            "discount_pct": discount_pct,
            "discount_amount": base_before_discount - base_after_discount,
            "surcharge": surcharge,
            "final_amount_before_surcharge": final_before_surcharge,
            "final_amount": final,
            "currency": rate.currency,
            "margin": margin,
        }

    if rate.mode == "sea":
        if request.quantity is None or request.container_size_ft is None:
            raise ValueError("Sea freight requires quantity and container_size_ft.")
        base_before_discount = float(request.quantity) * float(rate.base_rate)
        base_after_discount = base_before_discount * (1.0 - discount_pct)
        final_before_surcharge = base_after_discount * (1.0 + margin)
        final = final_before_surcharge + surcharge
        return {
            "base_amount": base_before_discount,
            "base_amount_after_discount": base_after_discount,
            "discount_pct": discount_pct,
            "discount_amount": base_before_discount - base_after_discount,
            "surcharge": surcharge,
            "final_amount_before_surcharge": final_before_surcharge,
            "final_amount": final,
            "currency": rate.currency,
            "margin": margin,
        }

    raise ValueError(f"Unsupported mode: {rate.mode}")
