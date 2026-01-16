from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Mode = Literal["air", "sea"]


@dataclass(frozen=True)
class SopProfile:
    customer_email: str
    customer_name: str
    allowed_modes: set[Mode] | None = None
    margin_override: float | None = None
    sea_discount_pct: float | None = None
    sea_discount_label: str | None = None
    origin_required: set[str] | None = None
    origin_fallbacks: dict[str, list[str]] = field(default_factory=dict)
    transit_warning_if_gt_days: int | None = None
    show_actual_and_chargeable_weight: bool = False
    hide_margin_percent: bool = True
    container_volume_discount_tiers: list[tuple[int, float]] = field(default_factory=list)


@dataclass(frozen=True)
class SopSurcharge:
    amount: float
    label: str


def load_sop_markdown(*, data_dir: Path) -> str:
    path = data_dir / "SOP.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_sop_profile(sender_email: str) -> SopProfile | None:
    sender = (sender_email or "").strip().casefold()
    profiles = _profiles_by_sender()
    return profiles.get(sender)


def get_global_surcharges(*, destination_canonical: str | None) -> list[SopSurcharge]:
    if not destination_canonical:
        return []
    if destination_canonical.strip().casefold() in {"melbourne"}:
        return [SopSurcharge(amount=150.0, label="Biosecurity surcharge (Australia)")]
    return []


def volume_discount_pct(profile: SopProfile, *, total_containers: int) -> float:
    if total_containers <= 0:
        return 0.0
    for threshold, pct in sorted(profile.container_volume_discount_tiers, key=lambda x: -x[0]):
        if total_containers >= threshold:
            return float(pct)
    return 0.0


def origin_fallbacks(profile: SopProfile, *, origin_canonical: str) -> list[str]:
    return list(profile.origin_fallbacks.get(origin_canonical, []))


def _profiles_by_sender() -> dict[str, SopProfile]:
    return {
        # SOP 1: Global Imports Ltd
        "sarah.chen@globalimports.com": SopProfile(
            customer_email="sarah.chen@globalimports.com",
            customer_name="Global Imports Ltd",
            allowed_modes={"sea"},
            sea_discount_pct=0.10,
            sea_discount_label="10% strategic partner discount (sea)",
            origin_fallbacks={"Shanghai": ["Ningbo"], "Ningbo": ["Shanghai"]},
        ),
        # SOP 2: TechParts Inc (air-only)
        "mike.johnson@techparts.io": SopProfile(
            customer_email="mike.johnson@techparts.io",
            customer_name="TechParts Inc",
            allowed_modes={"air"},
            transit_warning_if_gt_days=3,
            show_actual_and_chargeable_weight=True,
        ),
        # SOP 3: AutoSpares GmbH (volume discount across all routes)
        "david.mueller@autospares.de": SopProfile(
            customer_email="david.mueller@autospares.de",
            customer_name="AutoSpares GmbH",
            container_volume_discount_tiers=[(5, 0.12), (2, 0.05)],
        ),
        # Appendix: QuickShip UK (broker margin)
        "tom.bradley@quickship.co.uk": SopProfile(
            customer_email="tom.bradley@quickship.co.uk",
            customer_name="QuickShip UK",
            margin_override=0.08,
            hide_margin_percent=True,
        ),
        # Appendix: VietExport (origin restriction)
        "lisa.nguyen@vietexport.vn": SopProfile(
            customer_email="lisa.nguyen@vietexport.vn",
            customer_name="VietExport",
            origin_required={"Ho Chi Minh City"},
        ),
    }

