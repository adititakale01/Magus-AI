"""
Data models for the Freight Agent pipeline.

These dataclasses define the structure of data flowing through each step.
Using dataclasses for:
- Type safety
- Immutability (frozen=True)
- Easy serialization to/from JSON
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Email:
    """Raw email input from JSON file."""
    sender: str      # "from" field in JSON
    to: str
    subject: str
    body: str


@dataclass(frozen=True)
class Shipment:
    """
    A single shipment request extracted from an email.

    Note: Locations are kept RAW - normalization (HCMC -> Ho Chi Minh City)
    happens in a later step.
    """
    mode: Literal["sea", "air"] | None = None

    # Location (raw from email)
    origin_raw: str | None = None
    destination_raw: str | None = None

    # Sea freight specific
    container_size_ft: Literal[20, 40] | None = None
    quantity: int | None = None

    # Air freight specific
    actual_weight_kg: float | None = None
    volume_cbm: float | None = None

    # Optional
    commodity: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """
    Result of extracting shipment info from an email.

    This is the output of Step 1+2 (Read & Extract).
    """
    sender_email: str
    shipments: tuple[Shipment, ...] = field(default_factory=tuple)  # tuple for immutability
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    needs_clarification: bool = False

    # For debugging/tracing
    raw_email_subject: str | None = None
    raw_email_body: str | None = None
