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


# ============================================================================
# STEP 3: ENRICHMENT MODELS
# ============================================================================

@dataclass(frozen=True)
class CustomerSOP:
    """
    Customer-specific Standard Operating Procedures.

    Parsed from Qontext knowledge graph using GPT Structured Outputs.
    These rules determine how quotes are calculated and formatted.
    """
    customer_name: str

    # Pricing rules
    margin_percent: float = 15.0  # Default 15%, QuickShip gets 8%
    flat_discount_percent: float | None = None  # Global Imports: 10%
    volume_discount_tiers: tuple[tuple[int, float], ...] | None = None
    # AutoSpares: ((2, 5.0), (5, 12.0)) means 2+ containers = 5%, 5+ = 12%
    discount_before_margin: bool = True  # Apply discount before adding margin

    # Restrictions
    mode_restriction: Literal["sea", "air"] | None = None  # Global=sea, TechParts=air
    origin_restriction: str | None = None  # VietExport: "HCMC" only
    origin_equivalences: tuple[tuple[str, str], ...] = ()  # Global: Shanghai ↔ Ningbo

    # Output formatting requirements
    show_transit_time: bool = False       # Global: True
    show_chargeable_weight: bool = False  # TechParts: True
    show_subtotals: bool = False          # AutoSpares: True (multi-route)
    hide_margin: bool = False             # QuickShip: True (broker model)
    warn_transit_over_days: int | None = None  # TechParts: warn if > 3 days


@dataclass(frozen=True)
class Surcharge:
    """
    A surcharge that applies to a specific shipment.

    Example: Australia biosecurity fee of $150.
    """
    name: str      # "Australia Biosecurity"
    amount: float  # 150.0
    reason: str    # "Destination is Australia"


@dataclass(frozen=True)
class EnrichedShipment:
    """
    A shipment enriched with applicable surcharges.

    Wraps the original Shipment and adds any surcharges
    that apply based on destination, commodity, etc.
    """
    shipment: Shipment
    surcharges: tuple[Surcharge, ...] = ()


# ============================================================================
# VALIDATION MODELS (used by enrichment)
# ============================================================================

@dataclass(frozen=True)
class ValidationError:
    """
    A blocking validation error - cannot proceed with quote.

    References the SOP and provides a helpful suggestion.
    """
    error_type: str      # "mode_restriction", "origin_restriction", "missing_field"
    message: str         # User-friendly message referencing SOP
    suggestion: str      # What they can do instead
    shipment_index: int | None = None  # Which shipment (None = request-level)


@dataclass(frozen=True)
class ValidationWarning:
    """
    A non-blocking validation warning - include in quote but don't reject.

    Example: TechParts transit time > 3 days warning.
    """
    warning_type: str    # "transit_time", etc.
    message: str         # Warning text to include in quote
    shipment_index: int | None = None


# ============================================================================
# COMBINED ENRICHMENT + VALIDATION OUTPUT
# ============================================================================

@dataclass(frozen=True)
class EnrichedRequest:
    """
    Fully enriched and validated quote request.

    This is the combined output of Enrichment + Validation (one GPT call with tools).
    Contains customer info, SOP rules, enriched shipments, AND validation results.
    """
    sender_email: str
    customer_name: str
    customer_sop: CustomerSOP
    shipments: tuple[EnrichedShipment, ...]

    # Validation results (from tool calling)
    is_valid: bool = True
    validation_errors: tuple[ValidationError, ...] = ()
    validation_warnings: tuple[ValidationWarning, ...] = ()

    # Carried forward from ExtractionResult
    missing_fields: tuple[str, ...] = ()
    needs_clarification: bool = False


# ============================================================================
# STEP 5: RATE LOOKUP MODELS
# ============================================================================

@dataclass(frozen=True)
class RateMatch:
    """
    Result of looking up a rate in the Excel sheets.

    Contains the matched rate info and metadata about how the match was found.
    """
    origin: str                    # Normalized origin used for lookup
    destination: str               # Normalized destination used for lookup
    mode: Literal["sea", "air"]

    # Sea freight
    rate_per_container: float | None = None
    container_size_ft: Literal[20, 40] | None = None

    # Air freight
    rate_per_kg: float | None = None
    min_charge: float | None = None
    chargeable_weight_kg: float | None = None

    transit_days: int | None = None
    currency: str = "USD"

    # Metadata - how was this match found?
    source_sheet: Literal["easy", "medium", "hard"] | None = None
    matched_origin_alias: str | None = None   # What alias matched (if any)
    matched_dest_alias: str | None = None


# ============================================================================
# STEP 6: QUOTE CALCULATION MODELS
# ============================================================================

@dataclass(frozen=True)
class QuoteLineItem:
    """
    One line in the quote (one shipment).

    Contains the rate lookup result and all pricing calculations.
    """
    shipment_index: int
    description: str               # "Shanghai → Rotterdam, 2x 40ft"

    rate_match: RateMatch | None   # None if no rate found
    base_price: float | None = None
    discount_amount: float | None = None
    margin_amount: float | None = None
    surcharge_total: float | None = None
    line_total: float | None = None

    # SOP context for response formatting
    discount_reason: str | None = None  # e.g., "10% Strategic Partner discount"
    surcharges: tuple[Surcharge, ...] = ()  # Detailed surcharge breakdown

    # For display
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()   # e.g., "No rate found for this route"


@dataclass(frozen=True)
class Quote:
    """
    Complete quote ready for formatting.

    Contains all line items, totals, and display configuration from SOP.
    """
    customer_name: str
    customer_email: str

    line_items: tuple[QuoteLineItem, ...]

    subtotal: float | None = None
    total_surcharges: float | None = None
    grand_total: float | None = None

    # Display flags from CustomerSOP - controls what to show in response
    show_transit_time: bool = False
    show_chargeable_weight: bool = False
    show_subtotals: bool = False
    hide_margin: bool = False

    # SOP context for response formatting (reference in explanations)
    sop_summary: str | None = None  # e.g., "Strategic Partner - 10% discount, sea freight only"

    # Overall status
    is_complete: bool = True       # All shipments have rates
    has_warnings: bool = False
    has_errors: bool = False

    created_at: str = ""           # ISO timestamp


# ============================================================================
# STEP 7: RESPONSE FORMATTING MODELS
# ============================================================================

@dataclass(frozen=True)
class QuoteResponse:
    """
    Final formatted response ready to send.

    Contains the email text and keeps a reference to the underlying Quote data.
    """
    subject: str
    body: str
    quote: Quote  # Keep reference to underlying data for debugging/logging

    # Metadata
    generated_at: str = ""         # ISO timestamp
    model_used: str = "gpt-4o-mini"


# ============================================================================
# CONFIDENCE SCORE
# ============================================================================

@dataclass(frozen=True)
class ConfidenceScore:
    """
    Confidence level for the pipeline output.

    Helps frontend/humans know how much to trust the result:
    - HIGH: All data extracted, rates found, no errors → ready to send
    - MEDIUM: Needs clarification or partial data → human review recommended
    - LOW: Can't determine key info → escalate to human
    """
    level: Literal["high", "medium", "low"]
    reason: str  # Human-readable explanation

    # Detailed breakdown
    has_all_data: bool = True
    has_rates: bool = True
    has_validation_errors: bool = False
    needs_clarification: bool = False


def calculate_confidence(
    extraction: "ExtractionResult",
    enriched: "EnrichedRequest",
    quote: "Quote"
) -> ConfidenceScore:
    """
    Calculate confidence score based on pipeline results.

    HIGH: Complete quote with no issues
    MEDIUM: Partial quote or needs clarification
    LOW: Cannot produce a meaningful quote
    """
    has_all_data = len(extraction.missing_fields) == 0
    has_rates = quote.is_complete
    has_validation_errors = len(enriched.validation_errors) > 0
    needs_clarification = extraction.needs_clarification

    # Determine level
    if has_all_data and has_rates and not has_validation_errors and not needs_clarification:
        level = "high"
        reason = "Complete quote ready to send"
    elif needs_clarification:
        level = "medium"
        reason = f"Clarification needed: {', '.join(extraction.missing_fields) or 'ambiguous request'}"
    elif has_validation_errors:
        # SOP violations take precedence over rate issues
        level = "medium"
        reason = f"SOP validation issue: {enriched.validation_errors[0].message}"
    elif not has_rates and has_all_data:
        level = "medium"
        reason = "No rates found for this route"
    elif not has_all_data:
        level = "low"
        reason = f"Missing required data: {', '.join(extraction.missing_fields)}"
    else:
        level = "low"
        reason = "Unable to process request"

    return ConfidenceScore(
        level=level,
        reason=reason,
        has_all_data=has_all_data,
        has_rates=has_rates,
        has_validation_errors=has_validation_errors,
        needs_clarification=needs_clarification
    )


# ============================================================================
# PIPELINE RESULT (combines everything)
# ============================================================================

@dataclass(frozen=True)
class PipelineResult:
    """
    Complete result of processing an email through the pipeline.

    Contains all intermediate results for debugging/logging,
    plus the final response ready to send.
    """
    # Intermediate results
    extraction: ExtractionResult
    enriched: EnrichedRequest
    rate_matches: tuple[RateMatch | None, ...]
    quote: Quote

    # Final output
    response: QuoteResponse

    # Confidence score
    confidence: ConfidenceScore | None = None

    # Metadata
    processing_time_ms: int = 0
    gpt_calls: int = 3  # Should always be 3