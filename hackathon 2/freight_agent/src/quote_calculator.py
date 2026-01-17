"""
Quote Calculator

Takes an EnrichedRequest and rate matches, applies business logic
(discounts, margins, surcharges), and produces a Quote.

This is pure deterministic calculation - no AI/GPT involved.
"""

from datetime import datetime

from models import (
    EnrichedRequest,
    EnrichedShipment,
    CustomerSOP,
    RateMatch,
    QuoteLineItem,
    Quote,
)


def calculate_quote(
    enriched: EnrichedRequest,
    rate_matches: list[RateMatch | None],
) -> Quote:
    """
    Calculate complete quote with all pricing applied.

    Args:
        enriched: The enriched request with customer SOP and shipments
        rate_matches: List of rate matches (one per shipment, None if not found)

    Returns:
        Quote with all line items and totals calculated
    """
    line_items: list[QuoteLineItem] = []
    sop = enriched.customer_sop

    # Calculate TOTAL containers across all routes (for volume discount)
    # SOP: "Apply volume discount based on total containers across all routes"
    total_containers = sum(
        es.shipment.quantity or 1
        for es in enriched.shipments
        if es.shipment.mode == "sea"
    )

    for i, (enriched_shipment, rate_match) in enumerate(
        zip(enriched.shipments, rate_matches)
    ):
        line_item = _calculate_line_item(
            index=i,
            enriched_shipment=enriched_shipment,
            rate_match=rate_match,
            sop=sop,
            total_containers=total_containers,
        )
        line_items.append(line_item)

    # Calculate totals
    valid_totals = [li.line_total for li in line_items if li.line_total is not None]
    valid_surcharges = [li.surcharge_total for li in line_items if li.surcharge_total is not None]

    subtotal = sum(valid_totals) if valid_totals else None
    total_surcharges = sum(valid_surcharges) if valid_surcharges else None
    grand_total = subtotal  # Surcharges already included in line totals

    return Quote(
        customer_name=enriched.customer_name,
        customer_email=enriched.sender_email,
        line_items=tuple(line_items),
        subtotal=subtotal,
        total_surcharges=total_surcharges,
        grand_total=grand_total,
        # Display flags from SOP
        show_transit_time=sop.show_transit_time,
        show_chargeable_weight=sop.show_chargeable_weight,
        show_subtotals=sop.show_subtotals,
        hide_margin=sop.hide_margin,
        # Status
        is_complete=all(li.rate_match is not None for li in line_items),
        has_warnings=any(li.warnings for li in line_items),
        has_errors=any(li.errors for li in line_items),
        created_at=datetime.now().isoformat(),
    )


def _calculate_line_item(
    index: int,
    enriched_shipment: EnrichedShipment,
    rate_match: RateMatch | None,
    sop: CustomerSOP,
    total_containers: int = 1,
) -> QuoteLineItem:
    """Calculate a single line item."""
    shipment = enriched_shipment.shipment

    # Build description
    description = _build_description(shipment, rate_match)

    # No rate found - return error line item
    if rate_match is None:
        return QuoteLineItem(
            shipment_index=index,
            description=description,
            rate_match=None,
            errors=("No rate found for this route",),
        )

    # === STEP 1: Calculate base price ===
    if shipment.mode == "sea":
        quantity = shipment.quantity or 1
        base_price = rate_match.rate_per_container * quantity
    else:  # air
        if rate_match.chargeable_weight_kg and rate_match.rate_per_kg:
            base_price = rate_match.chargeable_weight_kg * rate_match.rate_per_kg
            # Apply minimum charge if applicable
            if rate_match.min_charge and base_price < rate_match.min_charge:
                base_price = rate_match.min_charge
        else:
            return QuoteLineItem(
                shipment_index=index,
                description=description,
                rate_match=rate_match,
                errors=("Missing weight or rate information for air freight",),
            )

    # === STEP 2: Calculate discount ===
    # Use TOTAL containers across all routes for volume discount (per SOP)
    discount_percent = _calculate_discount_percent(sop, total_containers)

    # === STEP 3: Apply discount and margin ===
    if sop.discount_before_margin:
        # Discount first, then margin on discounted price
        discount_amount = base_price * (discount_percent / 100)
        after_discount = base_price - discount_amount
        margin_amount = after_discount * (sop.margin_percent / 100)
        subtotal = after_discount + margin_amount
    else:
        # Margin first, then discount on margined price
        margin_amount = base_price * (sop.margin_percent / 100)
        after_margin = base_price + margin_amount
        discount_amount = after_margin * (discount_percent / 100)
        subtotal = after_margin - discount_amount

    # === STEP 4: Add surcharges ===
    surcharge_total = sum(s.amount for s in enriched_shipment.surcharges)
    line_total = subtotal + surcharge_total

    # === STEP 5: Build warnings ===
    warnings = _build_warnings(rate_match, sop)

    return QuoteLineItem(
        shipment_index=index,
        description=description,
        rate_match=rate_match,
        base_price=round(base_price, 2),
        discount_amount=round(discount_amount, 2),
        margin_amount=round(margin_amount, 2),
        surcharge_total=round(surcharge_total, 2),
        line_total=round(line_total, 2),
        warnings=tuple(warnings),
        errors=(),
    )


def _calculate_discount_percent(sop: CustomerSOP, quantity: int) -> float:
    """
    Determine discount percentage based on SOP rules.

    Priority:
    1. Flat discount (if set)
    2. Volume discount tiers (if set)
    3. No discount (0%)
    """
    # Flat discount takes priority
    if sop.flat_discount_percent is not None:
        return sop.flat_discount_percent

    # Volume-based tiers
    if sop.volume_discount_tiers:
        discount = 0.0
        for threshold, percent in sop.volume_discount_tiers:
            if quantity >= threshold:
                discount = percent  # Take highest qualifying tier
        return discount

    return 0.0


def _build_description(shipment, rate_match: RateMatch | None) -> str:
    """Build a human-readable description of the shipment."""
    origin = shipment.origin_raw or "Unknown"
    dest = shipment.destination_raw or "Unknown"

    if shipment.mode == "sea":
        quantity = shipment.quantity or 1
        size = shipment.container_size_ft or 40
        return f"{origin} -> {dest}, {quantity}x {size}ft"
    else:  # air
        weight = shipment.actual_weight_kg
        volume = shipment.volume_cbm
        parts = [f"{origin} -> {dest}"]
        if weight:
            parts.append(f"{weight}kg")
        if volume:
            parts.append(f"{volume}CBM")
        return ", ".join(parts)


def _build_warnings(rate_match: RateMatch, sop: CustomerSOP) -> list[str]:
    """Build list of warnings based on rate match and SOP rules."""
    warnings = []

    # Transit time warning
    if sop.warn_transit_over_days and rate_match.transit_days:
        if rate_match.transit_days > sop.warn_transit_over_days:
            warnings.append(
                f"Transit time ({rate_match.transit_days} days) exceeds "
                f"preferred maximum ({sop.warn_transit_over_days} days)"
            )

    return warnings
