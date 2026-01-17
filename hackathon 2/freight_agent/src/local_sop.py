"""
Local SOP Lookup

Ground truth customer SOPs from SOP.md.
Used as primary source, with optional Qontext validation.

This is faster and more reliable than external API calls.
"""

from models import CustomerSOP, Surcharge

# =============================================================================
# CUSTOMER EMAIL TO SOP MAPPING
# Based on SOP.md ground truth
# =============================================================================

# Customer identification by email domain
CUSTOMER_EMAIL_MAP = {
    "globalimports.com": "Global Imports Ltd",
    "techparts.io": "TechParts Inc",
    "autospares.de": "AutoSpares GmbH",
    "quickship.co.uk": "QuickShip UK",
    "vietexport.vn": "VietExport",
}

# Full SOP definitions per customer
CUSTOMER_SOPS = {
    "Global Imports Ltd": CustomerSOP(
        customer_name="Global Imports Ltd",
        margin_percent=15.0,  # Standard margin AFTER discount
        flat_discount_percent=10.0,  # 10% off all sea freight
        volume_discount_tiers=None,
        discount_before_margin=True,  # Discount applies to base rate first
        mode_restriction="sea",  # Sea freight ONLY
        origin_restriction=None,  # Shanghai/Ningbo interchangeable (handled in rate lookup)
        show_transit_time=True,
        show_chargeable_weight=False,
        show_subtotals=False,
        hide_margin=False,
    ),
    "TechParts Inc": CustomerSOP(
        customer_name="TechParts Inc",
        margin_percent=15.0,  # Standard margin
        flat_discount_percent=None,  # No discount
        volume_discount_tiers=None,
        discount_before_margin=True,
        mode_restriction="air",  # Air freight ONLY
        origin_restriction=None,
        show_transit_time=True,  # Warn if > 3 days
        show_chargeable_weight=True,  # Always show actual + chargeable weight
        show_subtotals=False,
        hide_margin=False,
    ),
    "AutoSpares GmbH": CustomerSOP(
        customer_name="AutoSpares GmbH",
        margin_percent=15.0,  # Standard margin
        flat_discount_percent=None,  # Volume discount instead
        # Volume tiers: (threshold, percent) - discount applies at threshold+ containers
        volume_discount_tiers=(
            (1, 0.0),    # 1 container: no discount
            (2, 5.0),    # 2-4 containers: 5% discount
            (5, 12.0),   # 5+ containers: 12% discount
        ),
        discount_before_margin=True,
        mode_restriction=None,  # No mode restriction
        origin_restriction=None,
        show_transit_time=True,
        show_chargeable_weight=False,
        show_subtotals=True,  # Show subtotal per route AND grand total
        hide_margin=False,
    ),
    "QuickShip UK": CustomerSOP(
        customer_name="QuickShip UK",
        margin_percent=8.0,  # Broker margin (lower)
        flat_discount_percent=None,
        volume_discount_tiers=None,
        discount_before_margin=True,
        mode_restriction=None,
        origin_restriction=None,
        show_transit_time=True,
        show_chargeable_weight=False,
        show_subtotals=False,
        hide_margin=True,  # Don't show margin % to customer
    ),
    "VietExport": CustomerSOP(
        customer_name="VietExport",
        margin_percent=15.0,  # Standard margin
        flat_discount_percent=None,
        volume_discount_tiers=None,
        discount_before_margin=True,
        mode_restriction=None,
        origin_restriction="hcmc",  # HCMC origin ONLY
        show_transit_time=True,
        show_chargeable_weight=False,
        show_subtotals=False,
        hide_margin=False,
    ),
}

# Default SOP for unknown customers
DEFAULT_SOP = CustomerSOP(
    customer_name="Unknown Customer",
    margin_percent=15.0,
    flat_discount_percent=None,
    volume_discount_tiers=None,
    discount_before_margin=True,
    mode_restriction=None,
    origin_restriction=None,
    show_transit_time=True,
    show_chargeable_weight=False,
    show_subtotals=False,
    hide_margin=False,
)


# =============================================================================
# DESTINATION-BASED SURCHARGES
# =============================================================================

def get_destination_surcharges(destination: str) -> list[Surcharge]:
    """
    Get surcharges based on destination.

    Per SOP.md: Australia destination = +$150 biosecurity fee (all customers)
    """
    surcharges = []

    # Normalize destination
    dest_lower = destination.lower().strip() if destination else ""

    # Australia destinations
    australia_keywords = ["australia", "sydney", "melbourne", "brisbane", "perth", "adelaide"]
    if any(kw in dest_lower for kw in australia_keywords):
        surcharges.append(Surcharge(
            name="Australia Biosecurity Fee",
            amount=150.0,
            reason="Required biosecurity inspection for all Australia-bound shipments",
        ))

    return surcharges


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================

def identify_customer(email: str) -> str | None:
    """
    Identify customer name from email address.

    Returns customer name or None if not found.
    """
    if not email:
        return None

    # Extract domain from email
    email_lower = email.lower().strip()
    if "@" not in email_lower:
        return None

    domain = email_lower.split("@")[1]

    # Look up in customer map
    return CUSTOMER_EMAIL_MAP.get(domain)


def lookup_sop(email: str) -> tuple[str, CustomerSOP]:
    """
    Look up customer SOP from email address.

    Returns (customer_name, CustomerSOP).
    Falls back to default SOP for unknown customers.
    """
    customer_name = identify_customer(email)

    if customer_name and customer_name in CUSTOMER_SOPS:
        sop = CUSTOMER_SOPS[customer_name]
        return customer_name, sop

    # Unknown customer - use default SOP
    return "Unknown Customer", DEFAULT_SOP


def lookup_sop_with_surcharges(
    email: str,
    destinations: list[str],
) -> tuple[str, CustomerSOP, list[Surcharge]]:
    """
    Look up customer SOP and any destination-based surcharges.

    Args:
        email: Customer email address
        destinations: List of destination locations in the request

    Returns:
        (customer_name, CustomerSOP, list of Surcharges)
    """
    customer_name, sop = lookup_sop(email)

    # Collect surcharges from all destinations
    all_surcharges = []
    seen_surcharges = set()  # Avoid duplicates

    for dest in destinations:
        for surcharge in get_destination_surcharges(dest):
            if surcharge.name not in seen_surcharges:
                all_surcharges.append(surcharge)
                seen_surcharges.add(surcharge.name)

    return customer_name, sop, all_surcharges


def compare_with_qontext(
    local_sop: CustomerSOP,
    qontext_sop: CustomerSOP,
) -> list[str]:
    """
    Compare local SOP with Qontext response and return discrepancies.

    Returns list of discrepancy descriptions (empty if they match).
    """
    discrepancies = []

    # Compare key fields
    if local_sop.margin_percent != qontext_sop.margin_percent:
        discrepancies.append(
            f"Margin mismatch: local={local_sop.margin_percent}%, "
            f"qontext={qontext_sop.margin_percent}%"
        )

    if local_sop.flat_discount_percent != qontext_sop.flat_discount_percent:
        discrepancies.append(
            f"Discount mismatch: local={local_sop.flat_discount_percent}%, "
            f"qontext={qontext_sop.flat_discount_percent}%"
        )

    if local_sop.mode_restriction != qontext_sop.mode_restriction:
        discrepancies.append(
            f"Mode restriction mismatch: local={local_sop.mode_restriction}, "
            f"qontext={qontext_sop.mode_restriction}"
        )

    if local_sop.origin_restriction != qontext_sop.origin_restriction:
        discrepancies.append(
            f"Origin restriction mismatch: local={local_sop.origin_restriction}, "
            f"qontext={qontext_sop.origin_restriction}"
        )

    return discrepancies


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test lookups
    test_emails = [
        "sarah.chen@globalimports.com",
        "mike.johnson@techparts.io",
        "david.mueller@autospares.de",
        "tom.bradley@quickship.co.uk",
        "lisa.nguyen@vietexport.vn",
        "random@unknown.com",
    ]

    print("=" * 60)
    print("LOCAL SOP LOOKUP TEST")
    print("=" * 60)

    for email in test_emails:
        customer, sop = lookup_sop(email)
        print(f"\n{email}")
        print(f"  Customer: {customer}")
        print(f"  Margin: {sop.margin_percent}%")
        print(f"  Discount: {sop.flat_discount_percent}%") if sop.flat_discount_percent else None
        print(f"  Mode: {sop.mode_restriction or 'any'}")
        print(f"  Origin: {sop.origin_restriction or 'any'}")

    # Test surcharges
    print("\n" + "=" * 60)
    print("DESTINATION SURCHARGE TEST")
    print("=" * 60)

    test_destinations = ["Sydney", "Rotterdam", "Melbourne", "Los Angeles", "Australia"]
    for dest in test_destinations:
        surcharges = get_destination_surcharges(dest)
        if surcharges:
            print(f"\n{dest}: {[f'{s.name}: ${s.amount}' for s in surcharges]}")
        else:
            print(f"\n{dest}: No surcharges")
