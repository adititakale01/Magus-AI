"""
Test Email 02 with Hard Rate Sheet + REAL SOP Enrichment

Email: email_02.json (TechParts - Air freight SFO -> Frankfurt)
Rate Sheet: 03_rates_hard.xlsx
SOP: TechParts Inc - Air only, 15% margin, no discount

Expected: $2,329 (with SOP applied - same as base since no discount)

Usage:
    cd freight_agent/src
    python ../tests/test_email02_hard.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extraction import load_email, extract_from_email
from enrichment import enrich_request
from rate_lookup import RateLookupService
from quote_calculator import calculate_quote


# Paths
BASE_DIR = Path(__file__).parent.parent.parent / "hackathon_data"
EMAIL_PATH = BASE_DIR / "emails" / "email_02.json"
RATE_SHEET = BASE_DIR / "rate_sheets" / "03_rates_hard.xlsx"

# Expected values from solution
# TechParts SOP: Air only, 15% margin, no discount
# So result should be same as base solution: $2,329
EXPECTED_TOTAL = 2329.0


def run_test():
    print("\n" + "=" * 60)
    print("TEST: Email 02 with Hard Rate Sheet + SOP")
    print("=" * 60)
    print(f"Email:      {EMAIL_PATH.name}")
    print(f"Rate Sheet: {RATE_SHEET.name}")
    print(f"Expected:   ${EXPECTED_TOTAL:.2f}")
    print("=" * 60)

    # Step 1: Load email
    print("\n[Step 1] Loading email...")
    email = load_email(EMAIL_PATH)
    print(f"  From: {email.sender}")
    print(f"  Subject: {email.subject}")

    # Step 2: Extract shipment details (GPT #1)
    print("\n[Step 2] Extracting shipment details...")
    extraction = extract_from_email(email)

    if not extraction.shipments:
        print("  ERROR: No shipments extracted!")
        return False

    shipment = extraction.shipments[0]
    print(f"  Mode: {shipment.mode}")
    print(f"  Origin: {shipment.origin_raw}")
    print(f"  Destination: {shipment.destination_raw}")
    print(f"  Weight: {shipment.actual_weight_kg} kg")
    print(f"  Volume: {shipment.volume_cbm} CBM")

    # Step 3: Enrich with SOP (GPT #2 - REAL enrichment!)
    print("\n[Step 3] Enriching with customer SOP...")
    enriched = enrich_request(extraction)

    print(f"  Customer: {enriched.customer_name}")
    print(f"  Mode restriction: {enriched.customer_sop.mode_restriction}")
    print(f"  Margin: {enriched.customer_sop.margin_percent}%")
    print(f"  Flat discount: {enriched.customer_sop.flat_discount_percent}")
    print(f"  Volume tiers: {enriched.customer_sop.volume_discount_tiers}")
    print(f"  is_valid: {enriched.is_valid}")

    if enriched.validation_errors:
        print(f"  Validation errors:")
        for err in enriched.validation_errors:
            print(f"    - {err.error_type}: {err.message}")

    # Step 4: Load rate service
    print("\n[Step 4] Loading rate sheet...")
    rate_service = RateLookupService(RATE_SHEET)
    print(f"  Detected format: {rate_service.format}")

    # Step 5: Look up rate
    print("\n[Step 5] Looking up rate...")
    rate_match = rate_service.lookup(
        origin=shipment.origin_raw or "",
        destination=shipment.destination_raw or "",
        mode=shipment.mode or "air",
        actual_weight_kg=shipment.actual_weight_kg,
        volume_cbm=shipment.volume_cbm,
    )

    if rate_match is None:
        print("  ERROR: No rate found!")
        print(f"  Tried: {shipment.origin_raw} -> {shipment.destination_raw}")
        return False

    print(f"  Found: {rate_match.origin} -> {rate_match.destination}")
    print(f"  Rate/kg: ${rate_match.rate_per_kg}")
    print(f"  Min charge: ${rate_match.min_charge}")
    print(f"  Chargeable weight: {rate_match.chargeable_weight_kg} kg")

    # Step 6: Calculate quote (with REAL SOP!)
    print("\n[Step 6] Calculating quote with SOP...")
    quote = calculate_quote(enriched, [rate_match])

    print(f"  Base price: ${quote.line_items[0].base_price:.2f}")
    print(f"  Discount: ${quote.line_items[0].discount_amount:.2f}")
    print(f"  Margin: ${quote.line_items[0].margin_amount:.2f}")
    print(f"  Grand Total: ${quote.grand_total:.2f}")

    # Step 7: Compare with expected
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Expected: ${EXPECTED_TOTAL:.2f}")
    print(f"  Actual:   ${quote.grand_total:.2f}")

    diff = abs(quote.grand_total - EXPECTED_TOTAL)
    tolerance = EXPECTED_TOTAL * 0.01  # 1% tolerance

    if diff <= tolerance:
        print(f"\n  PASS (diff: ${diff:.2f})")
        return True
    else:
        print(f"\n  FAIL (diff: ${diff:.2f})")
        return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    success = run_test()
    sys.exit(0 if success else 1)
