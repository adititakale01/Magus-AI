"""
End-to-End Pipeline Tests

Runs all 10 emails through the pipeline and compares results
with expected solutions.

Usage:
    cd freight_agent/src
    python -m pytest ../tests/test_e2e_pipeline.py -v

Or run directly:
    cd freight_agent/tests
    python test_e2e_pipeline.py
"""

import re
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extraction import load_email, extract_from_email
from rate_lookup import RateLookupService
from quote_calculator import calculate_quote
from models import EnrichedRequest, EnrichedShipment, CustomerSOP, Shipment


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent / "hackathon_data"
EMAILS_DIR = BASE_DIR / "emails"
RATE_SHEETS_DIR = BASE_DIR / "rate_sheets"
SOLUTIONS_DIR = BASE_DIR / "solutions"

# Use easy rate sheet for testing (solutions are based on this)
RATE_SHEET = RATE_SHEETS_DIR / "01_rates_easy.xlsx"


# ============================================================================
# SOLUTION PARSER
# ============================================================================

def parse_solution(solution_path: Path) -> dict:
    """
    Parse a solution markdown file to extract expected values.

    Returns dict with:
        - expected_total: float or None (if incomplete)
        - is_incomplete: bool
        - origin: str
        - destination: str
        - mode: str or None
        - container_size: int or None
        - quantity: int or None
        - weight_kg: float or None
        - volume_cbm: float or None
    """
    content = solution_path.read_text(encoding="utf-8")

    result = {
        "expected_total": None,
        "is_incomplete": False,
        "origin": None,
        "destination": None,
        "mode": None,
        "container_size": None,
        "quantity": None,
        "weight_kg": None,
        "volume_cbm": None,
        "chargeable_weight": None,
    }

    # Check if incomplete
    if "INCOMPLETE" in content or "Cannot provide quote" in content:
        result["is_incomplete"] = True

    # Extract total quote (various formats)
    # Priority: "Total: $X USD" format first (most specific), then TOTAL QUOTE
    total_patterns = [
        r"Total:\s*\$?([\d,]+(?:\.\d+)?)\s*USD",  # "Total: $7,360 USD"
        r"TOTAL QUOTE:.*?=\s*\$?([\d,]+(?:\.\d+)?)\s*$",  # "TOTAL QUOTE: $3,680 × 2 = $7,360"
        r"TOTAL QUOTE:\s*\$?([\d,]+(?:\.\d+)?)\s*$",  # "TOTAL QUOTE: $2,329"
        r"Grand Total.*?\$?([\d,]+(?:\.\d+)?)",
    ]
    for pattern in total_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            result["expected_total"] = float(match.group(1).replace(",", ""))
            break

    # Extract origin from table
    origin_match = re.search(r"\|\s*Origin\s*\|\s*([^|]+)\s*\|", content)
    if origin_match:
        result["origin"] = origin_match.group(1).strip()

    # Extract destination from table
    dest_match = re.search(r"\|\s*Destination\s*\|\s*([^|]+)\s*\|", content)
    if dest_match:
        result["destination"] = dest_match.group(1).strip()

    # Extract mode
    mode_match = re.search(r"\|\s*Mode\s*\|\s*([^|]+)\s*\|", content)
    if mode_match:
        mode_val = mode_match.group(1).strip().lower()
        if "sea" in mode_val:
            result["mode"] = "sea"
        elif "air" in mode_val:
            result["mode"] = "air"

    # Extract container size
    size_match = re.search(r"\|\s*Container Size\s*\|\s*(\d+)ft\s*\|", content)
    if size_match:
        result["container_size"] = int(size_match.group(1))

    # Extract quantity
    qty_match = re.search(r"\|\s*Quantity\s*\|\s*(\d+)\s*\|", content)
    if qty_match:
        result["quantity"] = int(qty_match.group(1))

    # Extract weight
    weight_match = re.search(r"\|\s*(?:Actual )?Weight\s*\|\s*([\d.]+)\s*kg\s*\|", content)
    if weight_match:
        result["weight_kg"] = float(weight_match.group(1))

    # Extract volume
    vol_match = re.search(r"\|\s*Volume\s*\|\s*([\d.]+)\s*CBM\s*\|", content)
    if vol_match:
        result["volume_cbm"] = float(vol_match.group(1))

    # Extract chargeable weight
    chg_match = re.search(r"Chargeable weight\s*=.*?=\s*([\d.]+)\s*kg", content)
    if chg_match:
        result["chargeable_weight"] = float(chg_match.group(1))

    return result


# ============================================================================
# TEST HELPERS
# ============================================================================

def create_mock_enriched_request(extraction_result, email) -> EnrichedRequest:
    """
    Create a mock EnrichedRequest for testing without GPT enrichment call.
    Uses default 15% margin, no discounts (matching solution expectations).
    """
    # Default SOP (no customer-specific rules - matches solution format)
    default_sop = CustomerSOP(
        customer_name="Test Customer",
        margin_percent=15.0,
        flat_discount_percent=None,
        volume_discount_tiers=None,
        discount_before_margin=True,
        mode_restriction=None,
        origin_restriction=None,
        show_transit_time=True,
        show_chargeable_weight=True,
        show_subtotals=False,
        hide_margin=False,
    )

    # Wrap shipments
    enriched_shipments = tuple(
        EnrichedShipment(shipment=s, surcharges=())
        for s in extraction_result.shipments
    )

    return EnrichedRequest(
        sender_email=extraction_result.sender_email,
        customer_name="Test Customer",
        customer_sop=default_sop,
        shipments=enriched_shipments,
        is_valid=True,
        validation_errors=(),
        validation_warnings=(),
        missing_fields=extraction_result.missing_fields,
        needs_clarification=extraction_result.needs_clarification,
    )


def run_single_test(email_num: int, rate_service: RateLookupService) -> dict:
    """
    Run pipeline for a single email and compare with solution.

    Returns dict with test results.
    """
    email_path = EMAILS_DIR / f"email_{email_num:02d}.json"
    solution_path = SOLUTIONS_DIR / f"solution_email_{email_num:02d}.md"

    # Load email and solution
    email = load_email(email_path)
    solution = parse_solution(solution_path)

    # Step 1-2: Extraction
    extraction = extract_from_email(email)

    result = {
        "email_num": email_num,
        "email_path": str(email_path),
        "passed": False,
        "expected_total": solution["expected_total"],
        "actual_total": None,
        "expected_incomplete": solution["is_incomplete"],
        "actual_incomplete": extraction.needs_clarification,
        "extraction": {
            "shipments": len(extraction.shipments),
            "needs_clarification": extraction.needs_clarification,
            "missing_fields": list(extraction.missing_fields),
        },
        "errors": [],
    }

    # Check if extraction matches expected incomplete status
    if solution["is_incomplete"]:
        # For incomplete emails, we just check that we detected it
        if extraction.needs_clarification or len(extraction.missing_fields) > 0:
            result["passed"] = True
            result["notes"] = "Correctly identified as incomplete"
        else:
            result["errors"].append("Should have flagged as needing clarification")
        return result

    # If no shipments extracted, fail
    if not extraction.shipments:
        result["errors"].append("No shipments extracted")
        return result

    # Create mock enriched request (without GPT call)
    enriched = create_mock_enriched_request(extraction, email)

    # Step 5: Rate lookup
    rate_matches = []
    for enriched_shipment in enriched.shipments:
        shipment = enriched_shipment.shipment
        match = rate_service.lookup(
            origin=shipment.origin_raw or "",
            destination=shipment.destination_raw or "",
            mode=shipment.mode or "sea",
            container_size_ft=shipment.container_size_ft,
            actual_weight_kg=shipment.actual_weight_kg,
            volume_cbm=shipment.volume_cbm,
        )
        rate_matches.append(match)

    # Check if rates found
    rates_found = [m is not None for m in rate_matches]
    result["rates_found"] = rates_found

    if not any(rates_found):
        result["errors"].append("No rates found for any shipment")
        return result

    # Step 6: Calculate quote
    quote = calculate_quote(enriched, rate_matches)
    result["actual_total"] = quote.grand_total

    # Compare totals (allow 1% tolerance for rounding)
    if solution["expected_total"] and quote.grand_total:
        diff = abs(quote.grand_total - solution["expected_total"])
        tolerance = solution["expected_total"] * 0.01  # 1% tolerance

        if diff <= tolerance:
            result["passed"] = True
        else:
            result["errors"].append(
                f"Total mismatch: expected ${solution['expected_total']:.2f}, "
                f"got ${quote.grand_total:.2f} (diff: ${diff:.2f})"
            )
    elif quote.grand_total:
        # No expected total but we got one - partial pass
        result["passed"] = True
        result["notes"] = "Got total but no expected value to compare"

    return result


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run tests for all 10 emails and print results."""

    print("\n" + "=" * 70)
    print("FREIGHT QUOTE AGENT - END-TO-END TESTS")
    print("=" * 70)
    print(f"\nEmails:     {EMAILS_DIR}")
    print(f"Rate Sheet: {RATE_SHEET}")
    print(f"Solutions:  {SOLUTIONS_DIR}")
    print("=" * 70)

    # Load rate service once
    rate_service = RateLookupService(RATE_SHEET)
    print(f"\nRate sheet format detected: {rate_service.format}")

    # Run tests
    results = []
    passed = 0
    failed = 0

    for i in range(1, 11):
        print(f"\n--- Email {i:02d} ---")
        try:
            result = run_single_test(i, rate_service)
            results.append(result)

            if result["passed"]:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            # Print summary
            print(f"Status: {status}")

            if result.get("expected_incomplete"):
                print(f"  Type: Incomplete request")
                print(f"  Detected: {result['actual_incomplete']}")
            else:
                print(f"  Expected: ${result['expected_total']:.2f}" if result['expected_total'] else "  Expected: N/A")
                print(f"  Actual:   ${result['actual_total']:.2f}" if result['actual_total'] else "  Actual: N/A")

            if result.get("notes"):
                print(f"  Notes: {result['notes']}")

            for error in result.get("errors", []):
                print(f"  ERROR: {error}")

        except Exception as e:
            failed += 1
            print(f"Status: ERROR")
            print(f"  Exception: {e}")
            results.append({
                "email_num": i,
                "passed": False,
                "errors": [str(e)],
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal:  {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Rate:   {passed/len(results)*100:.1f}%")

    # List failures
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["passed"]:
                print(f"  - Email {r['email_num']:02d}: {r.get('errors', ['Unknown error'])}")

    print("\n" + "=" * 70)

    return results


# ============================================================================
# PYTEST INTEGRATION (only when pytest is available)
# ============================================================================

try:
    import pytest

    @pytest.fixture(scope="module")
    def rate_service():
        """Load rate service once for all tests."""
        return RateLookupService(RATE_SHEET)

    @pytest.mark.parametrize("email_num", range(1, 11))
    def test_email(email_num, rate_service):
        """Test each email against its expected solution."""
        result = run_single_test(email_num, rate_service)

        if result["errors"]:
            pytest.fail(f"Email {email_num:02d}: {result['errors']}")

        assert result["passed"], f"Email {email_num:02d} did not pass"

except ImportError:
    # pytest not available - skip test definitions
    pass


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    results = run_all_tests()

    # Exit with error code if any failed
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failed > 0 else 0)
