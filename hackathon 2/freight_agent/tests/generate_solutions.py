"""
Generate Solution Files with Real SOP Calculations

This script runs the full pipeline for each email and generates
solution markdown files with accurate SOP-based calculations.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "src" / ".env")

from extraction import load_email, extract_from_email
from enrichment import enrich_request
from rate_lookup import RateLookupService
from quote_calculator import calculate_quote

# Force LOCAL SOP for consistent, reliable results
import enrichment
enrichment.USE_LOCAL_SOP = True
enrichment.VALIDATE_WITH_QONTEXT = False  # Skip Qontext entirely for speed

# Paths
BASE_DIR = Path(__file__).parent.parent.parent / "hackathon_data"
EMAILS_DIR = BASE_DIR / "emails"
RATE_SHEETS_DIR = BASE_DIR / "rate_sheets"
SOLUTIONS_DIR = BASE_DIR / "solutions_with_sops"  # New folder for SOP-based solutions

# Use easy rate sheet (same as original solutions)
RATE_SHEET = RATE_SHEETS_DIR / "01_rates_easy.xlsx"


def generate_solution(email_num: int, rate_service: RateLookupService) -> str:
    """Generate a solution markdown file for a single email."""

    email_path = EMAILS_DIR / f"email_{email_num:02d}.json"
    email = load_email(email_path)

    # Step 1-2: Extract
    extraction = extract_from_email(email)

    # Handle incomplete emails
    if extraction.needs_clarification or not extraction.shipments:
        return f"""# Solution: Email {email_num:02d} (with Real SOPs)

## Status: INCOMPLETE REQUEST

The email is missing required information and cannot be quoted.

**Missing Fields:** {', '.join(extraction.missing_fields) if extraction.missing_fields else 'Ambiguous request'}

## Required Action
Request clarification from the customer for the missing details.
"""

    # Step 3-4: Enrich with real SOPs
    enriched = enrich_request(extraction)

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

    # Step 6: Calculate quote
    quote = calculate_quote(enriched, rate_matches)

    # Build solution markdown
    md = f"""# Solution: Email {email_num:02d} (with Real SOPs)

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Customer Information
| Field | Value |
|-------|-------|
| Customer | {enriched.customer_name} |
| Email | {enriched.sender_email} |

## Customer SOP (from Qontext)
| Setting | Value |
|---------|-------|
| Margin | {enriched.customer_sop.margin_percent}% |
| Flat Discount | {enriched.customer_sop.flat_discount_percent or 'None'}{'%' if enriched.customer_sop.flat_discount_percent else ''} |
| Volume Discount Tiers | {enriched.customer_sop.volume_discount_tiers or 'None'} |
| Mode Restriction | {enriched.customer_sop.mode_restriction or 'None'} |
| Origin Restriction | {enriched.customer_sop.origin_restriction or 'None'} |
| Discount Before Margin | {enriched.customer_sop.discount_before_margin} |

"""

    # Add validation errors if any
    if enriched.validation_errors:
        md += """## Validation Errors
"""
        for err in enriched.validation_errors:
            md += f"- **{err.error_type}**: {err.message}\n"
            md += f"  - Suggestion: {err.suggestion}\n"
        md += "\n"

    # Add shipment details
    md += """## Extracted Shipments
"""
    for i, es in enumerate(enriched.shipments):
        s = es.shipment
        md += f"""
### Shipment {i + 1}
| Field | Value |
|-------|-------|
| Mode | {s.mode} |
| Origin | {s.origin_raw} |
| Destination | {s.destination_raw} |
"""
        if s.mode == "sea":
            md += f"| Container | {s.quantity or 1}x {s.container_size_ft}ft |\n"
        else:
            md += f"| Weight | {s.actual_weight_kg} kg |\n"
            md += f"| Volume | {s.volume_cbm} CBM |\n"

        if es.surcharges:
            md += f"| Surcharges | {', '.join(f'{sc.name}: ${sc.amount}' for sc in es.surcharges)} |\n"

    # Add rate lookup results
    md += """
## Rate Lookup
"""
    for i, match in enumerate(rate_matches):
        if match:
            md += f"""
### Route {i + 1}: {match.origin} → {match.destination}
| Field | Value |
|-------|-------|
| Mode | {match.mode} |
"""
            if match.mode == "sea":
                md += f"| Rate per Container | ${match.rate_per_container:,.2f} |\n"
            else:
                md += f"| Rate per kg | ${match.rate_per_kg:,.2f} |\n"
                md += f"| Chargeable Weight | {match.chargeable_weight_kg} kg |\n"
            md += f"| Transit | {match.transit_days} days |\n"
        else:
            md += f"\n### Route {i + 1}: NO RATE FOUND\n"

    # Add calculation breakdown
    md += """
## Calculation
```
"""
    for i, li in enumerate(quote.line_items):
        md += f"--- Shipment {i + 1}: {li.description} ---\n"
        if li.rate_match:
            md += f"Base Price: ${li.base_price:,.2f}\n"
            if li.discount_amount and li.discount_amount > 0:
                md += f"Discount ({li.discount_reason or 'SOP'}): -${li.discount_amount:,.2f}\n"
            md += f"Margin ({enriched.customer_sop.margin_percent}%): +${li.margin_amount:,.2f}\n"
            if li.surcharge_total and li.surcharge_total > 0:
                md += f"Surcharges: +${li.surcharge_total:,.2f}\n"
            md += f"Line Total: ${li.line_total:,.2f}\n"
        else:
            md += "NO RATE FOUND\n"
            for err in li.errors:
                md += f"  Error: {err}\n"
        md += "\n"

    if quote.grand_total:
        md += f"GRAND TOTAL: ${quote.grand_total:,.2f}\n"
    else:
        md += "GRAND TOTAL: N/A (incomplete quote)\n"
    md += "```\n"

    # Add final quote response
    md += f"""
## Quote Response
```
Customer: {quote.customer_name}
"""
    for li in quote.line_items:
        if li.line_total:
            md += f"{li.description}: ${li.line_total:,.2f}\n"
        else:
            md += f"{li.description}: NO RATE\n"

    if quote.grand_total:
        md += f"\nTotal: ${quote.grand_total:,.2f} USD\n"
    md += "```\n"

    return md


def main():
    """Generate solution files for all 10 emails."""

    # Create output directory
    SOLUTIONS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING SOLUTION FILES WITH REAL SOPs")
    print("=" * 60)
    print(f"\nOutput: {SOLUTIONS_DIR}")
    print(f"Rate Sheet: {RATE_SHEET}")
    print()

    # Load rate service
    rate_service = RateLookupService(RATE_SHEET)

    for i in range(1, 11):
        print(f"[{i:02d}/10] Generating solution for email_{i:02d}...", end=" ")
        try:
            solution = generate_solution(i, rate_service)

            # Write to file
            output_path = SOLUTIONS_DIR / f"solution_email_{i:02d}.md"
            output_path.write_text(solution, encoding="utf-8")
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    print("=" * 60)
    print("DONE! Solution files generated in:")
    print(f"  {SOLUTIONS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
