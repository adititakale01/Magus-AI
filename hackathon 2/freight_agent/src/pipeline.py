"""
Pipeline Orchestrator

The main entry point that chains all steps together:
1. Extraction (GPT #1)
2. Enrichment + Validation (GPT #2 with tools)
3. Rate Lookup (deterministic)
4. Quote Calculation (deterministic)
5. Response Formatting (GPT #3)

Total: 3 GPT calls (optimized from 5+)
"""

import os
import time
from pathlib import Path

from openai import OpenAI

from models import Email, PipelineResult
from extraction import extract_from_email, load_email
from enrichment import enrich_request
from qontext_client import QontextClient
from rate_lookup import RateLookupService
from quote_calculator import calculate_quote
from response_formatter import format_response_sync


def process_email(
    email: Email,
    rate_sheet_path: Path,
    openai_client: OpenAI | None = None,
    qontext_client: QontextClient | None = None,
) -> PipelineResult:
    """
    Main entry point: Email -> QuoteResponse

    Processes a freight quote request through the complete pipeline:
    1. Extract shipment details from email (GPT #1)
    2. Enrich with customer context and validate (GPT #2)
    3. Look up rates in Excel sheets (deterministic)
    4. Calculate quote with discounts/margins (deterministic)
    5. Format response email (GPT #3)

    Args:
        email: The customer's email request
        rate_sheet_path: Path to the Excel rate sheet
        openai_client: OpenAI client (created if not provided)
        qontext_client: Qontext client (created if not provided)

    Returns:
        PipelineResult with all intermediate and final results
    """
    start_time = time.monotonic()

    # Initialize clients
    if openai_client is None:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if qontext_client is None:
        qontext_client = QontextClient()

    # =========================================================================
    # STEP 1-2: EXTRACTION (GPT #1)
    # =========================================================================
    print("[Pipeline] Step 1-2: Extracting shipment details...")
    extraction = extract_from_email(email, openai_client)

    if extraction.needs_clarification:
        print(f"[Pipeline] WARNING: Needs clarification - {extraction.missing_fields}")
        # TODO: Could generate clarification email here instead

    # =========================================================================
    # STEP 3-4: ENRICHMENT + VALIDATION (GPT #2)
    # =========================================================================
    print("[Pipeline] Step 3-4: Enriching with customer context...")
    enriched = enrich_request(extraction, openai_client, qontext_client)

    if not enriched.is_valid:
        print(f"[Pipeline] WARNING: Validation errors - {enriched.validation_errors}")
        # TODO: Could generate rejection email here instead

    # =========================================================================
    # STEP 5: RATE LOOKUP (deterministic)
    # =========================================================================
    print(f"[Pipeline] Step 5: Looking up rates in {rate_sheet_path.name}...")
    rate_service = RateLookupService(rate_sheet_path)
    print(f"[Pipeline] Detected format: {rate_service.format}")

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
        if match:
            print(f"[Pipeline]   Found rate: {match.origin} -> {match.destination}")
        else:
            print(f"[Pipeline]   No rate found for: {shipment.origin_raw} -> {shipment.destination_raw}")

    # =========================================================================
    # STEP 6: QUOTE CALCULATION (deterministic)
    # =========================================================================
    print("[Pipeline] Step 6: Calculating quote...")
    quote = calculate_quote(enriched, rate_matches)
    print(f"[Pipeline]   Grand total: ${quote.grand_total:.2f}" if quote.grand_total else "[Pipeline]   No valid quote")

    # =========================================================================
    # STEP 7: RESPONSE FORMATTING (GPT #3)
    # =========================================================================
    print("[Pipeline] Step 7: Formatting response...")
    response = format_response_sync(quote, email, openai_client)
    print("[Pipeline] Response generated!")

    # Calculate total time
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    print(f"[Pipeline] Complete! Total time: {elapsed_ms}ms")

    return PipelineResult(
        extraction=extraction,
        enriched=enriched,
        rate_matches=tuple(rate_matches),
        quote=quote,
        response=response,
        processing_time_ms=elapsed_ms,
        gpt_calls=3,
    )


def process_email_file(
    email_path: Path,
    rate_sheet_path: Path,
    openai_client: OpenAI | None = None,
    qontext_client: QontextClient | None = None,
) -> PipelineResult:
    """
    Convenience function to process an email from a JSON file.

    Args:
        email_path: Path to the email JSON file
        rate_sheet_path: Path to the Excel rate sheet
        openai_client: OpenAI client (created if not provided)
        qontext_client: Qontext client (created if not provided)

    Returns:
        PipelineResult with all intermediate and final results
    """
    email = load_email(email_path)
    return process_email(email, rate_sheet_path, openai_client, qontext_client)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Default paths
    default_email = Path("../../hackathon_data/emails/01_simple.json")
    default_rates = Path("../../hackathon_data/rate_sheets/01_rates_easy.xlsx")

    # Parse args
    email_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_email
    rate_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_rates

    print(f"\n{'='*60}")
    print("FREIGHT QUOTE AGENT - PIPELINE TEST")
    print(f"{'='*60}")
    print(f"Email: {email_path}")
    print(f"Rate Sheet: {rate_path}")
    print(f"{'='*60}\n")

    # Run pipeline
    result = process_email_file(email_path, rate_path)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCustomer: {result.quote.customer_name}")
    print(f"Complete: {result.quote.is_complete}")
    print(f"Grand Total: ${result.quote.grand_total:.2f}" if result.quote.grand_total else "Grand Total: N/A")
    print(f"GPT Calls: {result.gpt_calls}")
    print(f"Processing Time: {result.processing_time_ms}ms")

    print(f"\n{'='*60}")
    print("GENERATED EMAIL")
    print(f"{'='*60}")
    print(f"\nSubject: {result.response.subject}")
    print(f"\n{result.response.body}")
