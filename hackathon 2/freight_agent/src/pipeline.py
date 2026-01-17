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

from models import Email, PipelineResult, calculate_confidence
from extraction import extract_from_email, load_email
from enrichment import enrich_request
from qontext_client import QontextClient
from rate_lookup import RateLookupService
from quote_calculator import calculate_quote
from response_formatter import format_response_sync, format_response_streaming_with_result


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

    # Calculate confidence score
    confidence = calculate_confidence(extraction, enriched, quote)
    print(f"[Pipeline] Confidence: {confidence.level.upper()} - {confidence.reason}")

    # Calculate total time
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    print(f"[Pipeline] Complete! Total time: {elapsed_ms}ms")

    return PipelineResult(
        extraction=extraction,
        enriched=enriched,
        rate_matches=tuple(rate_matches),
        quote=quote,
        response=response,
        confidence=confidence,
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


def process_email_streaming(
    email: Email,
    rate_sheet_path: Path,
    openai_client: OpenAI | None = None,
    qontext_client: QontextClient | None = None,
    on_chunk: callable = None,
) -> PipelineResult:
    """
    Streaming version of process_email.

    Same as process_email, but streams the response formatting step (GPT #3)
    for better perceived latency. Call on_chunk callback for each text chunk.

    Args:
        email: The customer's email request
        rate_sheet_path: Path to the Excel rate sheet
        openai_client: OpenAI client (created if not provided)
        qontext_client: Qontext client (created if not provided)
        on_chunk: Callback function called with each chunk of streamed text.
                  Signature: on_chunk(chunk: str) -> None
                  If None, chunks are printed to stdout.

    Returns:
        PipelineResult with all intermediate and final results
    """
    start_time = time.monotonic()

    # Default chunk handler: print to stdout
    if on_chunk is None:
        def on_chunk(chunk: str):
            print(chunk, end="", flush=True)

    # Initialize clients
    if openai_client is None:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if qontext_client is None:
        qontext_client = QontextClient()

    # =========================================================================
    # STEPS 1-6: Same as non-streaming version
    # =========================================================================
    print("[Pipeline] Step 1-2: Extracting shipment details...")
    extraction = extract_from_email(email, openai_client)

    if extraction.needs_clarification:
        print(f"[Pipeline] WARNING: Needs clarification - {extraction.missing_fields}")

    print("[Pipeline] Step 3-4: Enriching with customer context...")
    enriched = enrich_request(extraction, openai_client, qontext_client)

    if not enriched.is_valid:
        print(f"[Pipeline] WARNING: Validation errors - {enriched.validation_errors}")

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

    print("[Pipeline] Step 6: Calculating quote...")
    quote = calculate_quote(enriched, rate_matches)
    print(f"[Pipeline]   Grand total: ${quote.grand_total:.2f}" if quote.grand_total else "[Pipeline]   No valid quote")

    # =========================================================================
    # STEP 7: RESPONSE FORMATTING - STREAMING!
    # =========================================================================
    print("[Pipeline] Step 7: Formatting response (streaming)...")
    print("[Pipeline] " + "=" * 50)
    print()

    # Get streaming iterator and result getter
    stream, get_result = format_response_streaming_with_result(
        quote, email, openai_client
    )

    # Stream chunks to callback
    for chunk in stream():
        on_chunk(chunk)

    # Get final response after streaming completes
    response = get_result()

    print()
    print("[Pipeline] " + "=" * 50)
    print("[Pipeline] Response streaming complete!")

    # Calculate confidence score
    confidence = calculate_confidence(extraction, enriched, quote)
    print(f"[Pipeline] Confidence: {confidence.level.upper()} - {confidence.reason}")

    # Calculate total time
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    print(f"[Pipeline] Complete! Total time: {elapsed_ms}ms")

    return PipelineResult(
        extraction=extraction,
        enriched=enriched,
        rate_matches=tuple(rate_matches),
        quote=quote,
        response=response,
        confidence=confidence,
        processing_time_ms=elapsed_ms,
        gpt_calls=3,
    )


def process_email_file_streaming(
    email_path: Path,
    rate_sheet_path: Path,
    openai_client: OpenAI | None = None,
    qontext_client: QontextClient | None = None,
    on_chunk: callable = None,
) -> PipelineResult:
    """
    Convenience function to process an email from a JSON file with streaming.
    """
    email = load_email(email_path)
    return process_email_streaming(
        email, rate_sheet_path, openai_client, qontext_client, on_chunk
    )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Freight Quote Agent Pipeline")
    parser.add_argument("email", nargs="?", default="../../hackathon_data/emails/01_simple.json",
                        help="Path to email JSON file")
    parser.add_argument("rates", nargs="?", default="../../hackathon_data/rate_sheets/01_rates_easy.xlsx",
                        help="Path to rate sheet Excel file")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming output for response generation")
    args = parser.parse_args()

    email_path = Path(args.email)
    rate_path = Path(args.rates)
    use_streaming = args.stream

    print(f"\n{'='*60}")
    print("FREIGHT QUOTE AGENT - PIPELINE TEST")
    print(f"{'='*60}")
    print(f"Email: {email_path}")
    print(f"Rate Sheet: {rate_path}")
    print(f"Streaming: {'ENABLED' if use_streaming else 'DISABLED'}")
    print(f"{'='*60}\n")

    # Run pipeline (streaming or regular)
    if use_streaming:
        result = process_email_file_streaming(email_path, rate_path)
    else:
        result = process_email_file(email_path, rate_path)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCustomer: {result.quote.customer_name}")
    print(f"Complete: {result.quote.is_complete}")
    print(f"Grand Total: ${result.quote.grand_total:.2f}" if result.quote.grand_total else "Grand Total: N/A")
    print(f"Confidence: {result.confidence.level.upper()} - {result.confidence.reason}" if result.confidence else "Confidence: N/A")
    print(f"GPT Calls: {result.gpt_calls}")
    print(f"Processing Time: {result.processing_time_ms}ms")

    # Only print email body for non-streaming (streaming already printed it)
    if not use_streaming:
        print(f"\n{'='*60}")
        print("GENERATED EMAIL")
        print(f"{'='*60}")
        print(f"\nSubject: {result.response.subject}")
        print(f"\n{result.response.body}")
