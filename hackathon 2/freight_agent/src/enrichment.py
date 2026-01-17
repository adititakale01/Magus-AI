"""
Enrichment v2: Batched + Tool Calling

This module handles enrichment AND validation in a single GPT call:
1. Batch all Qontext queries (REST API, no GPT cost)
2. Single GPT call to parse ALL context
3. GPT calls validate_shipment tool for deterministic checks
4. GPT handles fuzzy matching (names, locations)

Optimized from 3+ GPT calls to just 1!
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from models import (
    ExtractionResult,
    EnrichedRequest,
    EnrichedShipment,
    CustomerSOP,
    Surcharge,
    ValidationError,
    ValidationWarning,
)
from qontext_client import QontextClient

# Load environment variables
load_dotenv()


# ============================================================================
# CUSTOMER DOMAIN LOOKUP (fast, reliable, no Qontext needed)
# ============================================================================

CUSTOMER_DOMAINS = {
    "globalimports.com": "Global Imports Ltd",
    "techparts.io": "TechParts Inc",
    "autospares.de": "AutoSpares GmbH",
    "quickship.co.uk": "QuickShip UK",
    "vietexport.vn": "VietExport",
}


def get_customer_by_domain(email: str) -> str | None:
    """
    Fast internal lookup - no Qontext call needed!
    Returns customer name if domain matches, None if unknown.
    """
    if "@" not in email:
        return None
    domain = email.split("@")[1].lower()
    return CUSTOMER_DOMAINS.get(domain)


# ============================================================================
# VALIDATION TOOL (called by GPT)
# ============================================================================

VALIDATION_TOOL = {
    "type": "function",
    "function": {
        "name": "validate_shipment",
        "description": "Check if a shipment passes customer SOP restrictions. Call this for EACH shipment after parsing the SOP rules. You MUST normalize locations before calling (e.g., 'Saigon' -> 'HCMC').",
        "parameters": {
            "type": "object",
            "properties": {
                "shipment_index": {
                    "type": "integer",
                    "description": "Index of the shipment (0-based)"
                },
                "shipment_mode": {
                    "type": "string",
                    "enum": ["sea", "air"],
                    "description": "The shipping mode requested"
                },
                "normalized_origin": {
                    "type": "string",
                    "description": "Origin normalized to standard name (e.g., 'HCMC' not 'Saigon' or 'Ho Chi Minh City')"
                },
                "mode_restriction": {
                    "type": ["string", "null"],
                    "description": "Customer's mode restriction from SOP ('sea', 'air', or null if none)"
                },
                "origin_restriction": {
                    "type": ["string", "null"],
                    "description": "Customer's origin restriction from SOP (normalized, e.g., 'HCMC'), or null if none"
                },
                "customer_name": {
                    "type": "string",
                    "description": "Customer name for error messages"
                }
            },
            "required": ["shipment_index", "shipment_mode", "normalized_origin", "mode_restriction", "origin_restriction", "customer_name"]
        }
    }
}


def handle_validate_shipment(
    shipment_index: int,
    shipment_mode: str,
    normalized_origin: str,
    mode_restriction: str | None,
    origin_restriction: str | None,
    customer_name: str
) -> dict:
    """
    Deterministic validation - no fuzzy logic, just exact checks.
    GPT already normalized the values before calling.
    """
    errors = []

    # Check mode restriction
    if mode_restriction and shipment_mode != mode_restriction:
        errors.append({
            "error_type": "mode_restriction",
            "message": f"Per your account agreement, {customer_name} is set up for {mode_restriction} freight only.",
            "suggestion": f"Would you like a {mode_restriction} freight quote instead?",
            "shipment_index": shipment_index
        })

    # Check origin restriction (exact match after normalization)
    if origin_restriction:
        if normalized_origin.upper() != origin_restriction.upper():
            errors.append({
                "error_type": "origin_restriction",
                "message": f"Per your account agreement, {customer_name} shipments must originate from {origin_restriction}.",
                "suggestion": f"Would you like a quote from {origin_restriction} instead?",
                "shipment_index": shipment_index
            })

    return {
        "shipment_index": shipment_index,
        "is_valid": len(errors) == 0,
        "errors": errors
    }


# ============================================================================
# JSON SCHEMA FOR FINAL OUTPUT
# ============================================================================

ENRICHMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "customer_name": {
            "type": "string",
            "description": "Name of the customer company"
        },
        "customer_sop": {
            "type": "object",
            "additionalProperties": False,
            "description": "Parsed SOP rules for the customer",
            "properties": {
                "margin_percent": {"type": "number"},
                "flat_discount_percent": {"type": ["number", "null"]},
                "volume_discount_tiers": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "discount_before_margin": {"type": "boolean"},
                "mode_restriction": {"type": ["string", "null"]},
                "origin_restriction": {"type": ["string", "null"]},
                "origin_equivalences": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "show_transit_time": {"type": "boolean"},
                "show_chargeable_weight": {"type": "boolean"},
                "show_subtotals": {"type": "boolean"},
                "hide_margin": {"type": "boolean"},
                "warn_transit_over_days": {"type": ["integer", "null"]}
            },
            "required": [
                "margin_percent", "flat_discount_percent", "volume_discount_tiers",
                "discount_before_margin", "mode_restriction", "origin_restriction",
                "origin_equivalences", "show_transit_time", "show_chargeable_weight",
                "show_subtotals", "hide_margin", "warn_transit_over_days"
            ]
        },
        "shipment_surcharges": {
            "type": "array",
            "description": "Surcharges for each shipment (same order as input shipments)",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "amount": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["name", "amount", "reason"]
                }
            }
        }
    },
    "required": ["customer_name", "customer_sop", "shipment_surcharges"]
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are parsing freight SOP rules and validating shipments.

The customer has already been identified (or marked as unknown) - use the name provided.

TASKS:
1. Use the customer name provided (KNOWN CUSTOMER or "Unknown Customer")
2. Parse SOP rules from context into structured format
3. Parse any surcharges that apply to each destination
4. For EACH shipment, call the validate_shipment tool to check SOP restrictions

LOCATION NORMALIZATION (do this BEFORE calling validate_shipment):
- "Ho Chi Minh City", "Saigon", "SGN", "HCM" → normalize to "HCMC"
- "Shanghai", "Pudong", "PVG" → normalize to "Shanghai"
- "Ningbo", "Ningpo" → normalize to "Ningbo"
- Keep other locations as-is but standardize capitalization

DEFAULT VALUES (use for unknown customers OR if not in context):
- margin_percent: 15
- flat_discount_percent: null
- volume_discount_tiers: null
- discount_before_margin: true
- mode_restriction: null
- origin_restriction: null
- origin_equivalences: []
- show_transit_time: false
- show_chargeable_weight: false
- show_subtotals: false
- hide_margin: false
- warn_transit_over_days: null

IMPORTANT:
- Call validate_shipment ONCE per shipment
- Normalize locations before calling the tool
- The tool does exact string matching, so normalization is critical
- For unknown customers, use default SOP values with no restrictions"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_domain(email: str) -> str:
    """Extract domain from email address."""
    if "@" not in email:
        return ""
    return email.split("@")[1].lower()


def collect_qontext_context(
    extraction: ExtractionResult,
    qontext_client: QontextClient,
    customer_name: str | None = None,
) -> dict:
    """
    Batch Qontext queries for SOP rules and surcharges.
    Customer lookup is now done internally via CUSTOMER_DOMAINS - much faster!

    Args:
        extraction: The extraction result
        qontext_client: Qontext client
        customer_name: Known customer name from domain lookup (None = unknown)
    """
    # Query 1: Customer SOP (only if we have a known customer)
    sop_context = []
    if customer_name:
        sop_response = qontext_client.retrieve(
            f"What are all the rules, discounts, restrictions, and requirements for {customer_name}?",
            limit=15,
            depth=2
        )
        if sop_response.success:
            sop_context = sop_response.context

    # Query 2: Surcharges for each unique destination
    destinations = set()
    for shipment in extraction.shipments:
        if shipment.destination_raw:
            destinations.add(shipment.destination_raw)

    surcharge_responses = {}
    for dest in destinations:
        response = qontext_client.get_destination_rules(dest)
        if response.success and response.context:
            surcharge_responses[dest] = response.context

    return {
        "customer_name": customer_name,  # Already known from domain lookup!
        "sop_context": sop_context,
        "surcharge_context": surcharge_responses
    }


# ============================================================================
# MAIN ENRICHMENT FUNCTION
# ============================================================================

def enrich_request(
    extraction: ExtractionResult,
    openai_client: OpenAI | None = None,
    qontext_client: QontextClient | None = None
) -> EnrichedRequest:
    """
    Enrich and validate an extraction result in a single GPT call.

    This is the main entry point - combines enrichment + validation.

    Args:
        extraction: The extraction result from Step 1+2
        openai_client: Optional OpenAI client
        qontext_client: Optional Qontext client

    Returns:
        EnrichedRequest with customer info, SOPs, surcharges, AND validation results
    """
    # Initialize clients
    if openai_client is None:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if qontext_client is None:
        qontext_client = QontextClient()

    # Step 1: Fast internal customer lookup (no Qontext needed!)
    customer_name = get_customer_by_domain(extraction.sender_email)

    # Step 2: Get SOP rules and surcharges from Qontext
    context = collect_qontext_context(extraction, qontext_client, customer_name)

    # Step 3: Build the prompt with all context
    shipments_info = []
    for i, s in enumerate(extraction.shipments):
        shipments_info.append({
            "index": i,
            "mode": s.mode,
            "origin": s.origin_raw,
            "destination": s.destination_raw
        })

    # Build customer info section
    if customer_name:
        customer_section = f"KNOWN CUSTOMER: {customer_name}"
    else:
        customer_section = "UNKNOWN CUSTOMER (use default SOP values)"

    user_prompt = f"""Parse the following context and validate the shipments.

{customer_section}

QONTEXT - SOP RULES:
{json.dumps(context['sop_context'], indent=2)}

QONTEXT - DESTINATION SURCHARGES:
{json.dumps(context['surcharge_context'], indent=2)}

SHIPMENTS TO VALIDATE:
{json.dumps(shipments_info, indent=2)}

Remember to:
1. Use the customer name provided above (or "Unknown Customer" if unknown)
2. Parse SOP rules from context (or use defaults if unknown customer)
3. Parse surcharges for each shipment's destination
4. Call validate_shipment tool for EACH shipment (normalize locations first!)
5. Return the final structured output"""

    # Step 3: Call GPT with tool
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # First call - GPT will call the validate_shipment tool
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[VALIDATION_TOOL],
        tool_choice="auto",
        temperature=0
    )

    # Process tool calls
    validation_results = []
    assistant_message = response.choices[0].message

    while assistant_message.tool_calls:
        # Add assistant message with tool calls
        messages.append(assistant_message)

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "validate_shipment":
                args = json.loads(tool_call.function.arguments)
                result = handle_validate_shipment(**args)
                validation_results.append(result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        # Continue the conversation
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[VALIDATION_TOOL],
            tool_choice="auto",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "enrichment_result",
                    "strict": True,
                    "schema": ENRICHMENT_SCHEMA
                }
            },
            temperature=0
        )
        assistant_message = response.choices[0].message

    # SAFEGUARD: Force validation for any shipments GPT skipped
    validated_indices = {vr.get("shipment_index") for vr in validation_results}
    expected_indices = set(range(len(extraction.shipments)))
    unvalidated_indices = expected_indices - validated_indices

    while unvalidated_indices:
        # Force GPT to validate remaining shipments
        missing_list = sorted(unvalidated_indices)
        messages.append({
            "role": "user",
            "content": f"You missed validating shipment(s) at index {missing_list}. Call validate_shipment for each one now."
        })

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[VALIDATION_TOOL],
            tool_choice={"type": "function", "function": {"name": "validate_shipment"}},  # Force tool call
            temperature=0
        )
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "validate_shipment":
                    args = json.loads(tool_call.function.arguments)
                    result = handle_validate_shipment(**args)
                    validation_results.append(result)
                    validated_indices.add(args.get("shipment_index"))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

        unvalidated_indices = expected_indices - validated_indices

    # Final call to get JSON response after forced validations (if any were forced)
    if len(validation_results) > 0 and not assistant_message.content:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[VALIDATION_TOOL],
            tool_choice="none",  # No more tool calls, just give us the JSON
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "enrichment_result",
                    "strict": True,
                    "schema": ENRICHMENT_SCHEMA
                }
            },
            temperature=0
        )
        assistant_message = response.choices[0].message

    # Parse final response
    result_json = json.loads(assistant_message.content)

    # Build CustomerSOP
    sop_data = result_json["customer_sop"]
    customer_sop = CustomerSOP(
        customer_name=result_json["customer_name"],
        margin_percent=sop_data["margin_percent"],
        flat_discount_percent=sop_data["flat_discount_percent"],
        volume_discount_tiers=tuple(tuple(t) for t in sop_data["volume_discount_tiers"]) if sop_data["volume_discount_tiers"] else None,
        discount_before_margin=sop_data["discount_before_margin"],
        mode_restriction=sop_data["mode_restriction"],
        origin_restriction=sop_data["origin_restriction"],
        origin_equivalences=tuple(tuple(e) for e in sop_data["origin_equivalences"]) if sop_data["origin_equivalences"] else (),
        show_transit_time=sop_data["show_transit_time"],
        show_chargeable_weight=sop_data["show_chargeable_weight"],
        show_subtotals=sop_data["show_subtotals"],
        hide_margin=sop_data["hide_margin"],
        warn_transit_over_days=sop_data["warn_transit_over_days"]
    )

    # Build EnrichedShipments with surcharges
    enriched_shipments = []
    for i, shipment in enumerate(extraction.shipments):
        surcharges = ()
        if i < len(result_json["shipment_surcharges"]):
            surcharges = tuple(
                Surcharge(name=s["name"], amount=s["amount"], reason=s["reason"])
                for s in result_json["shipment_surcharges"][i]
            )
        enriched_shipments.append(EnrichedShipment(shipment=shipment, surcharges=surcharges))

    # Collect validation errors from tool results
    all_errors = []
    for vr in validation_results:
        for err in vr.get("errors", []):
            all_errors.append(ValidationError(
                error_type=err["error_type"],
                message=err["message"],
                suggestion=err["suggestion"],
                shipment_index=err.get("shipment_index")
            ))

    # Determine overall validity
    request_level_errors = [e for e in all_errors if e.shipment_index is None]
    if request_level_errors:
        is_valid = False
    else:
        errored_shipments = {e.shipment_index for e in all_errors if e.shipment_index is not None}
        valid_count = len(extraction.shipments) - len(errored_shipments)
        is_valid = valid_count > 0

    return EnrichedRequest(
        sender_email=extraction.sender_email,
        customer_name=result_json["customer_name"],
        customer_sop=customer_sop,
        shipments=tuple(enriched_shipments),
        is_valid=is_valid,
        validation_errors=tuple(all_errors),
        validation_warnings=(),  # Future: add warnings
        missing_fields=extraction.missing_fields,
        needs_clarification=extraction.needs_clarification
    )


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_enrichment():
    """Test the new batched enrichment with tool calling."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    from extraction import extract_from_file

    print("Testing Enrichment v2 (Batched + Tool Calling)")
    print("=" * 60)

    # Test 1: TechParts with valid air freight
    print("\n[Test 1] TechParts - Air freight (valid)")
    print("-" * 40)
    extraction = extract_from_file("../hackathon_data/emails/email_02.json")
    result = enrich_request(extraction)

    print(f"Customer: {result.customer_name}")
    print(f"Mode restriction: {result.customer_sop.mode_restriction}")
    print(f"is_valid: {result.is_valid}")
    print(f"Errors: {len(result.validation_errors)}")
    print(f"✅ PASS" if result.is_valid else f"❌ FAIL")

    # Test 2: Global Imports (sea-only customer)
    print("\n[Test 2] Global Imports - Sea freight")
    print("-" * 40)
    extraction = extract_from_file("../hackathon_data/emails/email_01.json")
    result = enrich_request(extraction)

    print(f"Customer: {result.customer_name}")
    print(f"Mode restriction: {result.customer_sop.mode_restriction}")
    print(f"Discount: {result.customer_sop.flat_discount_percent}%")
    print(f"is_valid: {result.is_valid}")
    print(f"Errors: {len(result.validation_errors)}")
    for err in result.validation_errors:
        print(f"  - {err.error_type}: {err.message}")


if __name__ == "__main__":
    test_enrichment()
