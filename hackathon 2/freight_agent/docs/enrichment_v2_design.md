# Enrichment v2: Batched + Tool Calling Design

## Overview

Refactored enrichment that:
1. Batches all Qontext queries (REST API, no GPT cost)
2. Single GPT call to parse ALL context
3. Uses tool calling for deterministic validation
4. GPT handles fuzzy matching (names, locations)

## Architecture

```
ExtractionResult
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  QONTEXT QUERIES (REST API - no GPT)                       │
│                                                             │
│  1. Query: "Customer with domain @{domain}?"                │
│  2. Query: "Rules for {customer}?"                          │
│  3. Query: "Surcharges for {destination}?" (for each dest)  │
│                                                             │
│  All responses collected as strings                         │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  GPT CALL #2 (with tool calling)                           │
│                                                             │
│  Input:                                                     │
│    - All Qontext responses (combined)                       │
│    - Shipment details (from extraction)                     │
│                                                             │
│  GPT Tasks:                                                 │
│    1. Parse customer name from context                      │
│    2. Parse SOP rules into structured format                │
│    3. Parse surcharges per destination                      │
│    4. Normalize locations (HCMC = Saigon = Ho Chi Minh)     │
│    5. Call validate_shipment tool for each shipment         │
│                                                             │
│  Output:                                                    │
│    - customer_name                                          │
│    - customer_sop (structured)                              │
│    - enriched_shipments (with surcharges)                   │
│    - validation_errors                                      │
│    - validation_warnings                                    │
│    - is_valid                                               │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
EnrichedAndValidatedRequest
```

## Tool Definition

```python
VALIDATION_TOOL = {
    "type": "function",
    "function": {
        "name": "validate_shipment",
        "description": "Check if a shipment passes customer SOP restrictions. Call this for EACH shipment after parsing the SOP rules.",
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
                    "description": "Origin normalized to standard name (e.g., 'HCMC' not 'Saigon', 'Ho Chi Minh City')"
                },
                "mode_restriction": {
                    "type": ["string", "null"],
                    "description": "Customer's mode restriction from SOP, or null if none"
                },
                "origin_restriction": {
                    "type": ["string", "null"],
                    "description": "Customer's origin restriction from SOP (normalized), or null if none"
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
```

## Tool Implementation

```python
def validate_shipment(
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
            "suggestion": f"Would you like a {mode_restriction} freight quote instead?"
        })

    # Check origin restriction
    if origin_restriction and normalized_origin.upper() != origin_restriction.upper():
        errors.append({
            "error_type": "origin_restriction",
            "message": f"Per your account agreement, {customer_name} shipments must originate from {origin_restriction}.",
            "suggestion": f"Would you like a quote from {origin_restriction} instead?"
        })

    return {
        "shipment_index": shipment_index,
        "is_valid": len(errors) == 0,
        "errors": errors
    }
```

## GPT System Prompt

```
You are parsing freight customer data from a knowledge graph and validating shipments.

TASKS:
1. Parse the customer name from the context
2. Parse the SOP rules (discounts, margins, restrictions, output requirements)
3. Parse any surcharges that apply to destinations
4. For each shipment, normalize the origin location to a standard name:
   - "Ho Chi Minh City", "Saigon", "SGN" → "HCMC"
   - "Shanghai", "Pudong" → "Shanghai"
   - etc.
5. Call the validate_shipment tool for EACH shipment to check restrictions

IMPORTANT:
- Normalize locations BEFORE calling the validation tool
- The tool does exact string matching, so normalization is critical
- Call the tool once per shipment
```

## Output Schema

```python
@dataclass(frozen=True)
class EnrichedAndValidatedRequest:
    """Combined enrichment + validation result."""
    sender_email: str
    customer_name: str
    customer_sop: CustomerSOP
    shipments: tuple[EnrichedShipment, ...]

    # Validation results
    is_valid: bool
    validation_errors: tuple[ValidationError, ...] = ()
    validation_warnings: tuple[ValidationWarning, ...] = ()

    # Carried forward
    missing_fields: tuple[str, ...] = ()
    needs_clarification: bool = False
```

## Benefits

| Aspect | Before (3+ calls) | After (1 call + tools) |
|--------|-------------------|------------------------|
| GPT calls | 3+ | 1 |
| Location matching | Hardcoded | GPT (flexible) |
| Validation logic | GPT (might err) | Tool (deterministic) |
| Error messages | GPT (might vary) | Tool (consistent) |

## Flow Summary

```
Extraction (GPT #1)
       │
       ▼
Qontext queries (REST, free)
       │
       ▼
Enrichment + Validation (GPT #2 with tools)
       │
       ├─► GPT parses context
       ├─► GPT normalizes locations
       ├─► GPT calls validate_shipment tool (per shipment)
       └─► GPT compiles final result
       │
       ▼
EnrichedAndValidatedRequest
```
