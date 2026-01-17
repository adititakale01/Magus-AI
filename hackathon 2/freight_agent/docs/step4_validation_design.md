# Step 4: Validation Module Design

## Overview

The Validation module checks if an enriched request can be quoted based on customer SOPs.
It enforces restrictions (mode, origin) and collects warnings for the final quote.

## Data Structures

### ValidationError
Blocking error - cannot proceed with quote.

```python
@dataclass(frozen=True)
class ValidationError:
    error_type: str      # "mode_restriction", "origin_restriction", "missing_field"
    message: str         # User-friendly message referencing SOP
    suggestion: str      # What they can do instead
    shipment_index: int | None = None  # Which shipment (for multi-route)
```

### ValidationWarning
Non-blocking warning - include in quote but don't reject.

```python
@dataclass(frozen=True)
class ValidationWarning:
    warning_type: str    # "transit_time", etc.
    message: str         # Warning text for the quote
    shipment_index: int | None = None
```

### ValidationResult
Output of the validation step.

```python
@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool                           # False if any errors
    errors: tuple[ValidationError, ...] = ()
    warnings: tuple[ValidationWarning, ...] = ()
    request: EnrichedRequest | None = None   # Pass through if valid
```

## Validation Checks

In order:

1. **Missing Fields (request-level)**
   - Did extraction flag `needs_clarification`?
   - Any `missing_fields` from extraction?

2. **Mode Restriction (per shipment)**
   - Customer SOP says "sea only" but shipment is air?
   - Customer SOP says "air only" but shipment is sea?

3. **Origin Restriction (per shipment)**
   - VietExport: origin must be HCMC
   - Use fuzzy matching: "Ho Chi Minh" = "HCMC" = "HCM"

4. **Warnings (per shipment) - non-blocking**
   - TechParts: flag if transit > 3 days
   - (Deferred to Step 6 when we have transit data)

## Error Message Templates

| Check | Error Message | Suggestion |
|-------|---------------|------------|
| Mode restriction | "Per your account agreement, {customer} is set up for {allowed} freight only." | "Would you like a {allowed} freight quote instead?" |
| Origin restriction | "Per your account agreement, {customer} shipments must originate from {required_origin}." | "Would you like a quote from {required_origin} instead?" |
| Missing field | "We need additional information to provide a quote: {fields}" | "Please provide: {fields}" |

## Multi-Route Handling

For requests with multiple shipments:

```
Multi-Route Request: 3 shipments
  Shipment 1: Shanghai → Hamburg (sea)     ✅ Valid
  Shipment 2: Shanghai → Sydney (sea)      ✅ Valid
  Shipment 3: Shanghai → Tokyo (air)       ❌ Invalid (sea-only customer)

ValidationResult:
  is_valid: True  (some shipments are valid)
  errors: [ValidationError(shipment_index=2, ...)]
  request: EnrichedRequest with ALL shipments
```

**Key decisions:**
- `is_valid = True` if **at least one** shipment is valid
- `is_valid = False` only if **all** shipments are invalid
- Downstream steps skip invalid shipments but process valid ones

## Flow Diagram

```
EnrichedRequest
       │
       ▼
┌──────────────────────────────────────────────────┐
│              VALIDATION                          │
│                                                  │
│  validate_request(enriched: EnrichedRequest)    │
│       │                                          │
│       ├─► check_missing_fields()                │
│       ├─► check_mode_restrictions()             │
│       ├─► check_origin_restrictions()           │
│       └─► collect_warnings()                    │
│                                                  │
│  Returns: ValidationResult                       │
└──────────────────────────────────────────────────┘
       │
       ▼
   is_valid? ───No───► Format error response
       │                (reference SOP, give suggestions)
      Yes
       │
       ▼
   RATE LOOKUP (Step 5)
```

## File Structure

```
freight_agent/src/
├── models.py        # Add ValidationError, ValidationWarning, ValidationResult
├── extraction.py    # ✅ Done
├── enrichment.py    # ✅ Done
├── validation.py    # NEW - validation logic
└── qontext_client.py # ✅ Done
```
