# Step 3: Enrichment Module Design

## Overview

The Enrichment module takes an `ExtractionResult` (raw shipment data from email) and enriches it with:
- Customer identification (from email domain)
- Customer-specific rules/SOPs (from Qontext)
- Destination-based surcharges (from Qontext)

## Data Structures

### CustomerSOP
Holds all parsed rules for a customer.

```python
@dataclass(frozen=True)
class CustomerSOP:
    customer_name: str

    # Pricing
    margin_percent: float = 15.0              # Default 15%, QuickShip gets 8%
    flat_discount_percent: float | None = None # Global Imports: 10%
    volume_discount_tiers: tuple[tuple[int, float], ...] | None = None
    # AutoSpares: ((2, 5.0), (5, 12.0)) means 2+ containers = 5%, 5+ = 12%
    discount_before_margin: bool = True

    # Restrictions
    mode_restriction: Literal["sea", "air"] | None = None  # Global=sea, TechParts=air
    origin_restriction: str | None = None                   # VietExport: "HCMC"
    origin_equivalences: tuple[tuple[str, str], ...] = ()   # Global: (("Shanghai", "Ningbo"),)

    # Output formatting
    show_transit_time: bool = False      # Global: True
    show_chargeable_weight: bool = False # TechParts: True
    show_subtotals: bool = False         # AutoSpares: True
    hide_margin: bool = False            # QuickShip: True
    warn_transit_over_days: int | None = None  # TechParts: 3
```

### Surcharge
A surcharge that applies to a specific shipment.

```python
@dataclass(frozen=True)
class Surcharge:
    name: str           # "Australia Biosecurity"
    amount: float       # 150.0
    reason: str         # "Destination is Australia"
```

### EnrichedShipment
A shipment with its applicable surcharges.

```python
@dataclass(frozen=True)
class EnrichedShipment:
    shipment: Shipment
    surcharges: tuple[Surcharge, ...] = ()
```

### EnrichedRequest
The fully enriched request ready for validation & calculation.

```python
@dataclass(frozen=True)
class EnrichedRequest:
    sender_email: str
    customer_name: str
    customer_sop: CustomerSOP
    shipments: tuple[EnrichedShipment, ...]

    # Carried forward for debugging
    missing_fields: tuple[str, ...] = ()
    needs_clarification: bool = False
```

## Flow Diagram

```
ExtractionResult
       │
       ▼
┌──────────────────────────────────────────┐
│           ENRICHMENT MODULE              │
├──────────────────────────────────────────┤
│                                          │
│  1. Extract domain from sender_email     │
│     sarah.chen@globalimports.com         │
│              ↓                           │
│     domain = "globalimports.com"         │
│                                          │
│  2. Query Qontext for customer           │
│     "What customer uses @globalimports?" │
│              ↓                           │
│     customer_name = "Global Imports Ltd" │
│                                          │
│  3. Query Qontext for rules              │
│     "What are rules for Global Imports?" │
│              ↓                           │
│     raw_rules = ["10% discount...",      │
│                  "sea only...", ...]     │
│                                          │
│  4. GPT parses rules → CustomerSOP       │
│     (using Structured Outputs)           │
│              ↓                           │
│     CustomerSOP(margin=15,               │
│                 discount=10, ...)        │
│                                          │
│  5. For each shipment:                   │
│     Query destination surcharges         │
│     "Surcharges for Australia?"          │
│              ↓                           │
│     Surcharge(name="Biosecurity",        │
│               amount=150)                │
│                                          │
└──────────────────────────────────────────┘
       │
       ▼
EnrichedRequest
```

## GPT Structured Outputs

We use OpenAI's Structured Outputs feature (`response_format` with `json_schema`) to guarantee GPT returns data matching our exact schema.

### Why Structured Outputs?
- **Guaranteed valid JSON** - No parsing errors
- **Schema compliance** - Output matches our dataclass exactly
- **Constrained decoding** - Model literally can't output invalid tokens

### JSON Schema for CustomerSOP

```json
{
  "name": "CustomerSOPSchema",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "customer_name": {"type": "string"},
      "margin_percent": {"type": "number"},
      "flat_discount_percent": {"type": ["number", "null"]},
      "volume_discount_tiers": {
        "type": ["array", "null"],
        "items": {
          "type": "array",
          "items": [{"type": "integer"}, {"type": "number"}]
        }
      },
      "discount_before_margin": {"type": "boolean"},
      "mode_restriction": {"type": ["string", "null"], "enum": ["sea", "air", null]},
      "origin_restriction": {"type": ["string", "null"]},
      "origin_equivalences": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "show_transit_time": {"type": "boolean"},
      "show_chargeable_weight": {"type": "boolean"},
      "show_subtotals": {"type": "boolean"},
      "hide_margin": {"type": "boolean"},
      "warn_transit_over_days": {"type": ["integer", "null"]}
    },
    "required": [...all fields...],
    "additionalProperties": false
  }
}
```

## Edge Cases

| Scenario | Handling |
|----------|----------|
| **Unknown customer** | Return default SOP (15% margin, no discounts) |
| **Qontext returns nothing** | Same as unknown - use defaults |
| **No destination surcharges** | Empty surcharges tuple |
| **Multiple surcharges** | All get added to shipment |

## Customer Email Domain Mapping

From SOP.md, these are the known customer domains:

| Customer | Email Domain |
|----------|--------------|
| Global Imports Ltd | globalimports.com |
| TechParts Inc | techparts.io |
| AutoSpares GmbH | autospares.de |
| QuickShip UK | quickship.co.uk |
| VietExport | vietexport.vn |

Unknown domains get default rules (15% margin, no discounts, no restrictions).

## File Structure

```
freight_agent/src/
├── models.py        # Add CustomerSOP, Surcharge, EnrichedShipment, EnrichedRequest
├── extraction.py    # (already done)
├── enrichment.py    # NEW - the enrichment logic
└── qontext_client.py # (already done)
```
