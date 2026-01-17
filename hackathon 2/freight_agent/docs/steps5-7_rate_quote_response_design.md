# Steps 5-7: Rate Lookup, Quote Calculation & Response Formatting

**Date:** 2026-01-17
**Status:** Design Approved
**Author:** Claude + Jan

---

## Overview

This document describes the design for Steps 5-7 of the Freight Quote Agent pipeline:

| Step | Name | Type | Output |
|------|------|------|--------|
| 5 | Rate Lookup | Deterministic | `list[RateMatch \| None]` |
| 6 | Quote Calculation | Deterministic | `Quote` |
| 7 | Response Formatting | GPT Call #3 | `QuoteResponse` |

**Design Decisions:**
- Output format: Structured `Quote` dataclass (not raw email string)
- Rate sheets: Support ALL 3 difficulty levels (easy, medium, hard)
- Formatting: GPT-generated for natural tone matching
- Error handling: Graceful degradation (partial quotes OK)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STEPS 5-7 PIPELINE                               │
│                                                                         │
│   EnrichedRequest (from Step 4)                                         │
│       ↓                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ STEP 5: RATE LOOKUP (deterministic)                             │  │
│   │   rate_lookup/ → list[RateMatch | None]                         │  │
│   │   • Auto-detect sheet format (easy/medium/hard)                 │  │
│   │   • Parse to unified NormalizedRates format                     │  │
│   │   • Fuzzy match origins/destinations via aliases                │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│       ↓                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ STEP 6: QUOTE CALCULATION (deterministic)                       │  │
│   │   quote_calculator.py → Quote                                   │  │
│   │   • Base price × quantity                                       │  │
│   │   • Apply discounts (flat or volume-tiered)                     │  │
│   │   • Apply margin (respecting discount_before_margin flag)       │  │
│   │   • Add surcharges                                              │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│       ↓                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ STEP 7: FORMAT RESPONSE (GPT #3)                                │  │
│   │   response_formatter.py → QuoteResponse                         │  │
│   │   • Natural language email reply                                │  │
│   │   • Tone matching to customer's style                           │  │
│   │   • Graceful error explanations                                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│       ↓                                                                 │
│   QuoteResponse (ready to send!)                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## New Data Models

Add to `src/models.py`:

```python
@dataclass(frozen=True)
class RateMatch:
    """Result of looking up a rate in the Excel sheets"""
    origin: str                    # Normalized origin used for lookup
    destination: str               # Normalized destination used for lookup
    mode: Literal["sea", "air"]

    # Sea freight
    rate_per_container: float | None = None
    container_size_ft: int | None = None

    # Air freight
    rate_per_kg: float | None = None
    min_charge: float | None = None
    chargeable_weight_kg: float | None = None

    transit_days: int | None = None
    currency: str = "USD"

    # Metadata
    source_sheet: str | None = None       # "easy", "medium", "hard"
    matched_origin_alias: str | None = None
    matched_dest_alias: str | None = None


@dataclass(frozen=True)
class QuoteLineItem:
    """One line in the quote (one shipment)"""
    shipment_index: int
    description: str               # "Shanghai → Rotterdam, 2x 40ft"

    rate_match: RateMatch | None   # None if no rate found
    base_price: float | None
    discount_amount: float | None
    margin_amount: float | None
    surcharge_total: float | None
    line_total: float | None

    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quote:
    """Complete quote ready for formatting"""
    customer_name: str
    customer_email: str

    line_items: tuple[QuoteLineItem, ...]

    subtotal: float | None
    total_surcharges: float | None
    grand_total: float | None

    # Display flags from SOP
    show_transit_time: bool
    show_chargeable_weight: bool
    show_subtotals: bool
    hide_margin: bool

    # Status
    is_complete: bool
    has_warnings: bool
    has_errors: bool

    created_at: str


@dataclass(frozen=True)
class QuoteResponse:
    """Final formatted response ready to send"""
    subject: str
    body: str
    quote: Quote
    generated_at: str
    model_used: str
```

---

## Step 5: Rate Lookup

### File Structure

```
src/
├── rate_lookup/
│   ├── __init__.py
│   ├── detector.py      # Auto-detect format from Excel structure
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── easy.py      # Parse clean flat tables
│   │   ├── medium.py    # Parse multi-sheet with port codes
│   │   └── hard.py      # Parse messy data (ditto marks, etc.)
│   ├── models.py        # NormalizedRates internal model
│   └── service.py       # RateLookupService - main interface
```

### Format Detection

```python
def detect_format(excel_path: Path) -> Literal["easy", "medium", "hard"]:
    """Analyze Excel structure to determine format"""

    xl = pd.ExcelFile(excel_path)
    sheet_names = [s.lower() for s in xl.sheet_names]

    # Medium: Has port codes sheet
    if "port codes" in sheet_names or "codes" in sheet_names:
        return "medium"

    # Hard: Check for ditto marks or messy headers
    df = pd.read_excel(xl, sheet_name=0)
    ditto_patterns = ["''", '\"', "ditto", "-"]
    first_col = df.iloc[:, 0].astype(str)
    if first_col.str.contains('|'.join(ditto_patterns)).any():
        return "hard"

    # Also check for section headers (indicates hard format)
    if "GLOBAL FREIGHT" in str(df.iloc[0, 0]).upper():
        return "hard"

    return "easy"
```

### Unified Internal Format

```python
@dataclass
class NormalizedRates:
    """All rate sheets normalize to this format for lookup"""

    sea_rates: pd.DataFrame
    # Columns: origin, destination, rate_20ft, rate_40ft, transit_days

    air_rates: pd.DataFrame
    # Columns: origin, destination, rate_per_kg, min_charge, transit_days

    aliases: dict[str, list[str]]
    # {"ho chi minh city": ["hcmc", "saigon", "sgn"], ...}
```

### Parsing by Format

**Easy:** Direct column rename
```python
def parse_easy(excel_path: Path) -> NormalizedRates:
    sea = pd.read_excel(excel_path, sheet_name="Sea Freight Rates")
    air = pd.read_excel(excel_path, sheet_name="Air Freight Rates")
    # Standardize column names
    return NormalizedRates(sea_rates=sea, air_rates=air, aliases={})
```

**Medium:** JOIN port codes + extract aliases
```python
def parse_medium(excel_path: Path) -> NormalizedRates:
    codes = pd.read_excel(excel_path, sheet_name="Port Codes")
    sea = pd.read_excel(excel_path, sheet_name="Sea Rates")
    air = pd.read_excel(excel_path, sheet_name="Air Rates")

    # Build alias map from Aliases column
    aliases = {}
    for _, row in codes.iterrows():
        port_name = row["Port Name"].lower()
        alias_list = [a.strip().lower() for a in str(row["Aliases"]).split(",")]
        aliases[port_name] = alias_list + [row["Code"].lower()]

    # Merge codes into rates
    # ...
    return NormalizedRates(sea_rates=merged_sea, air_rates=merged_air, aliases=aliases)
```

**Hard:** Handle messy real-world data
```python
def parse_hard(excel_path: Path) -> NormalizedRates:
    df = pd.read_excel(excel_path, sheet_name="Master Rate Card Q1")

    # 1. Skip header rows until we hit actual data (look for "POL" header)
    # 2. Fill ditto marks ('', ", -, NaN) with value from row above
    # 3. Strip 'd' suffix from transit times ("28d" → 28)
    # 4. Handle section headers (ASIA - EUROPE, etc.) - skip these
    # 5. Parse combined ports (Gdansk/Gdynia → expand or match either)
    # 6. Extract inline aliases from notes (*Also: Saigon, HCMC)

    return NormalizedRates(sea_rates=cleaned_sea, air_rates=cleaned_air, aliases=extracted_aliases)
```

### Rate Sheet Data Analysis

**Easy (`01_rates_easy.xlsx`):**
- Sheets: `Sea Freight Rates`, `Air Freight Rates`
- Clean columns, direct lookup

**Medium (`02_rates_medium.xlsx`):**
- Sheets: `Port Codes`, `Sea Rates`, `Air Rates`
- Port Codes contains aliases: `SGN | Ho Chi Minh City | HCMC, SAIGON, HOCHIMINH`

**Hard (`03_rates_hard.xlsx`):**
- Sheets: `Master Rate Card Q1`, `Air Freight`
- Challenges:
  - Header rows: `GLOBAL FREIGHT SOLUTIONS - RATE CARD`
  - Section headers: `ASIA - EUROPE`, `ASIA - AMERICAS`
  - Ditto marks: `''`, `"`, `-`, empty cells
  - Transit format: `28d` (needs suffix strip)
  - Combined ports: `Gdansk/Gdynia`, `Yokohama/Tokyo`
  - Inline aliases: `HO CHI MINH*` with `*Also: Saigon, HCMC`

---

## Step 6: Quote Calculation

### File: `src/quote_calculator.py`

```python
def calculate_quote(
    enriched: EnrichedRequest,
    rate_matches: list[RateMatch | None],
) -> Quote:
    """Calculate complete quote with all pricing applied"""

    line_items = []

    for i, (enriched_shipment, rate_match) in enumerate(
        zip(enriched.shipments, rate_matches)
    ):
        shipment = enriched_shipment.shipment
        sop = enriched.customer_sop

        # No rate found - create error line item
        if rate_match is None:
            line_items.append(QuoteLineItem(
                shipment_index=i,
                description=f"{shipment.origin_raw} → {shipment.destination_raw}",
                rate_match=None,
                errors=("No rate found for this route",),
                # ... other fields None
            ))
            continue

        # STEP 1: Base Price
        if shipment.mode == "sea":
            base_price = rate_match.rate_per_container * shipment.quantity
        else:  # air
            base_price = rate_match.chargeable_weight_kg * rate_match.rate_per_kg
            if base_price < rate_match.min_charge:
                base_price = rate_match.min_charge

        # STEP 2: Discount
        discount_percent = calculate_discount(sop, shipment.quantity)

        # STEP 3: Margin (order depends on flag)
        if sop.discount_before_margin:
            discount_amount = base_price * (discount_percent / 100)
            after_discount = base_price - discount_amount
            margin_amount = after_discount * (sop.margin_percent / 100)
            subtotal = after_discount + margin_amount
        else:
            margin_amount = base_price * (sop.margin_percent / 100)
            after_margin = base_price + margin_amount
            discount_amount = after_margin * (discount_percent / 100)
            subtotal = after_margin - discount_amount

        # STEP 4: Surcharges
        surcharge_total = sum(s.amount for s in enriched_shipment.surcharges)
        line_total = subtotal + surcharge_total

        # Warnings
        warnings = []
        if sop.warn_transit_over_days and rate_match.transit_days:
            if rate_match.transit_days > sop.warn_transit_over_days:
                warnings.append(f"Transit time exceeds {sop.warn_transit_over_days} days")

        line_items.append(QuoteLineItem(...))

    return Quote(
        customer_name=enriched.customer_name,
        line_items=tuple(line_items),
        grand_total=sum(li.line_total for li in line_items if li.line_total),
        is_complete=all(li.rate_match is not None for li in line_items),
        # ... other fields
    )


def calculate_discount(sop: CustomerSOP, quantity: int) -> float:
    """Determine discount percentage based on SOP rules"""

    if sop.flat_discount_percent is not None:
        return sop.flat_discount_percent

    if sop.volume_discount_tiers:
        discount = 0.0
        for threshold, percent in sop.volume_discount_tiers:
            if quantity >= threshold:
                discount = percent
        return discount

    return 0.0
```

---

## Step 7: Response Formatting (GPT #3)

### File: `src/response_formatter.py`

**System Prompt:**
```
You are a freight quotation assistant. Generate a professional email reply.

DISPLAY RULES (from customer SOP):
- show_transit_time: Include transit days if true
- show_chargeable_weight: Show weight calc for air if true
- show_subtotals: Break down base/discount/margin if true
- hide_margin: Don't mention margin percentage if true

FORMATTING:
1. Warm greeting using customer's name
2. Reference their original request
3. Present quote clearly (one section per route)
4. Include WARNINGS prominently
5. Handle ERRORS gracefully (offer alternatives)
6. Professional sign-off

TONE:
Match the formality of the customer's original email.
```

**Implementation:**
```python
async def format_response(
    quote: Quote,
    original_email: Email,
    client: openai.AsyncOpenAI,
) -> QuoteResponse:
    """GPT call #3: Generate natural email response"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original: {original_email}\nQuote: {quote}"},
        ],
        temperature=0.7,
    )

    return QuoteResponse(
        subject=f"RE: {original_email.subject}",
        body=response.choices[0].message.content,
        quote=quote,
        generated_at=datetime.now().isoformat(),
        model_used="gpt-4o-mini",
    )
```

---

## Pipeline Integration

### File: `src/pipeline.py`

```python
async def process_email(
    email: Email,
    rate_sheet_path: Path,
    openai_client: openai.AsyncOpenAI,
    qontext_client: QontextClient,
) -> PipelineResult:
    """
    Main entry point: Email → QuoteResponse

    3 GPT calls total:
    1. Extraction (steps 1-2)
    2. Enrichment + Validation (steps 3-4)
    3. Response Formatting (step 7)
    """

    # Step 1-2: Extract
    extraction = await extract_shipments(email, openai_client)

    # Step 3-4: Enrich
    enriched = await enrich_request(extraction, qontext_client, openai_client)

    # Step 5: Rate Lookup
    rate_service = RateLookupService(rate_sheet_path)
    rate_matches = [
        rate_service.lookup(
            origin=s.shipment.origin_raw,
            destination=s.shipment.destination_raw,
            mode=s.shipment.mode,
            container_size_ft=s.shipment.container_size_ft,
            actual_weight_kg=s.shipment.actual_weight_kg,
            volume_cbm=s.shipment.volume_cbm,
        )
        for s in enriched.shipments
    ]

    # Step 6: Calculate
    quote = calculate_quote(enriched, rate_matches)

    # Step 7: Format
    response = await format_response(quote, email, openai_client)

    return PipelineResult(
        extraction=extraction,
        enriched=enriched,
        rate_matches=tuple(rate_matches),
        quote=quote,
        response=response,
    )
```

---

## Summary

| Component | File | Type |
|-----------|------|------|
| Models | `src/models.py` | Data structures |
| Format Detector | `src/rate_lookup/detector.py` | Auto-detect |
| Easy Parser | `src/rate_lookup/parsers/easy.py` | Direct load |
| Medium Parser | `src/rate_lookup/parsers/medium.py` | JOIN + aliases |
| Hard Parser | `src/rate_lookup/parsers/hard.py` | Messy data cleanup |
| Rate Service | `src/rate_lookup/service.py` | Main interface |
| Quote Calculator | `src/quote_calculator.py` | Pricing math |
| Response Formatter | `src/response_formatter.py` | GPT #3 |
| Pipeline | `src/pipeline.py` | Orchestration |

**Total GPT Calls: 3** (optimized from 5+)

---

## Next Steps

1. Implement rate lookup module (Step 5)
2. Implement quote calculator (Step 6)
3. Implement response formatter (Step 7)
4. Wire up pipeline orchestration
5. Test end-to-end with sample emails
