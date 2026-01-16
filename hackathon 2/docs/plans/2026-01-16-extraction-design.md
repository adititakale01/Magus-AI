# Freight Agent - Step 1+2: Extraction Design

**Date:** 2026-01-16
**Status:** Ready to implement

---

## Overview

Build a GPT-powered extraction step that reads freight quote request emails and outputs structured data.

**Approach:** Use OpenAI GPT to parse emails into a structured schema. Keep extracted data raw (no normalization) - fuzzy matching happens in later steps.

---

## Input

Raw email JSON from `hackathon_data/emails/`:

```json
{
    "from": "sarah.chen@globalimports.com",
    "to": "quotes@freightco.com",
    "subject": "Quote Request: Shanghai to Rotterdam",
    "body": "Hi,\n\nWe need a quote for:\n\nOrigin: Shanghai\nDestination: Rotterdam\nContainer: 2 x 40ft\nCommodity: Electronics\n\nPlease send your best rate.\n\nThanks,\nSarah"
}
```

---

## Output Schema

```python
{
    "sender_email": str,           # From email "from" field - needed for SOP lookup

    "shipments": [                 # Array - emails can have multiple routes (e.g., email_06)
        {
            "mode": "sea" | "air" | null,      # Inferred from context

            # Location (raw - no normalization yet)
            "origin_raw": str | null,           # "HCMC (Saigon)", "ningbo", etc.
            "destination_raw": str | null,      # "Tokyo Narita", "felixstowe", etc.

            # Sea freight specific
            "container_size_ft": 20 | 40 | null,
            "quantity": int | null,             # Number of containers

            # Air freight specific
            "actual_weight_kg": float | null,
            "volume_cbm": float | null,

            # Optional
            "commodity": str | null
        }
    ],

    "missing_fields": list[str],   # ["origin city", "container size", "mode"]
    "needs_clarification": bool    # True if we can't quote without more info
}
```

---

## Mode Detection Logic

GPT should infer mode from these signals:

| Signal | Mode |
|--------|------|
| "container", "20ft", "40ft", "FCL" | Sea |
| "kg", "weight", "CBM", "volume" | Air |
| "ocean", "sea freight" | Sea |
| "air", "air freight", "cargo" | Air |
| Airport codes (SFO, FRA, NRT) | Air |
| Port names only | Sea |

If unclear → set `mode: null` and add to `missing_fields`.

---

## Multi-Route Handling

Email 06 example has multiple routes in one request:
```
Rates from Busan, South Korea to:
1. Hamburg - 2 x 40ft
2. Rotterdam - 1 x 20ft
```

GPT must return multiple shipment objects in the `shipments` array.

---

## Missing Information Detection

If any of these are missing, add to `missing_fields`:

**Sea freight requires:**
- origin (specific city/port, not just "China")
- destination (specific city/port)
- container_size_ft (20 or 40)
- quantity

**Air freight requires:**
- origin
- destination
- actual_weight_kg
- volume_cbm

**Email 03 example** ("ship from China to Poland"):
```python
{
    "sender_email": "anna.kowalski@eurotrade.pl",
    "shipments": [{
        "mode": null,
        "origin_raw": "China",       # Too vague!
        "destination_raw": "Poland", # Too vague!
        ...
    }],
    "missing_fields": ["origin city", "destination city", "mode", "container size", "quantity"],
    "needs_clarification": true
}
```

---

## Implementation Plan

1. **models.py** - Define dataclasses: `Email`, `Shipment`, `ExtractionResult`
2. **extraction.py** - GPT extraction function:
   - Load email JSON
   - Build prompt with schema
   - Call OpenAI API with structured output
   - Parse response into dataclasses
3. **test_extraction.py** - Test against all 10 emails, compare to expected outputs

---

## GPT Prompt Structure

```
System: You are a freight quote extraction assistant. Extract shipping request details from emails.

User: Extract shipment details from this email:
From: {sender}
Subject: {subject}
Body: {body}

Return JSON matching this schema: {schema}

Rules:
- Extract ALL routes if multiple are mentioned
- Keep location names exactly as written (no normalization)
- Infer mode from context (container=sea, kg/CBM=air)
- Set needs_clarification=true if origin/destination are too vague (just country names)
```

---

## Success Criteria

- [ ] Correctly extracts all 10 hackathon emails
- [ ] Multi-route email (06) returns multiple shipments
- [ ] Incomplete email (03) sets `needs_clarification: true`
- [ ] Fuzzy locations kept raw: "HCMC (Saigon)" not normalized yet
- [ ] Mode correctly inferred for all emails

---

## Next Step

After extraction is built and tested, move to **Step 3: Customer Identification** (SOP lookup by sender email).
