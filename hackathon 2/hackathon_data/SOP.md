# Customer-Specific SOPs — Freight Quotation Rules & Exceptions

This document defines customer-specific rules that must be applied when generating freight quotations. If a customer SOP conflicts with default rules, the SOP takes precedence. SOPs may override pricing, restrict service types, or require special handling and quote formatting.

---

## General Application Order (unless overridden)

1. Parse the request: identify customer (by sender email), mode (sea/air), origin/destination, container size or weight, quantity, and whether it is multi-route.
2. Look up the base rate from the rate sheets.
3. Apply the customer SOP:
   - Mode restrictions (sea-only / air-only)
   - Location equivalence rules (e.g., Shanghai ↔ Ningbo)
   - Discounts/surcharges/margin rules (including ordering: discount before margin, etc.)
   - Output requirements (transit time, chargeable weight display, warnings, subtotals)
4. Calculate totals and format the quote accordingly.

---

## SOP 1: Global Imports Ltd

Customer Email: sarah.chen@globalimports.com  
Account Type: Strategic Partner  
Discount: 10% off all sea freight rates

Rules
- Apply the 10% discount **before** adding margin (discount applies to the base rate).
- Sea freight **only** — do not quote air freight.
- Shanghai and Ningbo are interchangeable as origins for rate matching.
- Always include transit time in the response.

Calculation Example
- Base rate: $3,200  
- After 10% discount: $2,880  
- After 15% margin: $3,312

---

## SOP 2: TechParts Inc

Customer Email: mike.johnson@techparts.io  
Account Type: Premium Air  
Restriction: Air freight only

Rules
- Air freight **only** — never quote sea freight.
- If transit time exceeds 3 days, add a warning to the quote.
- Always show both actual weight and chargeable weight in the quote.
- Apply the standard 15% margin (no discount).

---

## SOP 3: AutoSpares GmbH

Customer Email: david.mueller@autospares.de  
Account Type: Volume Shipper  
Discount: 5% off for 2+ containers, 12% off for 5+ containers

Rules
- Apply volume discount based on **total containers across all routes** in the request:
  - 1 container: No discount
  - 2–4 containers: 5% discount
  - 5+ containers: 12% discount
- For multi-route requests, list each route separately in the quote.
- Show subtotal per route **and** a grand total.

---

## Quick Reference

| Customer / Scope | Special Pricing | Restrictions / Notes |
|---|---:|---|
| Global Imports | 10% discount | Sea only |
| TechParts | Standard | Air only |
| AutoSpares | 5–12% volume discount | None |
| QuickShip UK | 8% margin (broker) | Hide margin % |
| VietExport | Standard | HCMC origin only |
| Australia destination | +$150 biosecurity | All shipments |

---

## Appendix — Items Mentioned in Quick Reference (details pending)

The following items appear in the Quick Reference but do not yet have full SOP blocks above. Implement as supplemental rules until detailed SOPs are provided.

1. QuickShip UK  
   - Use 8% margin (broker model).  
   - Hide the margin percentage in the customer-facing quote.

2. VietExport  
   - Origin restriction: HCMC only.

3. Australia destination (applies to all customers)  
   - Add a $150 biosecurity surcharge for any shipment with destination in Australia.

---
