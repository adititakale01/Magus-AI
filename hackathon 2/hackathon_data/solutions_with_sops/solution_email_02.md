# Solution: Email 02 (with Real SOPs)

Generated: 2026-01-17 12:19

## Customer Information
| Field | Value |
|-------|-------|
| Customer | TechParts Inc |
| Email | mike.johnson@techparts.io |

## Customer SOP (from Qontext)
| Setting | Value |
|---------|-------|
| Margin | 15.0% |
| Flat Discount | None |
| Volume Discount Tiers | None |
| Mode Restriction | air |
| Origin Restriction | None |
| Discount Before Margin | True |

## Extracted Shipments

### Shipment 1
| Field | Value |
|-------|-------|
| Mode | air |
| Origin | San Francisco (SFO) |
| Destination | Frankfurt (FRA) |
| Weight | 450 kg |
| Volume | 2 CBM |

## Rate Lookup

### Route 1: san francisco → frankfurt
| Field | Value |
|-------|-------|
| Mode | air |
| Rate per kg | $4.50 |
| Chargeable Weight | 450 kg |
| Transit | 3 days |

## Calculation
```
--- Shipment 1: San Francisco (SFO) -> Frankfurt (FRA), 450kg, 2CBM ---
Base Price: $2,025.00
Margin (15.0%): +$303.75
Line Total: $2,328.75

GRAND TOTAL: $2,328.75
```

## Quote Response
```
Customer: TechParts Inc
San Francisco (SFO) -> Frankfurt (FRA), 450kg, 2CBM: $2,328.75

Total: $2,328.75 USD
```
