# Solution: Email 06 (with Real SOPs)

Generated: 2026-01-17 12:19

## Customer Information
| Field | Value |
|-------|-------|
| Customer | AutoSpares GmbH |
| Email | david.mueller@autospares.de |

## Customer SOP (from Qontext)
| Setting | Value |
|---------|-------|
| Margin | 15.0% |
| Flat Discount | None |
| Volume Discount Tiers | ((1, 0.0), (2, 5.0), (5, 12.0)) |
| Mode Restriction | None |
| Origin Restriction | None |
| Discount Before Margin | True |

## Extracted Shipments

### Shipment 1
| Field | Value |
|-------|-------|
| Mode | sea |
| Origin | Busan, South Korea |
| Destination | Hamburg |
| Container | 2x 40ft |

### Shipment 2
| Field | Value |
|-------|-------|
| Mode | sea |
| Origin | Busan, South Korea |
| Destination | Rotterdam |
| Container | 1x 20ft |

## Rate Lookup

### Route 1: busan → hamburg
| Field | Value |
|-------|-------|
| Mode | sea |
| Rate per Container | $3,400.00 |
| Transit | 32 days |

### Route 2: busan → rotterdam
| Field | Value |
|-------|-------|
| Mode | sea |
| Rate per Container | $1,850.00 |
| Transit | 30 days |

## Calculation
```
--- Shipment 1: Busan, South Korea -> Hamburg, 2x 40ft ---
Base Price: $6,800.00
Discount (5% volume discount (3 containers, 2+ tier) per your account agreement): -$340.00
Margin (15.0%): +$969.00
Line Total: $7,429.00

--- Shipment 2: Busan, South Korea -> Rotterdam, 1x 20ft ---
Base Price: $1,850.00
Discount (5% volume discount (3 containers, 2+ tier) per your account agreement): -$92.50
Margin (15.0%): +$263.62
Line Total: $2,021.12

GRAND TOTAL: $9,450.12
```

## Quote Response
```
Customer: AutoSpares GmbH
Busan, South Korea -> Hamburg, 2x 40ft: $7,429.00
Busan, South Korea -> Rotterdam, 1x 20ft: $2,021.12

Total: $9,450.12 USD
```
