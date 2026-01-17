# Solution: Email 10 (with Real SOPs)

Generated: 2026-01-17 12:19

## Customer Information
| Field | Value |
|-------|-------|
| Customer | Unknown Customer |
| Email | jean.dubois@parislogistics.fr |

## Customer SOP (from Qontext)
| Setting | Value |
|---------|-------|
| Margin | 15.0% |
| Flat Discount | None |
| Volume Discount Tiers | None |
| Mode Restriction | None |
| Origin Restriction | None |
| Discount Before Margin | True |

## Extracted Shipments

### Shipment 1
| Field | Value |
|-------|-------|
| Mode | air |
| Origin | Tokyo Narita |
| Destination | Paris CDG |
| Weight | 850 kg |
| Volume | 4 CBM |

## Rate Lookup

### Route 1: tokyo → paris
| Field | Value |
|-------|-------|
| Mode | air |
| Rate per kg | $5.50 |
| Chargeable Weight | 850 kg |
| Transit | 3 days |

## Calculation
```
--- Shipment 1: Tokyo Narita -> Paris CDG, 850kg, 4CBM ---
Base Price: $4,675.00
Margin (15.0%): +$701.25
Line Total: $5,376.25

GRAND TOTAL: $5,376.25
```

## Quote Response
```
Customer: Unknown Customer
Tokyo Narita -> Paris CDG, 850kg, 4CBM: $5,376.25

Total: $5,376.25 USD
```
