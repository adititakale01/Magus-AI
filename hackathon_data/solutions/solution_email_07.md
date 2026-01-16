# Solution: Email 07

## Extracted Parameters
| Field | Value |
|-------|-------|
| Origin | Mumbai (BOM) |
| Destination | Chicago (ORD) |
| Actual Weight | 200 kg |
| Volume | 3 CBM |
| Mode | Air |

## Chargeable Weight Calculation
```
Volume weight = CBM × 167 = 3 × 167 = 501 kg
Actual weight = 200 kg
Chargeable weight = MAX(200, 501) = 501 kg
```

**Technical Challenge:** Volume weight significantly exceeds actual weight.

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Air Freight Rates"
**Match:** Row where Origin="Mumbai" AND Destination="Chicago"

| Rate/kg | Min Charge | Transit |
|---------|------------|---------|
| $5.20 | $180 | 4 days |

## Calculation
```
Base rate: 501 kg × $5.20 = $2,605
(Above minimum of $180 ✓)
Margin (15%): $2,605 × 1.15 = $2,996

TOTAL QUOTE: $2,996
```

## Quote Response
```
Mumbai (BOM) → Chicago (ORD)
Air Freight
Actual: 200 kg | Volume: 3 CBM
Chargeable weight: 501 kg (volumetric)
Rate: $2,996 USD
Transit: 4 days
```
