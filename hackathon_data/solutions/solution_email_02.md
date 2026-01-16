# Solution: Email 02

## Extracted Parameters
| Field | Value |
|-------|-------|
| Origin | San Francisco (SFO) |
| Destination | Frankfurt (FRA) |
| Actual Weight | 450 kg |
| Volume | 2 CBM |
| Mode | Air |

## Chargeable Weight Calculation
```
Volume weight = CBM × 167 = 2 × 167 = 334 kg
Actual weight = 450 kg
Chargeable weight = MAX(450, 334) = 450 kg
```

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Air Freight Rates"
**Match:** Row where Origin="San Francisco" AND Destination="Frankfurt"

| Rate/kg | Min Charge | Transit |
|---------|------------|---------|
| $4.50 | $150 | 3 days |

## Calculation
```
Base rate: 450 kg × $4.50 = $2,025
(Above minimum of $150 ✓)
Margin (15%): $2,025 × 1.15 = $2,329

TOTAL QUOTE: $2,329
```

## Quote Response
```
San Francisco (SFO) → Frankfurt (FRA)
Air Freight: 450 kg chargeable
Rate: $2,329 USD
Transit: 3 days
```
