# Solution: Email 10

## Extracted Parameters
| Field | Raw Text | Parsed Value |
|-------|----------|--------------|
| Origin | "Tokyo Narita" | Tokyo (NRT) |
| Destination | "Paris CDG" | Paris (CDG) |
| Actual Weight | 850 kg |
| Volume | 4 CBM |
| Mode | Air |

**Fuzzy Matching Challenge:**
- "Tokyo Narita" → NRT code or "Tokyo"
- "Paris CDG" → CDG code or "Paris"

## Chargeable Weight Calculation
```
Volume weight = CBM × 167 = 4 × 167 = 668 kg
Actual weight = 850 kg
Chargeable weight = MAX(850, 668) = 850 kg
```

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Air Freight Rates"
**Match:** Row where Origin="Tokyo" AND Destination="Paris"

| Rate/kg | Min Charge | Transit |
|---------|------------|---------|
| $5.50 | $200 | 3 days |

## Calculation
```
Base rate: 850 kg × $5.50 = $4,675
(Above minimum of $200 ✓)
Margin (15%): $4,675 × 1.15 = $5,376

TOTAL QUOTE: $5,376
```

## Quote Response
```
Tokyo (NRT) → Paris (CDG)
Air Freight: 850 kg chargeable
Rate: $5,376 USD
Transit: 3 days
```
