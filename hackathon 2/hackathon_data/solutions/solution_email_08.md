# Solution: Email 08

## Extracted Parameters
| Field | Raw Text | Parsed Value |
|-------|----------|--------------|
| Origin | "Manzanillo MX" | Manzanillo (Mexico) |
| Destination | "Tokyo/Yokohama area" | Yokohama |
| Container Size | "40ft" | 40ft |
| Quantity | "1 x" | 1 |
| Mode | (implied) | Sea |

**Fuzzy Matching Challenge:**
- "Tokyo/Yokohama area" → match to "Yokohama" in rate sheet
- "Manzanillo MX" → ensure Mexico (not Colombia)

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"
**Match:** Row where Origin="Manzanillo" AND Destination="Yokohama"

| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $2,100 | $3,800 | 22 days |

## Calculation
```
Rate: $3,800
Margin (15%): $3,800 × 1.15 = $4,370

TOTAL QUOTE: $4,370
```

## Quote Response
```
Manzanillo (Mexico) → Yokohama (Japan)
1 x 40ft container
Rate: $4,370 USD
Transit: 22 days
```
