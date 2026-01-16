# Solution: Email 04

## Extracted Parameters (from informal text)
| Field | Raw Text | Parsed Value |
|-------|----------|--------------|
| Origin | "ningbo" | Ningbo |
| Destination | "felixstowe" | Felixstowe |
| Container Size | "1x40ft" | 40ft |
| Quantity | "1x" | 1 |
| Mode | (implied) | Sea |

**NLP Challenge:** Handle lowercase, typos ("urgant"), casual language ("can u", "cheers")

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"
**Match:** Row where Origin="Ningbo" AND Destination="Felixstowe"

| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,850 | $3,300 | 30 days |

## Calculation
```
Rate: $3,300
Margin (15%): $3,300 × 1.15 = $3,795

TOTAL QUOTE: $3,795
```

## Quote Response
```
Ningbo → Felixstowe
1 x 40ft container
Rate: $3,795 USD
Transit: 30 days
```
