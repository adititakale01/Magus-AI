# Solution: Email 01

## Extracted Parameters
| Field | Value |
|-------|-------|
| Origin | Shanghai |
| Destination | Rotterdam |
| Container Size | 40ft |
| Quantity | 2 |
| Mode | Sea |

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"
**Match:** Row where Origin="Shanghai" AND Destination="Rotterdam"

| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,800 | $3,200 | 28 days |

## Calculation
```
Rate per container: $3,200
Quantity: 2
Margin (15%): $3,200 × 1.15 = $3,680 per container

TOTAL QUOTE: $3,680 × 2 = $7,360
```

## Quote Response
```
Shanghai → Rotterdam
2 x 40ft containers
Rate: $3,680 per container
Total: $7,360 USD
Transit: 28 days
```
