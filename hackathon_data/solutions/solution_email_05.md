# Solution: Email 05

## Extracted Parameters
| Field | Raw Text | Parsed Value |
|-------|----------|--------------|
| Origin | "HCMC (Saigon)" | Ho Chi Minh City |
| Destination | "Los Angeles" | Los Angeles |
| Container Size | "40'" | 40ft |
| Quantity | "1 x" | 1 |
| Mode | "ocean freight" | Sea |

**Fuzzy Matching Challenge:** 
- "HCMC" = Ho Chi Minh City
- "Saigon" = Ho Chi Minh City (old name)
- Must match to "Ho Chi Minh City" in rate sheet

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"
**Match:** Row where Origin="Ho Chi Minh City" AND Destination="Los Angeles"

| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,700 | $3,000 | 21 days |

## Calculation
```
Rate: $3,000
Margin (15%): $3,000 × 1.15 = $3,450

TOTAL QUOTE: $3,450
```

## Quote Response
```
Ho Chi Minh City → Los Angeles
1 x 40ft container
Rate: $3,450 USD
Transit: 21 days
```
