# Solution: Email 06

## Extracted Parameters
| Route | Origin | Destination | Size | Qty |
|-------|--------|-------------|------|-----|
| 1 | Busan | Hamburg | 40ft | 2 |
| 2 | Busan | Rotterdam | 20ft | 1 |

**Multi-Route Challenge:** Must look up 2 separate routes and aggregate.

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"

**Route 1:** Busan → Hamburg
| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,900 | $3,400 | 32 days |

**Route 2:** Busan → Rotterdam
| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,850 | $3,300 | 30 days |

## Calculation
```
Route 1: Hamburg (40ft × 2)
  Base: $3,400 × 2 = $6,800
  + Margin (15%): $6,800 × 1.15 = $7,820

Route 2: Rotterdam (20ft × 1)
  Base: $1,850 × 1 = $1,850
  + Margin (15%): $1,850 × 1.15 = $2,128

TOTAL QUOTE: $7,820 + $2,128 = $9,948
```

## Quote Response
```
Route 1: Busan → Hamburg
  2 x 40ft containers
  Rate: $3,910 per container
  Subtotal: $7,820
  Transit: 32 days

Route 2: Busan → Rotterdam
  1 x 20ft container
  Rate: $2,128
  Transit: 30 days

GRAND TOTAL: $9,948 USD
```
