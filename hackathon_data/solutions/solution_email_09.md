# Solution: Email 09

## Extracted Parameters
| Field | Value |
|-------|-------|
| Origin | Qingdao |
| Destination | Melbourne |
| Volume | 50 CBM |
| Container Size | ❌ NOT SPECIFIED (must infer) |
| Mode | Sea |

## Container Size Inference
```
50 CBM cargo volume
40ft container ≈ 67 CBM capacity
20ft container ≈ 33 CBM capacity

50 CBM fits in: 1 x 40ft (with space) OR 2 x 20ft

RECOMMENDATION: 1 x 40ft container (most cost-effective)
```

**Technical Challenge:** Agent must infer container type from cargo volume.

## Rate Lookup
**File:** `01_rates_easy.xlsx` → Sheet: "Sea Freight Rates"
**Match:** Row where Origin="Qingdao" AND Destination="Melbourne"

| 20ft Price | 40ft Price | Transit |
|------------|------------|---------|
| $1,500 | $2,700 | 18 days |

## Calculation (assuming 1 x 40ft)
```
Rate: $2,700
Margin (15%): $2,700 × 1.15 = $3,105

TOTAL QUOTE: $3,105
```

## Quote Response
```
Qingdao → Melbourne
Cargo: 50 CBM (furniture)
Recommended: 1 x 40ft container
Rate: $3,105 USD
Transit: 18 days

Note: 50 CBM fits comfortably in one 40ft container.
Alternative: 2 x 20ft @ $1,725 each = $3,450 total
```
