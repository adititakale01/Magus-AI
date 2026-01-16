# Freight Rate Quotation Agent - Hackathon Challenge

## 🎯 Challenge Overview

Build an AI agent that processes freight rate requests from emails and returns accurate quotes by looking up rates in Excel sheets.

**Focus:** Data extraction and matching (not freight domain expertise)

---

## 📦 What's Included

```
hackathon_data/
├── emails/           # 10 JSON email requests
├── rate_sheets/      # 3 Excel files (Easy → Hard)
└── solutions/        # Expected outputs for validation
```

---

## 🧠 The Task

1. **Parse** the email to extract: origin, destination, container size (or weight/volume for air), quantity
2. **Match** to the correct rate in the Excel sheet
3. **Calculate** the quote: `rate × quantity × 1.15` (15% margin)
4. **Return** a formatted quote

---

## 💰 Pricing Model (Simplified)

### Sea Freight
- All-in price per container (20ft or 40ft)
- Just look up: Origin + Destination + Size → Price

### Air Freight
- Price per kg
- **Chargeable weight** = MAX(actual_kg, volume_CBM × 167)
- Apply minimum charge if below threshold

### Margin
- Add 15% to all rates: `final_price = base_rate × 1.15`

---

## 📊 Rate Sheet Difficulty Levels

### Level 1: Easy (`01_rates_easy.xlsx`)
- Clean flat tables
- Direct column lookup
- No data cleaning needed

```
| Origin | Destination | 20ft Price | 40ft Price | Transit |
|--------|-------------|------------|------------|---------|
| Shanghai | Rotterdam | 1800 | 3200 | 28 |
```

### Level 2: Medium (`02_rates_medium.xlsx`)
- Separate sheets for port codes and rates
- Must JOIN port codes to look up rates
- Port aliases for fuzzy matching (e.g., "HCMC" = "Ho Chi Minh City")

```
Port Codes Sheet:
| Code | Port Name | Aliases |
|------|-----------|---------|
| SGN | Ho Chi Minh City | HCMC, SAIGON |

Rates Sheet:
| Origin Code | Dest Code | 20ft | 40ft |
|-------------|-----------|------|------|
| SGN | LAX | 1700 | 3000 |
```

### Level 3: Hard (`03_rates_hard.xlsx`)
- Messy real-world formatting
- Ditto marks (`''`, `"`, `-`) meaning "same as above"
- Merged section headers
- Notes embedded in cells
- Mixed port name formats

```
| POL | POD | 20' | 40' | Notes |
|-----|-----|-----|-----|-------|
| SHANGHAI | Rotterdam | 1800 | 3200 | |
| '' | Hamburg | 1850 | 3300 | |  ← '' means Shanghai
| '' | Felixstowe | 1900 | 3400 | UK port |
```

---

## 📧 Email Challenges

| Email | Challenge Type | What to Handle |
|-------|---------------|----------------|
| 01 | Simple | Clean, structured request |
| 02 | Air freight | Volume weight calculation |
| 03 | Incomplete | Missing info → ask for clarification |
| 04 | Informal | Typos, casual language, abbreviations |
| 05 | Fuzzy match | "HCMC/Saigon" → "Ho Chi Minh City" |
| 06 | Multi-route | 2 different routes in one request |
| 07 | Volume weight | Volume > actual weight |
| 08 | Port variations | "Tokyo/Yokohama area" → Yokohama |
| 09 | Infer size | Given CBM, must recommend container |
| 10 | City names | "Tokyo Narita" → Tokyo |

---

## ✅ Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Extraction Accuracy** | 30% | Correctly parse origin, destination, size, qty |
| **Fuzzy Matching** | 25% | Handle port name variations |
| **Rate Lookup** | 25% | Find correct rate in Excel |
| **Edge Cases** | 20% | Missing data detection, multi-route, inference |

---

## 🔧 Technical Hints

### Port Name Matching
Consider these variations as equivalent:
- `Ho Chi Minh City` = `HCMC` = `Saigon` = `SGN`
- `Los Angeles` = `LA` = `LAX` = `Long Beach`
- `Tokyo` = `Narita` = `NRT` = `Yokohama`

### Parsing Ditto Marks
In the hard rate sheet, these all mean "same origin as above":
- `''`
- `"`
- `-`
- Empty cell

### Volume Weight Formula
```python
chargeable_kg = max(actual_kg, volume_cbm * 167)
```

### Container Recommendation
When only volume is given:
- Under 33 CBM → 20ft container
- 33-67 CBM → 40ft container
- Over 67 CBM → Multiple containers

---

Good luck! 🎉
