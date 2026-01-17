# Freight Quote Agent API (v1)

This service exposes the demo pipeline as a JSON API for a separate frontend.

## Base
- Base URL: `/api/v1`
- Content type: `application/json` (except file upload/download)
- OpenAPI: `/docs` and `/openapi.json`

API tester UI: `hackathon 2/demo_web_gui/api_tester/index.html`

ngrok (SDK): `hackathon 2/demo_web_gui/run_with_ngrok.py` (reads `NGROK_AUTHTOKEN`)

## Auth / OpenAI key
- **Header:** `X-OpenAI-Api-Key: <your-openai-key>`
- **Alias (optional):** `X-API-Key: <your-openai-key>`

If an endpoint is invoked with `use_openai=true` (or `sop_parse_mode=llm`) and the OpenAI key is missing, the API returns `401`.

## Common types

### Difficulty
- `easy` | `medium` | `hard`

### Email payload
```json
{
  "email_id": "optional-string",
  "from": "sender@example.com",
  "to": "quotes@freightco.com",
  "subject": "Quote request",
  "body": "..."
}
```

## Endpoints

### Health
`GET /api/v1/health`

Returns `{ "ok": true }`.

---

### Upload / replace a rate sheet (xlsx)
`PUT /api/v1/rate-sheets/{difficulty}`

- `difficulty`: `easy|medium|hard`
- `multipart/form-data` field: `file` (required)

Response:
```json
{
  "difficulty": "easy",
  "path": ".../hackathon_data/rate_sheets/01_rates_easy.xlsx",
  "bytes_written": 12345
}
```

---

### Download current rate sheet (xlsx)
`GET /api/v1/rate-sheets/{difficulty}/source`

Returns the `.xlsx` file as `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`.

---

### Normalize a rate sheet (and cache it)
`POST /api/v1/rate-sheets/{difficulty}/normalize`

Response:
```json
{
  "difficulty": "easy",
  "source_path": "...",
  "source_mtime": 1234567890.0,
  "sheet_names": ["Sea Freight Rates", "Air Freight Rates"],
  "warnings": [],
  "counts": { "sea_rows": 12, "air_rows": 12 },
  "canonical_locations": {
    "sea_origins": ["Shanghai"],
    "sea_destinations": ["Rotterdam"],
    "air_origins": ["Tokyo"],
    "air_destinations": ["Paris"]
  }
}
```

---

### Get normalized rate tables (Sea + Air)
`GET /api/v1/rate-sheets/{difficulty}/normalized`

Query params:
- `include_rows` (default `true`): if `false`, returns only metadata (no `sea_rows` / `air_rows` arrays).

Response (abridged):
```json
{
  "difficulty": "easy",
  "warnings": [],
  "sea_rows": [{ "Origin": "Shanghai", "Destination": "Rotterdam", "40ft Price (USD)": 3200, "Transit (Days)": 28 }],
  "air_rows": [{ "Origin": "Tokyo", "Destination": "Paris", "Rate per kg (USD)": 4.5 }]
}
```

---

### Get current SOP markdown
`GET /api/v1/sop`

Response:
```json
{ "path": ".../hackathon_data/SOP.md", "markdown": "..." }
```

---

### Update SOP markdown
`PUT /api/v1/sop`

Body:
```json
{ "markdown": "..." }
```

Response:
```json
{ "path": ".../hackathon_data/SOP.md", "bytes_written": 1234 }
```

---

### Parse SOP (cache per difficulty)
`POST /api/v1/sop/parse`

Body:
```json
{
  "difficulty": "easy",
  "sop_parse_mode": "auto",
  "force_refresh": false
}
```

Notes:
- `sop_parse_mode`:
  - `auto`: use LLM if `X-OpenAI-Api-Key` is present, else rule-based.
  - `llm`: require `X-OpenAI-Api-Key`.
  - `rule_based`: never call LLM.
- Response includes `llm_raw` when LLM was used.

---

### Quote: generate the reply text for an email request
`POST /api/v1/quote`

Body (two supported formats):

1) Structured (legacy):
```json
{
  "difficulty": "easy",
  "enable_sop": true,
  "use_openai": true,
  "email": {
    "from": "sarah.chen@globalimports.com",
    "to": "quotes@freightco.com",
    "subject": "Quote Request",
    "body": "..."
  },
  "force_sop_refresh": false,
  "include_trace": true
}
```

2) Raw email text (recommended for frontend): the server parses `From:` / `To:` / `Subject:` and uses the rest as body.
```json
{
  "difficulty": "easy",
  "enable_sop": true,
  "use_openai": true,
  "email_id": "optional-string",
  "email": "From: sarah.chen@globalimports.com\\nTo: quotes@freightco.com\\nSubject: Quote Request\\n\\nHi team, ...",
  "force_sop_refresh": false,
  "include_trace": true
}
```

Response (abridged):
```json
{
  "quote_text": "Hi,\\n\\nThank you ...",
  "error": null,
  "trace": { "llm_usage": { "calls": 2, "total_tokens": 1234 }, "steps": [ ... ] }
}
```

Behavior:
- Rate sheets are normalized once and cached per difficulty (auto invalidated on file upload).
- When `enable_sop=true`, the server uses the **cached parsed SOP** for that difficulty if available; otherwise it parses SOP once and reuses it for subsequent quotes.
- When `use_openai=true`, missing `X-OpenAI-Api-Key` returns `401`.
