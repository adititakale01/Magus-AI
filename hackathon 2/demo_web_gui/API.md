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

## Persistence (Supabase / Postgres)
Optional. Enables storing each email quote record (email + reply + trace + config + status).

- Recommended: configure Supabase REST (HTTPS 443) via `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` in `hackathon 2/demo_web_gui/.env` (see `hackathon 2/demo_web_gui/.env.example`)
- Alternative: configure direct Postgres via `DATABASE_URL` (Supabase Postgres requires SSL `sslmode=require`; if your `DATABASE_URL` host is non-local and `sslmode` is missing, the server auto-adds `sslmode=require`)
- Schema SQL: `hackathon 2/demo_web_gui/sql/001_email_reply_records.sql`

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

### DB health (requires persistence config)
`GET /api/v1/db/health`

Response:
```json
{ "ok": true, "mode": "supabase", "newest_created_at": "2026-01-01T12:34:56.789+00:00" }
```

---

### DB init / create tables (idempotent)
`POST /api/v1/db/init`

Notes:
- If configured with `SUPABASE_URL` (REST mode), this endpoint only verifies the table exists and is reachable (it cannot run DDL over REST).

Response:
```json
{ "ok": true, "mode": "psycopg" }
```

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
  "include_trace": true,
  "persist": false,
  "record_type": "auto",
  "record_status": "needs_human_decision"
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
  "include_trace": true,
  "persist": false,
  "record_type": "auto",
  "record_status": "needs_human_decision"
}
```

Response (abridged):
```json
{
  "email": {
    "email_id": "email_api",
    "from": "sarah.chen@globalimports.com",
    "to": "quotes@freightco.com",
    "subject": "Quote Request",
    "body": "..."
  },
  "quote_text": "Hi,\\n\\nThank you ...",
  "error": null,
  "type": "Auto",
  "trace": { "llm_usage": { "calls": 2, "total_tokens": 1234 }, "steps": [ ... ] }
}
```

Behavior:
- Rate sheets are normalized once and cached per difficulty (auto invalidated on file upload).
- When `enable_sop=true`, the server uses the **cached parsed SOP** for that difficulty if available; otherwise it parses SOP once and reuses it for subsequent quotes.
- When `use_openai=true`, missing `X-OpenAI-Api-Key` returns `401`.
- When `persist=true`, the server stores an `email_quote_records` row and adds `record_id` to the response.
- `type` is `Auto` or `HITL` (human-in-the-loop). The server returns `HITL` if SOP was not matched, if SOP conflicts/parse errors exist, if the order amount is very large (env: `HITL_LARGE_ORDER_USD`, default `20000`), or if special requests are detected (notes/keywords).

Status values (for `record_status` / stored `status`):
- `unprocessed` (没处理)
- `auto_processed` (已自动处理)
- `needs_human_decision` (等待人类决策)
- `human_confirmed_replied` (已人工确认并回复)
- `human_rejected` (已人工拒绝)

---

### Create an email record (write)
`POST /api/v1/email-records`

Body:
```json
{
  "email_id": "optional-string",
  "from": "sender@example.com",
  "to": "quotes@freightco.com",
  "subject": "Quote request",
  "body": "...",
  "reply": "...",
  "trace": { "steps": [] },
  "type": "auto",
  "status": "needs_human_decision",
  "config": { "difficulty": "easy", "enable_sop": true }
}
```

---

### Update record status (and optionally reply/trace/config)
`PATCH /api/v1/email-records/{record_id}`

Body (any subset):
```json
{
  "status": "human_confirmed_replied",
  "type": "human",
  "reply": "Final reply text..."
}
```

---

### List email records (paged)
`GET /api/v1/email-records`

Query params:
- `limit` (default `50`)
- `offset` (default `0`)

Response:
```json
{
  "items": [
    {
      "id": "uuid",
      "time": "2026-01-01T12:34:56.789+00:00",
      "from": "sarah.chen@globalimports.com",
      "to": "quotes@freightco.com",
      "subject": "Quote Request",
      "body": "...",
      "reply": "...",
      "type": "auto",
      "status": "needs_human_decision",
      "config": { "difficulty": "easy", "enable_sop": true }
    }
  ],
  "limit": 50,
  "offset": 0,
  "count": 1
}
```

---

### Status counts
`GET /api/v1/email-records/status-counts`

Response:
```json
{
  "counts": {
    "unprocessed": 0,
    "auto_processed": 3,
    "needs_human_decision": 5,
    "human_confirmed_replied": 2,
    "total": 10
  }
}
```

---

### Needs human decision
`GET /api/v1/email-records/needs-human-decision`

Response shape is the same as `/api/v1/email-records` (paged list), filtered to `status=needs_human_decision`.

---

### Human decision (accept/reject + optional LLM refine + webhook)
`POST /api/v1/email-records/decision`

Body:
```json
{
  "id": "uuid",
  "decision": "accept",
  "refined_quote": "Human edited quote text...",
  "comment": "Optional instructions to refine the quote via LLM"
}
```

Behavior:
- If `comment` is non-empty, the server calls OpenAI to rewrite the quote using `email.body + refined_quote + comment` (requires `X-OpenAI-Api-Key` or `OPENAI_API_KEY`).
- Always updates the record `type` to `human` and stores the decision in `trace`.
- If `decision=accept`, calls `MAKE_WEBHOOK_URL` (default is the Make.com hook) and then sets `status=human_confirmed_replied`.
- If `decision=reject`, does not call the webhook and sets `status=human_rejected`.
