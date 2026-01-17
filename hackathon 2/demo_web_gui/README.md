# Freight Rate Quotation Agent (Demo Web GUI)

This is a small Streamlit demo for the **Hackathon Challenge** in `hackathon 2/hackathon_data/`.

It auto-loads:
- Email requests from `hackathon_data/emails/*.json`
- Rate sheets from `hackathon_data/rate_sheets/*.xlsx`

And runs a step-by-step quote pipeline (with a visible trace) to produce a formatted quote response.

It also includes:
- **SOP parsing & caching** (`SOP.md`) with LLM raw output display
- **FastAPI JSON API server** (`api_server.py`) for external frontend integration
- **Persistence to Supabase / Postgres** (`email_quote_records` table) with derived fields (origin/destination/mode/price/route)
- **HITL decision endpoint** (accept/reject + optional LLM rewrite + Make.com webhook)
- **Static API tester UI** (`api_tester/`) to quickly test all endpoints

## Quickstart

1) Install dependencies:
```powershell
cd "hackathon 2/demo_web_gui"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

2) Add your OpenAI API key in `hackathon 2/demo_web_gui/.env`:
```
OPENAI_API_KEY=...
```

Optional (recommended) DB persistence via Supabase REST (HTTPS 443):
```
SUPABASE_URL=https://<PROJECT_REF>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
```

Optional Make.com webhook override (default is hardcoded in the API server):
```
MAKE_WEBHOOK_URL=...
```

3) Run the web app:
```bash
python -m streamlit run "hackathon 2/demo_web_gui/app.py"
```

## API server (for frontend integration)
Run from the `hackathon 2/demo_web_gui/` folder:
```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

- API docs: `hackathon 2/demo_web_gui/API.md`
- Base URL: `http://localhost:8000/api/v1`
- Quote endpoint: `POST /api/v1/quote` (returns inferred fields like `origin_city`, `price`, `transport_type`, `has_route`)
- Persistence endpoints: `/api/v1/email-records/*`
- HITL decision endpoint: `POST /api/v1/email-records/decision` (accept/reject + optional LLM rewrite + webhook)

## Expose via ngrok (Python SDK)
If `ngrok http 8000` is not working on your machine, you can use the ngrok Python SDK instead.

From the `hackathon 2/demo_web_gui/` folder:
```powershell
$env:NGROK_AUTHTOKEN = "<your-token>"
python run_with_ngrok.py
```
It prints the public base URL (and keeps running until Ctrl+C).

## API tester UI (static frontend)
Run from the `hackathon 2/demo_web_gui/` folder:
```bash
python -m http.server 5173
```
Then open `http://localhost:5173/api_tester/` and point it to your API base URL (default: `http://localhost:8000/api/v1`).

The tester supports:
- Rate sheet upload/normalize
- SOP view/parse
- Quote generation (with optional persistence)
- Email record list/counts/needs-human queue
- Human decision submit (accept/reject + comment + refined quote)

## Pages
- **Freight Quote Agent Demo**: pick an email and run the pipeline with a full trace.
- **Workflow & Methodology**: explains the end-to-end flow and includes a batch validation runner.

## Notes
- The demo is wired for **OpenAI**-based extraction, with a rule-based fallback parser for basic emails.
- Before quoting emails, the UI requires **rate sheet normalization** (it converts Excel formats into a canonical Sea/Air table).
- Rate lookup + calculation is implemented for **Easy / Medium / Hard** sheets (they normalize into the same internal schema).
- To enable DB persistence, ensure your Supabase table schema matches `hackathon 2/demo_web_gui/sql/001_email_reply_records.sql`.
