# Freight Rate Quotation Agent (Demo Web GUI)

This is a small Streamlit demo for the **Hackathon Challenge** in `hackathon 2/hackathon_data/`.

It auto-loads:
- Email requests from `hackathon_data/emails/*.json`
- Rate sheets from `hackathon_data/rate_sheets/*.xlsx`

And runs a step-by-step quote pipeline (with a visible trace) to produce a formatted quote response.

## Quickstart

1) Install dependencies:
```bash
python -m pip install --user -r "hackathon 2/demo_web_gui/requirements.txt"
```

2) Add your OpenAI API key in `hackathon 2/demo_web_gui/.env`:
```
OPENAI_API_KEY=...
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

## Pages
- **Freight Quote Agent Demo**: pick an email and run the pipeline with a full trace.
- **Workflow & Methodology**: explains the end-to-end flow and includes a batch validation runner.

## Notes
- The demo is wired for **OpenAI**-based extraction, with a rule-based fallback parser for basic emails.
- Before quoting emails, the UI requires **rate sheet normalization** (it converts Excel formats into a canonical Sea/Air table).
- Rate lookup + calculation is implemented for **Easy / Medium / Hard** sheets (they normalize into the same internal schema).
