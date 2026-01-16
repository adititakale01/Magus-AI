from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_app_config
from src.data_loader import list_emails, load_email
from src.pipeline import run_quote_pipeline
from src.rate_sheets import normalize_rate_sheets


APP_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(page_title="Workflow", layout="wide")

    st.title("Workflow & Methodology")
    st.caption("How this demo goes from an email request to a validated quote.")

    st.header("High-level workflow")
    st.markdown(
        """
This demo implements a freight quotation agent with a transparent, step-by-step pipeline:

0. **Normalize the rate sheet (required)**
   - Load the selected Excel file (easy/medium/hard)
   - Convert it into a clean, canonical internal table (Sea + Air)
1. **Load inputs**
   - Email requests from `hackathon_data/emails/*.json`
2. **Extract structured shipment requests**
   - Primary: OpenAI (optional)
   - Fallback: rule-based regex parser (handles the 10 hackathon emails, including multi-route)
3. **Normalize**
   - Normalize email fields (trim noise like country suffixes, remove IATA codes in parentheses, etc.)
4. **Match**
   - Deterministic match against canonical lists
   - Alias match (from config + from rate sheet, when available)
   - Fuzzy match (local)
   - Optional OpenAI fallback for last-mile mapping when matching fails
5. **Compute**
   - Sea: `final = rate_per_container * qty * (1 + 0.15)`
   - Air: `chargeable_kg = max(actual_kg, cbm * 167)` then margin
6. **Respond**
   - Format a quote (or ask clarification questions if required inputs are missing)
   - Emit a trace so you can audit each step and intermediate artifact
"""
    )

    st.header("Key principles")
    st.markdown(
        """
- **Canonical names win**: the rate sheet is treated as the source of truth; inputs are mapped onto it.
- **Fail-soft**: when required fields are missing, the agent returns a clarification email instead of crashing.
- **Explainability**: every run emits a trace (extraction -> normalization -> lookup -> calculation -> formatting).
- **LLM only as needed**: OpenAI can be used for extraction and (optionally) for matching fallback; otherwise it runs locally.
"""
    )

    st.header("Batch validation (10 hackathon emails)")
    config = load_app_config(APP_DIR / "config.toml")
    email_index = list_emails(config.data.data_dir)

    col_a, col_b = st.columns([0.35, 0.65], gap="large")
    with col_a:
        difficulty = st.selectbox("Difficulty", options=["easy", "medium", "hard"], index=0)
        use_openai = st.toggle("Use OpenAI (extraction + matching fallback)", value=False)
        run_btn = st.button("Run batch validation", type="primary", disabled=not bool(email_index))

    with col_b:
        st.write(f"Data dir: `{config.data.data_dir}`")
        st.write(f"Emails found: **{len(email_index)}**")
        st.write("OpenAI API key: " + ("loaded" if config.openai.api_key else "missing"))

    if run_btn:
        with st.spinner(f"Normalizing {difficulty} rate sheet..."):
            rates = normalize_rate_sheets(config=config, difficulty=difficulty)

        rows: list[dict] = []
        failures: list[tuple[str, str]] = []
        with st.spinner("Running all emails..."):
            for email_id in sorted(email_index.keys()):
                email = load_email(email_index[email_id])
                result = run_quote_pipeline(
                    email=email,
                    config=config,
                    difficulty=difficulty,
                    use_openai=use_openai,
                    rate_sheets=rates,
                )
                ok = (result.error is None) and bool(result.quote_text)
                rows.append(
                    {
                        "email": email_id,
                        "ok": ok,
                        "error": result.error or "",
                        "preview": (result.quote_text or "").splitlines()[0:2],
                    }
                )
                if not ok:
                    failures.append((email_id, result.error or "Unknown error"))

        st.dataframe(rows, use_container_width=True)
        if failures:
            st.error("Some emails failed validation.")
            for email_id, err in failures:
                st.write(f"- `{email_id}`: {err}")
        else:
            st.success("All 10 emails produced a quote or clarification response successfully.")


if __name__ == "__main__":
    main()

