from __future__ import annotations

from pathlib import Path
import time

import streamlit as st

from src.config import load_app_config
from src.data_loader import list_emails, load_email
from src.pipeline import run_quote_pipeline
from src.prompts import (
    EXTRACTION_RESPONSE_SCHEMA,
    EXTRACTION_SYSTEM_PROMPT,
    LOCATION_MATCH_RESPONSE_SCHEMA,
    LOCATION_MATCH_SYSTEM_PROMPT,
    extraction_user_prompt,
    location_match_user_prompt,
)
from src.rate_sheets import normalize_rate_sheets


APP_DIR = Path(__file__).resolve().parents[1]


def _load_solution_md(*, data_dir: Path, email_id: str) -> str:
    path = (data_dir / "solutions" / f"solution_{email_id}.md").resolve()
    if not path.exists():
        return f"_No solution file found at `{path}`_"
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    st.set_page_config(page_title="Workflow", layout="wide")

    st.title("Workflow & Methodology")
    st.caption("Detailed pipeline + prompts (with schemas) + test examples + ground-truth solutions.")

    config = load_app_config(APP_DIR / "config.toml")
    email_index = list_emails(config.data.data_dir)

    st.header("Pipeline (detailed)")

    with st.expander("0) Rate sheet normalization (Excel → canonical tables)", expanded=False):
        st.markdown(
            """
The agent does **not** quote directly from raw Excel. It first normalizes each difficulty level into two clean tables:

- `Sea` rates: canonical `Origin`, `Destination`, and per-container prices.
- `Air` rates: canonical `Origin`, `Destination`, `rate_per_kg`, `min_charge`, and `transit_days`.

This makes later steps deterministic and explainable.
"""
        )
        st.markdown("**What changes across difficulties**")
        st.markdown(
            """
- **Easy**: direct lookup.
- **Medium**: joins from port codes/aliases into canonical names.
- **Hard**: de-ditto (`''`, `"`, `-`, empty), strip notes embedded in cells, and normalize messy text.
"""
        )
        st.markdown("**Example (Hard ditto normalization)**")
        st.code(
            "Input rows:\n"
            "Origin      Destination   40'\n"
            "Shanghai    Rotterdam     3200\n"
            "''          Hamburg       3300\n"
            "\n"
            "Normalized rows:\n"
            "Origin      Destination   40ft Price (USD)\n"
            "Shanghai    Rotterdam     3200\n"
            "Shanghai    Hamburg       3300\n",
            language="text",
        )

    with st.expander("1) Inputs (emails)", expanded=False):
        st.markdown("Emails are loaded from `hackathon_data/emails/*.json` with a simple schema:")
        st.code('{"from": "...", "to": "...", "subject": "...", "body": "..."}', language="json")
        if "email_04" in email_index:
            example = load_email(email_index["email_04"])
            st.markdown("**Test input example (`email_04`)**")
            st.code(f"SUBJECT:\n{example.subject}\n\nBODY:\n{example.body}", language="text")

    with st.expander("2) Extraction (rule-based + optional OpenAI)", expanded=False):
        st.markdown(
            """
Goal: convert unstructured email text into a list of structured `ShipmentRequest`s.

- **Rule-based extractor (default)**: regex parsing for the 10 hackathon emails (including multi-route).
- **OpenAI extractor (optional)**: uses a strict JSON-output prompt and returns clarification questions when fields are missing.
"""
        )
        st.markdown("**OpenAI prompt scheme (Chat Completions)**")
        st.json(
            {
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": extraction_user_prompt(subject="{subject}", body="{body}"),
                    },
                ],
                "response_format": {"type": "json_object"},
            }
        )
        st.markdown("**OpenAI response schema**")
        st.json(EXTRACTION_RESPONSE_SCHEMA)
        st.markdown("**Test I/O example**")
        st.code(
            "Input:\n"
            "From: San Francisco (SFO)\n"
            "To: Frankfurt (FRA)\n"
            "Weight: 450 kg\n"
            "Dimensions: 2 CBM\n"
            "\n"
            "Expected output (JSON):\n"
            "{\n"
            '  \"shipments\": [\n'
            "    {\n"
            '      \"mode\": \"air\",\n'
            '      \"origin_raw\": \"San Francisco (SFO)\",\n'
            '      \"destination_raw\": \"Frankfurt (FRA)\",\n'
            '      \"actual_weight_kg\": 450,\n'
            '      \"volume_cbm\": 2,\n'
            '      \"quantity\": null,\n'
            '      \"container_size_ft\": null,\n'
            '      \"commodity\": null,\n'
            '      \"notes\": null\n'
            "    }\n"
            "  ],\n"
            '  \"clarification_questions\": []\n'
            "}\n",
            language="text",
        )

    with st.expander("3) Normalize (locations, codes, noise removal)", expanded=False):
        st.markdown(
            """
The agent normalizes locations before lookup:

- Extract IATA code when present (e.g., `San Francisco (SFO)`).
- Strip suffix noise like country names or `area`.
- Apply aliases (from config + from rate sheet).
- Fuzzy match locally.
- Optional OpenAI fallback (only when enabled and local matching fails).
"""
        )
        st.markdown("**Test input/output examples**")
        st.code(
            "Input:  Tokyo Narita\nOutput: Tokyo\n\n"
            "Input:  Manzanillo MX\nOutput: Manzanillo\n\n"
            "Input:  Busan, South Korea\nOutput: Busan\n",
            language="text",
        )

    with st.expander("4) Match (exact/alias/fuzzy + optional OpenAI fallback)", expanded=False):
        st.markdown(
            """
After normalization, matching is deterministic:

1. Exact match to canonical list
2. Alias match (config + sheet-derived aliases)
3. Fuzzy match (local)
4. Optional OpenAI fallback *only if* 1–3 fail
"""
        )
        st.markdown("**OpenAI fallback prompt scheme**")
        st.json(
            {
                "messages": [
                    {"role": "system", "content": LOCATION_MATCH_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": location_match_user_prompt(raw="{raw}", choices=["{choice1}", "{choice2}", "..."]),
                    },
                ],
                "response_format": {"type": "json_object"},
            }
        )
        st.markdown("**OpenAI response schema**")
        st.json(LOCATION_MATCH_RESPONSE_SCHEMA)
        st.markdown("**Test I/O example**")
        st.code(
            "Input choices:\n- Tokyo\n- Yokohama\n\nRAW: Tokyo/Yokohama area\n\nExpected output:\n{\"canonical\": \"Yokohama\"}\n",
            language="text",
        )

    with st.expander("5) Compute (pricing formulas)", expanded=False):
        st.markdown("**Sea freight**: look up per-container price and apply margin.")
        st.code("final = base_rate * quantity * (1 + margin)", language="python")
        st.markdown("**Air freight**: compute chargeable weight then apply per-kg rate and margin.")
        st.code(
            "chargeable_kg = max(actual_kg, volume_cbm * 167)\nfinal = max(chargeable_kg * rate_per_kg, min_charge) * (1 + margin)",
            language="python",
        )
        st.markdown("**Test example (air chargeable weight)**")
        st.code(
            "actual_kg = 450\nvolume_cbm = 2\nvolume_weight = 2 * 167 = 334\nchargeable = max(450, 334) = 450\n",
            language="text",
        )

    with st.expander("6) SOP rules (optional, from `SOP.md`)", expanded=False):
        st.markdown(
            """
SOPs are **optional**. When enabled, they override default quoting rules for specific customers.

**Implementation overview**
- The SOP spec lives in `hackathon_data/SOP.md`.
- This demo implements those rules as **code-backed profiles** (not a full SOP.md parser yet):
  - Profiles + helpers: `src/sop.py`
  - SOP wiring + enforcement: `src/pipeline.py` (`enable_sop=True`)
  - Pricing order (discount → margin → surcharge): `src/quote_logic.py` (`compute_quote_amounts`)

**How a SOP is applied (high level)**
1. Identify the customer by sender email → load a `SopProfile`.
2. For each extracted shipment:
   - Enforce restrictions (mode/origin rules).
   - Normalize + match locations and look up the base rate.
   - Apply pricing overrides (discounts, margin override, surcharges) in the SOP-defined order.
   - Apply output requirements (warnings / weights / subtotals / email reply wrapper).

---

### A) Customer identification
`get_sop_profile(sender_email)` returns a `SopProfile` keyed by sender email (case-insensitive).

---

### B) Restrictions / eligibility rules (hard gates)
These rules prevent quoting until the customer confirms a valid request:

**1) Mode restriction (sea-only / air-only)**
- If a profile defines `allowed_modes` and the email asks for a different mode:
  - The pipeline **skips quoting** that shipment.
  - Adds a clarification: “Per your SOP, we can quote sea/air only…”

**2) Origin restriction**
- Example: VietExport requires origin to be **Ho Chi Minh City**.
- If origin doesn’t match `origin_required`, the pipeline asks the user to confirm origin.

---

### C) Location equivalence rules (matching helpers)
These rules help find a rate without changing the user’s text:

**1) Origin equivalence fallback**
- Example: Global Imports treats **Shanghai ↔ Ningbo** as interchangeable for lookup.
- If lookup fails for the requested origin, the pipeline tries the SOP fallback origins.
- A note is added: “Origin equivalence applied…”

---

### D) Pricing overrides (discounts / margin / surcharges)
All SOP pricing rules follow the same computation pipeline:

```text
base_amount = rate * qty
base_after_discount = base_amount * (1 - discount_pct)
final_before_surcharge = base_after_discount * (1 + margin)
final = final_before_surcharge + fixed_surcharge
```

**1) Discount before margin**
- Global Imports: `sea_discount_pct = 10%` (sea only)
- Applied to the base rate **before** adding margin.

**2) Volume discounts across multi-route requests**
- AutoSpares: discount tier is based on **total containers across all sea routes**.
- The demo estimates total containers as:
  - If `quantity` exists → use it
  - Else if `volume_cbm` exists → infer recommended containers (`recommend_containers`)
  - Else → ask clarification (“confirm total container quantities…”)

**3) Margin override**
- QuickShip UK uses `margin_override = 8%` (broker model).
- Margin percent is **not shown** in customer-facing text (we only show final prices).

**4) Global fixed surcharge**
- Australia destination: +$150 biosecurity surcharge (currently triggered for destination canonical `"Melbourne"`).
- Added as a separate note line in the quote.

---

### E) Output requirements (format + extra fields)
These rules only affect how the answer is presented:

**1) Transit-time warnings**
- TechParts: if `transit_days > 3`, add a warning line.

**2) Weight display requirements**
- TechParts: show actual weight + chargeable weight (and volume).

**3) Multi-route presentation**
- AutoSpares: show per-route subtotals + a grand total (the multi-route formatter always does this).

**4) SOP-aware reply email wrapper**
- When SOP is enabled, the final output is wrapped as an English email reply and states that SOP was applied.
"""
        )
        st.markdown("**Concrete examples**")
        st.code(
            "Global Imports (sea-only, 10% discount before margin)\n"
            "Base (40ft): 3200\n"
            "After 10% discount: 2880\n"
            "After 15% margin: 3312\n\n"
            "QuickShip UK (8% margin override)\n"
            "Base (40ft): 3300\n"
            "After 8% margin: 3564\n",
            language="text",
        )

    with st.expander("7) Respond (quote vs clarification email)", expanded=False):
        st.markdown(
            """
The agent either:

- Returns a formatted quote, or
- Returns a clarification email when required fields are missing.

When SOP is enabled, the output is wrapped as an email reply and SOP-required fields are included.
"""
        )
        st.markdown("**Test example (clarification)**")
        st.code(
            "Hi,\n\nThanks for your inquiry. To provide an accurate quote, could you please confirm:\n"
            "1. Is this sea freight or air freight?\n"
            "2. How much cargo is this (container size/quantity, or weight and volume)?\n\n"
            "Best regards,\n",
            language="text",
        )

    st.header("Ground-truth solutions (reference)")
    if not email_index:
        st.warning(f"No emails found under: `{config.data.data_dir}`")
    else:
        sol_tabs = st.tabs([eid.replace("email_", "Email ") for eid in sorted(email_index.keys())])
        for tab, email_id in zip(sol_tabs, sorted(email_index.keys()), strict=True):
            with tab:
                st.markdown(_load_solution_md(data_dir=config.data.data_dir, email_id=email_id))

    st.header("Batch validation (10 hackathon emails)")

    col_a, col_b = st.columns([0.35, 0.65], gap="large")
    with col_a:
        difficulty = st.selectbox("Difficulty", options=["easy", "medium", "hard"], index=0)
        use_openai = st.toggle("Use OpenAI (extraction + matching fallback)", value=False)
        enable_sop = st.toggle("Enable SOP rules (`SOP.md`)", value=False)
        run_btn = st.button("Run batch validation", type="primary", disabled=not bool(email_index))

    with col_b:
        st.write(f"Data dir: `{config.data.data_dir}`")
        st.write(f"Emails found: **{len(email_index)}**")
        st.write("OpenAI API key: " + ("loaded" if config.openai.api_key else "missing"))

    if run_btn:
        with st.spinner(f"Normalizing {difficulty} rate sheet..."):
            rates = normalize_rate_sheets(config=config, difficulty=difficulty)

        rows: list[dict] = []
        outputs: dict[str, str] = {}
        failures: list[tuple[str, str]] = []
        with st.spinner("Running all emails..."):
            for email_id in sorted(email_index.keys()):
                email = load_email(email_index[email_id])
                t0 = time.perf_counter()
                result = run_quote_pipeline(
                    email=email,
                    config=config,
                    difficulty=difficulty,
                    use_openai=use_openai,
                    rate_sheets=rates,
                    enable_sop=enable_sop,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                ok = (result.error is None) and bool(result.quote_text)
                outputs[email_id] = result.quote_text or ""
                rows.append(
                    {
                        "email": email_id,
                        "ok": ok,
                        "latency_ms": round(elapsed_ms, 1),
                        "llm_calls": result.trace.llm_usage.calls,
                        "llm_total_tokens": result.trace.llm_usage.total_tokens,
                        "error": result.error or "",
                        "preview": (result.quote_text or "").splitlines()[0:2],
                    }
                )
                if not ok:
                    failures.append((email_id, result.error or "Unknown error"))

        st.dataframe(rows, use_container_width=True)

        st.subheader("Per-email output vs standard solution")
        for email_id in sorted(email_index.keys()):
            with st.expander(email_id, expanded=False):
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("**Output (this run)**")
                    st.code(outputs.get(email_id, ""), language="text")
                with right:
                    st.markdown("**Standard answer (`solutions/`)**")
                    st.markdown(_load_solution_md(data_dir=config.data.data_dir, email_id=email_id))

        if failures:
            st.error("Some emails failed validation.")
            for email_id, err in failures:
                st.write(f"- `{email_id}`: {err}")
        else:
            st.success("All 10 emails produced a quote or clarification response successfully.")


if __name__ == "__main__":
    main()
