from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_app_config
from src.data_loader import list_emails, load_email
from src.pipeline import run_quote_pipeline
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.toml"


def _get_rate_sheet_paths(config) -> dict[str, Path]:
    return {
        "easy": (config.data.data_dir / config.rates.easy_file).resolve(),
        "medium": (config.data.data_dir / config.rates.medium_file).resolve(),
        "hard": (config.data.data_dir / config.rates.hard_file).resolve(),
    }


def _get_rates_cache() -> dict[str, NormalizedRateSheets]:
    if "normalized_rate_sheets" not in st.session_state:
        st.session_state["normalized_rate_sheets"] = {}
    return st.session_state["normalized_rate_sheets"]


def _get_cached_rates(
    *,
    rates_cache: dict[str, NormalizedRateSheets],
    difficulty: str,
    source_path: Path,
) -> NormalizedRateSheets | None:
    rates = rates_cache.get(difficulty)
    if not rates:
        return None
    if not source_path.exists():
        rates_cache.pop(difficulty, None)
        return None
    if rates.source_mtime != source_path.stat().st_mtime:
        rates_cache.pop(difficulty, None)
        return None
    return rates


def main() -> None:
    st.set_page_config(page_title="Freight Quote Agent Demo", layout="wide")

    st.title("Freight Rate Quotation Agent (Demo)")
    st.caption("Normalize rate sheets first, then run the quote pipeline with a full trace.")

    config = load_app_config(CONFIG_PATH)
    email_index = list_emails(config.data.data_dir)
    paths = _get_rate_sheet_paths(config)
    rates_cache = _get_rates_cache()

    with st.sidebar:
        st.header("Settings")
        difficulty = st.selectbox(
            "Rate Sheet Difficulty (for quoting)",
            options=["easy", "medium", "hard"],
            index=0,
        )
        use_openai = st.toggle(
            "Use OpenAI for extraction",
            value=True,
            help="If disabled (or no API key), a rule-based extractor is used.",
        )

        st.divider()
        st.subheader("OpenAI")
        st.write(f"Model: `{config.openai.model}`")
        st.write("API key: " + ("loaded" if config.openai.api_key else "missing"))
        if use_openai and not config.openai.api_key:
            st.warning("OpenAI is enabled but `OPENAI_API_KEY` is missing in `.env`.")

        st.divider()
        st.subheader("Data")
        st.write(f"Data dir: `{config.data.data_dir}`")
        st.write(f"Emails: **{len(email_index)}**")
        st.markdown("**Rate sheets**")
        for diff in ["easy", "medium", "hard"]:
            st.write(f"- {diff.title()}: `{paths[diff]}`" + (" (found)" if paths[diff].exists() else " (missing)"))

    if not email_index:
        st.error(f"No emails found under: `{config.data.data_dir}`")
        return

    st.subheader("0) Rate Sheet Normalization (required)")
    st.caption("Normalize the Excel sheets and review the canonicalized tables before quoting emails.")

    cols = st.columns([0.75, 0.25])
    with cols[1]:
        if st.button("Normalize all", type="secondary"):
            for diff in ["easy", "medium", "hard"]:
                if paths[diff].exists():
                    rates_cache[diff] = normalize_rate_sheets(config=config, difficulty=diff)
            st.rerun()

    tabs = st.tabs(["Easy", "Medium", "Hard"])
    for tab, diff in zip(tabs, ["easy", "medium", "hard"], strict=True):
        with tab:
            source_path = paths[diff]
            rates = _get_cached_rates(rates_cache=rates_cache, difficulty=diff, source_path=source_path)

            top = st.columns([0.7, 0.3])
            with top[0]:
                st.write(f"Source: `{source_path}`")
                st.write("Status: " + ("Normalized" if rates else "Not normalized"))
            with top[1]:
                if st.button(f"Normalize {diff.title()}", key=f"normalize_{diff}", disabled=not source_path.exists()):
                    with st.spinner(f"Normalizing {diff}..."):
                        rates_cache[diff] = normalize_rate_sheets(config=config, difficulty=diff)
                    st.rerun()

            if not rates:
                continue

            if rates.warnings:
                st.warning("Warnings:\n- " + "\n- ".join(rates.warnings))

            st.markdown("**Normalized: Sea**")
            st.dataframe(rates.sea, use_container_width=True)
            st.markdown("**Normalized: Air**")
            st.dataframe(rates.air, use_container_width=True)

            with st.expander("Mappings (codes / aliases)", expanded=False):
                st.write({"codes": rates.codes, "aliases": rates.aliases})

            with st.expander("Canonical locations (derived)", expanded=False):
                st.write(
                    {
                        "sea_origins": sorted(rates.sea["Origin"].unique().tolist()),
                        "sea_destinations": sorted(rates.sea["Destination"].unique().tolist()),
                        "air_origins": sorted(rates.air["Origin"].unique().tolist()),
                        "air_destinations": sorted(rates.air["Destination"].unique().tolist()),
                    }
                )

    selected_rates = _get_cached_rates(rates_cache=rates_cache, difficulty=difficulty, source_path=paths[difficulty])
    if not selected_rates:
        st.info(f"Normalize the `{difficulty}` sheet to enable quoting.")

    col_left, col_right = st.columns([0.45, 0.55], gap="large")

    with col_left:
        st.subheader("1) Select an Email")
        email_id = st.selectbox("Email", options=sorted(email_index.keys()))
        email = load_email(email_index[email_id])

        st.markdown("**Subject**")
        st.code(email.subject, language="text")

        st.markdown("**Body**")
        st.code(email.body, language="text")

        run_btn = st.button("Run Quote Pipeline", type="primary", disabled=selected_rates is None)

    with col_right:
        st.subheader("2) Output")
        if run_btn and selected_rates is not None:
            with st.spinner("Running pipeline..."):
                result = run_quote_pipeline(
                    email=email,
                    config=config,
                    difficulty=difficulty,
                    use_openai=use_openai,
                    rate_sheets=selected_rates,
                )

            if result.error:
                st.error(result.error)
            else:
                st.markdown("**Formatted Quote**")
                st.code(result.quote_text or "", language="text")

            st.markdown("**Trace (steps + artifacts)**")
            for step in result.trace.steps:
                with st.expander(step.title, expanded=False):
                    if step.summary:
                        st.caption(step.summary)
                    if step.data is not None:
                        st.json(step.data)


if __name__ == "__main__":
    main()
