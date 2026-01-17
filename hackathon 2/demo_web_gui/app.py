from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

import streamlit as st

from src.config import load_app_config
from src.data_loader import list_emails, load_email
from src.db_store import (
    DbNotConfiguredError,
    count_email_records_by_status,
    healthcheck as db_healthcheck,
    init_schema as db_init_schema,
    insert_email_record,
    list_email_records,
    list_needs_human_decision,
)
from api_server import _confidence_from_reply, _decide_reply_type
from src.pipeline import run_quote_pipeline
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets
from src.sop import SopParseResult, load_sop_markdown, parse_sop


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


def _get_sop_parse_cache() -> dict[str, SopParseResult]:
    if "sop_parse_results" not in st.session_state:
        st.session_state["sop_parse_results"] = {}
    return st.session_state["sop_parse_results"]


def _sop_parse_key(
    *,
    sop_markdown: str,
    model: str,
    canonical_origins: list[str],
    canonical_destinations: list[str],
) -> str:
    payload = {
        "sop": sop_markdown,
        "model": model,
        "origins": sorted(set([str(x) for x in canonical_origins])),
        "destinations": sorted(set([str(x) for x in canonical_destinations])),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _list_known_senders(email_index: dict[str, Path]) -> list[str]:
    senders: list[str] = []
    for path in email_index.values():
        try:
            msg = load_email(path)
        except Exception:
            continue
        sender = str(msg.sender or "").strip()
        if sender:
            senders.append(sender)
    return sorted(set(senders))


def _get_trace_step(*, trace_payload: dict, title: str) -> dict | None:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and str(step.get("title") or "") == title:
            return step
    return None


def _extract_shipments_from_trace(trace_payload: dict) -> list[dict]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Extraction:"):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        shipments = data.get("shipments")
        if isinstance(shipments, list):
            return [s for s in shipments if isinstance(s, dict)]
    return []


def _extract_canonical_origins_from_trace(trace_payload: dict) -> list[str]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    origins: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Normalize Locations ("):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        origin = data.get("origin")
        if isinstance(origin, dict):
            canonical = origin.get("canonical")
            if canonical and str(canonical).strip():
                origins.append(str(canonical).strip())
    return origins


def _extract_canonical_destinations_from_trace(trace_payload: dict) -> list[str]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    destinations: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Normalize Locations ("):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        destination = data.get("destination")
        if isinstance(destination, dict):
            canonical = destination.get("canonical")
            if canonical and str(canonical).strip():
                destinations.append(str(canonical).strip())
    return destinations


def _extract_final_amounts_from_trace(trace_payload: dict) -> list[float]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    out: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Calculate Quote ("):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        amt = data.get("final_amount")
        if isinstance(amt, (int, float)):
            out.append(float(amt))
    return out


def _extract_currencies_from_trace(trace_payload: dict) -> list[str]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    out: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Calculate Quote ("):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        currency = data.get("currency")
        if currency and str(currency).strip():
            out.append(str(currency).strip())
    return out


def _extract_transit_days_from_trace(trace_payload: dict) -> list[int]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    out: list[int] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if not title.startswith("Rate Lookup ("):
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        transit_days = data.get("transit_days")
        if isinstance(transit_days, bool):
            continue
        if isinstance(transit_days, int):
            out.append(int(transit_days))
        elif isinstance(transit_days, float) and transit_days.is_integer():
            out.append(int(transit_days))
    return out


def _extract_transport_types_from_trace(trace_payload: dict) -> list[str]:
    modes: list[str] = []
    shipments = _extract_shipments_from_trace(trace_payload)
    for sh in shipments:
        mode = str(sh.get("mode") or "").strip().casefold()
        if mode in {"air", "sea"}:
            modes.append(mode)

    if modes:
        return modes

    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        if title == "Normalize Locations (Air)":
            modes.append("air")
        elif title == "Normalize Locations (Sea)":
            modes.append("sea")
    return modes


def _join_unique(values: list[str], *, sep: str = "; ") -> str | None:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = str(value or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    if not out:
        return None
    return sep.join(out)


def _infer_quote_fields_from_trace(trace_payload: dict) -> dict:
    origins = _extract_canonical_origins_from_trace(trace_payload)
    destinations = _extract_canonical_destinations_from_trace(trace_payload)
    origin_city = _join_unique(origins)
    destination_city = _join_unique(destinations)

    transport_types = _extract_transport_types_from_trace(trace_payload)
    unique_types: list[str] = []
    for t in transport_types:
        t_norm = str(t or "").strip().casefold()
        if t_norm not in {"air", "sea"}:
            continue
        if t_norm not in unique_types:
            unique_types.append(t_norm)
    transport_type: str | None = None
    if len(unique_types) == 1:
        transport_type = unique_types[0]
    elif len(unique_types) > 1:
        transport_type = "mixed"

    amounts = _extract_final_amounts_from_trace(trace_payload)
    rounded_amounts = [float(round(x)) for x in amounts]
    price: float | None = None
    if rounded_amounts:
        price = float(sum(rounded_amounts))

    currencies = _extract_currencies_from_trace(trace_payload)
    currency: str | None = None
    if currencies:
        first = str(currencies[0]).strip()
        if first:
            currency = first

    has_route = bool(_extract_transit_days_from_trace(trace_payload))

    return {
        "origin_city": origin_city,
        "destination_city": destination_city,
        "price": price,
        "currency": currency,
        "transport_type": transport_type,
        "has_route": has_route,
    }


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
        enable_sop = st.toggle(
            "Enable SOP rules (`SOP.md`)",
            value=False,
            help="Applies customer-specific SOP overrides (mode restrictions, discounts, surcharges, formatting).",
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
        sop_path = (config.data.data_dir / "SOP.md").resolve()
        st.write("SOP: " + ("found" if sop_path.exists() else "missing") + f" (`{sop_path}`)")
        st.markdown("**Rate sheets**")
        for diff in ["easy", "medium", "hard"]:
            st.write(f"- {diff.title()}: `{paths[diff]}`" + (" (found)" if paths[diff].exists() else " (missing)"))

        st.divider()
        st.subheader("Persistence (optional)")
        has_supabase = bool(os.getenv("SUPABASE_URL")) and bool(
            os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        )
        has_pg = bool(os.getenv("DATABASE_URL"))
        st.write("Mode: " + ("Supabase REST" if has_supabase else ("Postgres" if has_pg else "disabled")))
        auto_persist = st.toggle(
            "Auto-persist latest run",
            value=False,
            help="Stores email + quote + trace + inferred fields into `email_quote_records` (requires DB config).",
        )

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

            with st.expander("Normalized: Sea", expanded=False):
                st.dataframe(rates.sea, use_container_width=True)
            with st.expander("Normalized: Air", expanded=False):
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

    sop_parse_result: SopParseResult | None = None
    sop_config_for_pipeline = None
    if enable_sop:
        st.markdown("**SOP parsing (once, reused for all emails)**")
        st.caption("Parses `SOP.md` once using canonical locations from the normalized sheet, and reuses it for all email runs.")
        if selected_rates is None:
            st.info("Normalize the selected difficulty first to parse SOP.")
        else:
            df_air = selected_rates.air
            df_sea = selected_rates.sea
            canonical_origins = sorted(
                set(
                    [str(x) for x in df_air["Origin"].dropna().tolist()]
                    + [str(x) for x in df_sea["Origin"].dropna().tolist()]
                )
            )
            canonical_destinations = sorted(
                set(
                    [str(x) for x in df_air["Destination"].dropna().tolist()]
                    + [str(x) for x in df_sea["Destination"].dropna().tolist()]
                )
            )
            sop_markdown = load_sop_markdown(data_dir=config.data.data_dir)
            key = _sop_parse_key(
                sop_markdown=sop_markdown,
                model=config.openai.model,
                canonical_origins=canonical_origins,
                canonical_destinations=canonical_destinations,
            )
            sop_cache = _get_sop_parse_cache()
            sop_parse_result = sop_cache.get(key)

            left, right = st.columns([0.25, 0.75])
            with left:
                parse_now = st.button("Parse SOP", type="secondary", key="parse_sop_now")
            with right:
                st.write(
                    {
                        "origins": len(canonical_origins),
                        "destinations": len(canonical_destinations),
                        "sop_md_loaded": bool(sop_markdown.strip()),
                    }
                )

            if sop_parse_result is None or parse_now:
                with st.spinner("Parsing SOP..."):
                    sop_parse_result = parse_sop(
                        config=config,
                        canonical_origins=canonical_origins,
                        canonical_destinations=canonical_destinations,
                        known_senders=_list_known_senders(email_index),
                        force_refresh=bool(parse_now),
                    )
                sop_cache[key] = sop_parse_result

            if sop_parse_result:
                sop_config_for_pipeline = sop_parse_result.sop_config
                st.write(
                    {
                        "sop_path": str(sop_parse_result.sop_path),
                        "sop_loaded": sop_parse_result.sop_loaded,
                        "parse_source": sop_parse_result.sop_config.source,
                        "profiles": len(sop_parse_result.sop_config.profiles),
                        "global_surcharge_rules": len(sop_parse_result.sop_config.global_surcharge_rules),
                        "cached": sop_parse_result.cached,
                    }
                )
                if sop_parse_result.errors:
                    st.warning("SOP parse notes:\n- " + "\n- ".join(sop_parse_result.errors))
                if sop_parse_result.llm_usage:
                    st.caption(
                        "LLM usage: "
                        + ", ".join(
                            [
                                f"calls={sop_parse_result.llm_usage.get('calls')}",
                                f"prompt={sop_parse_result.llm_usage.get('prompt_tokens')}",
                                f"completion={sop_parse_result.llm_usage.get('completion_tokens')}",
                                f"total={sop_parse_result.llm_usage.get('total_tokens')}",
                            ]
                        )
                    )
                if sop_parse_result.llm_raw:
                    with st.expander("SOP parser LLM raw output (JSON)", expanded=False):
                        st.json(sop_parse_result.llm_raw)

    col_left, col_right = st.columns([0.45, 0.55], gap="large")

    with col_left:
        st.subheader("1) Select an Email")
        with st.expander("Create & save a custom email", expanded=False):
            emails_dir = (config.data.data_dir / "emails").resolve()
            st.caption(f"Saves to: `{emails_dir}`")

            if "custom_email_id" not in st.session_state:
                st.session_state["custom_email_id"] = "email_custom_01"

            def _sanitize_email_stem(raw: str) -> str:
                s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip()).strip("_")
                if not s:
                    s = f"email_custom_{int(time.time())}"
                if not s.casefold().startswith("email_"):
                    s = f"email_{s}"
                return s

            custom_email_id = st.text_input(
                "Email ID (filename stem, must start with `email_`)",
                key="custom_email_id",
                help="Will be saved as `email_*.json` so it appears in the dropdown.",
            )
            custom_from = st.text_input("From", value="user@example.com", key="custom_from")
            custom_to = st.text_input("To", value="quotes@freightco.com", key="custom_to")
            custom_subject = st.text_input("Subject", value="Custom quote request", key="custom_subject")
            custom_body = st.text_area("Body", value="", height=180, key="custom_body")
            overwrite = st.toggle("Overwrite if file exists", value=False, key="custom_overwrite")

            save_btn = st.button("Save email to local folder", type="secondary", key="save_custom_email")
            if save_btn:
                stem = _sanitize_email_stem(custom_email_id)
                emails_dir.mkdir(parents=True, exist_ok=True)
                path = (emails_dir / f"{stem}.json").resolve()
                if path.exists() and not overwrite:
                    st.error(f"File already exists: `{path}` (enable overwrite to replace).")
                else:
                    payload = {
                        "from": str(custom_from or "").strip(),
                        "to": str(custom_to or "").strip(),
                        "subject": str(custom_subject or "").strip(),
                        "body": str(custom_body or "").strip(),
                    }
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success(f"Saved: `{path}`")
                    st.session_state["selected_email_id"] = stem
                    st.rerun()

        email_id = st.selectbox("Email", options=sorted(email_index.keys()), key="selected_email_id")
        email = load_email(email_index[email_id])

        st.markdown("**Subject**")
        st.code(email.subject, language="text")

        st.markdown("**Body**")
        st.code(email.body, language="text")

        sol_path = (config.data.data_dir / "solutions" / f"solution_{email_id}.md").resolve()
        with st.expander("Standard answer (ground truth)", expanded=False):
            if sol_path.exists():
                st.markdown(sol_path.read_text(encoding="utf-8", errors="replace"))
            else:
                st.info(f"No solution file found at `{sol_path}`")

        run_btn = st.button("Run Quote Pipeline", type="primary", disabled=selected_rates is None)

    with col_right:
        st.subheader("2) Output")
        if run_btn and selected_rates is not None:
            with st.spinner("Running pipeline..."):
                t0 = time.perf_counter()
                result = run_quote_pipeline(
                    email=email,
                    config=config,
                    difficulty=difficulty,
                    use_openai=use_openai,
                    rate_sheets=selected_rates,
                    enable_sop=enable_sop,
                    sop_config=sop_config_for_pipeline,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            trace_payload = asdict(result.trace)
            inferred = _infer_quote_fields_from_trace(trace_payload)
            st.session_state["last_run"] = {
                "email_id": email.email_id,
                "difficulty": difficulty,
                "enable_sop": bool(enable_sop),
                "use_openai": bool(use_openai),
                "quote_text": result.quote_text,
                "error": result.error,
                "elapsed_ms": elapsed_ms,
                "trace": trace_payload,
                "inferred": inferred,
            }

            st.markdown("**Run metrics**")
            st.write(f"Latency: {elapsed_ms:.0f} ms")
            st.write(
                {
                    "llm_calls": result.trace.llm_usage.calls,
                    "prompt_tokens": result.trace.llm_usage.prompt_tokens,
                    "completion_tokens": result.trace.llm_usage.completion_tokens,
                    "total_tokens": result.trace.llm_usage.total_tokens,
                }
            )

            if result.error:
                st.error(result.error)
            else:
                st.markdown("**Formatted Quote**")
                st.code(result.quote_text or "", language="text")

            st.markdown("**Inferred fields (for DB/API)**")
            st.write(inferred)

            st.markdown("**Trace (steps + artifacts)**")
            for step in result.trace.steps:
                marker = "LLM" if step.used_llm else "local"
                with st.expander(f"[{marker}] {step.title}", expanded=False):
                    if step.summary:
                        st.caption(step.summary)
                    if step.llm_usage:
                        st.caption(
                            "LLM usage: "
                            + ", ".join(
                                [
                                    f"calls={step.llm_usage.get('calls')}",
                                    f"prompt={step.llm_usage.get('prompt_tokens')}",
                                    f"completion={step.llm_usage.get('completion_tokens')}",
                                    f"total={step.llm_usage.get('total_tokens')}",
                                ]
                            )
                        )
                    if step.data is not None:
                        st.json(step.data)

            if auto_persist:
                st.markdown("**DB auto-persist**")
                config_payload = {
                    "difficulty": difficulty,
                    "enable_sop": bool(enable_sop),
                    "use_openai": bool(use_openai),
                }
                try:
                    reply_type, hitl_reasons = _decide_reply_type(
                        trace_payload=trace_payload,
                        email_subject=str(email.subject or ""),
                        email_body=str(email.body or ""),
                    )
                    if result.error:
                        hitl_reasons.append({"code": "pipeline_error", "detail": str(result.error)})
                    if not result.quote_text:
                        hitl_reasons.append({"code": "missing_quote_text"})
                    if hitl_reasons:
                        reply_type = "HITL"
                    confidence = _confidence_from_reply(reply_type=reply_type, hitl_reasons=hitl_reasons)

                    record_id = insert_email_record(
                        email_id=email.email_id,
                        email_from=email.sender,
                        email_to=email.to,
                        subject=email.subject,
                        body=email.body,
                        reply=result.quote_text,
                        trace=trace_payload,
                        record_type="auto",
                        status=("unprocessed" if result.error else "needs_human_decision"),
                        confidence=confidence,
                        config=config_payload,
                        origin_city=inferred.get("origin_city"),
                        destination_city=inferred.get("destination_city"),
                        price=inferred.get("price"),
                        currency=inferred.get("currency"),
                        transport_type=inferred.get("transport_type"),
                        has_route=inferred.get("has_route"),
                    )
                    st.success(f"Saved to DB: `{record_id}`")
                except DbNotConfiguredError as e:
                    st.warning(str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"DB persist failed: {e.__class__.__name__}: {e}")

        with st.expander("3) Persistence dashboard (optional)", expanded=False):
            col_a, col_b = st.columns([0.25, 0.75])
            with col_a:
                if st.button("DB healthcheck", type="secondary", key="db_healthcheck"):
                    try:
                        st.write(db_healthcheck())
                    except DbNotConfiguredError as e:
                        st.warning(str(e))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{e.__class__.__name__}: {e}")
                if st.button("DB init/verify", type="secondary", key="db_init"):
                    try:
                        st.write(db_init_schema())
                    except DbNotConfiguredError as e:
                        st.warning(str(e))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{e.__class__.__name__}: {e}")

            with col_b:
                st.markdown("**Status counts**")
                try:
                    st.write(count_email_records_by_status())
                except DbNotConfiguredError as e:
                    st.warning(str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"{e.__class__.__name__}: {e}")

            st.markdown("**Needs human decision (latest 50)**")
            try:
                items = list_needs_human_decision(limit=50, offset=0)
                st.dataframe(items, use_container_width=True)
            except DbNotConfiguredError as e:
                st.warning(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"{e.__class__.__name__}: {e}")

            st.markdown("**All records (latest 50)**")
            try:
                items = list_email_records(limit=50, offset=0)
                st.dataframe(items, use_container_width=True)
            except DbNotConfiguredError as e:
                st.warning(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"{e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
