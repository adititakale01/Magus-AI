from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from fastapi import APIRouter, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import AppConfig, OpenAIConfig, load_app_config
from src.data_loader import EmailMessage, list_emails, load_email
from src.db_store import DbNotConfiguredError, count_email_records_by_status, get_email_record, healthcheck as db_healthcheck
from src.db_store import init_schema as db_init_schema
from src.db_store import insert_email_record, list_email_records, list_needs_human_decision, update_email_record
from src.llm_usage import usage_from_response
from src.pipeline import run_quote_pipeline
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets
from src.sop import SopConfig, SopParseResult, parse_sop


CONFIG_PATH = APP_DIR / "config.toml"


class SopParseMode(str):
    pass


Difficulty = Literal["easy", "medium", "hard"]
SopParseModeLiteral = Literal["auto", "llm", "rule_based"]
EmailRecordTypeLiteral = Literal["auto", "human"]
EmailRecordStatusLiteral = Literal[
    "unprocessed",
    "auto_processed",
    "needs_human_decision",
    "human_confirmed_replied",
    "human_rejected",
]
ConfidenceLiteral = Literal["high", "medium", "low"]


class EmailPayload(BaseModel):
    email_id: str | None = None
    from_: str = Field(alias="from")
    to: str
    subject: str
    body: str

    model_config = {"populate_by_name": True}


class SopUpdateRequest(BaseModel):
    markdown: str


class SopParseRequest(BaseModel):
    difficulty: Difficulty
    sop_parse_mode: SopParseModeLiteral = "auto"
    force_refresh: bool = False


class QuoteRequest(BaseModel):
    difficulty: Difficulty
    enable_sop: bool = False
    use_openai: bool = False
    email: EmailPayload | str
    email_id: str | None = None
    sop_parse_mode: SopParseModeLiteral = "auto"
    force_sop_refresh: bool = False
    include_trace: bool = True
    persist: bool = False
    record_type: EmailRecordTypeLiteral = "auto"
    record_status: EmailRecordStatusLiteral = "needs_human_decision"


class EmailRecordCreateRequest(BaseModel):
    email_id: str | None = None
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    subject: str | None = None
    body: str | None = None
    reply: str | None = None
    trace: dict[str, Any] | None = None
    type: EmailRecordTypeLiteral = "auto"
    status: EmailRecordStatusLiteral = "unprocessed"
    confidence: ConfidenceLiteral | None = None
    config: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class EmailRecordUpdateRequest(BaseModel):
    status: EmailRecordStatusLiteral | None = None
    type: EmailRecordTypeLiteral | None = None
    reply: str | None = None
    trace: dict[str, Any] | None = None
    confidence: ConfidenceLiteral | None = None
    config: dict[str, Any] | None = None


DecisionLiteral = Literal["accept", "reject"]


class EmailRecordDecisionRequest(BaseModel):
    id: str
    decision: DecisionLiteral
    refined_quote: str | None = None
    comment: str | None = None


_RATES_CACHE: dict[str, NormalizedRateSheets] = {}
_RATES_LOCK = threading.Lock()

_SOP_PARSE_CACHE: dict[str, tuple[str, SopParseResult]] = {}
_SOP_LOCK = threading.Lock()

_SPECIAL_REQUEST_KEYWORDS = [
    r"\bddp\b",
    r"\bdap\b",
    r"\bddu\b",
    r"\bdoor\s*to\s*door\b",
    r"\bdoor[-\s]*to[-\s]*door\b",
    r"\bcustoms\b",
    r"\bclearance\b",
    r"\bduty\b",
    r"\binsurance\b",
    r"\bhazard(?:ous)?\b",
    r"\bdg\b",
    r"\bdangerous\s*goods\b",
    r"\blithium\b",
    r"\bbattery\b",
    r"\bmsds\b",
    r"\btemperature\b",
    r"\bcold\s*chain\b",
    r"\breefer\b",
    r"\burgent\b",
    r"\basap\b",
    r"\bdeadline\b",
]

_COMMON_GENERIC_EMAIL_DOMAINS = {
    "163.com",
    "126.com",
    "qq.com",
    "foxmail.com",
    "sina.com",
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "msn.com",
    "yahoo.com",
    "icloud.com",
    "gmx.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "yandex.com",
    "yandex.ru",
}


def _resolve_rate_sheet_path(*, config: AppConfig, difficulty: Difficulty) -> Path:
    if difficulty == "easy":
        rel = config.rates.easy_file
    elif difficulty == "medium":
        rel = config.rates.medium_file
    elif difficulty == "hard":
        rel = config.rates.hard_file
    else:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return (config.data.data_dir / rel).resolve()


def _load_base_config() -> AppConfig:
    return load_app_config(CONFIG_PATH)


def _with_openai_key(*, config: AppConfig, openai_api_key: str | None) -> AppConfig:
    key = str(openai_api_key).strip() if openai_api_key else None
    openai_cfg = OpenAIConfig(
        api_key=key or config.openai.api_key,
        model=config.openai.model,
        temperature=config.openai.temperature,
    )
    return AppConfig(
        data=config.data,
        rates=config.rates,
        pricing=config.pricing,
        openai=openai_cfg,
        aliases=config.aliases,
        codes=config.codes,
    )


def _require_openai_key(*, config: AppConfig, provided: str | None, reason: str) -> AppConfig:
    effective = _with_openai_key(config=config, openai_api_key=provided)
    if not effective.openai.api_key:
        raise HTTPException(status_code=401, detail=f"Missing OpenAI API key for {reason}. Provide `X-OpenAI-Api-Key`.")
    return effective


def _get_openai_key(
    *,
    x_openai_api_key: str | None,
    x_api_key: str | None,
) -> str | None:
    if x_openai_api_key and str(x_openai_api_key).strip():
        return str(x_openai_api_key).strip()
    if x_api_key and str(x_api_key).strip():
        return str(x_api_key).strip()
    return None


def _refine_quote_with_openai(
    *,
    config: AppConfig,
    email_subject: str,
    email_body: str,
    refined_quote: str,
    comment: str,
) -> tuple[str, dict[str, Any] | None, str]:
    client = OpenAI(api_key=config.openai.api_key)

    system = (
        "You are a freight forwarding quoting assistant. "
        "Improve the draft quote email using the human reviewer comment and the original customer email. "
        "Be concise, professional, and include only the final quote message. "
        "Do not include meta commentary, analysis, or JSON."
    )
    user = (
        "Original customer email:\n"
        f"Subject: {email_subject}\n"
        f"Body:\n{email_body}\n\n"
        "Draft quote:\n"
        f"{refined_quote}\n\n"
        "Human reviewer comment / requested changes:\n"
        f"{comment}\n\n"
        "Task: Rewrite the quote email accordingly. Output only the final quote text."
    )

    resp = client.chat.completions.create(
        model=config.openai.model,
        temperature=config.openai.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return content, usage_from_response(resp, model=config.openai.model, calls=1), content


def _post_make_webhook(*, url: str, payload: dict[str, Any], timeout_s: float = 20.0) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            return {"status_code": int(getattr(resp, "status", 0) or 0), "response_body": resp_body}
    except HTTPError as e:
        resp_body = ""
        try:
            resp_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"status_code": int(getattr(e, "code", 0) or 0), "error": str(e), "response_body": resp_body}
    except URLError as e:
        return {"status_code": 0, "error": str(e)}


_HEADER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\s*:\s*(.*)$")


def _parse_raw_email_text(raw: str) -> dict[str, str]:
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("\n")
    if not text.strip():
        return {"from": "", "to": "", "subject": "", "body": ""}

    lines = text.split("\n")

    header_lines: list[str] = []
    body_start_idx: int | None = None

    for idx, line in enumerate(lines):
        if idx == 0 and not _HEADER_LINE_RE.match(line):
            body_start_idx = 0
            break

        if not line.strip():
            body_start_idx = idx + 1
            break

        if _HEADER_LINE_RE.match(line) or (line.startswith(" ") or line.startswith("\t")):
            header_lines.append(line)
            continue

        body_start_idx = idx
        break

    if body_start_idx is None:
        body_start_idx = len(lines)

    headers: dict[str, str] = {}
    current_key: str | None = None
    for line in header_lines:
        if (line.startswith(" ") or line.startswith("\t")) and current_key:
            headers[current_key] = (headers[current_key] + " " + line.strip()).strip()
            continue

        m = _HEADER_LINE_RE.match(line)
        if not m:
            continue
        key = str(m.group(1) or "").strip().casefold()
        value = str(m.group(2) or "").strip()
        if not key:
            continue
        if key in headers and value:
            headers[key] = f"{headers[key]}, {value}"
        else:
            headers[key] = value
        current_key = key

    has_any_expected_header = any(headers.get(k) for k in ("from", "to", "subject"))
    if not has_any_expected_header:
        return {"from": "", "to": "", "subject": "", "body": text}

    body = "\n".join(lines[body_start_idx:]).strip("\n")
    return {
        "from": str(headers.get("from") or ""),
        "to": str(headers.get("to") or ""),
        "subject": str(headers.get("subject") or ""),
        "body": body,
    }


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    df2 = df.copy()
    df2 = df2.where(pd.notna(df2), None)
    records: list[dict[str, Any]] = []
    for rec in df2.to_dict(orient="records"):
        out: dict[str, Any] = {}
        for k, v in rec.items():
            if hasattr(v, "item") and callable(v.item):  # numpy scalars
                try:
                    v = v.item()
                except Exception:
                    pass
            out[str(k)] = v
        records.append(out)
    return records


def _email_record_api_shape(item: dict[str, Any]) -> dict[str, Any]:
    trace_value = item.get("trace")
    if isinstance(trace_value, str):
        parsed = _coerce_trace_payload(trace_value)
        if parsed:
            trace_value = parsed
    return {
        "id": item.get("id"),
        "time": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "email_id": item.get("email_id"),
        "from": item.get("email_from"),
        "to": item.get("email_to"),
        "subject": item.get("subject"),
        "body": item.get("body"),
        "reply": item.get("reply"),
        "trace": trace_value,
        "type": item.get("type"),
        "status": item.get("status"),
        "confidence": item.get("confidence"),
        "config": item.get("config"),
        "origin_city": item.get("origin_city"),
        "destination_city": item.get("destination_city"),
        "price": item.get("price"),
        "currency": item.get("currency"),
        "transport_type": item.get("transport_type"),
        "has_route": item.get("has_route"),
    }


def _first_email_address(text: str | None) -> str | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    m = re.search(r"(?i)\b([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b", s)
    if not m:
        return None
    return str(m.group(1)).strip()


def _email_domain(text: str | None) -> str | None:
    email = _first_email_address(text)
    if not email:
        return None
    parts = email.split("@", 1)
    if len(parts) != 2:
        return None
    domain = parts[1].strip().casefold()
    return domain or None


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _split_semicolon_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    parts = re.split(r"\s*;\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _coerce_trace_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}

    if isinstance(value, str):
        raw = str(value)
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except Exception:  # noqa: BLE001
            return {"raw_trace": raw}

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"steps": parsed}
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
            except Exception:  # noqa: BLE001
                return {"raw_trace": raw}
            if isinstance(parsed2, dict):
                return parsed2
            if isinstance(parsed2, list):
                return {"steps": parsed2}

        return {"raw_trace": raw}

    return {"raw_trace": str(value)}


def _hitl_summary_from_trace(trace_payload: dict[str, Any]) -> str:
    sop_step = _get_trace_step(trace_payload=trace_payload, title="SOP")
    if not isinstance(sop_step, dict):
        return "Requires human review based on heuristic signals."
    sop_data = sop_step.get("data")
    if not isinstance(sop_data, dict):
        return "Requires human review based on heuristic signals."
    matched_profile = sop_data.get("matched_profile")
    if isinstance(matched_profile, dict):
        customer_name = str(matched_profile.get("customer_name") or "").strip()
        if customer_name:
            return f"Requires human review (related to SOP VIP member: {customer_name})."
        return "Requires human review (related to SOP VIP member)."
    return "Requires human review based on heuristic signals."


def _extract_clarification_questions(trace_payload: dict[str, Any]) -> list[str]:
    step = _get_trace_step(trace_payload=trace_payload, title="Clarification Required")
    if not isinstance(step, dict):
        return []
    data = step.get("data")
    if not isinstance(data, dict):
        return []
    questions = data.get("questions")
    if not isinstance(questions, list):
        return []
    return [str(x).strip() for x in questions if str(x).strip()]


def _classify_clarifications(questions: list[str]) -> set[str]:
    if not questions:
        return set()
    text = "\n".join(questions).casefold()
    reasons: set[str] = set()
    if re.search(r"\borigin\b", text):
        reasons.add("missing_origin")
    if re.search(r"\bdestination\b", text):
        reasons.add("missing_destination")
    if re.search(r"\bsea\b|\bocean\b|\bair\b|\bfreight\b", text) and re.search(r"\bconfirm\b|\bwhether\b", text):
        reasons.add("missing_mode")
    if re.search(r"\bweight\b|\bkg\b|\bcbm\b|\bvolume\b", text):
        reasons.add("missing_weight_volume")
    if re.search(r"\bcontainer\b|\b20ft\b|\b40ft\b|\bquantity\b", text):
        reasons.add("missing_container_info")
    if not reasons:
        reasons.add("clarification_required")
    return reasons


def _parse_mode_from_title(title: str) -> str | None:
    m = re.search(r"\((Air|Sea)\)\s*$", str(title))
    if not m:
        return None
    return str(m.group(1)).strip().casefold()


def _find_last_route_index(routes: list[dict[str, Any]], mode: str | None, *, need_field: str) -> int | None:
    if not mode:
        return None
    for i in range(len(routes) - 1, -1, -1):
        r = routes[i]
        if str(r.get("mode") or "") != mode:
            continue
        if r.get(need_field) is None:
            return i
    return None


def _extract_routes_from_trace(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return []

    routes: list[dict[str, Any]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")

        if title.startswith("Normalize Locations ("):
            mode = _parse_mode_from_title(title)
            data = step.get("data")
            origin = None
            destination = None
            if isinstance(data, dict):
                o = data.get("origin")
                d = data.get("destination")
                if isinstance(o, dict):
                    origin = str(o.get("canonical") or "").strip() or None
                if isinstance(d, dict):
                    destination = str(d.get("canonical") or "").strip() or None
            routes.append(
                {
                    "mode": mode,
                    "origin": origin,
                    "destination": destination,
                    "rate_found": None,
                    "transit_days": None,
                    "final_amount": None,
                    "currency": None,
                    "chargeable_weight_kg": None,
                    "actual_weight_kg": None,
                    "volume_cbm": None,
                    "quantity": None,
                    "container_size_ft": None,
                }
            )
            continue

        if title.startswith("Rate Lookup ("):
            mode = _parse_mode_from_title(title)
            idx = _find_last_route_index(routes, mode, need_field="rate_found")
            if idx is None:
                continue
            r = routes[idx]
            data = step.get("data") if isinstance(step.get("data"), dict) else {}
            summary = str(step.get("summary") or "")

            if isinstance(data, dict):
                transit_days = data.get("transit_days")
                if isinstance(transit_days, bool):
                    transit_days = None
                transit_num = _coerce_number(transit_days)
                if transit_num is not None and float(transit_num).is_integer():
                    r["transit_days"] = int(transit_num)

            if re.search(r"(?i)\bno\s+direct\b", summary):
                r["rate_found"] = False
            else:
                matched = bool(re.search(r"(?i)\bmatched\b", summary))
                has_rate_fields = isinstance(data, dict) and any(
                    k in data for k in ["base_rate", "Rate per kg (USD)", "20ft Price (USD)", "40ft Price (USD)"]
                )
                r["rate_found"] = bool(matched or has_rate_fields)
            continue

        if title.startswith("Calculate Quote ("):
            mode = _parse_mode_from_title(title)
            idx = _find_last_route_index(routes, mode, need_field="final_amount")
            if idx is None:
                continue
            r = routes[idx]
            data = step.get("data")
            if not isinstance(data, dict):
                continue

            amt = _coerce_number(data.get("final_amount"))
            if amt is not None:
                r["final_amount"] = float(amt)
            currency = str(data.get("currency") or "").strip()
            if currency:
                r["currency"] = currency
            chargeable = _coerce_number(data.get("chargeable_weight_kg"))
            if chargeable is not None:
                r["chargeable_weight_kg"] = float(chargeable)
            continue

    shipments = _extract_shipments_from_trace(trace_payload)
    by_mode: dict[str, list[dict[str, Any]]] = {"air": [], "sea": []}
    for sh in shipments:
        mode = str(sh.get("mode") or "").strip().casefold()
        if mode in by_mode:
            by_mode[mode].append(sh)

    ptr: dict[str, int] = {"air": 0, "sea": 0}
    for r in routes:
        mode = str(r.get("mode") or "")
        if mode not in by_mode:
            continue
        i = ptr[mode]
        if i >= len(by_mode[mode]):
            continue
        sh = by_mode[mode][i]
        ptr[mode] = i + 1

        r["actual_weight_kg"] = _coerce_number(sh.get("actual_weight_kg"))
        r["volume_cbm"] = _coerce_number(sh.get("volume_cbm"))
        qty = _coerce_number(sh.get("quantity"))
        r["quantity"] = int(qty) if qty is not None and float(qty).is_integer() else None
        size = _coerce_number(sh.get("container_size_ft"))
        r["container_size_ft"] = int(size) if size is not None and float(size).is_integer() else None

    return routes


def _get_trace_step(*, trace_payload: dict[str, Any], title: str) -> dict[str, Any] | None:
    steps = trace_payload.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and str(step.get("title") or "") == title:
            return step
    return None


def _extract_shipments_from_trace(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
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


def _extract_canonical_origins_from_trace(trace_payload: dict[str, Any]) -> list[str]:
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


def _extract_final_amounts_from_trace(trace_payload: dict[str, Any]) -> list[float]:
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


def _extract_canonical_destinations_from_trace(trace_payload: dict[str, Any]) -> list[str]:
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


def _extract_transport_types_from_trace(trace_payload: dict[str, Any]) -> list[str]:
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


def _extract_currencies_from_trace(trace_payload: dict[str, Any]) -> list[str]:
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


def _extract_transit_days_from_trace(trace_payload: dict[str, Any]) -> list[int]:
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


def _infer_quote_fields_from_trace(trace_payload: dict[str, Any]) -> dict[str, Any]:
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


def _decide_reply_type(*, trace_payload: dict[str, Any], email_subject: str, email_body: str) -> tuple[str, list[dict[str, Any]]]:
    reasons: list[dict[str, Any]] = []

    sop_step = _get_trace_step(trace_payload=trace_payload, title="SOP")
    sop_data = sop_step.get("data") if isinstance(sop_step, dict) else None

    sop_enabled = False
    sop_loaded = False
    sop_matched = False
    sop_allowed_modes: set[str] | None = None
    sop_origin_required: set[str] | None = None
    sop_parse_errors: list[str] = []

    if isinstance(sop_data, dict):
        sop_enabled = bool(sop_data.get("enabled"))
        sop_loaded = bool(sop_data.get("sop_loaded"))
        matched_profile = sop_data.get("matched_profile")
        sop_matched = isinstance(matched_profile, dict)
        if isinstance(matched_profile, dict):
            allowed_modes = matched_profile.get("allowed_modes")
            if isinstance(allowed_modes, list) and allowed_modes:
                sop_allowed_modes = {str(x) for x in allowed_modes if str(x).strip()}
            origin_required = matched_profile.get("origin_required")
            if isinstance(origin_required, list) and origin_required:
                sop_origin_required = {str(x) for x in origin_required if str(x).strip()}

        parse_errors = sop_data.get("parse_errors")
        if isinstance(parse_errors, list):
            sop_parse_errors = [str(x) for x in parse_errors if str(x).strip()]

    shipments = _extract_shipments_from_trace(trace_payload)
    if sop_enabled and sop_parse_errors:
        reasons.append({"code": "sop_parse_errors", "detail": sop_parse_errors})

    if sop_enabled and sop_loaded and sop_matched and sop_allowed_modes:
        for sh in shipments:
            mode = str(sh.get("mode") or "").strip()
            if mode and mode not in sop_allowed_modes:
                reasons.append({"code": "sop_mode_conflict", "detail": {"mode": mode, "allowed_modes": sorted(sop_allowed_modes)}})
                break

    if sop_enabled and sop_loaded and sop_matched and sop_origin_required:
        for origin in _extract_canonical_origins_from_trace(trace_payload):
            if origin not in sop_origin_required:
                reasons.append(
                    {"code": "sop_origin_conflict", "detail": {"origin_city": origin, "origin_required": sorted(sop_origin_required)}}
                )
                break

    clar_step = _get_trace_step(trace_payload=trace_payload, title="Clarification Required")
    if isinstance(clar_step, dict):
        clar_data = clar_step.get("data") if isinstance(clar_step.get("data"), dict) else None
        reasons.append({"code": "clarification_required", "detail": clar_data})

    threshold = float(os.getenv("HITL_LARGE_ORDER_USD") or 20000)
    amounts = _extract_final_amounts_from_trace(trace_payload)
    if amounts and (max(amounts) >= threshold or sum(amounts) >= threshold):
        reasons.append(
            {
                "code": "large_order",
                "detail": {"threshold_usd": threshold, "max_amount": max(amounts), "sum_amount": sum(amounts), "amounts": amounts},
            }
        )

    note_values = [str(sh.get("notes") or "").strip() for sh in shipments if str(sh.get("notes") or "").strip()]
    if note_values:
        note_matches: list[str] = []
        for note in note_values:
            if any(re.search(pat, note, flags=re.IGNORECASE) for pat in _SPECIAL_REQUEST_KEYWORDS):
                note_matches.append(note)
        if note_matches:
            reasons.append({"code": "special_request_notes", "detail": note_matches})

    text = f"{email_subject}\n{email_body}"
    keyword_matches: list[dict[str, str]] = []
    for pat in _SPECIAL_REQUEST_KEYWORDS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            keyword_matches.append({"pattern": pat, "match": m.group(0)})
    if keyword_matches:
        reasons.append({"code": "special_request_keywords", "detail": keyword_matches})

    if reasons:
        return "HITL", reasons
    return "Auto", []


def _confidence_from_reply(*, reply_type: str, hitl_reasons: list[dict[str, Any]]) -> str:
    if str(reply_type) == "Auto":
        return "high"

    low_codes = {"clarification_required", "pipeline_error", "missing_quote_text"}
    medium_codes = {
        "sop_parse_errors",
        "sop_mode_conflict",
        "sop_origin_conflict",
        "large_order",
        "special_request_notes",
        "special_request_keywords",
    }

    codes = {str(r.get("code") or "") for r in hitl_reasons if isinstance(r, dict)}
    if any(c in low_codes for c in codes):
        return "low"
    if any(c in medium_codes for c in codes):
        return "medium"
    return "medium"


def _rates_payload(*, rates: NormalizedRateSheets, include_rows: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "difficulty": rates.difficulty,
        "source_path": str(rates.source_path),
        "source_mtime": float(rates.source_mtime),
        "sheet_names": list(rates.sheet_names),
        "warnings": list(rates.warnings),
        "aliases": rates.aliases,
        "codes": rates.codes,
        "counts": {"sea_rows": int(rates.sea.shape[0]), "air_rows": int(rates.air.shape[0])},
    }
    if include_rows:
        payload["sea_rows"] = _df_to_records(rates.sea)
        payload["air_rows"] = _df_to_records(rates.air)
    return payload


def _sop_config_payload(sop: SopConfig) -> dict[str, Any]:
    return {
        "source": sop.source,
        "profiles": [
            {
                "customer_name": p.customer_name,
                "match_emails": sorted(list(p.match_emails)),
                "match_domains": sorted(list(p.match_domains)),
                "match_domain_keywords": sorted(list(p.match_domain_keywords)),
                "allowed_modes": sorted(list(p.allowed_modes)) if p.allowed_modes else None,
                "margin_override": p.margin_override,
                "sea_discount_pct": p.sea_discount_pct,
                "sea_discount_label": p.sea_discount_label,
                "origin_required": sorted(list(p.origin_required)) if p.origin_required else None,
                "origin_fallbacks": p.origin_fallbacks,
                "transit_warning_if_gt_days": p.transit_warning_if_gt_days,
                "show_actual_and_chargeable_weight": p.show_actual_and_chargeable_weight,
                "hide_margin_percent": p.hide_margin_percent,
                "container_volume_discount_tiers": p.container_volume_discount_tiers,
            }
            for p in sop.profiles
        ],
        "global_surcharge_rules": [
            {
                "amount": r.amount,
                "label": r.label,
                "destination_canonical_in": sorted(list(r.destination_canonical_in)),
            }
            for r in sop.global_surcharge_rules
        ],
    }


def _known_senders(*, config: AppConfig) -> list[str]:
    email_index = list_emails(config.data.data_dir)
    senders: list[str] = []
    for path in email_index.values():
        try:
            msg = load_email(path)
        except Exception:
            continue
        s = str(msg.sender or "").strip()
        if s:
            senders.append(s)
    return sorted(set(senders))


def _get_rates_cached(*, config: AppConfig, difficulty: Difficulty, force: bool = False) -> NormalizedRateSheets:
    source_path = _resolve_rate_sheet_path(config=config, difficulty=difficulty)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Rate sheet not found: {source_path}")
    mtime = source_path.stat().st_mtime

    with _RATES_LOCK:
        cached = _RATES_CACHE.get(difficulty)
        if cached and (not force) and cached.source_mtime == mtime:
            return cached

    rates = normalize_rate_sheets(config=config, difficulty=difficulty)
    with _RATES_LOCK:
        _RATES_CACHE[difficulty] = rates
    return rates


def _invalidate_rates_cache(difficulty: Difficulty) -> None:
    with _RATES_LOCK:
        _RATES_CACHE.pop(difficulty, None)
    with _SOP_LOCK:
        _SOP_PARSE_CACHE.pop(difficulty, None)


def _sop_parse_key(
    *,
    sop_markdown: str,
    model: str,
    canonical_origins: list[str],
    canonical_destinations: list[str],
) -> str:
    import json

    payload = {
        "sop": sop_markdown,
        "model": model,
        "origins": sorted(set([str(x) for x in canonical_origins])),
        "destinations": sorted(set([str(x) for x in canonical_destinations])),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _get_sop_parsed_cached(
    *,
    config: AppConfig,
    difficulty: Difficulty,
    sop_parse_mode: SopParseModeLiteral,
    force_refresh: bool,
    openai_api_key: str | None,
) -> SopParseResult:
    rates = _get_rates_cached(config=config, difficulty=difficulty)
    df_air = rates.air
    df_sea = rates.sea
    canonical_origins = sorted(set([str(x) for x in df_air["Origin"].dropna().tolist()] + [str(x) for x in df_sea["Origin"].dropna().tolist()]))
    canonical_destinations = sorted(
        set([str(x) for x in df_air["Destination"].dropna().tolist()] + [str(x) for x in df_sea["Destination"].dropna().tolist()])
    )

    sop_path = (config.data.data_dir / "SOP.md").resolve()
    sop_markdown = sop_path.read_text(encoding="utf-8", errors="replace") if sop_path.exists() else ""

    key = _sop_parse_key(
        sop_markdown=sop_markdown,
        model=config.openai.model,
        canonical_origins=canonical_origins,
        canonical_destinations=canonical_destinations,
    )

    with _SOP_LOCK:
        cached = _SOP_PARSE_CACHE.get(difficulty)
        if cached and (not force_refresh) and cached[0] == key:
            return cached[1]

    if sop_parse_mode == "rule_based":
        config_for_parse = _with_openai_key(config=config, openai_api_key=None)
    elif sop_parse_mode == "llm":
        config_for_parse = _require_openai_key(config=config, provided=openai_api_key, reason="SOP parsing (llm)")
    else:
        config_for_parse = _with_openai_key(config=config, openai_api_key=openai_api_key)

    result = parse_sop(
        config=config_for_parse,
        canonical_origins=canonical_origins,
        canonical_destinations=canonical_destinations,
        known_senders=_known_senders(config=config_for_parse),
        force_refresh=bool(force_refresh),
    )

    with _SOP_LOCK:
        _SOP_PARSE_CACHE[difficulty] = (key, result)
    return result


def create_app() -> FastAPI:
    app = FastAPI(title="Freight Quote Agent API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @router.get("/db/health")
    def db_health() -> dict[str, Any]:
        _load_base_config()  # loads .env
        try:
            return db_healthcheck()
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @router.post("/db/init")
    def db_init() -> dict[str, Any]:
        _load_base_config()  # loads .env
        try:
            return db_init_schema()
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @router.put("/rate-sheets/{difficulty}")
    async def upload_rate_sheet(
        difficulty: Difficulty,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        config = _load_base_config()
        dest = _resolve_rate_sheet_path(config=config, difficulty=difficulty)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        dest.write_bytes(content)
        _invalidate_rates_cache(difficulty)
        return {"difficulty": difficulty, "path": str(dest), "bytes_written": len(content)}

    @router.get("/rate-sheets/{difficulty}/source")
    def download_rate_sheet(difficulty: Difficulty) -> FileResponse:
        config = _load_base_config()
        path = _resolve_rate_sheet_path(config=config, difficulty=difficulty)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Rate sheet not found: {path}")
        return FileResponse(
            path=str(path),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=path.name,
        )

    @router.post("/rate-sheets/{difficulty}/normalize")
    def normalize_rates(difficulty: Difficulty) -> dict[str, Any]:
        config = _load_base_config()
        rates = _get_rates_cached(config=config, difficulty=difficulty, force=True)
        return _rates_payload(rates=rates, include_rows=False) | {
            "canonical_locations": {
                "sea_origins": sorted(rates.sea["Origin"].unique().tolist()),
                "sea_destinations": sorted(rates.sea["Destination"].unique().tolist()),
                "air_origins": sorted(rates.air["Origin"].unique().tolist()),
                "air_destinations": sorted(rates.air["Destination"].unique().tolist()),
            }
        }

    @router.get("/rate-sheets/{difficulty}/normalized")
    def get_normalized_rates(difficulty: Difficulty, include_rows: bool = True) -> dict[str, Any]:
        config = _load_base_config()
        rates = _get_rates_cached(config=config, difficulty=difficulty)
        return _rates_payload(rates=rates, include_rows=bool(include_rows))

    @router.get("/sop")
    def get_sop() -> dict[str, Any]:
        config = _load_base_config()
        path = (config.data.data_dir / "SOP.md").resolve()
        markdown = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        return {"path": str(path), "markdown": markdown}

    @router.put("/sop")
    def put_sop(payload: SopUpdateRequest) -> dict[str, Any]:
        config = _load_base_config()
        path = (config.data.data_dir / "SOP.md").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = str(payload.markdown or "")
        path.write_text(data, encoding="utf-8")
        with _SOP_LOCK:
            _SOP_PARSE_CACHE.clear()
        return {"path": str(path), "bytes_written": len(data.encode("utf-8"))}

    @router.post("/sop/parse")
    def parse_sop_endpoint(
        payload: SopParseRequest,
        x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        base_config = _load_base_config()
        openai_api_key = _get_openai_key(x_openai_api_key=x_openai_api_key, x_api_key=x_api_key)

        config_for_parse = base_config
        if payload.sop_parse_mode == "llm":
            config_for_parse = _require_openai_key(config=base_config, provided=openai_api_key, reason="SOP parsing (llm)")
        else:
            config_for_parse = _with_openai_key(config=base_config, openai_api_key=openai_api_key)

        result = _get_sop_parsed_cached(
            config=config_for_parse,
            difficulty=payload.difficulty,
            sop_parse_mode=payload.sop_parse_mode,
            force_refresh=payload.force_refresh,
            openai_api_key=openai_api_key,
        )
        return {
            "difficulty": payload.difficulty,
            "sop_path": str(result.sop_path),
            "sop_loaded": result.sop_loaded,
            "cached": result.cached,
            "errors": result.errors,
            "llm_usage": result.llm_usage,
            "llm_raw": result.llm_raw,
            "sop_config": _sop_config_payload(result.sop_config),
        }

    @router.post("/quote")
    def quote(
        payload: QuoteRequest,
        x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        base_config = _load_base_config()
        openai_api_key = _get_openai_key(x_openai_api_key=x_openai_api_key, x_api_key=x_api_key)

        config = base_config
        if payload.use_openai:
            config = _require_openai_key(config=base_config, provided=openai_api_key, reason="use_openai=true (extraction/location match)")
        else:
            config = _with_openai_key(config=base_config, openai_api_key=openai_api_key)

        rates = _get_rates_cached(config=config, difficulty=payload.difficulty)

        sop_cfg: SopConfig | None = None
        sop_parse: SopParseResult | None = None
        if payload.enable_sop:
            sop_parse = _get_sop_parsed_cached(
                config=config,
                difficulty=payload.difficulty,
                sop_parse_mode=payload.sop_parse_mode,
                force_refresh=payload.force_sop_refresh,
                openai_api_key=openai_api_key,
            )
            sop_cfg = sop_parse.sop_config

        if isinstance(payload.email, EmailPayload):
            email_id = payload.email.email_id or payload.email_id or "email_api"
            email = EmailMessage(
                email_id=email_id,
                sender=payload.email.from_,
                to=payload.email.to,
                subject=payload.email.subject,
                body=payload.email.body,
            )
        else:
            parsed = _parse_raw_email_text(str(payload.email or ""))
            email_id = payload.email_id or "email_api"
            email = EmailMessage(
                email_id=email_id,
                sender=parsed.get("from", ""),
                to=parsed.get("to", ""),
                subject=parsed.get("subject", ""),
                body=parsed.get("body", ""),
            )

        result = run_quote_pipeline(
            email=email,
            config=config,
            difficulty=payload.difficulty,
            use_openai=payload.use_openai,
            rate_sheets=rates,
            enable_sop=payload.enable_sop,
            sop_config=sop_cfg,
        )

        out: dict[str, Any] = {
            "quote_text": result.quote_text,
            "error": result.error,
            "email": {
                "email_id": email.email_id,
                "from": email.sender,
                "to": email.to,
                "subject": email.subject,
                "body": email.body,
            },
        }

        trace_payload = asdict(result.trace)
        inferred = _infer_quote_fields_from_trace(trace_payload)
        out.update(inferred)
        reply_type, hitl_reasons = _decide_reply_type(
            trace_payload=trace_payload,
            email_subject=email.subject,
            email_body=email.body,
        )
        if result.error:
            hitl_reasons.append({"code": "pipeline_error", "detail": str(result.error)})
        if not result.quote_text:
            hitl_reasons.append({"code": "missing_quote_text"})
        if hitl_reasons:
            reply_type = "HITL"

        if reply_type == "HITL":
            decision_step = {
                "title": "HITL Decision",
                "summary": _hitl_summary_from_trace(trace_payload),
                "data": {"type": "HITL", "reasons": hitl_reasons},
                "used_llm": False,
                "llm_usage": None,
            }
            steps = trace_payload.get("steps")
            if isinstance(steps, list):
                trace_payload["steps"] = [*steps, decision_step]
            else:
                trace_payload["steps"] = [decision_step]

            out["hitl_reasons"] = hitl_reasons

        out["type"] = reply_type
        out["confidence"] = _confidence_from_reply(reply_type=reply_type, hitl_reasons=hitl_reasons)

        if payload.include_trace:
            out["trace"] = trace_payload

        if sop_parse is not None:
            out["sop"] = {
                "cached": sop_parse.cached,
                "errors": sop_parse.errors,
                "llm_usage": sop_parse.llm_usage,
                "llm_raw": sop_parse.llm_raw,
                "sop_config": _sop_config_payload(sop_parse.sop_config),
            }

        if payload.persist:
            config_payload: dict[str, Any] = {
                "difficulty": payload.difficulty,
                "enable_sop": bool(payload.enable_sop),
                "use_openai": bool(payload.use_openai),
                "sop_parse_mode": str(payload.sop_parse_mode),
                "force_sop_refresh": bool(payload.force_sop_refresh),
            }
            try:
                record_id = insert_email_record(
                    email_id=email.email_id,
                    email_from=email.sender,
                    email_to=email.to,
                    subject=email.subject,
                    body=email.body,
                    reply=result.quote_text,
                    trace=trace_payload,
                    record_type=payload.record_type,
                    status=("unprocessed" if result.error else payload.record_status),
                    confidence=str(out.get("confidence") or "").strip() or None,
                    config=config_payload,
                    origin_city=inferred.get("origin_city"),
                    destination_city=inferred.get("destination_city"),
                    price=inferred.get("price"),
                    currency=inferred.get("currency"),
                    transport_type=inferred.get("transport_type"),
                    has_route=inferred.get("has_route"),
                )
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Failed to persist email record: {e.__class__.__name__}: {e}") from e

            out["record_id"] = str(record_id)

        return out

    @router.post("/email-records")
    def create_email_record(payload: EmailRecordCreateRequest) -> dict[str, Any]:
        _load_base_config()  # loads .env
        inferred = _infer_quote_fields_from_trace(payload.trace or {})
        trace_payload = payload.trace if isinstance(payload.trace, dict) else {}
        confidence = payload.confidence
        if confidence is None:
            reply_type, hitl_reasons = _decide_reply_type(
                trace_payload=trace_payload,
                email_subject=str(payload.subject or ""),
                email_body=str(payload.body or ""),
            )
            if not str(payload.reply or "").strip():
                hitl_reasons.append({"code": "missing_quote_text"})
            if hitl_reasons:
                reply_type = "HITL"
            confidence = str(_confidence_from_reply(reply_type=reply_type, hitl_reasons=hitl_reasons))
        try:
            record_id = insert_email_record(
                email_id=payload.email_id,
                email_from=payload.from_,
                email_to=payload.to,
                subject=payload.subject,
                body=payload.body,
                reply=payload.reply,
                trace=payload.trace,
                record_type=payload.type,
                status=payload.status,
                confidence=confidence,
                config=payload.config,
                origin_city=inferred.get("origin_city"),
                destination_city=inferred.get("destination_city"),
                price=inferred.get("price"),
                currency=inferred.get("currency"),
                transport_type=inferred.get("transport_type"),
                has_route=inferred.get("has_route"),
            )
            record = get_email_record(record_id=str(record_id))
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to create email record: {e.__class__.__name__}: {e}") from e

        if not record:
            raise HTTPException(status_code=500, detail="Email record was created but could not be loaded.")
        return _email_record_api_shape(record)

    @router.patch("/email-records/{record_id}")
    def patch_email_record(record_id: str, payload: EmailRecordUpdateRequest) -> dict[str, Any]:
        _load_base_config()  # loads .env
        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update.")
        try:
            record = update_email_record(
                record_id=record_id,
                status=updates.get("status"),
                record_type=updates.get("type"),
                reply=updates.get("reply"),
                trace=updates.get("trace"),
                confidence=updates.get("confidence"),
                config=updates.get("config"),
            )
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to update email record: {e.__class__.__name__}: {e}") from e

        if not record:
            raise HTTPException(status_code=404, detail="Email record not found.")
        return _email_record_api_shape(record)

    @router.get("/email-records")
    def list_email_record_items(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        _load_base_config()  # loads .env
        try:
            items = [_email_record_api_shape(x) for x in list_email_records(limit=limit, offset=offset)]
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {"items": items, "limit": int(limit), "offset": int(offset), "count": len(items)}

    @router.get("/email-records/status-counts")
    def get_email_record_status_counts() -> dict[str, Any]:
        _load_base_config()  # loads .env
        try:
            counts = count_email_records_by_status()
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {"counts": counts}

    @router.get("/email-records/needs-human-decision")
    def list_needs_human_decision_items(limit: int = 200, offset: int = 0) -> dict[str, Any]:
        _load_base_config()  # loads .env
        try:
            items = [_email_record_api_shape(x) for x in list_needs_human_decision(limit=limit, offset=offset)]
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {"items": items, "limit": int(limit), "offset": int(offset), "count": len(items)}

    @router.get("/stats/senders")
    def stats_senders(top: int = 50, max_rows: int = 20000, batch_size: int = 1000) -> dict[str, Any]:
        _load_base_config()  # loads .env

        top_i = max(int(top), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        counts: dict[str, int] = {}
        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break
            for row in batch:
                sender = _first_email_address(row.get("email_from")) or str(row.get("email_from") or "").strip()
                if not sender:
                    continue
                counts[sender] = int(counts.get(sender) or 0) + 1
            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        items = [{"sender": k, "count": v} for k, v in counts.items()]
        items.sort(key=lambda x: (-int(x["count"]), str(x["sender"]).casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "unique_senders": len(counts),
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/domains")
    def stats_domains(
        top: int = 50,
        exclude_common: bool = True,
        max_rows: int = 20000,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        _load_base_config()  # loads .env

        top_i = max(int(top), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        excluded = set(_COMMON_GENERIC_EMAIL_DOMAINS) if exclude_common else set()
        counts: dict[str, int] = {}
        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break
            for row in batch:
                domain = _email_domain(row.get("email_from"))
                if not domain:
                    continue
                if domain in excluded:
                    continue
                counts[domain] = int(counts.get(domain) or 0) + 1
            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        items = [{"domain": k, "count": v} for k, v in counts.items()]
        items.sort(key=lambda x: (-int(x["count"]), str(x["domain"]).casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "unique_domains": len(counts),
            "excluded_common": bool(exclude_common),
            "excluded_common_list": sorted(list(excluded)) if exclude_common else [],
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/origins")
    def stats_origins(top: int = 50, max_rows: int = 20000, batch_size: int = 1000) -> dict[str, Any]:
        _load_base_config()  # loads .env

        top_i = max(int(top), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        counts: dict[str, int] = {}
        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break

            for row in batch:
                origins: set[str] = set()
                trace = row.get("trace")
                if isinstance(trace, dict):
                    for r in _extract_routes_from_trace(trace):
                        origin = str(r.get("origin") or "").strip()
                        if origin:
                            origins.add(origin)
                if not origins:
                    for origin in _split_semicolon_values(row.get("origin_city")):
                        origins.add(origin)

                for origin in origins:
                    counts[origin] = int(counts.get(origin) or 0) + 1

            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        items = [{"origin_city": k, "count": v} for k, v in counts.items()]
        items.sort(key=lambda x: (-int(x["count"]), str(x["origin_city"]).casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "unique_origins": len(counts),
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/destinations-weight")
    def stats_destinations_weight(top: int = 50, max_rows: int = 20000, batch_size: int = 1000) -> dict[str, Any]:
        base_config = _load_base_config()  # loads .env
        air_factor = float(base_config.pricing.air_volume_factor or 0.0)

        top_i = max(int(top), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        stats: dict[str, dict[str, Any]] = {}
        record_hits: dict[str, set[str]] = {}

        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break

            for row in batch:
                record_id = str(row.get("id") or "")
                trace = row.get("trace")
                if not isinstance(trace, dict):
                    continue
                routes = _extract_routes_from_trace(trace)
                for r in routes:
                    if str(r.get("mode") or "") != "air":
                        continue
                    destination = str(r.get("destination") or "").strip()
                    if not destination:
                        continue
                    entry = stats.get(destination)
                    if entry is None:
                        entry = {
                            "destination_city": destination,
                            "route_count": 0,
                            "record_count": 0,
                            "sum_actual_weight_kg": 0.0,
                            "sum_volume_cbm": 0.0,
                            "sum_chargeable_weight_kg": 0.0,
                        }
                        stats[destination] = entry

                    entry["route_count"] = int(entry.get("route_count") or 0) + 1

                    actual = _coerce_number(r.get("actual_weight_kg"))
                    volume = _coerce_number(r.get("volume_cbm"))
                    if actual is not None:
                        entry["sum_actual_weight_kg"] = float(entry.get("sum_actual_weight_kg") or 0.0) + float(actual)
                    if volume is not None:
                        entry["sum_volume_cbm"] = float(entry.get("sum_volume_cbm") or 0.0) + float(volume)

                    cw = _coerce_number(r.get("chargeable_weight_kg"))
                    if cw is None and actual is not None and volume is not None and air_factor > 0:
                        cw = max(float(actual), float(volume) * float(air_factor))
                    if cw is not None:
                        entry["sum_chargeable_weight_kg"] = float(entry.get("sum_chargeable_weight_kg") or 0.0) + float(cw)

                    if record_id:
                        record_hits.setdefault(destination, set()).add(record_id)

            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        for dest, ids in record_hits.items():
            if dest in stats:
                stats[dest]["record_count"] = len(ids)

        items = list(stats.values())
        items.sort(key=lambda x: (-float(x.get("sum_chargeable_weight_kg") or 0.0), str(x.get("destination_city") or "").casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "unique_destinations": len(stats),
            "units": {"weight": "kg", "volume": "cbm"},
            "air_volume_factor": air_factor,
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/routes")
    def stats_routes(top: int = 100, max_rows: int = 20000, batch_size: int = 1000) -> dict[str, Any]:
        _load_base_config()  # loads .env

        top_i = max(int(top), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        counts: dict[str, int] = {}
        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break

            for row in batch:
                trace = row.get("trace")
                route_keys: set[str] = set()
                if isinstance(trace, dict):
                    for r in _extract_routes_from_trace(trace):
                        mode = str(r.get("mode") or "").strip() or "unknown"
                        origin = str(r.get("origin") or "").strip() or "unknown"
                        destination = str(r.get("destination") or "").strip() or "unknown"
                        route_keys.add(f"{mode}: {origin} -> {destination}")
                if not route_keys:
                    modes = _split_semicolon_values(row.get("transport_type"))
                    origins = _split_semicolon_values(row.get("origin_city"))
                    destinations = _split_semicolon_values(row.get("destination_city"))
                    mode = modes[0] if modes else (str(row.get("transport_type") or "").strip() or "unknown")
                    for o in origins or ["unknown"]:
                        for d in destinations or ["unknown"]:
                            route_keys.add(f"{mode}: {o} -> {d}")

                for k in route_keys:
                    counts[k] = int(counts.get(k) or 0) + 1

            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        items = [{"route": k, "count": v} for k, v in counts.items()]
        items.sort(key=lambda x: (-int(x["count"]), str(x["route"]).casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "unique_routes": len(counts),
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/types")
    def stats_types(max_rows: int = 20000, batch_size: int = 1000) -> dict[str, Any]:
        _load_base_config()  # loads .env

        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"record_type_counts": {}, "transport_type_counts": {}, "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        record_type_counts: dict[str, int] = {}
        transport_type_counts: dict[str, int] = {}

        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break

            for row in batch:
                rt = str(row.get("type") or "unknown").strip() or "unknown"
                record_type_counts[rt] = int(record_type_counts.get(rt) or 0) + 1

                tt = str(row.get("transport_type") or "unknown").strip() or "unknown"
                transport_type_counts[tt] = int(transport_type_counts.get(tt) or 0) + 1

            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        return {
            "record_type_counts": record_type_counts,
            "transport_type_counts": transport_type_counts,
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.get("/stats/unsatisfied-routes")
    def stats_unsatisfied_routes(
        top: int = 100,
        sample_per_route: int = 3,
        max_rows: int = 20000,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        _load_base_config()  # loads .env

        top_i = max(int(top), 0)
        sample_i = max(min(int(sample_per_route), 20), 0)
        max_rows_i = max(int(max_rows), 0)
        batch_i = max(min(int(batch_size), 5000), 1)
        if max_rows_i == 0:
            return {"items": [], "reason_counts": {}, "scanned": 0, "max_rows": 0, "reached_max_rows": False}

        per_route: dict[str, dict[str, Any]] = {}
        reason_counts: dict[str, int] = {}

        offset = 0
        scanned = 0
        while scanned < max_rows_i:
            limit = min(batch_i, max_rows_i - scanned)
            try:
                batch = list_email_records(limit=limit, offset=offset)
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not batch:
                break

            for row in batch:
                record_id = str(row.get("id") or "")
                trace = row.get("trace")
                if not isinstance(trace, dict):
                    continue
                routes = _extract_routes_from_trace(trace)
                clarifications = _extract_clarification_questions(trace)
                clar_reasons = _classify_clarifications(clarifications)

                for r in routes:
                    mode = str(r.get("mode") or "").strip() or "unknown"
                    origin = str(r.get("origin") or "").strip() or "unknown"
                    destination = str(r.get("destination") or "").strip() or "unknown"
                    route_key = f"{mode}: {origin} -> {destination}"

                    issues: set[str] = set()
                    if r.get("rate_found") is False:
                        issues.add("rate_not_found")
                    if r.get("rate_found") is True and r.get("transit_days") is None:
                        issues.add("missing_transit_days")
                    if r.get("final_amount") is None:
                        if "missing_origin" in clar_reasons or "missing_destination" in clar_reasons:
                            issues.add("missing_location")
                        if mode == "air" and "missing_weight_volume" in clar_reasons:
                            issues.add("missing_weight_volume")
                        if mode == "sea" and "missing_container_info" in clar_reasons:
                            issues.add("missing_container_info")
                        if not issues and clar_reasons:
                            issues |= set(clar_reasons)
                        if not issues:
                            issues.add("quote_not_generated")

                    if not issues:
                        continue

                    entry = per_route.get(route_key)
                    if entry is None:
                        entry = {
                            "route": route_key,
                            "mode": mode,
                            "origin_city": origin,
                            "destination_city": destination,
                            "count": 0,
                            "reasons": {},
                            "sample_record_ids": [],
                        }
                        per_route[route_key] = entry

                    entry["count"] = int(entry.get("count") or 0) + 1
                    reasons_dict: dict[str, int] = entry.get("reasons") if isinstance(entry.get("reasons"), dict) else {}
                    for reason in sorted(list(issues)):
                        reasons_dict[reason] = int(reasons_dict.get(reason) or 0) + 1
                        reason_counts[reason] = int(reason_counts.get(reason) or 0) + 1
                    entry["reasons"] = reasons_dict

                    if sample_i and record_id:
                        samples: list[str] = entry.get("sample_record_ids") if isinstance(entry.get("sample_record_ids"), list) else []
                        if record_id not in samples and len(samples) < sample_i:
                            entry["sample_record_ids"] = [*samples, record_id]

            scanned += len(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        items = list(per_route.values())
        items.sort(key=lambda x: (-int(x.get("count") or 0), str(x.get("route") or "").casefold()))
        if top_i:
            items = items[:top_i]

        return {
            "items": items,
            "reason_counts": reason_counts,
            "unique_routes": len(per_route),
            "scanned": scanned,
            "max_rows": max_rows_i,
            "reached_max_rows": scanned >= max_rows_i,
        }

    @router.post("/email-records/decision")
    def decide_email_record(
        payload: EmailRecordDecisionRequest,
        x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        base_config = _load_base_config()
        openai_api_key = _get_openai_key(x_openai_api_key=x_openai_api_key, x_api_key=x_api_key)

        try:
            record = get_email_record(record_id=payload.id)
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to load email record: {e.__class__.__name__}: {e}") from e

        if not record:
            raise HTTPException(status_code=404, detail="Email record not found.")

        subject = str(record.get("subject") or "")
        body = str(record.get("body") or "")
        refined_quote = str(payload.refined_quote or record.get("reply") or "").strip()
        comment = str(payload.comment or "").strip()

        final_quote = refined_quote
        llm_usage: dict[str, Any] | None = None
        llm_raw: str | None = None
        if comment:
            cfg_for_llm = _require_openai_key(config=base_config, provided=openai_api_key, reason="human quote refinement (comment)")
            try:
                final_quote, llm_usage, llm_raw = _refine_quote_with_openai(
                    config=cfg_for_llm,
                    email_subject=subject,
                    email_body=body,
                    refined_quote=refined_quote,
                    comment=comment,
                )
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"LLM refine failed: {e.__class__.__name__}: {e}") from e

            if not str(final_quote or "").strip():
                final_quote = refined_quote

        existing_trace = record.get("trace")
        trace_payload = _coerce_trace_payload(existing_trace)

        step = {
            "title": "Human Decision",
            "summary": f"Decision={payload.decision}",
            "data": {
                "decision": payload.decision,
                "refined_quote": refined_quote,
                "comment": comment,
                "final_quote": final_quote,
                "llm_raw": llm_raw,
            },
            "used_llm": bool(llm_usage),
            "llm_usage": llm_usage,
        }
        steps = trace_payload.get("steps")
        if isinstance(steps, list):
            trace_payload["steps"] = [*steps, step]
        else:
            trace_payload["steps"] = [step]
        if llm_usage:
            totals = trace_payload.get("llm_usage")
            if not isinstance(totals, dict):
                totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                totals[k] = int(totals.get(k) or 0) + int(llm_usage.get(k) or 0)
            trace_payload["llm_usage"] = totals

        try:
            update_email_record(
                record_id=payload.id,
                record_type="human",
                reply=final_quote if final_quote else None,
                trace=trace_payload,
            )
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to update email record: {e.__class__.__name__}: {e}") from e

        if payload.decision == "reject":
            try:
                update_email_record(record_id=payload.id, status="human_rejected", record_type="human")
            except DbNotConfiguredError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Failed to set status: {e.__class__.__name__}: {e}") from e

            return {"ok": True, "id": payload.id, "decision": payload.decision, "sent": False, "quote_text": final_quote}

        webhook_url = os.getenv("MAKE_WEBHOOK_URL") or "https://hook.eu2.make.com/5l92e29fxygd0ognqqvjas2j40xmpkbl"
        webhook_payload = {
            "data": {
                "quote_text": final_quote,
                "email": {
                    "email_id": record.get("email_id") or "email_api",
                    "from": record.get("email_from") or "",
                    "to": record.get("email_to") or "",
                    "subject": subject,
                    "body": body,
                },
            }
        }

        webhook_result = _post_make_webhook(url=webhook_url, payload=webhook_payload)
        status_code = int(webhook_result.get("status_code") or 0)
        if status_code < 200 or status_code >= 300:
            raise HTTPException(status_code=502, detail={"message": "Webhook call failed.", "webhook": webhook_result})

        try:
            update_email_record(record_id=payload.id, status="human_confirmed_replied", record_type="human")
        except DbNotConfiguredError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to set status: {e.__class__.__name__}: {e}") from e

        return {
            "ok": True,
            "id": payload.id,
            "decision": payload.decision,
            "sent": True,
            "quote_text": final_quote,
            "webhook": {"status_code": status_code},
        }

    app.include_router(router)
    return app


app = create_app()
