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
    config: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class EmailRecordUpdateRequest(BaseModel):
    status: EmailRecordStatusLiteral | None = None
    type: EmailRecordTypeLiteral | None = None
    reply: str | None = None
    trace: dict[str, Any] | None = None
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
        "trace": item.get("trace"),
        "type": item.get("type"),
        "status": item.get("status"),
        "config": item.get("config"),
    }


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


def _decide_reply_type(*, trace_payload: dict[str, Any], email_subject: str, email_body: str) -> str:
    sop_step = _get_trace_step(trace_payload=trace_payload, title="SOP")
    sop_data = sop_step.get("data") if isinstance(sop_step, dict) else None

    sop_matched = False
    sop_allowed_modes: set[str] | None = None
    sop_origin_required: set[str] | None = None
    sop_parse_errors: list[str] = []

    if isinstance(sop_data, dict):
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

    if not sop_matched:
        return "HITL"
    if sop_parse_errors:
        return "HITL"

    shipments = _extract_shipments_from_trace(trace_payload)
    if sop_allowed_modes:
        for sh in shipments:
            mode = str(sh.get("mode") or "").strip()
            if mode and mode not in sop_allowed_modes:
                return "HITL"

    if sop_origin_required:
        for origin in _extract_canonical_origins_from_trace(trace_payload):
            if origin not in sop_origin_required:
                return "HITL"

    steps = trace_payload.get("steps")
    if isinstance(steps, list) and any(
        isinstance(s, dict) and str(s.get("title") or "") == "Clarification Required" for s in steps
    ):
        return "HITL"

    threshold = float(os.getenv("HITL_LARGE_ORDER_USD") or 20000)
    amounts = _extract_final_amounts_from_trace(trace_payload)
    if amounts and (max(amounts) >= threshold or sum(amounts) >= threshold):
        return "HITL"

    for sh in shipments:
        notes = sh.get("notes")
        if notes and str(notes).strip():
            return "HITL"

    text = f"{email_subject}\n{email_body}"
    for pat in _SPECIAL_REQUEST_KEYWORDS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return "HITL"

    return "Auto"


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
        reply_type = _decide_reply_type(trace_payload=trace_payload, email_subject=email.subject, email_body=email.body)
        if result.error or not result.quote_text:
            reply_type = "HITL"
        out["type"] = reply_type

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
                    config=config_payload,
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
                config=payload.config,
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
        if isinstance(existing_trace, dict):
            trace_payload: dict[str, Any] = dict(existing_trace)
        elif existing_trace is None:
            trace_payload = {}
        else:
            trace_payload = {"raw_trace": existing_trace}

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
