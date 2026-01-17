from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import sys
import threading
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import AppConfig, OpenAIConfig, load_app_config
from src.data_loader import EmailMessage, list_emails, load_email
from src.pipeline import run_quote_pipeline
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets
from src.sop import SopConfig, SopParseResult, parse_sop


CONFIG_PATH = APP_DIR / "config.toml"


class SopParseMode(str):
    pass


Difficulty = Literal["easy", "medium", "hard"]
SopParseModeLiteral = Literal["auto", "llm", "rule_based"]


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


_RATES_CACHE: dict[str, NormalizedRateSheets] = {}
_RATES_LOCK = threading.Lock()

_SOP_PARSE_CACHE: dict[str, tuple[str, SopParseResult]] = {}
_SOP_LOCK = threading.Lock()


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
        }

        if payload.include_trace:
            out["trace"] = asdict(result.trace)

        if sop_parse is not None:
            out["sop"] = {
                "cached": sop_parse.cached,
                "errors": sop_parse.errors,
                "llm_usage": sop_parse.llm_usage,
                "llm_raw": sop_parse.llm_raw,
                "sop_config": _sop_config_payload(sop_parse.sop_config),
            }

        return out

    app.include_router(router)
    return app


app = create_app()
