from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

from openai import OpenAI

from src.config import AppConfig
from src.data_loader import EmailMessage
from src.llm_usage import usage_from_response
from src.prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt
from src.trace import RunTrace


Mode = Literal["air", "sea"]


@dataclass(frozen=True)
class ShipmentRequest:
    mode: Mode
    origin_raw: str | None
    destination_raw: str | None
    quantity: int | None = None
    container_size_ft: int | None = None
    actual_weight_kg: float | None = None
    volume_cbm: float | None = None
    commodity: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    shipments: list[ShipmentRequest] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)


_OPTIONAL_CLARIFICATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bcommodity\b"),
    re.compile(r"(?i)\bhs\s*code\b"),
    re.compile(r"(?i)\bharmonized\s+system\b"),
    re.compile(r"(?i)\bproduct\s+type\b"),
    re.compile(r"(?i)\btype\s+of\s+goods\b"),
]

_DANGEROUS_GOODS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bdg\b"),
    re.compile(r"(?i)\bdangerous\s*goods\b"),
    re.compile(r"(?i)\bhazard(?:ous)?\b"),
    re.compile(r"(?i)\blithium\b"),
    re.compile(r"(?i)\bbattery\b"),
    re.compile(r"(?i)\bmsds\b"),
]


def _filter_optional_clarification_questions(questions: list[str]) -> list[str]:
    out: list[str] = []
    for q in questions:
        text = str(q or "").strip()
        if not text:
            continue
        if any(p.search(text) for p in _OPTIONAL_CLARIFICATION_PATTERNS) and not any(
            p.search(text) for p in _DANGEROUS_GOODS_PATTERNS
        ):
            continue
        out.append(text)
    return out


def _filter_mode_irrelevant_questions(*, shipments: list[ShipmentRequest], questions: list[str]) -> list[str]:
    if not questions:
        return []

    modes = {str(s.mode or "").strip().casefold() for s in shipments if getattr(s, "mode", None)}
    if not modes:
        return questions

    out: list[str] = []
    for q in questions:
        text = str(q or "").strip()
        if not text:
            continue
        t = text.casefold()

        # If we're doing SEA, weight/volume are not required for pricing in this demo.
        if modes == {"sea"}:
            if any(k in t for k in ["actual weight", "weight", " kg", "volume", " cbm"]):
                continue

        # If we're doing AIR, container size/quantity are irrelevant.
        if modes == {"air"}:
            if "container" in t or "20ft" in t or "40ft" in t:
                continue

        out.append(text)

    return out


def _infer_qty_container_from_text(text: str) -> tuple[int | None, int | None]:
    return _match_qty_container(str(text or ""))


def _postprocess_shipments_latest_override(*, email_body: str, shipments: list[ShipmentRequest]) -> list[ShipmentRequest]:
    if not shipments:
        return []
    latest_body, _older_body = _split_latest_context_sections(email_body)
    qty_hint, size_hint = _infer_qty_container_from_text(latest_body)
    if qty_hint is None and size_hint is None:
        return shipments

    out: list[ShipmentRequest] = []
    for s in shipments:
        if s.mode != "sea":
            out.append(s)
            continue
        out.append(
            ShipmentRequest(
                mode=s.mode,
                origin_raw=s.origin_raw,
                destination_raw=s.destination_raw,
                quantity=s.quantity if s.quantity is not None else qty_hint,
                container_size_ft=s.container_size_ft if s.container_size_ft is not None else size_hint,
                actual_weight_kg=s.actual_weight_kg,
                volume_cbm=s.volume_cbm,
                commodity=s.commodity,
                notes=s.notes,
            )
        )
    return out


def _split_latest_context_sections(body: str) -> tuple[str, str]:
    text = str(body or "")
    marker = "\n\nOLDER CONTEXT (use only if missing in latest):\n"
    if marker in text:
        latest, older = text.split(marker, 1)
        latest = latest.replace("LATEST MESSAGE (authoritative, override older values):\n", "").strip()
        return latest, older.strip()
    return text, ""


def _merge_shipments_latest_over_old(*, latest: ShipmentRequest, old: ShipmentRequest) -> ShipmentRequest:
    return ShipmentRequest(
        mode=latest.mode or old.mode,
        origin_raw=latest.origin_raw if latest.origin_raw is not None else old.origin_raw,
        destination_raw=latest.destination_raw if latest.destination_raw is not None else old.destination_raw,
        quantity=latest.quantity if latest.quantity is not None else old.quantity,
        container_size_ft=latest.container_size_ft if latest.container_size_ft is not None else old.container_size_ft,
        actual_weight_kg=latest.actual_weight_kg if latest.actual_weight_kg is not None else old.actual_weight_kg,
        volume_cbm=latest.volume_cbm if latest.volume_cbm is not None else old.volume_cbm,
        commodity=latest.commodity if latest.commodity is not None else old.commodity,
        notes=latest.notes if latest.notes is not None else old.notes,
    )


def extract_requests(
    *,
    email: EmailMessage,
    config: AppConfig,
    trace: RunTrace,
    use_openai: bool,
) -> ExtractionResult:
    if use_openai and config.openai.api_key:
        try:
            res, llm_usage = _extract_with_openai(email=email, config=config)
            res = ExtractionResult(shipments=_postprocess_shipments_latest_override(email_body=email.body, shipments=res.shipments), clarification_questions=res.clarification_questions)
            filtered_questions = _filter_mode_irrelevant_questions(
                shipments=res.shipments,
                questions=_filter_optional_clarification_questions(res.clarification_questions),
            )
            res = ExtractionResult(shipments=res.shipments, clarification_questions=filtered_questions)
            trace.add(
                "Extraction: OpenAI",
                summary="Parsed structured request(s) via OpenAI.",
                data={
                    "shipments": [asdict(s) for s in res.shipments],
                    "clarification_questions": res.clarification_questions,
                },
                used_llm=True,
                llm_usage=llm_usage,
            )
            if not res.shipments:
                trace.add(
                    "Extraction: OpenAI (empty)",
                    summary="LLM returned no shipments; falling back to rule-based extraction.",
                )
                fallback = _extract_rule_based(email=email)
                filtered_questions = _filter_mode_irrelevant_questions(
                    shipments=fallback.shipments,
                    questions=_filter_optional_clarification_questions(res.clarification_questions),
                )
                trace.add(
                    "Extraction: Rule-based",
                    summary="Parsed structured request(s) via regex rules (fallback after empty LLM parse).",
                    data={
                        "shipments": [asdict(s) for s in fallback.shipments],
                        "clarification_questions": filtered_questions,
                    },
                )
                return ExtractionResult(shipments=fallback.shipments, clarification_questions=filtered_questions)
            return res
        except Exception as e:  # noqa: BLE001
            trace.add(
                "Extraction: OpenAI (failed)",
                summary=f"Falling back to rule-based extraction: {e.__class__.__name__}: {e}",
                used_llm=True,
            )

    res = _extract_rule_based(email=email)
    res = ExtractionResult(
        shipments=res.shipments,
        clarification_questions=_filter_mode_irrelevant_questions(
            shipments=res.shipments,
            questions=_filter_optional_clarification_questions(res.clarification_questions),
        ),
    )
    trace.add(
        "Extraction: Rule-based",
        summary="Parsed structured request(s) via regex rules.",
        data={"shipments": [asdict(s) for s in res.shipments], "clarification_questions": res.clarification_questions},
    )
    return res


def _extract_with_openai(*, email: EmailMessage, config: AppConfig) -> tuple[ExtractionResult, dict[str, Any]]:
    client = OpenAI(api_key=config.openai.api_key)

    system = EXTRACTION_SYSTEM_PROMPT
    user = extraction_user_prompt(subject=email.subject, body=email.body)

    kwargs: dict[str, Any] = dict(
        model=config.openai.model,
        temperature=config.openai.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    try:
        resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except TypeError:
        resp = client.chat.completions.create(**kwargs)

    llm_usage = usage_from_response(resp, model=config.openai.model, calls=1)
    content = resp.choices[0].message.content or ""
    data = _coerce_json_object(content)

    if "shipments" in data and isinstance(data.get("shipments"), list):
        shipments_raw = data.get("shipments") or []
        shipments: list[ShipmentRequest] = []
        for item in shipments_raw:
            if not isinstance(item, dict):
                continue
            mode = item.get("mode")
            if mode not in ("air", "sea"):
                mode = _infer_mode(email.subject + "\n" + email.body)
            shipments.append(
                ShipmentRequest(
                    mode=mode,
                    origin_raw=_none_if_blank(item.get("origin_raw")),
                    destination_raw=_none_if_blank(item.get("destination_raw")),
                    quantity=_to_int_or_none(item.get("quantity")),
                    container_size_ft=_to_int_or_none(item.get("container_size_ft")),
                    actual_weight_kg=_to_float_or_none(item.get("actual_weight_kg")),
                    volume_cbm=_to_float_or_none(item.get("volume_cbm")),
                    commodity=_none_if_blank(item.get("commodity")),
                    notes=_none_if_blank(item.get("notes")),
                )
            )
        questions = data.get("clarification_questions")
        if not isinstance(questions, list):
            questions = []
        questions_out = _filter_optional_clarification_questions([str(q) for q in questions])
        return (
            ExtractionResult(shipments=shipments, clarification_questions=questions_out),
            llm_usage,
        )

    mode = data.get("mode")
    if mode not in ("air", "sea"):
        mode = _infer_mode(email.subject + "\n" + email.body)
    single = ShipmentRequest(
        mode=mode,
        origin_raw=_none_if_blank(data.get("origin_raw")),
        destination_raw=_none_if_blank(data.get("destination_raw")),
        quantity=_to_int_or_none(data.get("quantity")),
        container_size_ft=_to_int_or_none(data.get("container_size_ft")),
        actual_weight_kg=_to_float_or_none(data.get("actual_weight_kg")),
        volume_cbm=_to_float_or_none(data.get("volume_cbm")),
        commodity=_none_if_blank(data.get("commodity")),
        notes=_none_if_blank(data.get("notes")),
    )
    questions = data.get("clarification_questions")
    if not isinstance(questions, list):
        questions = []
    questions_out = _filter_optional_clarification_questions([str(q) for q in questions])
    return ExtractionResult(shipments=[single], clarification_questions=questions_out), llm_usage


def _extract_rule_based(*, email: EmailMessage) -> ExtractionResult:
    latest_body, older_body = _split_latest_context_sections(email.body)
    text_latest = f"{email.subject}\n{latest_body}"
    text_full = f"{email.subject}\n{email.body}"
    mode = _infer_mode(text_full)

    multi = _extract_multi_route_sea(latest_body)
    if multi:
        return ExtractionResult(shipments=multi, clarification_questions=[])

    origin_raw = _match_any_field(latest_body, ["origin", "from"])
    destination_raw = _match_any_field(latest_body, ["destination", "to"])

    if not origin_raw or not destination_raw:
        pair = _match_from_to_pair(latest_body)
        if pair:
            origin_raw, destination_raw = pair

    actual_weight_kg = _match_number(text_latest, r"(\d+(?:\.\d+)?)\s*kg")
    volume_cbm = _match_number(text_latest, r"(\d+(?:\.\d+)?)\s*(?:cbm|cubic\s+meters?|m3)\b")

    commodity = _match_any_field(latest_body, ["commodity"])
    if not commodity and re.search(r"(?i)\bfurniture\b", text_latest):
        commodity = "furniture"

    quantity, container_size_ft = _match_qty_container(text_latest)

    shipments: list[ShipmentRequest] = [
        ShipmentRequest(
            mode=mode,
            origin_raw=origin_raw,
            destination_raw=destination_raw,
            quantity=quantity,
            container_size_ft=container_size_ft,
            actual_weight_kg=actual_weight_kg,
            volume_cbm=volume_cbm,
            commodity=commodity,
            notes=None,
        )
    ]

    if older_body:
        older_email = EmailMessage(
            email_id=email.email_id,
            sender=email.sender,
            to=email.to,
            subject=email.subject,
            body=older_body,
        )
        older_res = _extract_rule_based(email=older_email)
        if older_res.shipments and shipments:
            shipments = [_merge_shipments_latest_over_old(latest=shipments[0], old=older_res.shipments[0])]

    return ExtractionResult(shipments=shipments, clarification_questions=[])


def _infer_mode(text: str) -> Mode:
    t = text.casefold()
    if "air" in t or "kg" in t:
        return "air"
    return "sea"


def _match_line_value(body: str, field_name: str) -> str | None:
    m = re.search(rf"(?im)^{re.escape(field_name)}\s*:\s*(.+?)\s*$", body)
    if not m:
        return None
    return m.group(1).strip() or None


def _match_any_field(body: str, field_names: list[str]) -> str | None:
    for field_name in field_names:
        m = re.search(rf"(?im)^\s*[-*]?\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$", body)
        if m:
            value = m.group(1).strip()
            if value:
                return value
    return None


def _match_from_to_pair(body: str) -> tuple[str, str] | None:
    m = re.search(
        r"(?is)\bfrom\s+(?P<o>[a-z0-9\s/().,'-]+?)\s+to\s+(?P<d>[a-z0-9\s/().,'-]+?)(?:[?.!\n]|$)",
        body,
    )
    if not m:
        return None
    o = m.group("o").strip()
    d = m.group("d").strip()
    if not o or not d:
        return None
    return o, d


def _match_qty_container(text: str) -> tuple[int | None, int | None]:
    m = re.search(r"(?i)\b(\d+)\s*x\s*(20|40)\s*(?:ft|')(?=\s|$|[.,;:!?])", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?i)\b(20|40)\s*(?:ft|')(?=\s|$|[.,;:!?])", text)
    size = int(m.group(1)) if m else None
    m = re.search(r"(?i)\b(\d+)\s*x\b", text)
    qty = int(m.group(1)) if m else None
    return qty, size


def _extract_multi_route_sea(body: str) -> list[ShipmentRequest] | None:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    origin = None
    start = 0
    for i, ln in enumerate(lines):
        m = re.search(r"(?i)rates\s+from\s+(.+?)\s+to\s*:", ln)
        if m:
            origin = m.group(1).strip()
            start = i + 1
            break
    if not origin:
        return None

    routes: list[ShipmentRequest] = []
    for ln in lines[start:]:
        m = re.search(r"(?i)^\s*\d+\.\s*(.+?)\s*-\s*(\d+)\s*x\s*(20|40)\s*ft\b", ln)
        if not m:
            continue
        dest = m.group(1).strip()
        qty = int(m.group(2))
        size = int(m.group(3))
        routes.append(
            ShipmentRequest(
                mode="sea",
                origin_raw=origin,
                destination_raw=dest,
                quantity=qty,
                container_size_ft=size,
            )
        )
    return routes or None


def _match_number(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _match_int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coerce_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return JSON.")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object.")
    return data
