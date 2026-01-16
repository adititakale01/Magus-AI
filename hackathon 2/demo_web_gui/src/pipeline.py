from __future__ import annotations

from dataclasses import asdict, dataclass

from src.agent import ShipmentRequest, extract_requests
from src.config import AppConfig
from src.data_loader import EmailMessage
from src.normalize import LocationResolution, clean_text, recommend_containers, resolve_location
from src.quote_logic import (
    compute_quote_amounts,
    lookup_easy_air_rate,
    lookup_easy_sea_rate,
)
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets
from src.trace import RunTrace


@dataclass(frozen=True)
class PipelineResult:
    quote_text: str | None
    trace: RunTrace
    error: str | None = None


@dataclass(frozen=True)
class RouteQuote:
    mode: str
    origin: str
    destination: str
    origin_display: str
    destination_display: str
    transit_days: int | None
    currency: str
    final_amount: float
    chargeable_weight_kg: float | None = None
    actual_weight_kg: float | None = None
    volume_cbm: float | None = None
    commodity: str | None = None
    quantity: int | None = None
    container_size_ft: int | None = None
    per_container_amount: float | None = None
    recommendation: str | None = None
    alternative: str | None = None
    notes: list[str] | None = None


def run_quote_pipeline(
    *,
    email: EmailMessage,
    config: AppConfig,
    difficulty: str,
    use_openai: bool,
    rate_sheets: NormalizedRateSheets | None = None,
) -> PipelineResult:
    trace = RunTrace()

    trace.add(
        "Load Email",
        summary=f"Loaded {email.email_id}: {email.subject}",
        data={"email_id": email.email_id, "from": email.sender, "to": email.to},
    )

    extraction = extract_requests(email=email, config=config, trace=trace, use_openai=use_openai)

    if rate_sheets is None:
        rate_sheets = normalize_rate_sheets(config=config, difficulty=difficulty)
        trace.add(
            "Rate Sheet Normalization",
            summary="Loaded and normalized the selected rate sheet.",
            data={
                "difficulty": rate_sheets.difficulty,
                "source_path": str(rate_sheets.source_path),
                "sheets": rate_sheets.sheet_names,
                "warnings": rate_sheets.warnings,
            },
        )
    else:
        trace.add(
            "Rate Sheet Normalization",
            summary="Using pre-normalized rate sheet from UI.",
            data={
                "difficulty": rate_sheets.difficulty,
                "source_path": str(rate_sheets.source_path),
                "sheets": rate_sheets.sheet_names,
                "warnings": rate_sheets.warnings,
            },
        )

    df_air = rate_sheets.air
    df_sea = rate_sheets.sea

    air_origins = sorted({str(x) for x in df_air["Origin"].dropna().tolist()})
    air_destinations = sorted({str(x) for x in df_air["Destination"].dropna().tolist()})
    sea_origins = sorted({str(x) for x in df_sea["Origin"].dropna().tolist()})
    sea_destinations = sorted({str(x) for x in df_sea["Destination"].dropna().tolist()})

    merged_aliases = _merge_aliases(config.aliases, rate_sheets.aliases)
    merged_codes = {**rate_sheets.codes, **config.codes}

    trace.add(
        "Config Match",
        summary="Resolved paths and pricing config.",
        data={
            "data_dir": str(config.data.data_dir),
            "difficulty": difficulty,
            "rate_sheet": str(rate_sheets.source_path),
            "margin": config.pricing.margin,
            "currency": config.pricing.currency,
            "air_volume_factor": config.pricing.air_volume_factor,
            "known_air_origins": air_origins,
            "known_air_destinations": air_destinations,
            "known_sea_origins": sea_origins,
            "known_sea_destinations": sea_destinations,
            "rate_sheet_warnings": rate_sheets.warnings,
        },
    )

    trace.add(
        "Load Rate Sheets",
        summary="Loaded normalized rate tables for lookup + canonical locations.",
        data={
            "air_rows": int(df_air.shape[0]),
            "sea_rows": int(df_sea.shape[0]),
        },
    )

    clarifications: list[str] = list(extraction.clarification_questions)
    quotes: list[RouteQuote] = []
    for shipment in extraction.shipments:
        if shipment.mode == "air":
            origin_res = _resolve(
                raw=shipment.origin_raw,
                known=air_origins,
                config=config,
                use_openai=use_openai,
                aliases=merged_aliases,
            )
            dest_res = _resolve(
                raw=shipment.destination_raw,
                known=air_destinations,
                config=config,
                use_openai=use_openai,
                aliases=merged_aliases,
            )
            trace.add(
                "Normalize Locations (Air)",
                summary="Resolved origin/destination for air shipment.",
                data={"origin": asdict(origin_res), "destination": asdict(dest_res)},
            )

            missing_location = False
            if not origin_res.canonical:
                clarifications.append("Which airport/city is the origin?")
                missing_location = True
            if not dest_res.canonical:
                clarifications.append("Which airport/city is the destination?")
                missing_location = True
            if missing_location:
                continue
            if shipment.actual_weight_kg is None or shipment.volume_cbm is None:
                clarifications.append("For air freight, please confirm actual weight (kg) and volume (CBM).")
                continue

            rate = lookup_easy_air_rate(
                df_air=df_air,
                origin=origin_res.canonical,
                destination=dest_res.canonical,
                currency=config.pricing.currency,
            )
            if not rate:
                trace.add(
                    "Rate Lookup (Air)",
                    summary="No direct air rate found after normalization.",
                    data={"origin": origin_res.canonical, "destination": dest_res.canonical},
                )
                clarifications.append(
                    f"No air rate found for {origin_res.canonical} -> {dest_res.canonical}. Please confirm the exact airports."
                )
                continue

            trace.add("Rate Lookup (Air)", summary="Matched air rate row.", data=asdict(rate))
            amounts = compute_quote_amounts(request=shipment, rate=rate, config=config)
            trace.add("Calculate Quote (Air)", summary="Computed chargeable weight and price.", data=amounts)

            final_amount = float(round(float(amounts["final_amount"])))
            chargeable = float(round(float(amounts["chargeable_weight_kg"])))
            o_disp = _display_location(origin_res, codes=merged_codes, show_code=True)
            d_disp = _display_location(dest_res, codes=merged_codes, show_code=True)

            notes: list[str] = []
            if shipment.actual_weight_kg is not None and chargeable > float(shipment.actual_weight_kg) + 0.1:
                notes.append("volumetric")

            quotes.append(
                RouteQuote(
                    mode="air",
                    origin=origin_res.canonical,
                    destination=dest_res.canonical,
                    origin_display=o_disp,
                    destination_display=d_disp,
                    transit_days=rate.transit_days,
                    currency=config.pricing.currency,
                    final_amount=final_amount,
                    chargeable_weight_kg=chargeable,
                    actual_weight_kg=shipment.actual_weight_kg,
                    volume_cbm=shipment.volume_cbm,
                    commodity=shipment.commodity,
                    notes=notes or None,
                )
            )
            continue

        if shipment.mode == "sea":
            origin_res = _resolve(
                raw=shipment.origin_raw,
                known=sea_origins,
                config=config,
                use_openai=use_openai,
                aliases=merged_aliases,
            )
            dest_res = _resolve(
                raw=shipment.destination_raw,
                known=sea_destinations,
                config=config,
                use_openai=use_openai,
                aliases=merged_aliases,
            )
            trace.add(
                "Normalize Locations (Sea)",
                summary="Resolved origin/destination for sea shipment.",
                data={"origin": asdict(origin_res), "destination": asdict(dest_res)},
            )

            missing_location = False
            if not origin_res.canonical:
                if shipment.origin_raw and clean_text(shipment.origin_raw) == "china":
                    clarifications.append("Which city in China? (e.g., Shanghai, Ningbo, Shenzhen)")
                else:
                    clarifications.append("Which port/city is the origin?")
                missing_location = True
            if not dest_res.canonical:
                if shipment.destination_raw and clean_text(shipment.destination_raw) == "poland":
                    clarifications.append("Which destination in Poland? (e.g., Gdansk)")
                else:
                    clarifications.append("Which port/city is the destination?")
                missing_location = True
            if missing_location:
                continue

            effective = shipment
            fit_note: str | None = None
            recommendation: str | None = None
            alternative: str | None = None

            if effective.container_size_ft not in (20, 40) or effective.quantity is None:
                if effective.volume_cbm is not None:
                    rec = recommend_containers(float(effective.volume_cbm))
                    effective = ShipmentRequest(
                        mode="sea",
                        origin_raw=shipment.origin_raw,
                        destination_raw=shipment.destination_raw,
                        quantity=rec.recommended_quantity,
                        container_size_ft=rec.recommended_size_ft,
                        volume_cbm=shipment.volume_cbm,
                        commodity=shipment.commodity,
                        notes=shipment.notes,
                    )
                    recommendation = f"{rec.recommended_quantity} x {rec.recommended_size_ft}ft container"
                    if rec.recommended_size_ft == 40 and float(effective.volume_cbm) <= 67:
                        fit_note = f"{float(effective.volume_cbm):g} CBM fits comfortably in one 40ft container."
                    if rec.alternative:
                        alt_size = int(rec.alternative["size_ft"])
                        alt_qty = int(rec.alternative["quantity"])
                        alt_rate = lookup_easy_sea_rate(
                            df_sea=df_sea,
                            origin=origin_res.canonical,
                            destination=dest_res.canonical,
                            container_size_ft=alt_size,
                            currency=config.pricing.currency,
                        )
                        if alt_rate:
                            alt_per = round(float(alt_rate.base_rate) * (1.0 + config.pricing.margin))
                            alt_total = round(float(alt_rate.base_rate) * alt_qty * (1.0 + config.pricing.margin))
                            alternative = f"{alt_qty} x {alt_size}ft @ ${int(alt_per):,} each = ${int(alt_total):,} total"
                else:
                    clarifications.append(
                        f"For sea freight {origin_res.canonical} -> {dest_res.canonical}, please confirm container size (20ft/40ft) and quantity."
                    )
                    continue

            rate = lookup_easy_sea_rate(
                df_sea=df_sea,
                origin=origin_res.canonical,
                destination=dest_res.canonical,
                container_size_ft=int(effective.container_size_ft or 0),
                currency=config.pricing.currency,
            )
            if not rate:
                trace.add(
                    "Rate Lookup (Sea)",
                    summary="No direct sea rate found after normalization.",
                    data={
                        "origin": origin_res.canonical,
                        "destination": dest_res.canonical,
                        "container_size_ft": effective.container_size_ft,
                    },
                )
                clarifications.append(
                    f"No sea rate found for {origin_res.canonical} -> {dest_res.canonical}. Please confirm the exact ports."
                )
                continue

            trace.add("Rate Lookup (Sea)", summary="Matched sea rate row.", data=asdict(rate))
            amounts = compute_quote_amounts(request=effective, rate=rate, config=config)
            trace.add("Calculate Quote (Sea)", summary="Computed total price.", data=amounts)

            final_amount = float(round(float(amounts["final_amount"])))
            per_container = float(round(float(rate.base_rate) * (1.0 + config.pricing.margin)))
            o_disp = _display_location(origin_res, codes=merged_codes, show_code=False)
            d_disp = _display_location(dest_res, codes=merged_codes, show_code=False)

            quotes.append(
                RouteQuote(
                    mode="sea",
                    origin=origin_res.canonical,
                    destination=dest_res.canonical,
                    origin_display=o_disp,
                    destination_display=d_disp,
                    transit_days=rate.transit_days,
                    currency=config.pricing.currency,
                    final_amount=final_amount,
                    quantity=effective.quantity,
                    container_size_ft=effective.container_size_ft,
                    per_container_amount=per_container,
                    volume_cbm=shipment.volume_cbm,
                    commodity=shipment.commodity,
                    recommendation=recommendation,
                    alternative=alternative,
                    notes=[fit_note] if fit_note else None,
                )
            )
            continue

        clarifications.append("Please confirm whether this is sea freight or air freight.")

    if not quotes:
        if not clarifications:
            return PipelineResult(quote_text=None, trace=trace, error="No shipments could be parsed from the email.")
        email_text = f"{email.subject}\n{email.body}".casefold()
        has_mode_signal = any(k in email_text for k in ["air", "ocean", "sea", "container"])
        has_quantity_signal = bool(clean_text(email_text) and any(k in email_text for k in ["20ft", "40ft", "cbm", "kg"]))
        if not has_mode_signal:
            clarifications.append("Is this sea freight or air freight?")
        if not has_quantity_signal:
            clarifications.append("How much cargo is this (container size/quantity, or weight and volume)?")
        trace.add("Clarification Required", summary="Missing information prevents quoting.", data={"questions": clarifications})
        return PipelineResult(quote_text=_format_clarification_response(email=email, questions=clarifications), trace=trace)

    quote_text = _format_quotes(quotes=quotes, clarifications=clarifications, config=config)
    trace.add("Format Response", summary="Generated formatted quote text.")
    return PipelineResult(quote_text=quote_text, trace=trace)


def _resolve(
    *,
    raw: str | None,
    known: list[str],
    aliases: dict[str, list[str]],
    config: AppConfig,
    use_openai: bool,
) -> LocationResolution:
    return resolve_location(raw=raw, known_locations=known, aliases=aliases, use_openai=use_openai, config=config)


def _display_location(res: LocationResolution, *, codes: dict[str, str], show_code: bool) -> str:
    if not res.canonical:
        return res.raw
    if not show_code:
        return res.canonical
    code = res.extracted_code or codes.get(res.canonical)
    return f"{res.canonical} ({code})" if code else res.canonical


def _merge_aliases(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {k: list(v) for k, v in a.items()}
    for canonical, aliases in b.items():
        merged.setdefault(canonical, [])
        merged[canonical].extend(list(aliases))
    # de-dupe, preserve order
    out: dict[str, list[str]] = {}
    for canonical, aliases in merged.items():
        seen: set[str] = set()
        deduped: list[str] = []
        for alias in aliases:
            key = str(alias).casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(str(alias))
        out[canonical] = deduped
    return out


def _format_clarification_response(*, email: EmailMessage, questions: list[str]) -> str:
    lines = [
        "Hi,",
        "",
        "Thanks for your inquiry. To provide an accurate quote, could you please confirm:",
        "",
    ]
    for i, q in enumerate(dict.fromkeys([q.strip() for q in questions if q.strip()]), start=1):
        lines.append(f"{i}. {q}")
    lines += ["", "Best regards,"]
    return "\n".join(lines)


def _format_quotes(*, quotes: list[RouteQuote], clarifications: list[str], config: AppConfig) -> str:
    if len(quotes) == 1:
        return _format_single_quote(quotes[0], clarifications=clarifications, config=config)
    return _format_multi_route(quotes, clarifications=clarifications, config=config)


def _format_single_quote(q: RouteQuote, *, clarifications: list[str], config: AppConfig) -> str:
    lines: list[str] = [f"{q.origin_display} -> {q.destination_display}"]
    if q.mode == "air":
        lines.append("Air Freight")
        if q.actual_weight_kg is not None and q.volume_cbm is not None:
            lines.append(f"Actual: {int(q.actual_weight_kg)} kg | Volume: {q.volume_cbm:g} CBM")
        suffix = " (volumetric)" if q.notes and "volumetric" in q.notes else ""
        lines.append(f"Chargeable weight: {int(q.chargeable_weight_kg or 0)} kg{suffix}")
        lines.append(f"Rate: ${int(q.final_amount):,} {q.currency}")
        if q.transit_days is not None:
            lines.append(f"Transit: {q.transit_days} days")
    else:
        qty = int(q.quantity or 0)
        size = int(q.container_size_ft or 0)
        if q.volume_cbm is not None:
            cargo = f"Cargo: {q.volume_cbm:g} CBM"
            if q.commodity:
                cargo += f" ({q.commodity})"
            lines.append(cargo)
        if q.recommendation:
            lines.append(f"Recommended: {q.recommendation}")
        if not q.recommendation:
            lines.append(f"{qty} x {size}ft container" if qty == 1 else f"{qty} x {size}ft containers")
        if qty > 1 and q.per_container_amount is not None:
            lines.append(f"Rate: ${int(q.per_container_amount):,} per container")
            lines.append(f"Total: ${int(q.final_amount):,} {q.currency}")
        else:
            lines.append(f"Rate: ${int(q.final_amount):,} {q.currency}")
        if q.transit_days is not None:
            lines.append(f"Transit: {q.transit_days} days")
        if q.notes:
            for note in q.notes:
                lines.append(f"Note: {note}")
        if q.alternative:
            lines.append(f"Alternative: {q.alternative}")

    if clarifications:
        lines.append("")
        lines.append("Questions:")
        for qn in dict.fromkeys([x.strip() for x in clarifications if x.strip()]):
            lines.append(f"- {qn}")
    return "\n".join(lines)


def _format_multi_route(quotes: list[RouteQuote], *, clarifications: list[str], config: AppConfig) -> str:
    lines: list[str] = []
    grand_total = 0.0
    for idx, q in enumerate(quotes, start=1):
        lines.append(f"Route {idx}: {q.origin_display} -> {q.destination_display}")
        qty = int(q.quantity or 0)
        size = int(q.container_size_ft or 0)
        if q.mode == "sea":
            lines.append(f"  {qty} x {size}ft container" if qty == 1 else f"  {qty} x {size}ft containers")
            if q.per_container_amount is not None:
                if qty > 1:
                    lines.append(f"  Rate: ${int(q.per_container_amount):,} per container")
                else:
                    lines.append(f"  Rate: ${int(q.per_container_amount):,}")
            lines.append(f"  Subtotal: ${int(q.final_amount):,}")
            if q.transit_days is not None:
                lines.append(f"  Transit: {q.transit_days} days")
            if q.notes:
                for note in q.notes:
                    lines.append(f"  Note: {note}")
        else:
            lines.append(f"  Air Freight: {int(q.chargeable_weight_kg or 0)} kg chargeable")
            lines.append(f"  Subtotal: ${int(q.final_amount):,}")
            if q.transit_days is not None:
                lines.append(f"  Transit: {q.transit_days} days")
        lines.append("")
        grand_total += q.final_amount

    lines.append(f"GRAND TOTAL: ${int(round(grand_total)):,} {config.pricing.currency}")

    if clarifications:
        lines.append("")
        lines.append("Questions:")
        for qn in dict.fromkeys([x.strip() for x in clarifications if x.strip()]):
            lines.append(f"- {qn}")
    return "\n".join(lines)
