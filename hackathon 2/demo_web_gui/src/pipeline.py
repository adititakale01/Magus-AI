from __future__ import annotations

from dataclasses import asdict, dataclass

from src.agent import ShipmentRequest, extract_requests
from src.config import AppConfig
from src.data_loader import EmailMessage, list_emails, load_email
from src.normalize import LocationResolution, clean_text, recommend_containers, resolve_location
from src.quote_logic import (
    compute_quote_amounts,
    lookup_easy_air_rate,
    lookup_easy_sea_rate,
)
from src.rate_sheets import NormalizedRateSheets, normalize_rate_sheets
from src.llm_usage import sum_usage
from src.sop import (
    SopConfig,
    SopParseResult,
    SopProfile,
    get_global_surcharges,
    get_sop_profile,
    load_sop_markdown,
    origin_fallbacks as sop_origin_fallbacks,
    parse_sop,
    volume_discount_pct,
)
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
    warnings: list[str] | None = None


def run_quote_pipeline(
    *,
    email: EmailMessage,
    config: AppConfig,
    difficulty: str,
    use_openai: bool,
    rate_sheets: NormalizedRateSheets | None = None,
    enable_sop: bool = False,
    sop_config: SopConfig | None = None,
) -> PipelineResult:
    trace = RunTrace()

    trace.add(
        "Load Email",
        summary=f"Loaded {email.email_id}: {email.subject}",
        data={"email_id": email.email_id, "from": email.sender, "to": email.to},
    )

    extraction = extract_requests(email=email, config=config, trace=trace, use_openai=use_openai)

    sop_result: SopParseResult | None = None
    active_sop_config = SopConfig()
    sop_profile: SopProfile | None = None
    sop_preparsed = False

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

    if enable_sop and sop_config is not None:
        active_sop_config = sop_config
        sop_profile = get_sop_profile(sop_config=active_sop_config, sender_email=email.sender)
        sop_preparsed = True
    elif enable_sop:
        email_index = list_emails(config.data.data_dir)
        known_senders: list[str] = []
        for path in email_index.values():
            try:
                msg = load_email(path)
            except Exception:
                continue
            sender = str(msg.sender or "").strip()
            if sender:
                known_senders.append(sender)
        known_senders = sorted(set(known_senders))

        canonical_origins = sorted(set(air_origins + sea_origins))
        canonical_destinations = sorted(set(air_destinations + sea_destinations))
        sop_result = parse_sop(
            config=config,
            canonical_origins=canonical_origins,
            canonical_destinations=canonical_destinations,
            known_senders=known_senders,
        )
        active_sop_config = sop_result.sop_config
        sop_profile = get_sop_profile(sop_config=active_sop_config, sender_email=email.sender)

    effective_margin = float(config.pricing.margin)
    if enable_sop and sop_profile and sop_profile.margin_override is not None:
        effective_margin = float(sop_profile.margin_override)

    sop_path = (config.data.data_dir / "SOP.md").resolve()
    sop_found = sop_path.exists()
    sop_loaded = False
    if enable_sop:
        sop_loaded = bool(sop_result.sop_loaded) if sop_result else bool(load_sop_markdown(data_dir=config.data.data_dir).strip())
    trace.add(
        "SOP",
        summary="Applied customer-specific SOP rules." if enable_sop else "SOP disabled.",
        data={
            "enabled": enable_sop,
            "sop_path": str(sop_path),
            "sop_found": sop_found,
            "sop_loaded": sop_loaded,
            "preparsed": sop_preparsed if enable_sop else None,
            "parse_source": active_sop_config.source if enable_sop else None,
            "parse_cached": sop_result.cached if sop_result else None,
            "parse_errors": sop_result.errors if sop_result else None,
            "customer_email": email.sender,
            "matched_profile": {
                "customer_name": sop_profile.customer_name,
                "allowed_modes": sorted(list(sop_profile.allowed_modes)) if sop_profile and sop_profile.allowed_modes else None,
                "margin_override": sop_profile.margin_override if sop_profile else None,
                "sea_discount_pct": sop_profile.sea_discount_pct if sop_profile else None,
                "container_volume_discount_tiers": sop_profile.container_volume_discount_tiers if sop_profile else None,
                "origin_required": sorted(list(sop_profile.origin_required)) if sop_profile and sop_profile.origin_required else None,
                "match_emails": sorted(list(sop_profile.match_emails)) if sop_profile and sop_profile.match_emails else None,
                "match_domains": sorted(list(sop_profile.match_domains)) if sop_profile and sop_profile.match_domains else None,
                "match_domain_keywords": sorted(list(sop_profile.match_domain_keywords))
                if sop_profile and sop_profile.match_domain_keywords
                else None,
            }
            if sop_profile
            else None,
        },
        used_llm=bool(sop_result and sop_result.llm_usage),
        llm_usage=sop_result.llm_usage if sop_result else None,
    )

    trace.add(
        "Config Match",
        summary="Resolved paths and pricing config.",
        data={
            "data_dir": str(config.data.data_dir),
            "difficulty": difficulty,
            "rate_sheet": str(rate_sheets.source_path),
            "margin_default": config.pricing.margin,
            "margin_effective": effective_margin,
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
    container_discount_pct = 0.0
    container_discount_label: str | None = None
    container_qty_estimates: list[int | None] = [None] * len(extraction.shipments)
    total_containers_estimate = 0
    if enable_sop and sop_profile and sop_profile.container_volume_discount_tiers:
        missing_container_counts = False
        for idx, sh in enumerate(extraction.shipments):
            if sh.mode != "sea":
                continue
            qty = sh.quantity
            if qty is None and sh.volume_cbm is not None:
                qty = recommend_containers(float(sh.volume_cbm)).recommended_quantity
            if qty is None:
                missing_container_counts = True
            else:
                qty_int = int(qty)
                container_qty_estimates[idx] = qty_int
                total_containers_estimate += qty_int

        container_discount_pct = volume_discount_pct(sop_profile, total_containers=total_containers_estimate)
        if container_discount_pct > 0:
            container_discount_label = (
                f"{int(container_discount_pct * 100)}% volume discount ({total_containers_estimate} containers total)"
            )
        if missing_container_counts:
            clarifications.append("Please confirm total container quantities across all routes so we can apply your volume discount.")
    quotes: list[RouteQuote] = []
    for shipment_idx, shipment in enumerate(extraction.shipments):
        if enable_sop and sop_profile and sop_profile.allowed_modes and shipment.mode not in sop_profile.allowed_modes:
            allowed = " / ".join(
                ["air freight" if m == "air" else "sea freight" for m in sorted(sop_profile.allowed_modes)]
            )
            clarifications.append(
                f"Per your SOP ({sop_profile.customer_name}), we can quote {allowed} only. Please confirm if you'd like a {allowed} quote."
            )
            continue
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
            loc_llm_usage = sum_usage([origin_res.llm_usage, dest_res.llm_usage], model=config.openai.model)
            trace.add(
                "Normalize Locations (Air)",
                summary="Resolved origin/destination for air shipment.",
                data={"origin": asdict(origin_res), "destination": asdict(dest_res)},
                used_llm=bool(loc_llm_usage),
                llm_usage=loc_llm_usage,
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
            surcharges = (
                get_global_surcharges(sop_config=active_sop_config, destination_canonical=dest_res.canonical)
                if enable_sop
                else []
            )
            surcharge_total = sum(float(s.amount) for s in surcharges)
            amounts = compute_quote_amounts(
                request=shipment,
                rate=rate,
                config=config,
                margin_override=effective_margin if enable_sop else None,
                surcharge=surcharge_total,
            )
            trace.add("Calculate Quote (Air)", summary="Computed chargeable weight and price.", data=amounts)

            final_amount = float(round(float(amounts["final_amount"])))
            chargeable = float(round(float(amounts["chargeable_weight_kg"])))
            o_disp = _display_location(origin_res, codes=merged_codes, show_code=True)
            d_disp = _display_location(dest_res, codes=merged_codes, show_code=True)

            notes: list[str] = []
            if shipment.actual_weight_kg is not None and chargeable > float(shipment.actual_weight_kg) + 0.1:
                notes.append("volumetric")
            if enable_sop and surcharges:
                for s in surcharges:
                    notes.append(f"Surcharge: ${int(round(float(s.amount))):,} {s.label}")

            warnings: list[str] = []
            if (
                enable_sop
                and sop_profile
                and sop_profile.transit_warning_if_gt_days is not None
                and rate.transit_days is not None
                and rate.transit_days > int(sop_profile.transit_warning_if_gt_days)
            ):
                warnings.append(
                    f"Transit time is {int(rate.transit_days)} days (exceeds {int(sop_profile.transit_warning_if_gt_days)} days)."
                )

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
                    warnings=warnings or None,
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
            loc_llm_usage = sum_usage([origin_res.llm_usage, dest_res.llm_usage], model=config.openai.model)
            trace.add(
                "Normalize Locations (Sea)",
                summary="Resolved origin/destination for sea shipment.",
                data={"origin": asdict(origin_res), "destination": asdict(dest_res)},
                used_llm=bool(loc_llm_usage),
                llm_usage=loc_llm_usage,
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

            if enable_sop and sop_profile and sop_profile.origin_required:
                required = {x.strip() for x in sop_profile.origin_required if x.strip()}
                if not origin_res.canonical or origin_res.canonical not in required:
                    allowed = " / ".join(sorted(required))
                    clarifications.append(
                        f"Per your SOP ({sop_profile.customer_name}), origin must be {allowed}. Please confirm the origin."
                    )
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
                        alt_discount_pct = 0.0
                        alt_discount_label: str | None = None
                        if enable_sop and sop_profile and sop_profile.container_volume_discount_tiers:
                            base_qty = container_qty_estimates[shipment_idx]
                            if base_qty is not None:
                                alt_total_containers = int(total_containers_estimate) - int(base_qty) + int(alt_qty)
                                alt_discount_pct = volume_discount_pct(sop_profile, total_containers=alt_total_containers)
                                if alt_discount_pct > 0:
                                    alt_discount_label = (
                                        f"{int(alt_discount_pct * 100)}% volume discount ({alt_total_containers} containers total)"
                                    )

                        alt_rate, _, _ = _lookup_sea_rate_with_sop(
                            df_sea=df_sea,
                            origin=origin_res.canonical,
                            destination=dest_res.canonical,
                            container_size_ft=alt_size,
                            currency=config.pricing.currency,
                            sop_profile=sop_profile if enable_sop else None,
                        )
                        if alt_rate:
                            surcharges = (
                                get_global_surcharges(
                                    sop_config=active_sop_config,
                                    destination_canonical=dest_res.canonical,
                                )
                                if enable_sop
                                else []
                            )
                            surcharge_total = sum(float(s.amount) for s in surcharges)
                            alt_effective = ShipmentRequest(
                                mode="sea",
                                origin_raw=shipment.origin_raw,
                                destination_raw=shipment.destination_raw,
                                quantity=alt_qty,
                                container_size_ft=alt_size,
                                volume_cbm=shipment.volume_cbm,
                                commodity=shipment.commodity,
                                notes=shipment.notes,
                            )

                            discount_pct = 0.0
                            if enable_sop and sop_profile:
                                if sop_profile.sea_discount_pct:
                                    discount_pct = float(sop_profile.sea_discount_pct)
                                elif alt_discount_pct > 0:
                                    discount_pct = float(alt_discount_pct)
                                elif container_discount_pct > 0:
                                    discount_pct = float(container_discount_pct)

                            alt_amounts = compute_quote_amounts(
                                request=alt_effective,
                                rate=alt_rate,
                                config=config,
                                margin_override=effective_margin if enable_sop else None,
                                discount_pct=discount_pct,
                                surcharge=surcharge_total,
                            )
                            alt_subtotal = float(round(float(alt_amounts["final_amount_before_surcharge"])))
                            alt_discount = float(alt_amounts.get("discount_pct") or 0.0)
                            alt_margin = float(alt_amounts.get("margin") or config.pricing.margin)
                            alt_per = float(round(float(alt_rate.base_rate) * (1.0 - alt_discount) * (1.0 + alt_margin)))
                            alt_total = float(round(float(alt_amounts["final_amount"])))

                            suffix = ""
                            if enable_sop and alt_discount_label:
                                suffix += f" ({alt_discount_label})"
                            if enable_sop and surcharges:
                                surcharge_label = ", ".join([f"${int(round(float(s.amount))):,} {s.label}" for s in surcharges])
                                suffix += f" (+{surcharge_label} = {_format_money(alt_total)} total)"

                            if enable_sop and surcharges:
                                alternative = (
                                    f"{alt_qty} x {alt_size}ft @ {_format_money(alt_per)} each = {_format_money(alt_subtotal)} subtotal{suffix}"
                                )
                            else:
                                alternative = f"{alt_qty} x {alt_size}ft @ {_format_money(alt_per)} each = {_format_money(alt_subtotal)} total{suffix}"
                else:
                    clarifications.append(
                        f"For sea freight {origin_res.canonical} -> {dest_res.canonical}, please confirm container size (20ft/40ft) and quantity."
                    )
                    continue

            rate, origin_used, origin_fallback_note = _lookup_sea_rate_with_sop(
                df_sea=df_sea,
                origin=origin_res.canonical,
                destination=dest_res.canonical,
                container_size_ft=int(effective.container_size_ft or 0),
                currency=config.pricing.currency,
                sop_profile=sop_profile if enable_sop else None,
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

            lookup_data = asdict(rate)
            if origin_used != origin_res.canonical:
                lookup_data["origin_equivalence"] = {"requested": origin_res.canonical, "matched": origin_used}
            trace.add(
                "Rate Lookup (Sea)",
                summary="Matched sea rate row." if origin_used == origin_res.canonical else "Matched sea rate row (SOP origin equivalence applied).",
                data=lookup_data,
            )

            discount_pct = 0.0
            discount_note: str | None = None
            if enable_sop and sop_profile:
                if sop_profile.sea_discount_pct:
                    discount_pct = float(sop_profile.sea_discount_pct)
                    discount_note = sop_profile.sea_discount_label or f"{int(discount_pct * 100)}% discount"
                elif container_discount_pct > 0:
                    discount_pct = float(container_discount_pct)
                    discount_note = container_discount_label

            surcharges = (
                get_global_surcharges(sop_config=active_sop_config, destination_canonical=dest_res.canonical)
                if enable_sop
                else []
            )
            surcharge_total = sum(float(s.amount) for s in surcharges)

            amounts = compute_quote_amounts(
                request=effective,
                rate=rate,
                config=config,
                margin_override=effective_margin if enable_sop else None,
                discount_pct=discount_pct,
                surcharge=surcharge_total,
            )
            trace.add("Calculate Quote (Sea)", summary="Computed total price.", data=amounts)

            final_amount = float(round(float(amounts["final_amount"])))
            discount_used = float(amounts.get("discount_pct") or 0.0)
            margin_used = float(amounts.get("margin") or config.pricing.margin)
            per_container = float(round(float(rate.base_rate) * (1.0 - discount_used) * (1.0 + margin_used)))
            o_disp = _display_location(origin_res, codes=merged_codes, show_code=False)
            d_disp = _display_location(dest_res, codes=merged_codes, show_code=False)

            notes_out: list[str] = []
            if fit_note:
                notes_out.append(fit_note)
            if origin_fallback_note:
                notes_out.append(origin_fallback_note)
            if enable_sop and discount_note:
                notes_out.append(f"Discount applied: {discount_note}")
            if enable_sop and surcharges:
                for s in surcharges:
                    notes_out.append(f"Surcharge: ${int(round(float(s.amount))):,} {s.label}")

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
                    notes=notes_out or None,
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

    quote_body = _format_quotes(quotes=quotes, clarifications=clarifications, config=config)
    quote_text = (
        _format_quote_email_reply(email=email, quote_body=quote_body, sop_profile=sop_profile)
        if enable_sop
        else quote_body
    )
    trace.add("Format Response", summary="Generated formatted quote text.")
    return PipelineResult(quote_text=quote_text, trace=trace)


def _lookup_sea_rate_with_sop(
    *,
    df_sea,
    origin: str,
    destination: str,
    container_size_ft: int,
    currency: str,
    sop_profile: SopProfile | None,
):
    rate = lookup_easy_sea_rate(
        df_sea=df_sea,
        origin=origin,
        destination=destination,
        container_size_ft=container_size_ft,
        currency=currency,
    )
    if rate:
        return rate, origin, None

    if not sop_profile:
        return None, origin, None

    for alt_origin in sop_origin_fallbacks(sop_profile, origin_canonical=origin):
        alt_rate = lookup_easy_sea_rate(
            df_sea=df_sea,
            origin=alt_origin,
            destination=destination,
            container_size_ft=container_size_ft,
            currency=currency,
        )
        if alt_rate:
            note = f"Origin equivalence applied: matched using {alt_origin} per SOP."
            return alt_rate, alt_origin, note

    return None, origin, None


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


def _format_money(amount: float) -> str:
    value = float(amount)
    nearest_int = round(value)
    if abs(value - nearest_int) < 1e-6:
        return f"${int(nearest_int):,}"
    return f"${value:,.2f}"


def _format_quote_email_reply(*, email: EmailMessage, quote_body: str, sop_profile: SopProfile | None) -> str:
    greeting = "Hi,"
    intro = "Thank you for your inquiry. Please find our quote below."
    if sop_profile:
        intro = f"Thank you for your inquiry. We have applied your SOP ({sop_profile.customer_name}). Please find our quote below."

    lines: list[str] = [
        greeting,
        "",
        intro,
        "",
        quote_body.strip(),
        "",
        "Best regards,",
    ]
    return "\n".join(lines).strip() + "\n"


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
        lines.append(f"Rate: {_format_money(q.final_amount)} {q.currency}")
        if q.transit_days is not None:
            lines.append(f"Transit: {q.transit_days} days")
        if q.warnings:
            for warning in q.warnings:
                lines.append(f"Warning: {warning}")
        if q.notes:
            for note in q.notes:
                if note.strip().casefold() == "volumetric":
                    continue
                lines.append(f"Note: {note}")
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
            lines.append(f"Rate: {_format_money(q.per_container_amount)} per container")
            lines.append(f"Total: {_format_money(q.final_amount)} {q.currency}")
        else:
            lines.append(f"Rate: {_format_money(q.final_amount)} {q.currency}")
        if q.transit_days is not None:
            lines.append(f"Transit: {q.transit_days} days")
        if q.warnings:
            for warning in q.warnings:
                lines.append(f"Warning: {warning}")
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
                    lines.append(f"  Rate: {_format_money(q.per_container_amount)} per container")
                else:
                    lines.append(f"  Rate: {_format_money(q.per_container_amount)}")
            lines.append(f"  Subtotal: {_format_money(q.final_amount)}")
            if q.transit_days is not None:
                lines.append(f"  Transit: {q.transit_days} days")
            if q.warnings:
                for warning in q.warnings:
                    lines.append(f"  Warning: {warning}")
            if q.notes:
                for note in q.notes:
                    lines.append(f"  Note: {note}")
        else:
            lines.append(f"  Air Freight: {int(q.chargeable_weight_kg or 0)} kg chargeable")
            if q.actual_weight_kg is not None and q.volume_cbm is not None:
                lines.append(f"  Actual: {int(q.actual_weight_kg)} kg | Volume: {q.volume_cbm:g} CBM")
            lines.append(f"  Subtotal: {_format_money(q.final_amount)}")
            if q.transit_days is not None:
                lines.append(f"  Transit: {q.transit_days} days")
            if q.warnings:
                for warning in q.warnings:
                    lines.append(f"  Warning: {warning}")
            if q.notes:
                for note in q.notes:
                    if note.strip().casefold() == "volumetric":
                        continue
                    lines.append(f"  Note: {note}")
        lines.append("")
        grand_total += q.final_amount

    lines.append(f"GRAND TOTAL: {_format_money(grand_total)} {config.pricing.currency}")

    if clarifications:
        lines.append("")
        lines.append("Questions:")
        for qn in dict.fromkeys([x.strip() for x in clarifications if x.strip()]):
            lines.append(f"- {qn}")
    return "\n".join(lines)
