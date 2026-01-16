from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal

from src.config import AppConfig
from src.llm_usage import usage_from_response
from src.prompts import SOP_PARSE_SYSTEM_PROMPT, sop_parse_user_prompt


Mode = Literal["air", "sea"]


@dataclass(frozen=True)
class SopProfile:
    customer_name: str
    match_emails: set[str] = field(default_factory=set)
    match_domains: set[str] = field(default_factory=set)
    match_domain_keywords: set[str] = field(default_factory=set)
    allowed_modes: set[Mode] | None = None
    margin_override: float | None = None
    sea_discount_pct: float | None = None
    sea_discount_label: str | None = None
    origin_required: set[str] | None = None
    origin_fallbacks: dict[str, list[str]] = field(default_factory=dict)
    transit_warning_if_gt_days: int | None = None
    show_actual_and_chargeable_weight: bool = False
    hide_margin_percent: bool = True
    container_volume_discount_tiers: list[tuple[int, float]] = field(default_factory=list)


@dataclass(frozen=True)
class SopSurchargeRule:
    amount: float
    label: str
    destination_canonical_in: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SopSurcharge:
    amount: float
    label: str


@dataclass(frozen=True)
class SopConfig:
    profiles: list[SopProfile] = field(default_factory=list)
    global_surcharge_rules: list[SopSurchargeRule] = field(default_factory=list)
    source: Literal["llm", "rule_based"] = "rule_based"


@dataclass(frozen=True)
class SopParseResult:
    sop_config: SopConfig
    sop_loaded: bool
    sop_path: Path
    cached: bool = False
    llm_usage: dict[str, Any] | None = None
    llm_raw: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


_SOP_CACHE: dict[str, SopConfig] = {}


def load_sop_markdown(*, data_dir: Path) -> str:
    path = data_dir / "SOP.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_sop(
    *,
    config: AppConfig,
    canonical_origins: list[str],
    canonical_destinations: list[str],
    known_senders: list[str],
    force_refresh: bool = False,
) -> SopParseResult:
    sop_path = (config.data.data_dir / "SOP.md").resolve()
    sop_markdown = load_sop_markdown(data_dir=config.data.data_dir)
    sop_loaded = bool(sop_markdown.strip())

    key = _cache_key(
        sop_markdown=sop_markdown,
        model=config.openai.model,
        canonical_origins=canonical_origins,
        canonical_destinations=canonical_destinations,
    )
    if not force_refresh and key in _SOP_CACHE:
        cached_config = _SOP_CACHE[key]
        return SopParseResult(
            sop_config=cached_config,
            sop_loaded=sop_loaded,
            sop_path=sop_path,
            cached=True,
            llm_usage=None,
            errors=[],
        )

    if not sop_loaded:
        empty = SopConfig(profiles=[], global_surcharge_rules=[], source="rule_based")
        _SOP_CACHE[key] = empty
        return SopParseResult(sop_config=empty, sop_loaded=False, sop_path=sop_path, cached=False, errors=["SOP.md missing/empty"])

    if config.openai.api_key:
        try:
            parsed, llm_usage = _parse_sop_with_openai(
                sop_markdown=sop_markdown,
                config=config,
                canonical_origins=canonical_origins,
                canonical_destinations=canonical_destinations,
                known_senders=known_senders,
            )
            sop_config, errors = _build_sop_config_from_parsed(
                parsed=parsed,
                canonical_origins=canonical_origins,
                canonical_destinations=canonical_destinations,
            )
            out = SopConfig(
                profiles=sop_config.profiles,
                global_surcharge_rules=sop_config.global_surcharge_rules,
                source="llm",
            )
            rule_based = _parse_sop_rule_based(
                sop_markdown=sop_markdown,
                canonical_origins=canonical_origins,
                canonical_destinations=canonical_destinations,
            )
            if not out.profiles and not out.global_surcharge_rules:
                if rule_based.profiles or rule_based.global_surcharge_rules:
                    out = SopConfig(
                        profiles=rule_based.profiles,
                        global_surcharge_rules=rule_based.global_surcharge_rules,
                        source="rule_based",
                    )
                    errors = list(errors) + ["LLM parse output could not be converted to SOP rules, used rule-based fallback."]
            else:
                merged_profiles = list(out.profiles)
                email_to_idx: dict[str, int] = {}
                name_to_idx: dict[str, int] = {}
                for idx, p in enumerate(merged_profiles):
                    if p.customer_name:
                        name_to_idx[p.customer_name.strip().casefold()] = idx
                    for e in p.match_emails:
                        if e:
                            email_to_idx[e.casefold()] = idx

                for rb in rule_based.profiles:
                    rb_name = rb.customer_name.strip().casefold() if rb.customer_name else ""
                    matched_idx: int | None = None
                    if rb.match_emails:
                        for e in rb.match_emails:
                            idx = email_to_idx.get(str(e).casefold())
                            if idx is not None:
                                matched_idx = idx
                                break
                    if matched_idx is None and rb_name:
                        matched_idx = name_to_idx.get(rb_name)

                    if matched_idx is None:
                        merged_profiles.append(rb)
                        matched_idx = len(merged_profiles) - 1
                    else:
                        merged_profiles[matched_idx] = _merge_profile(primary=merged_profiles[matched_idx], fallback=rb)

                    p = merged_profiles[matched_idx]
                    if p.customer_name:
                        name_to_idx[p.customer_name.strip().casefold()] = matched_idx
                    for e in p.match_emails:
                        if e:
                            email_to_idx[e.casefold()] = matched_idx

                merged_surcharges = list(out.global_surcharge_rules)
                surcharge_to_idx: dict[tuple[float, str], int] = {}
                for idx, r in enumerate(merged_surcharges):
                    key = (round(float(r.amount), 2), str(r.label or "").strip().casefold())
                    surcharge_to_idx[key] = idx
                for rb in rule_based.global_surcharge_rules:
                    key = (round(float(rb.amount), 2), str(rb.label or "").strip().casefold())
                    idx = surcharge_to_idx.get(key)
                    if idx is None:
                        merged_surcharges.append(rb)
                        surcharge_to_idx[key] = len(merged_surcharges) - 1
                        continue
                    existing = merged_surcharges[idx]
                    merged_surcharges[idx] = SopSurchargeRule(
                        amount=float(existing.amount),
                        label=str(existing.label),
                        destination_canonical_in=set(existing.destination_canonical_in) | set(rb.destination_canonical_in),
                    )

                out = SopConfig(
                    profiles=merged_profiles,
                    global_surcharge_rules=merged_surcharges,
                    source=out.source,
                )
            _SOP_CACHE[key] = out
            return SopParseResult(
                sop_config=out,
                sop_loaded=True,
                sop_path=sop_path,
                cached=False,
                llm_usage=llm_usage,
                llm_raw=parsed,
                errors=errors,
            )
        except Exception as e:  # noqa: BLE001
            rule_based = _parse_sop_rule_based(
                sop_markdown=sop_markdown,
                canonical_origins=canonical_origins,
                canonical_destinations=canonical_destinations,
            )
            out = SopConfig(
                profiles=rule_based.profiles,
                global_surcharge_rules=rule_based.global_surcharge_rules,
                source="rule_based",
            )
            _SOP_CACHE[key] = out
            return SopParseResult(
                sop_config=out,
                sop_loaded=True,
                sop_path=sop_path,
                cached=False,
                llm_usage=None,
                errors=[f"LLM parse failed, used rule-based fallback: {e.__class__.__name__}: {e}"],
            )

    rule_based = _parse_sop_rule_based(
        sop_markdown=sop_markdown,
        canonical_origins=canonical_origins,
        canonical_destinations=canonical_destinations,
    )
    out = SopConfig(
        profiles=rule_based.profiles,
        global_surcharge_rules=rule_based.global_surcharge_rules,
        source="rule_based",
    )
    _SOP_CACHE[key] = out
    return SopParseResult(sop_config=out, sop_loaded=True, sop_path=sop_path, cached=False, llm_usage=None, errors=[])


def _merge_profile(*, primary: SopProfile, fallback: SopProfile) -> SopProfile:
    customer_name = primary.customer_name or fallback.customer_name
    match_emails = set(primary.match_emails) | set(fallback.match_emails)
    match_domains = set(primary.match_domains) | set(fallback.match_domains)
    match_domain_keywords = set(primary.match_domain_keywords) | set(fallback.match_domain_keywords)

    allowed_modes = primary.allowed_modes if primary.allowed_modes else fallback.allowed_modes
    margin_override = primary.margin_override if primary.margin_override is not None else fallback.margin_override

    sea_discount_pct = primary.sea_discount_pct if primary.sea_discount_pct is not None else fallback.sea_discount_pct
    if primary.sea_discount_pct is not None:
        sea_discount_label = primary.sea_discount_label if primary.sea_discount_label is not None else fallback.sea_discount_label
    else:
        sea_discount_label = fallback.sea_discount_label

    origin_required = primary.origin_required if primary.origin_required else fallback.origin_required

    origin_fallbacks: dict[str, list[str]] = {}
    for src in [fallback.origin_fallbacks, primary.origin_fallbacks]:
        for loc, alts in (src or {}).items():
            loc_key = str(loc).strip()
            if not loc_key:
                continue
            origin_fallbacks.setdefault(loc_key, [])
            for alt in alts or []:
                alt_key = str(alt).strip()
                if not alt_key:
                    continue
                if alt_key not in origin_fallbacks[loc_key]:
                    origin_fallbacks[loc_key].append(alt_key)

    transit_warning_if_gt_days = (
        primary.transit_warning_if_gt_days
        if primary.transit_warning_if_gt_days is not None
        else fallback.transit_warning_if_gt_days
    )
    show_actual_and_chargeable_weight = bool(
        primary.show_actual_and_chargeable_weight or fallback.show_actual_and_chargeable_weight
    )
    hide_margin_percent = bool(primary.hide_margin_percent or fallback.hide_margin_percent)
    container_volume_discount_tiers = (
        list(primary.container_volume_discount_tiers)
        if primary.container_volume_discount_tiers
        else list(fallback.container_volume_discount_tiers)
    )

    return SopProfile(
        customer_name=customer_name,
        match_emails=match_emails,
        match_domains=match_domains,
        match_domain_keywords=match_domain_keywords,
        allowed_modes=allowed_modes,
        margin_override=margin_override,
        sea_discount_pct=sea_discount_pct,
        sea_discount_label=sea_discount_label,
        origin_required=origin_required,
        origin_fallbacks=origin_fallbacks,
        transit_warning_if_gt_days=transit_warning_if_gt_days,
        show_actual_and_chargeable_weight=show_actual_and_chargeable_weight,
        hide_margin_percent=hide_margin_percent,
        container_volume_discount_tiers=container_volume_discount_tiers,
    )


def get_sop_profile(*, sop_config: SopConfig, sender_email: str) -> SopProfile | None:
    sender = _normalize_email(sender_email)
    domain = _extract_domain(sender)
    if not sender and not domain:
        return None

    best: SopProfile | None = None
    best_score = 0
    for profile in sop_config.profiles:
        score = _match_score(profile=profile, sender_email=sender, sender_domain=domain)
        if score > best_score:
            best_score = score
            best = profile
    return best


def get_global_surcharges(*, sop_config: SopConfig, destination_canonical: str | None) -> list[SopSurcharge]:
    if not destination_canonical:
        return []
    dest = destination_canonical.strip().casefold()
    out: list[SopSurcharge] = []
    for rule in sop_config.global_surcharge_rules:
        if not rule.destination_canonical_in:
            continue
        if dest in {d.casefold() for d in rule.destination_canonical_in}:
            out.append(SopSurcharge(amount=float(rule.amount), label=str(rule.label)))
    return out


def volume_discount_pct(profile: SopProfile, *, total_containers: int) -> float:
    if total_containers <= 0:
        return 0.0
    for threshold, pct in sorted(profile.container_volume_discount_tiers, key=lambda x: -x[0]):
        if total_containers >= threshold:
            return float(pct)
    return 0.0


def origin_fallbacks(profile: SopProfile, *, origin_canonical: str) -> list[str]:
    return list(profile.origin_fallbacks.get(origin_canonical, []))


def _cache_key(
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


def _parse_sop_with_openai(
    *,
    sop_markdown: str,
    config: AppConfig,
    canonical_origins: list[str],
    canonical_destinations: list[str],
    known_senders: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.openai.api_key)
    user = sop_parse_user_prompt(
        sop_markdown=sop_markdown,
        canonical_origins=canonical_origins,
        canonical_destinations=canonical_destinations,
        known_senders=known_senders,
    )

    resp = client.chat.completions.create(
        model=config.openai.model,
        temperature=0,
        messages=[
            {"role": "system", "content": SOP_PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    llm_usage = usage_from_response(resp, model=config.openai.model, calls=1)
    content = resp.choices[0].message.content or ""
    data = _coerce_json_object(content)
    return data, llm_usage


def _build_sop_config_from_parsed(
    *,
    parsed: dict[str, Any],
    canonical_origins: list[str],
    canonical_destinations: list[str],
) -> tuple[SopConfig, list[str]]:
    errors: list[str] = []

    parsed = _normalize_llm_sop_payload(
        parsed=parsed,
        canonical_destinations=canonical_destinations,
    )

    origin_lookup = _casefold_lookup(canonical_origins)
    destination_lookup = _casefold_lookup(canonical_destinations)

    profiles: list[SopProfile] = []
    customers = parsed.get("customers") or []
    if not isinstance(customers, list):
        customers = []
        errors.append("Parsed SOP: customers is not a list.")

    for item in customers:
        if not isinstance(item, dict):
            continue

        customer_name = str(item.get("customer_name") or "").strip()
        if not customer_name:
            continue

        match = item.get("match") if isinstance(item.get("match"), dict) else {}
        emails = [_normalize_email(x) for x in (match.get("emails") or []) if str(x).strip()]
        domains = [str(x).strip().casefold() for x in (match.get("domains") or []) if str(x).strip()]
        keywords = [str(x).strip().casefold() for x in (match.get("domain_keywords") or []) if str(x).strip()]

        for email in list(emails):
            dom = _extract_domain(email)
            if dom:
                domains.append(dom)

        match_domains = {d for d in domains if d}
        match_emails = {e for e in emails if e}
        match_keywords = {k for k in keywords if k}
        if not (match_emails or match_domains or match_keywords):
            match_keywords = _keywords_from_name(customer_name)

        allowed_modes = _parse_allowed_modes(item.get("allowed_modes"))
        margin_override = _normalize_pct(item.get("margin_override_pct"))
        hide_margin = _bool_or_none(item.get("hide_margin_percent"))

        sea_discount_pct, sea_discount_label = _pick_sea_discount(item.get("discounts"))

        origin_required: set[str] | None = None
        raw_origin_required = item.get("origin_required")
        if isinstance(raw_origin_required, list) and raw_origin_required:
            resolved = {_pick_from_lookup(str(x), origin_lookup) for x in raw_origin_required if str(x).strip()}
            origin_required = {x for x in resolved if x}

        origin_fallbacks: dict[str, list[str]] = {}
        for grp in item.get("origin_equivalence_groups") or []:
            if not isinstance(grp, dict):
                continue
            if str(grp.get("scope") or "").strip().casefold() != "origin":
                continue
            locs = grp.get("locations") or []
            if not isinstance(locs, list) or len(locs) < 2:
                continue
            resolved = [_pick_from_lookup(str(x), origin_lookup) for x in locs if str(x).strip()]
            resolved = [x for x in resolved if x]
            for loc in resolved:
                others = [x for x in resolved if x != loc]
                if others:
                    origin_fallbacks.setdefault(loc, [])
                    origin_fallbacks[loc].extend([x for x in others if x not in origin_fallbacks[loc]])

        transit_warn = _int_or_none(item.get("transit_warning_if_gt_days"))
        show_weights = bool(item.get("show_actual_and_chargeable_weight") or False)

        tiers: list[tuple[int, float]] = []
        for tier in item.get("container_volume_discount_tiers") or []:
            if not isinstance(tier, dict):
                continue
            th = _int_or_none(tier.get("min_total_containers"))
            pct = _normalize_pct(tier.get("discount_pct"))
            if th and pct:
                tiers.append((int(th), float(pct)))
        tiers = sorted(list({(int(t[0]), float(t[1])) for t in tiers}), key=lambda x: -x[0])

        profiles.append(
            SopProfile(
                customer_name=customer_name,
                match_emails=match_emails,
                match_domains=match_domains,
                match_domain_keywords=match_keywords,
                allowed_modes=allowed_modes,
                margin_override=margin_override,
                sea_discount_pct=sea_discount_pct,
                sea_discount_label=sea_discount_label,
                origin_required=origin_required,
                origin_fallbacks=origin_fallbacks,
                transit_warning_if_gt_days=transit_warn,
                show_actual_and_chargeable_weight=show_weights,
                hide_margin_percent=True if hide_margin is None else bool(hide_margin),
                container_volume_discount_tiers=tiers,
            )
        )

    global_rules: list[SopSurchargeRule] = []
    surcharges = parsed.get("global_surcharges") or []
    if not isinstance(surcharges, list):
        surcharges = []
        errors.append("Parsed SOP: global_surcharges is not a list.")

    for item in surcharges:
        if not isinstance(item, dict):
            continue
        amount = _float_or_none(item.get("amount"))
        label = str(item.get("label") or "").strip()
        if amount is None or not label:
            continue
        dests_raw = item.get("destination_canonical_in") or []
        dests: set[str] = set()
        if isinstance(dests_raw, list):
            for d in dests_raw:
                picked = _pick_from_lookup(str(d), destination_lookup)
                if picked:
                    dests.add(picked)
        global_rules.append(SopSurchargeRule(amount=float(amount), label=label, destination_canonical_in=dests))

    return SopConfig(profiles=profiles, global_surcharge_rules=global_rules, source="llm"), errors


def _normalize_llm_sop_payload(*, parsed: dict[str, Any], canonical_destinations: list[str]) -> dict[str, Any]:
    if isinstance(parsed.get("customers"), list) and isinstance(parsed.get("global_surcharges"), list):
        return parsed

    def key_norm(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())

    def coerce_text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, dict):
            items: list[tuple[Any, Any]] = list(value.items())

            def sort_key(item: tuple[Any, Any]) -> tuple[int, Any]:
                k = item[0]
                try:
                    return (0, int(k))
                except (TypeError, ValueError):
                    return (1, str(k))

            items = sorted(items, key=sort_key)
            return [str(v).strip() for _, v in items if str(v).strip()]
        return []

    def get_any(mapping: Any, *names: str) -> Any:
        if not isinstance(mapping, dict) or not names:
            return None
        for name in names:
            if name in mapping:
                return mapping.get(name)
        targets = {key_norm(n) for n in names}
        for k, v in mapping.items():
            if key_norm(k) in targets:
                return v
        return None

    def get_ci(name: str) -> Any:
        return get_any(parsed, name)

    customer_specific_rules = get_ci("customer_specific_rules")
    global_rules = get_ci("global_rules")
    if not isinstance(customer_specific_rules, dict) and not isinstance(global_rules, dict):
        customer_specific_rules = get_ci("customerSpecificRules")
        global_rules = get_ci("globalRules")
    if not isinstance(customer_specific_rules, dict) and not isinstance(global_rules, dict):
        return parsed

    out_customers: list[dict[str, Any]] = []
    out_global_surcharges: list[dict[str, Any]] = []

    if isinstance(customer_specific_rules, dict):
        for raw_name, raw in customer_specific_rules.items():
            if not isinstance(raw, dict):
                continue
            customer_name = str(raw.get("customer_name") or raw_name or "").strip()
            if not customer_name:
                continue

            cust: dict[str, Any] = {"customer_name": customer_name}
            email = get_any(raw, "email", "customer_email", "customerEmail")
            if str(email or "").strip():
                cust["match"] = {"emails": [str(email).strip()]}

            rules = get_any(raw, "rules")
            rules = rules if isinstance(rules, dict) else {}
            mode = str(get_any(rules, "mode") or get_any(raw, "mode") or "").strip().casefold()
            if mode in {"air", "sea"}:
                cust["allowed_modes"] = [mode]

            disc = _normalize_pct(get_any(raw, "discount", "discount_pct", "discountPct"))
            if disc and disc > 0:
                disc_mode: str | None = None
                if isinstance(cust.get("allowed_modes"), list) and cust["allowed_modes"]:
                    disc_mode = str(cust["allowed_modes"][0])
                cust["discounts"] = [
                    {
                        "mode": disc_mode,
                        "discount_pct": float(disc),
                        "apply_before_margin": True,
                        "label": None,
                    }
                ]

            tiers: dict[int, float] = {}
            discounts_obj = get_any(raw, "discounts")
            if isinstance(discounts_obj, dict):
                for k, v in discounts_obj.items():
                    pct = _normalize_pct(v)
                    if not pct or pct <= 0:
                        continue
                    m = re.search(r"(\d+)", str(k))
                    if not m:
                        continue
                    th = _int_or_none(m.group(1))
                    if not th:
                        continue
                    tiers[int(th)] = max(float(pct), float(tiers.get(int(th), 0.0)))
            if tiers:
                cust["container_volume_discount_tiers"] = [
                    {"min_total_containers": int(th), "discount_pct": float(pct)} for th, pct in sorted(tiers.items(), key=lambda x: -x[0])
                ]

            locs = coerce_text_list(get_any(rules, "location_equivalence", "locationEquivalence", "locationEquivalenceList"))
            if len(locs) >= 2:
                cust["origin_equivalence_groups"] = [{"scope": "origin", "locations": locs}]

            req_texts = coerce_text_list(get_any(rules, "output_requirements", "outputRequirements"))
            if req_texts:
                transit_warn = _infer_transit_warning_days(req_texts)
                if transit_warn is not None:
                    cust["transit_warning_if_gt_days"] = int(transit_warn)
                if _infer_show_weights(req_texts):
                    cust["show_actual_and_chargeable_weight"] = True
                if any("subtotal" in t.casefold() for t in req_texts) or any("grand total" in t.casefold() for t in req_texts):
                    cust["require_route_subtotals_and_grand_total"] = True

            if bool(get_any(rules, "multi_route", "multiRoute")):
                cust["require_route_subtotals_and_grand_total"] = True

            out_customers.append(cust)

    if isinstance(global_rules, dict):
        au_dests = _australia_destinations(canonical_destinations)
        for raw_name, raw in global_rules.items():
            if not isinstance(raw, dict):
                continue
            name = str(raw_name or raw.get("customer_name") or "").strip()
            if not name:
                continue

            bio = _float_or_none(get_any(raw, "biosecurity_surcharge", "biosecuritySurcharge"))
            if bio and bio > 0:
                if au_dests:
                    out_global_surcharges.append(
                        {"amount": float(bio), "label": "Biosecurity surcharge (Australia)", "destination_canonical_in": au_dests}
                    )
                continue

            cust: dict[str, Any] = {"customer_name": name}
            margin = _normalize_pct(get_any(raw, "margin", "margin_override_pct", "marginOverridePct"))
            if margin is not None:
                cust["margin_override_pct"] = float(margin)
            hide_margin = _bool_or_none(get_any(raw, "hide_margin", "hideMargin", "hide_margin_percent", "hideMarginPercent"))
            if hide_margin is not None:
                cust["hide_margin_percent"] = bool(hide_margin)
            origin = str(get_any(raw, "origin", "originRestriction", "origin_required", "originRequired") or "").strip()
            if origin:
                cust["origin_required"] = [origin]
            out_customers.append(cust)

    return {"customers": out_customers, "global_surcharges": out_global_surcharges}


def _infer_transit_warning_days(requirements: list[str]) -> int | None:
    for text in requirements:
        s = str(text or "").strip()
        if not s:
            continue
        if "transit" not in s.casefold():
            continue
        m = re.search(r"(?i)exceed(?:s)?\s*(\d+)\s*day", s)
        if m:
            return _int_or_none(m.group(1))
        m = re.search(r"(?i)(\d+)\s*day", s)
        if m:
            return _int_or_none(m.group(1))
    return None


def _infer_show_weights(requirements: list[str]) -> bool:
    actual = False
    chargeable = False
    for text in requirements:
        s = str(text or "").strip().casefold()
        if not s:
            continue
        if "actual" in s and "weight" in s:
            actual = True
        if "chargeable" in s and "weight" in s:
            chargeable = True
        if "show" in s and "chargeable" in s and "actual" in s:
            return True
    return actual and chargeable


def _australia_destinations(canonical_destinations: list[str]) -> list[str]:
    australian_city_tokens = {"melbourne", "sydney", "brisbane", "perth", "adelaide"}
    destinations = [str(d) for d in canonical_destinations if str(d).strip().casefold() in australian_city_tokens]
    if destinations:
        return destinations
    melbourne = _pick_from_lookup("Melbourne", _casefold_lookup(canonical_destinations))
    return [melbourne] if melbourne else []


def _parse_sop_rule_based(
    *,
    sop_markdown: str,
    canonical_origins: list[str],
    canonical_destinations: list[str],
) -> SopConfig:
    origin_lookup = _casefold_lookup(canonical_origins)
    destination_lookup = _casefold_lookup(canonical_destinations)

    profiles: list[SopProfile] = []

    for heading, block in _iter_markdown_sections(sop_markdown):
        email = _match_one(block, r"(?im)^Customer Email:\s*([^\s]+@[^\s]+)\s*$")
        if not email:
            continue

        name = heading
        if ":" in name:
            name = name.split(":", 1)[1].strip()
        name = name.strip()
        match_email = _normalize_email(email)
        match_domain = _extract_domain(match_email) or ""

        allowed_modes = None
        if re.search(r"(?i)\bSea freight\b.*\bonly\b", block) or re.search(r"(?i)\bSea freight\b\s*\*\*only\*\*", block):
            allowed_modes = {"sea"}
        if re.search(r"(?i)\bAir freight\b.*\bonly\b", block) or re.search(r"(?i)\bAir freight\b\s*\*\*only\*\*", block):
            allowed_modes = {"air"}

        sea_discount_pct = None
        sea_discount_label = None
        disc = _match_one(block, r"(?i)^Discount:\s*([0-9]{1,2})%.*$", flags=re.MULTILINE)
        if disc:
            sea_discount_pct = _normalize_pct(disc)
            if sea_discount_pct:
                sea_discount_label = f"{int(round(sea_discount_pct * 100))}% discount (sea)"

        origin_fallbacks: dict[str, list[str]] = {}
        m = re.search(r"(?i)\b(.+?)\s+and\s+(.+?)\s+are\s+interchangeable\s+as\s+origins", block)
        if m:
            a = _pick_from_lookup(m.group(1), origin_lookup)
            b = _pick_from_lookup(m.group(2), origin_lookup)
            if a and b and a != b:
                origin_fallbacks = {a: [b], b: [a]}

        transit_warn = None
        m = re.search(r"(?i)transit\s+time\s+exceeds\s+(\d+)\s+days", block)
        if m:
            transit_warn = _int_or_none(m.group(1))

        show_weights = bool(re.search(r"(?i)show\s+both\s+actual\s+weight\s+and\s+chargeable\s+weight", block))

        tiers: list[tuple[int, float]] = []
        for pct_raw, th_raw in re.findall(r"(?i)(\d+)%\s+off\s+for\s+(\d+)\+\s+containers", block):
            th = _int_or_none(th_raw)
            pct = _normalize_pct(pct_raw)
            if th and pct:
                tiers.append((int(th), float(pct)))
        tiers = sorted(list({(int(t[0]), float(t[1])) for t in tiers}), key=lambda x: -x[0])

        profiles.append(
            SopProfile(
                customer_name=name or match_email,
                match_emails={match_email} if match_email else set(),
                match_domains={match_domain} if match_domain else set(),
                match_domain_keywords=_keywords_from_name(name),
                allowed_modes=allowed_modes,
                sea_discount_pct=sea_discount_pct,
                sea_discount_label=sea_discount_label,
                origin_fallbacks=origin_fallbacks,
                transit_warning_if_gt_days=transit_warn,
                show_actual_and_chargeable_weight=show_weights,
                container_volume_discount_tiers=tiers,
            )
        )

    appendix = _extract_appendix_items(sop_markdown)
    for item in appendix:
        name = item["name"]
        block = item["block"]
        if not name:
            continue
        if "australia destination" in name.casefold():
            continue

        margin_override = None
        m = re.search(r"(?i)use\s+(\d+)%\s+margin", block)
        if m:
            margin_override = _normalize_pct(m.group(1))

        origin_required: set[str] | None = None
        m = re.search(r"(?i)origin\s+restriction\s*:\s*(.+?)\s+only", block)
        if m:
            raw_loc = m.group(1).strip()
            picked = _pick_from_lookup(raw_loc, origin_lookup)
            if picked:
                origin_required = {picked}

        profiles.append(
            SopProfile(
                customer_name=name,
                match_domain_keywords=_keywords_from_name(name),
                margin_override=margin_override,
                origin_required=origin_required,
                hide_margin_percent=True,
            )
        )

    global_surcharge_rules: list[SopSurchargeRule] = []
    if re.search(r"(?i)australia destination", sop_markdown) and re.search(r"(?i)\$\s*150", sop_markdown):
        australian_city_tokens = {"melbourne", "sydney", "brisbane", "perth", "adelaide"}
        destinations = {d for d in canonical_destinations if d.casefold() in australian_city_tokens}
        if destinations:
            global_surcharge_rules.append(
                SopSurchargeRule(amount=150.0, label="Biosecurity surcharge (Australia)", destination_canonical_in=destinations)
            )
        else:
            melbourne = _pick_from_lookup("Melbourne", destination_lookup)
            if melbourne:
                global_surcharge_rules.append(
                    SopSurchargeRule(amount=150.0, label="Biosecurity surcharge (Australia)", destination_canonical_in={melbourne})
                )

    return SopConfig(profiles=profiles, global_surcharge_rules=global_surcharge_rules, source="rule_based")


def _iter_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines()
    indices: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            indices.append((idx, line[3:].strip()))

    sections: list[tuple[str, str]] = []
    for i, (start, title) in enumerate(indices):
        end = indices[i + 1][0] if i + 1 < len(indices) else len(lines)
        block = "\n".join(lines[start + 1 : end]).strip()
        sections.append((title, block))
    return sections


def _extract_appendix_items(markdown: str) -> list[dict[str, str]]:
    m = re.search(r"(?im)^##\s+Appendix\b.*$", markdown)
    if not m:
        return []
    tail = markdown[m.start() :]
    items: list[dict[str, str]] = []
    for match in re.finditer(r"(?m)^\s*\d+\.\s+(.+?)\s*$", tail):
        name = match.group(1).strip()
        start = match.end()
        next_m = re.search(r"(?m)^\s*\d+\.\s+.+\s*$", tail[start:])
        end = start + (next_m.start() if next_m else len(tail[start:]))
        block = tail[start:end].strip()
        items.append({"name": name, "block": block})
    return items


def _match_one(text: str, pattern: str, *, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags=flags)
    if not m:
        return None
    return m.group(1).strip() if m.group(1) else None


def _coerce_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return JSON.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object.")
    return obj


def _casefold_lookup(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values:
        key = str(v).strip().casefold()
        if key:
            out[key] = str(v)
    return out


def _pick_from_lookup(raw: str, lookup: dict[str, str]) -> str | None:
    key = str(raw or "").strip()
    if not key:
        return None
    cf = key.casefold()
    if cf in lookup:
        return lookup[cf]
    cf2 = re.sub(r"[^a-z0-9]+", " ", cf).strip()
    if cf2 in lookup:
        return lookup[cf2]
    return None


def _normalize_email(value: str | None) -> str:
    return str(value or "").strip().casefold()


def _extract_domain(email: str | None) -> str | None:
    e = str(email or "").strip().casefold()
    if "@" not in e:
        return None
    return e.split("@", 1)[1].strip() or None


def _keywords_from_name(name: str) -> set[str]:
    words = [re.sub(r"[^a-z0-9]+", "", w.casefold()) for w in str(name or "").split()]
    words = [w for w in words if len(w) >= 4]
    return set(words[:3]) if words else set()


def _parse_allowed_modes(value: Any) -> set[Mode] | None:
    if not value:
        return None
    if not isinstance(value, list):
        return None
    modes: set[Mode] = set()
    for m in value:
        s = str(m).strip().casefold()
        if s in {"air", "sea"}:
            modes.add(s)  # type: ignore[arg-type]
    return modes or None


def _normalize_pct(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        s = str(value).strip()
        if not s:
            return None
        s = s.replace("%", "")
        try:
            v = float(s)
        except ValueError:
            return None
    if v > 1.0:
        v = v / 100.0
    if v < 0:
        return None
    return float(v)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().casefold()
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_sea_discount(discounts: Any) -> tuple[float | None, str | None]:
    if not discounts or not isinstance(discounts, list):
        return None, None
    best_pct = 0.0
    best_label: str | None = None
    for d in discounts:
        if not isinstance(d, dict):
            continue
        mode = d.get("mode")
        if mode is not None and str(mode).strip().casefold() not in {"sea"}:
            continue
        pct = _normalize_pct(d.get("discount_pct"))
        if not pct or pct <= 0:
            continue
        apply_before_margin = bool(d.get("apply_before_margin", True))
        if not apply_before_margin:
            continue
        label = str(d.get("label") or "").strip() or None
        if pct > best_pct:
            best_pct = pct
            best_label = label
    return (best_pct or None), best_label


def _match_score(*, profile: SopProfile, sender_email: str | None, sender_domain: str | None) -> int:
    sender = sender_email or ""
    domain = sender_domain or ""
    has_explicit_match = bool(profile.match_emails or profile.match_domains)
    if sender and sender in {e.casefold() for e in profile.match_emails}:
        return 3
    if domain and domain in {d.casefold() for d in profile.match_domains}:
        return 2
    if (not has_explicit_match) and domain and profile.match_domain_keywords:
        for kw in profile.match_domain_keywords:
            if kw and kw.casefold() in domain:
                return 1
    return 0
