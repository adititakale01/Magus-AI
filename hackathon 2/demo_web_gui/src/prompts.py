from __future__ import annotations


EXTRACTION_SYSTEM_PROMPT = (
    "You extract shipping rate request details from emails.\n"
    "Return ONLY valid JSON (no markdown) with this schema:\n"
    "{\n"
    '  "shipments": [\n'
    "    {\n"
    '      "mode": "air"|"sea",\n'
    '      "origin_raw": string|null,\n'
    '      "destination_raw": string|null,\n'
    '      "quantity": integer|null,\n'
    '      "container_size_ft": 20|40|null,\n'
    '      "actual_weight_kg": number|null,\n'
    '      "volume_cbm": number|null,\n'
    '      "commodity": string|null,\n'
    '      "notes": string|null\n'
    "    }\n"
    "  ],\n"
    '  "clarification_questions": [string]\n'
    "}\n"
    "If an email contains multiple routes, return multiple shipments.\n"
    "If info is missing, use null and add a clarification question.\n"
    "Do not invent values."
)


EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "shipments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["air", "sea"]},
                    "origin_raw": {"type": ["string", "null"]},
                    "destination_raw": {"type": ["string", "null"]},
                    "quantity": {"type": ["integer", "null"]},
                    "container_size_ft": {"type": ["integer", "null"], "enum": [20, 40, None]},
                    "actual_weight_kg": {"type": ["number", "null"]},
                    "volume_cbm": {"type": ["number", "null"]},
                    "commodity": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["mode", "origin_raw", "destination_raw"],
            },
        },
        "clarification_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["shipments", "clarification_questions"],
}


def extraction_user_prompt(*, subject: str, body: str) -> str:
    return f"SUBJECT:\n{subject}\n\nBODY:\n{body}"


LOCATION_MATCH_SYSTEM_PROMPT = (
    "You map a raw location string to the best matching canonical location.\n"
    "Return ONLY JSON: {\"canonical\": \"...\"}\n"
    "If none matches, return {\"canonical\": null}."
)


LOCATION_MATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"canonical": {"type": ["string", "null"]}},
    "required": ["canonical"],
}


def location_match_user_prompt(*, raw: str, choices: list[str]) -> str:
    return "RAW:\n" + str(raw) + "\n\nCHOICES:\n" + "\n".join(f"- {c}" for c in choices)


SOP_PARSE_SYSTEM_PROMPT = (
    "You convert a Markdown SOP document into a machine-readable configuration for a freight quotation system.\n"
    "Return ONLY valid JSON (no markdown).\n"
    "\n"
    "Key rules:\n"
    "- Extract customer-specific rules and global rules.\n"
    "- Use decimals for percentages (e.g., 10% -> 0.10; 8% -> 0.08).\n"
    "- For any location fields, choose canonical names EXACTLY from the provided canonical lists.\n"
    "- If the SOP does not explicitly specify a customer email, infer match criteria via sender domain keywords.\n"
    "- Do not invent discounts, margins, or surcharges that are not present in the SOP.\n"
)


SOP_PARSE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "customers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "match": {
                        "type": "object",
                        "properties": {
                            "emails": {"type": "array", "items": {"type": "string"}},
                            "domains": {"type": "array", "items": {"type": "string"}},
                            "domain_keywords": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "allowed_modes": {"type": ["array", "null"], "items": {"type": "string", "enum": ["air", "sea"]}},
                    "margin_override_pct": {"type": ["number", "null"]},
                    "hide_margin_percent": {"type": ["boolean", "null"]},
                    "discounts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": ["string", "null"], "enum": ["air", "sea", None]},
                                "discount_pct": {"type": "number"},
                                "apply_before_margin": {"type": "boolean"},
                                "label": {"type": ["string", "null"]},
                            },
                            "required": ["discount_pct", "apply_before_margin"],
                        },
                    },
                    "origin_equivalence_groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scope": {"type": "string", "enum": ["origin"]},
                                "locations": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["scope", "locations"],
                        },
                    },
                    "origin_required": {"type": "array", "items": {"type": "string"}},
                    "transit_warning_if_gt_days": {"type": ["integer", "null"]},
                    "show_actual_and_chargeable_weight": {"type": ["boolean", "null"]},
                    "container_volume_discount_tiers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "min_total_containers": {"type": "integer"},
                                "discount_pct": {"type": "number"},
                            },
                            "required": ["min_total_containers", "discount_pct"],
                        },
                    },
                    "require_route_subtotals_and_grand_total": {"type": ["boolean", "null"]},
                    "require_transit_time": {"type": ["boolean", "null"]},
                },
                "required": ["customer_name"],
            },
        },
        "global_surcharges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "label": {"type": "string"},
                    "destination_canonical_in": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["amount", "label"],
            },
        },
    },
    "required": ["customers", "global_surcharges"],
}


def sop_parse_user_prompt(
    *,
    sop_markdown: str,
    canonical_origins: list[str],
    canonical_destinations: list[str],
    known_senders: list[str],
) -> str:
    return (
        "SOP_MARKDOWN:\n"
        + sop_markdown
        + "\n\nCANONICAL_ORIGINS:\n"
        + "\n".join(f"- {x}" for x in canonical_origins)
        + "\n\nCANONICAL_DESTINATIONS:\n"
        + "\n".join(f"- {x}" for x in canonical_destinations)
        + "\n\nKNOWN_SENDER_EMAILS:\n"
        + "\n".join(f"- {x}" for x in known_senders)
    )
