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
