"""
GPT-powered extraction for freight quote emails.

This module handles Step 1+2 of the pipeline:
- Read email
- Extract shipment details using OpenAI GPT

The extraction keeps data RAW - normalization happens later.
"""
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from .models import Email, Shipment, ExtractionResult


# Load environment variables
load_dotenv()


# JSON schema for GPT structured output
# Note: additionalProperties: false is required at all levels for OpenAI strict mode
EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "shipments": {
            "type": "array",
            "description": "List of shipment requests found in the email",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {
                        "type": ["string", "null"],
                        "enum": ["sea", "air", None],
                        "description": "Shipping mode: 'sea' for containers/ocean, 'air' for air cargo"
                    },
                    "origin_raw": {
                        "type": ["string", "null"],
                        "description": "Origin location exactly as written in email"
                    },
                    "destination_raw": {
                        "type": ["string", "null"],
                        "description": "Destination location exactly as written in email"
                    },
                    "container_size_ft": {
                        "type": ["integer", "null"],
                        "enum": [20, 40, None],
                        "description": "Container size in feet (sea freight only)"
                    },
                    "quantity": {
                        "type": ["integer", "null"],
                        "description": "Number of containers (sea freight only)"
                    },
                    "actual_weight_kg": {
                        "type": ["number", "null"],
                        "description": "Actual weight in kg (air freight only)"
                    },
                    "volume_cbm": {
                        "type": ["number", "null"],
                        "description": "Volume in cubic meters (air freight, or for container inference)"
                    },
                    "commodity": {
                        "type": ["string", "null"],
                        "description": "Type of goods being shipped"
                    }
                },
                "required": ["mode", "origin_raw", "destination_raw", "container_size_ft", "quantity", "actual_weight_kg", "volume_cbm", "commodity"]
            }
        },
        "missing_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of required fields that are missing or unclear"
        },
        "needs_clarification": {
            "type": "boolean",
            "description": "True if we cannot provide a quote without more information"
        }
    },
    "required": ["shipments", "missing_fields", "needs_clarification"]
}


SYSTEM_PROMPT = """You are a freight quote extraction assistant. Your job is to extract shipping request details from customer emails.

RULES:
1. Extract ALL shipment routes if multiple are mentioned (e.g., "Rates to: 1. Hamburg 2. Rotterdam" = 2 shipments)
2. Keep location names EXACTLY as written - do not normalize (keep "HCMC", "ningbo", "Tokyo Narita" as-is)
3. Infer shipping mode from context:
   - SEA freight signals:
     * Mentions "container", "20ft", "40ft", "FCL", "ocean", "sea freight"
     * Large volume WITHOUT weight (e.g., "50 CBM of furniture") - this needs a container!
     * Bulky goods like furniture, machinery, vehicles
   - AIR freight signals:
     * Mentions "air freight", "air cargo", "air shipment"
     * Weight AND volume together (e.g., "450 kg, 2 CBM") - for volume weight calculation
     * Airport codes (SFO, FRA, NRT, BOM, ORD)
     * Small urgent shipments with kg specified
4. Set needs_clarification=true if:
   - Origin is just a country name (e.g., "China" instead of "Shanghai")
   - Destination is just a country name (e.g., "Poland" instead of "Gdansk")
   - Mode cannot be determined
   - For sea: missing container size or quantity (unless volume given for inference)
   - For air: missing weight or volume
5. Add to missing_fields any required information that's not provided

EXAMPLES OF MODE INFERENCE:
- "2 x 40ft container" → mode: "sea"
- "450 kg, 2 CBM" → mode: "air" (weight + volume = air freight volume weight calc)
- "50 CBM of furniture" → mode: "sea" (large volume, no weight = needs container)
- "ocean freight" → mode: "sea"
- "air cargo from Tokyo Narita" → mode: "air"
- No clear signals → mode: null, add "mode" to missing_fields"""


def load_email(filepath: str | Path) -> Email:
    """Load an email from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Email(
        sender=data["from"],
        to=data["to"],
        subject=data["subject"],
        body=data["body"]
    )


def extract_from_email(email: Email, client: OpenAI | None = None) -> ExtractionResult:
    """
    Extract shipment details from an email using GPT.

    Args:
        email: The email to extract from
        client: Optional OpenAI client (creates one if not provided)

    Returns:
        ExtractionResult with extracted shipments and any missing fields
    """
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build the user prompt
    user_prompt = f"""Extract shipment details from this email:

From: {email.sender}
Subject: {email.subject}

Body:
{email.body}

Return a JSON object with the extracted information."""

    # Call GPT with structured output
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_result",
                "strict": True,
                "schema": EXTRACTION_SCHEMA
            }
        },
        temperature=0  # Deterministic for consistent extraction
    )

    # Parse the response
    result_json = json.loads(response.choices[0].message.content)

    # Convert to dataclasses
    shipments = tuple(
        Shipment(
            mode=s.get("mode"),
            origin_raw=s.get("origin_raw"),
            destination_raw=s.get("destination_raw"),
            container_size_ft=s.get("container_size_ft"),
            quantity=s.get("quantity"),
            actual_weight_kg=s.get("actual_weight_kg"),
            volume_cbm=s.get("volume_cbm"),
            commodity=s.get("commodity")
        )
        for s in result_json["shipments"]
    )

    return ExtractionResult(
        sender_email=email.sender,
        shipments=shipments,
        missing_fields=tuple(result_json["missing_fields"]),
        needs_clarification=result_json["needs_clarification"],
        raw_email_subject=email.subject,
        raw_email_body=email.body
    )


def extract_from_file(filepath: str | Path, client: OpenAI | None = None) -> ExtractionResult:
    """Convenience function to extract from a file path."""
    email = load_email(filepath)
    return extract_from_email(email, client)
