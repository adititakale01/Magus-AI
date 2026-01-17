"""
REST API for the Freight Agent Pipeline.

Exposes the full pipeline output to frontends with all data including:
- Extraction results
- Enrichment + validation
- Rate matches
- Quote calculations
- Response email
- Confidence score

Run with: python api.py
"""

import os
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from models import Email, PipelineResult, Shipment, RateMatch, Surcharge
from pipeline import process_email

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Default rate sheet path (can be overridden per request)
DEFAULT_RATE_SHEET = Path(__file__).parent.parent / "hackathon_data" / "rate_sheets" / "03_rates_hard.xlsx"


def serialize_pipeline_result(result: PipelineResult) -> dict:
    """
    Convert PipelineResult to a JSON-serializable dictionary.

    Handles frozen dataclasses, tuples, and nested objects.
    """
    def convert(obj):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, tuple):
            return [convert(item) for item in obj]
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if hasattr(obj, '__dataclass_fields__'):
            # It's a dataclass - convert to dict
            return {k: convert(v) for k, v in asdict(obj).items()}
        # Fallback: try to convert to string
        return str(obj)

    return convert(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "freight-agent-api",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/api/quote", methods=["POST"])
def process_quote():
    """
    Process a freight quote request and return full pipeline data.

    Request body:
    {
        "email": {
            "from": "sender@example.com",
            "to": "freight@company.com",
            "subject": "Quote request",
            "body": "Email body text..."
        },
        "rate_sheet": "03_rates_hard.xlsx"  // optional
    }

    Response:
    {
        "success": true,
        "data": {
            "extraction": { ... },
            "enriched": { ... },
            "rate_matches": [ ... ],
            "quote": { ... },
            "response": { ... },
            "confidence": {
                "level": "high" | "medium" | "low",
                "reason": "...",
                ...
            },
            "processing_time_ms": 1234,
            "gpt_calls": 3
        }
    }
    """
    try:
        data = request.get_json()

        if not data or "email" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'email' field in request body"
            }), 400

        email_data = data["email"]

        # Validate email fields
        required_fields = ["from", "to", "subject", "body"]
        for field in required_fields:
            if field not in email_data:
                return jsonify({
                    "success": False,
                    "error": f"Missing '{field}' field in email"
                }), 400

        # Create Email object
        email = Email(
            sender=email_data["from"],
            to=email_data["to"],
            subject=email_data["subject"],
            body=email_data["body"]
        )

        # Get rate sheet path
        rate_sheet_name = data.get("rate_sheet", "03_rates_hard.xlsx")
        rate_sheet_path = Path(__file__).parent.parent / "hackathon_data" / "rate_sheets" / rate_sheet_name

        if not rate_sheet_path.exists():
            # Try the parent path structure
            rate_sheet_path = Path(__file__).parent.parent.parent / "hackathon_data" / "rate_sheets" / rate_sheet_name

        if not rate_sheet_path.exists():
            return jsonify({
                "success": False,
                "error": f"Rate sheet not found: {rate_sheet_name}"
            }), 400

        # Run pipeline
        result = process_email(email, rate_sheet_path)

        # Serialize and return
        return jsonify({
            "success": True,
            "data": serialize_pipeline_result(result)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/quote/file", methods=["POST"])
def process_quote_file():
    """
    Process a freight quote from a predefined email file.

    Request body:
    {
        "email_file": "email_01.json",
        "rate_sheet": "03_rates_hard.xlsx"
    }
    """
    try:
        data = request.get_json()

        email_file = data.get("email_file", "email_01.json")
        rate_sheet_name = data.get("rate_sheet", "03_rates_hard.xlsx")

        # Find paths
        base_path = Path(__file__).parent.parent.parent / "hackathon_data"
        email_path = base_path / "emails" / email_file
        rate_sheet_path = base_path / "rate_sheets" / rate_sheet_name

        if not email_path.exists():
            return jsonify({
                "success": False,
                "error": f"Email file not found: {email_file}"
            }), 400

        if not rate_sheet_path.exists():
            return jsonify({
                "success": False,
                "error": f"Rate sheet not found: {rate_sheet_name}"
            }), 400

        # Load email
        with open(email_path, "r", encoding="utf-8") as f:
            email_data = json.load(f)

        email = Email(
            sender=email_data["from"],
            to=email_data["to"],
            subject=email_data["subject"],
            body=email_data["body"]
        )

        # Run pipeline
        result = process_email(email, rate_sheet_path)

        # Serialize and return
        return jsonify({
            "success": True,
            "data": serialize_pipeline_result(result)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/emails", methods=["GET"])
def list_emails():
    """List available email files."""
    base_path = Path(__file__).parent.parent.parent / "hackathon_data" / "emails"
    emails = [f.name for f in base_path.glob("*.json") if f.is_file()]
    return jsonify({
        "success": True,
        "emails": sorted(emails)
    })


@app.route("/api/rate-sheets", methods=["GET"])
def list_rate_sheets():
    """List available rate sheet files."""
    base_path = Path(__file__).parent.parent.parent / "hackathon_data" / "rate_sheets"
    sheets = [f.name for f in base_path.glob("*.xlsx") if f.is_file()]
    return jsonify({
        "success": True,
        "rate_sheets": sorted(sheets)
    })


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5001))
    debug = os.getenv("API_DEBUG", "true").lower() == "true"

    print(f"\n{'='*60}")
    print("FREIGHT AGENT API")
    print(f"{'='*60}")
    print(f"Running on: http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"\nEndpoints:")
    print(f"  GET  /health           - Health check")
    print(f"  POST /api/quote        - Process email (raw JSON)")
    print(f"  POST /api/quote/file   - Process email (from file)")
    print(f"  GET  /api/emails       - List email files")
    print(f"  GET  /api/rate-sheets  - List rate sheet files")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
