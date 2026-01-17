"""
Test the extraction module against all 10 hackathon emails.

Run from the freight_agent directory:
    python test_extraction.py
"""
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extraction import extract_from_file

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "hackathon_data"
EMAILS_DIR = DATA_DIR / "emails"


def test_all_emails():
    """Test extraction on all 10 emails and print results."""
    print("=" * 60)
    print("FREIGHT AGENT - EXTRACTION TEST")
    print("=" * 60)

    for i in range(1, 11):
        email_file = EMAILS_DIR / f"email_{i:02d}.json"

        if not email_file.exists():
            print(f"\n[X] Email {i:02d}: File not found")
            continue

        print(f"\n{'-' * 60}")
        print(f"EMAIL {i:02d}")
        print(f"{'-' * 60}")

        try:
            result = extract_from_file(email_file)

            print(f"Sender: {result.sender_email}")
            print(f"Subject: {result.raw_email_subject}")
            print(f"Needs Clarification: {result.needs_clarification}")

            if result.missing_fields:
                print(f"Missing Fields: {', '.join(result.missing_fields)}")

            print(f"\nShipments ({len(result.shipments)}):")
            for j, shipment in enumerate(result.shipments, 1):
                print(f"\n  [{j}] Mode: {shipment.mode}")
                print(f"      Origin: {shipment.origin_raw}")
                print(f"      Destination: {shipment.destination_raw}")

                if shipment.mode == "sea":
                    print(f"      Container: {shipment.quantity}x{shipment.container_size_ft}ft")
                elif shipment.mode == "air":
                    print(f"      Weight: {shipment.actual_weight_kg} kg")
                    print(f"      Volume: {shipment.volume_cbm} CBM")

                if shipment.commodity:
                    print(f"      Commodity: {shipment.commodity}")

            print(f"\n[OK] Email {i:02d} extracted successfully")

        except Exception as e:
            print(f"\n[X] Email {i:02d} FAILED: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_all_emails()
