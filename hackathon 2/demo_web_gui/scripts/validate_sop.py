from __future__ import annotations

import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from src.config import load_app_config  # noqa: E402
from src.data_loader import list_emails, load_email  # noqa: E402
from src.pipeline import run_quote_pipeline  # noqa: E402
from src.rate_sheets import normalize_rate_sheets  # noqa: E402


def _money_values(text: str) -> list[float]:
    hits = re.findall(r"\$([0-9][0-9,]*(?:\\.[0-9]{1,2})?)", text)
    values: list[float] = []
    for h in hits:
        try:
            values.append(float(h.replace(",", "")))
        except ValueError:
            continue
    return values


def main() -> int:
    config = load_app_config(APP_DIR / "config.toml")
    email_index = list_emails(config.data.data_dir)

    checks = {
        # Global Imports: 10% discount before 15% margin -> 3312 per 40ft; total 6624
        "email_01": {"amounts": [3312, 6624], "phrases": ["applied your sop (global imports ltd)"]},
        # TechParts: SOP should be detected and reply should be email-style
        "email_02": {"phrases": ["applied your sop (techparts inc)"]},
        # QuickShip UK: 8% margin override -> 3564 for 1x40ft Ningbo->Felixstowe
        "email_04": {"amounts": [3564], "phrases": ["applied your sop (quickship uk)"]},
        # AutoSpares: 3 containers total => 5% discount across routes; grand total should be 9450
        "email_06": {"amounts": [7429, 2021, 9450], "phrases": ["applied your sop (autospares gmbh)"]},
        # Australia destination: +$150 biosecurity surcharge; main quote becomes 3600
        "email_09": {"amounts": [150, 3600], "phrases": ["biosecurity surcharge"]},
    }

    failures: list[str] = []

    for difficulty in ["easy", "medium", "hard"]:
        rates = normalize_rate_sheets(config=config, difficulty=difficulty)
        for email_id, expectation in checks.items():
            email = load_email(email_index[email_id])
            result = run_quote_pipeline(
                email=email,
                config=config,
                difficulty=difficulty,
                use_openai=False,
                rate_sheets=rates,
                enable_sop=True,
            )
            if result.error or not result.quote_text:
                failures.append(f"{difficulty}/{email_id}: error={result.error!r}")
                continue

            text = result.quote_text
            values = _money_values(text)
            for amt in expectation.get("amounts", []):
                if float(amt) not in values:
                    failures.append(f"{difficulty}/{email_id}: missing amount {amt} (found={sorted(set(values))})")

            for phrase in expectation.get("phrases", []):
                if phrase.casefold() not in text.casefold():
                    failures.append(f"{difficulty}/{email_id}: missing phrase {phrase!r}")

    if failures:
        print("SOP VALIDATION FAILED")
        for f in failures:
            print("-", f)
        return 1

    print("SOP VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

