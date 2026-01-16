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


EXPECTED_DOLLARS: dict[str, list[int]] = {
    "email_01": [3680, 7360],
    "email_02": [2329],
    "email_04": [3795],
    "email_05": [3450],
    "email_06": [7820, 2128, 9948],
    "email_07": [2996],
    "email_08": [4370],
    "email_09": [3105, 3450],
    "email_10": [5376],
}


def main() -> int:
    config = load_app_config(APP_DIR / "config.toml")
    email_index = list_emails(config.data.data_dir)

    failures: list[str] = []

    for difficulty in ["easy", "medium", "hard"]:
        rates = normalize_rate_sheets(config=config, difficulty=difficulty)
        for email_id in sorted(email_index.keys()):
            email = load_email(email_index[email_id])
            result = run_quote_pipeline(
                email=email,
                config=config,
                difficulty=difficulty,
                use_openai=False,
                rate_sheets=rates,
            )
            if result.error or not result.quote_text:
                failures.append(f"{difficulty}/{email_id}: error={result.error!r}")
                continue

            if email_id == "email_03":
                required_phrases = ["Which city in China", "Which destination in Poland", "sea freight or air freight"]
                if not all(p.casefold() in result.quote_text.casefold() for p in required_phrases):
                    failures.append(f"{difficulty}/{email_id}: missing expected clarification phrases")
                continue

            expected = EXPECTED_DOLLARS.get(email_id, [])
            found = [int(x.replace(",", "")) for x in re.findall(r"\$([0-9][0-9,]*)", result.quote_text)]
            missing = [amt for amt in expected if amt not in found]
            if missing:
                failures.append(f"{difficulty}/{email_id}: missing amounts {missing} (found={sorted(set(found))})")

    if failures:
        print("VALIDATION FAILED")
        for f in failures:
            print("-", f)
        return 1

    print("VALIDATION OK (10 emails x 3 difficulties)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
