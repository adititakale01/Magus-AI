from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EmailMessage:
    email_id: str
    sender: str
    to: str
    subject: str
    body: str


def list_emails(data_dir: Path) -> dict[str, Path]:
    emails_dir = data_dir / "emails"
    if not emails_dir.exists():
        return {}
    email_files = sorted(emails_dir.glob("email_*.json"))
    index: dict[str, Path] = {}
    for path in email_files:
        index[path.stem] = path
    return index


def load_email(path: Path) -> EmailMessage:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return EmailMessage(
        email_id=path.stem,
        sender=str(payload.get("from", "")),
        to=str(payload.get("to", "")),
        subject=str(payload.get("subject", "")),
        body=str(payload.get("body", "")),
    )


def load_excel_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def get_excel_sheet_names(excel_path: Path) -> list[str]:
    return pd.ExcelFile(excel_path).sheet_names
