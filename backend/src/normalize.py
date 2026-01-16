from __future__ import annotations

from dataclasses import dataclass
import difflib
import re

from openai import OpenAI

from src.config import AppConfig


_IATA_RE = re.compile(r"\(([A-Z]{3})\)")


@dataclass(frozen=True)
class LocationResolution:
    raw: str
    canonical: str | None
    extracted_code: str | None
    method: str
    score: float | None = None


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def extract_iata_code(raw: str) -> tuple[str, str | None]:
    raw = raw.strip()
    m = _IATA_RE.search(raw)
    if m:
        code = m.group(1)
        stripped = _IATA_RE.sub("", raw).strip()
        stripped = re.sub(r"\s+", " ", stripped).strip(" -")
        return stripped, code

    tail = re.search(r"(?<![a-z])([A-Z]{3})(?![a-z])\s*$", raw)
    if tail and len(raw.split()) >= 2:
        code = tail.group(1)
        stripped = raw[: tail.start(1)].strip().strip(" -")
        return stripped, code

    return raw, None


def strip_location_noise(raw: str) -> str:
    s = raw.strip()

    if "," in s:
        s = s.split(",", 1)[0].strip()

    tokens = s.split()
    if len(tokens) >= 2 and len(tokens[-1]) == 2 and tokens[-1].isalpha() and tokens[-1].isupper():
        if tokens[-1] not in {"LA"}:
            s = " ".join(tokens[:-1]).strip()

    s = re.sub(r"\barea\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_location(
    *,
    raw: str | None,
    known_locations: list[str],
    aliases: dict[str, list[str]],
    use_openai: bool,
    config: AppConfig,
) -> LocationResolution:
    if not raw or not raw.strip():
        return LocationResolution(raw=str(raw or ""), canonical=None, extracted_code=None, method="missing")

    name, extracted_code = extract_iata_code(raw)
    name = strip_location_noise(name)
    cleaned = clean_text(name)

    if cleaned in {"china", "poland", "europe", "asia"}:
        return LocationResolution(raw=raw, canonical=None, extracted_code=extracted_code, method="too_broad")

    by_clean = {clean_text(loc): loc for loc in known_locations}
    if cleaned in by_clean:
        return LocationResolution(raw=raw, canonical=by_clean[cleaned], extracted_code=extracted_code, method="exact", score=1.0)

    for canonical, alias_list in aliases.items():
        for alias in [canonical, *alias_list]:
            alias_clean = clean_text(alias)
            if not alias_clean:
                continue
            if alias_clean in cleaned or cleaned in alias_clean:
                if canonical in known_locations:
                    return LocationResolution(
                        raw=raw,
                        canonical=canonical,
                        extracted_code=extracted_code,
                        method="alias",
                        score=0.95,
                    )

    best_loc = None
    best_score = 0.0
    for loc in known_locations:
        score = difflib.SequenceMatcher(None, cleaned, clean_text(loc)).ratio()
        if score > best_score:
            best_loc = loc
            best_score = score
    if best_loc and best_score >= 0.80:
        return LocationResolution(
            raw=raw,
            canonical=best_loc,
            extracted_code=extracted_code,
            method="fuzzy",
            score=best_score,
        )

    if use_openai and config.openai.api_key and known_locations:
        try:
            canonical = _llm_pick_location(raw=raw, choices=known_locations, config=config)
            if canonical in known_locations:
                return LocationResolution(
                    raw=raw,
                    canonical=canonical,
                    extracted_code=extracted_code,
                    method="llm",
                    score=None,
                )
        except Exception:
            pass

    return LocationResolution(raw=raw, canonical=None, extracted_code=extracted_code, method="unmatched", score=best_score or None)


def _llm_pick_location(*, raw: str, choices: list[str], config: AppConfig) -> str:
    client = OpenAI(api_key=config.openai.api_key)

    system = (
        "You map a raw location string to the best matching canonical location.\n"
        "Return ONLY JSON: {\"canonical\": \"...\"}\n"
        "If none matches, return {\"canonical\": null}."
    )
    user = f"RAW:\n{raw}\n\nCHOICES:\n" + "\n".join(f"- {c}" for c in choices)

    kwargs = dict(
        model=config.openai.model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    try:
        resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except TypeError:
        resp = client.chat.completions.create(**kwargs)

    content = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return ""
    data = json_loads_safe(m.group(0))
    canonical = data.get("canonical")
    return str(canonical) if canonical else ""


def json_loads_safe(text: str) -> dict:
    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


@dataclass(frozen=True)
class ContainerRecommendation:
    recommended_size_ft: int
    recommended_quantity: int
    alternative: dict | None = None


def recommend_containers(volume_cbm: float) -> ContainerRecommendation:
    if volume_cbm <= 33:
        return ContainerRecommendation(recommended_size_ft=20, recommended_quantity=1)
    if 33 < volume_cbm <= 67:
        return ContainerRecommendation(
            recommended_size_ft=40,
            recommended_quantity=1,
            alternative={"size_ft": 20, "quantity": 2},
        )
    qty = int((volume_cbm + 66.9999) // 67)  # ceil-ish without importing math
    return ContainerRecommendation(recommended_size_ft=40, recommended_quantity=max(1, qty))
