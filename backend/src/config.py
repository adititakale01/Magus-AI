from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib

from dotenv import load_dotenv


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path


@dataclass(frozen=True)
class RatesConfig:
    easy_file: str
    medium_file: str
    hard_file: str


@dataclass(frozen=True)
class PricingConfig:
    margin: float
    currency: str
    air_volume_factor: float


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None
    model: str
    temperature: float


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    rates: RatesConfig
    pricing: PricingConfig
    openai: OpenAIConfig
    aliases: dict[str, list[str]]
    codes: dict[str, str]


def load_app_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    app_dir = config_path.parent

    env_path = app_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    data_dir = (app_dir / raw["data"]["data_dir"]).resolve()
    pricing = raw.get("pricing", {})
    openai_raw = raw.get("openai", {})

    return AppConfig(
        data=DataConfig(data_dir=data_dir),
        rates=RatesConfig(
            easy_file=raw["rates"]["easy_file"],
            medium_file=raw["rates"]["medium_file"],
            hard_file=raw["rates"]["hard_file"],
        ),
        pricing=PricingConfig(
            margin=float(pricing.get("margin", 0.15)),
            currency=str(pricing.get("currency", "USD")),
            air_volume_factor=float(pricing.get("air_volume_factor", 167.0)),
        ),
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY") or None,
            model=str(openai_raw.get("model", "gpt-4o-mini")),
            temperature=float(openai_raw.get("temperature", 0)),
        ),
        aliases={str(k): [str(x) for x in v] for k, v in raw.get("aliases", {}).items()},
        codes={str(k): str(v) for k, v in raw.get("codes", {}).items()},
    )

