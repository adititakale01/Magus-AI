from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import uuid

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except Exception:  # noqa: BLE001
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    Jsonb = None  # type: ignore[assignment]

try:
    from supabase import Client as SupabaseClient
    from supabase import create_client as create_supabase_client
except Exception:  # noqa: BLE001
    SupabaseClient = None  # type: ignore[assignment]
    create_supabase_client = None  # type: ignore[assignment]


EmailRecordType = Literal["auto", "human"]
EmailRecordStatus = Literal["unprocessed", "auto_processed", "needs_human_decision", "human_confirmed_replied", "human_rejected"]

ALL_EMAIL_RECORD_STATUSES: tuple[EmailRecordStatus, ...] = (
    "unprocessed",
    "auto_processed",
    "needs_human_decision",
    "human_confirmed_replied",
    "human_rejected",
)


class DbNotConfiguredError(RuntimeError):
    pass


def _is_local_host(host: str | None) -> bool:
    if not host:
        return True
    host = host.strip().casefold()
    return host in {"localhost", "127.0.0.1", "::1"}


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url or not str(url).strip():
        raise DbNotConfiguredError("DATABASE_URL is not set.")
    return _normalize_database_url(str(url).strip())


def _normalize_database_url(url: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if "connect_timeout" not in params:
        params["connect_timeout"] = "8"
    if not _is_local_host(parsed.hostname) and "sslmode" not in params:
        params["sslmode"] = "require"

    query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=query))


def connect():
    if psycopg is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")
    return psycopg.connect(get_database_url())


def _get_supabase_config() -> tuple[str, str] | None:
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    if not url or not str(url).strip():
        return None
    if not key or not str(key).strip():
        return None
    return (str(url).strip(), str(key).strip())


_SUPABASE_CLIENT: SupabaseClient | None = None
_SUPABASE_CLIENT_CFG: tuple[str, str] | None = None


def supabase_client() -> SupabaseClient:
    cfg = _get_supabase_config()
    if cfg is None:
        raise DbNotConfiguredError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (recommended).")
    if create_supabase_client is None or SupabaseClient is None:
        raise DbNotConfiguredError("supabase is not installed. Install demo_web_gui requirements to enable Supabase REST DB features.")

    global _SUPABASE_CLIENT, _SUPABASE_CLIENT_CFG
    if _SUPABASE_CLIENT is None or _SUPABASE_CLIENT_CFG != cfg:
        url, key = cfg
        _SUPABASE_CLIENT = create_supabase_client(url, key)
        _SUPABASE_CLIENT_CFG = cfg
    return _SUPABASE_CLIENT


def _use_supabase() -> bool:
    return _get_supabase_config() is not None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:  # noqa: BLE001
            pass

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    return str(value)


_SCHEMA_STATEMENTS: list[str] = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
    """
    CREATE TABLE IF NOT EXISTS public.email_quote_records (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),

        email_id text NULL,
        email_from text NULL,
        email_to text NULL,
        subject text NULL,
        body text NULL,

        reply text NULL,
        trace jsonb NULL,

        type text NOT NULL DEFAULT 'auto',
        status text NOT NULL DEFAULT 'unprocessed',
        config jsonb NULL
    );
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'email_quote_records_type_chk'
        ) THEN
            ALTER TABLE public.email_quote_records
                ADD CONSTRAINT email_quote_records_type_chk
                CHECK (type IN ('auto', 'human'));
        END IF;
    END $$;
    """,
    """
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'email_quote_records_status_chk') THEN
            ALTER TABLE public.email_quote_records DROP CONSTRAINT email_quote_records_status_chk;
        END IF;

        ALTER TABLE public.email_quote_records
            ADD CONSTRAINT email_quote_records_status_chk
            CHECK (status IN ('unprocessed', 'auto_processed', 'needs_human_decision', 'human_confirmed_replied', 'human_rejected'));
    END $$;
    """,
    "CREATE INDEX IF NOT EXISTS email_quote_records_created_at_idx ON public.email_quote_records (created_at DESC);",
    "CREATE INDEX IF NOT EXISTS email_quote_records_status_idx ON public.email_quote_records (status);",
    "CREATE INDEX IF NOT EXISTS email_quote_records_type_idx ON public.email_quote_records (type);",
    "CREATE INDEX IF NOT EXISTS email_quote_records_email_id_idx ON public.email_quote_records (email_id);",
    """
    CREATE OR REPLACE FUNCTION public.set_updated_at()
    RETURNS trigger AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_email_quote_records_set_updated_at') THEN
            CREATE TRIGGER trg_email_quote_records_set_updated_at
            BEFORE UPDATE ON public.email_quote_records
            FOR EACH ROW
            EXECUTE FUNCTION public.set_updated_at();
        END IF;
    END $$;
    """,
]


def init_schema() -> dict[str, Any]:
    if _use_supabase():
        client = supabase_client()
        try:
            client.table("email_quote_records").select("id").limit(1).execute()
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST schema check failed: {e.__class__.__name__}: {e}") from e
        return {"ok": True, "mode": "supabase", "note": "Schema init via REST is not supported; verified table is reachable."}

    with connect() as conn:
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)
    return {"ok": True, "mode": "psycopg"}


def healthcheck() -> dict[str, Any]:
    if _use_supabase():
        client = supabase_client()
        try:
            resp = client.table("email_quote_records").select("created_at").order("created_at", desc=True).limit(1).execute()
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST healthcheck failed: {e.__class__.__name__}: {e}") from e

        data = getattr(resp, "data", None)
        newest = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            newest = data[0].get("created_at")
        return {"ok": True, "mode": "supabase", "newest_created_at": newest}

    with connect() as conn:
        row = conn.execute("SELECT now() AS now").fetchone()
    return {"ok": True, "mode": "psycopg", "now": row[0] if row else None}


def insert_email_record(
    *,
    email_id: str | None,
    email_from: str | None,
    email_to: str | None,
    subject: str | None,
    body: str | None,
    reply: str | None,
    trace: dict[str, Any] | None,
    record_type: EmailRecordType = "auto",
    status: EmailRecordStatus = "needs_human_decision",
    config: dict[str, Any] | None,
) -> uuid.UUID:
    record_id = uuid.uuid4()
    if _use_supabase():
        client = supabase_client()
        try:
            client.table("email_quote_records").insert(
                {
                    "id": str(record_id),
                    "email_id": email_id,
                    "email_from": email_from,
                    "email_to": email_to,
                    "subject": subject,
                    "body": body,
                    "reply": reply,
                    "trace": _jsonable(trace) if trace is not None else None,
                    "type": str(record_type),
                    "status": str(status),
                    "config": _jsonable(config) if config is not None else None,
                }
            ).execute()
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST insert failed: {e.__class__.__name__}: {e}") from e
        return record_id

    if Jsonb is None or dict_row is None:
        raise DbNotConfiguredError("psycopg JSON helpers are unavailable. Install demo_web_gui requirements to enable DB features.")

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO public.email_quote_records (
                    id, email_id, email_from, email_to, subject, body,
                    reply, trace,
                    type, status, config
                ) VALUES (
                    %(id)s, %(email_id)s, %(email_from)s, %(email_to)s, %(subject)s, %(body)s,
                    %(reply)s, %(trace)s,
                    %(type)s, %(status)s, %(config)s
                );
                """,
                {
                    "id": record_id,
                    "email_id": email_id,
                    "email_from": email_from,
                    "email_to": email_to,
                    "subject": subject,
                    "body": body,
                    "reply": reply,
                    "trace": Jsonb(_jsonable(trace)) if trace is not None else None,
                    "type": str(record_type),
                    "status": str(status),
                    "config": Jsonb(_jsonable(config)) if config is not None else None,
                },
            )
    return record_id


_EMAIL_RECORD_SELECT_COLUMNS = """
    id,
    created_at,
    updated_at,
    email_id,
    email_from,
    email_to,
    subject,
    body,
    reply,
    trace,
    type,
    status,
    config
"""


def _coerce_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


def _fetch_email_record(*, cur, record_id: uuid.UUID) -> dict[str, Any] | None:
    cur.execute(
        f"""
        SELECT {_EMAIL_RECORD_SELECT_COLUMNS}
        FROM public.email_quote_records
        WHERE id = %(id)s
        """,
        {"id": record_id},
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_email_record(*, record_id: str) -> dict[str, Any] | None:
    record_uuid = _coerce_uuid(record_id)
    if record_uuid is None:
        return None

    if _use_supabase():
        client = supabase_client()
        try:
            resp = client.table("email_quote_records").select("*").eq("id", str(record_uuid)).limit(1).execute()
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST get failed: {e.__class__.__name__}: {e}") from e
        data = getattr(resp, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return dict(data[0])
        return None

    if dict_row is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            return _fetch_email_record(cur=cur, record_id=record_uuid)


def list_email_records(*, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    if _use_supabase():
        client = supabase_client()
        start = max(int(offset), 0)
        limit_i = max(int(limit), 0)
        if limit_i == 0:
            return []
        end = start + limit_i - 1
        try:
            resp = (
                client.table("email_quote_records")
                .select("*")
                .order("created_at", desc=True)
                .range(start, end)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST list failed: {e.__class__.__name__}: {e}") from e
        data = getattr(resp, "data", None)
        if isinstance(data, list):
            return [dict(r) for r in data if isinstance(r, dict)]
        return []

    if dict_row is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    created_at,
                    updated_at,
                    email_id,
                    email_from,
                    email_to,
                    subject,
                    body,
                    reply,
                    trace,
                    type,
                    status,
                    config
                FROM public.email_quote_records
                ORDER BY created_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                {"limit": int(limit), "offset": int(offset)},
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def count_email_records_by_status() -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in ALL_EMAIL_RECORD_STATUSES}
    if _use_supabase():
        client = supabase_client()
        for status in ALL_EMAIL_RECORD_STATUSES:
            try:
                resp = client.table("email_quote_records").select("id", count="exact").eq("status", status).range(0, 0).execute()
            except Exception as e:  # noqa: BLE001
                raise DbNotConfiguredError(f"Supabase REST count failed: {e.__class__.__name__}: {e}") from e
            count = getattr(resp, "count", None)
            if isinstance(count, int):
                counts[status] = int(count)

        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        return counts

    if dict_row is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT status, COUNT(*)::int AS count
                FROM public.email_quote_records
                GROUP BY status
                """
            )
            for row in cur.fetchall():
                status = str(row.get("status") or "")
                if status in counts:
                    counts[status] = int(row.get("count") or 0)

    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def list_needs_human_decision(*, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    if _use_supabase():
        client = supabase_client()
        start = max(int(offset), 0)
        limit_i = max(int(limit), 0)
        if limit_i == 0:
            return []
        end = start + limit_i - 1
        try:
            resp = (
                client.table("email_quote_records")
                .select("*")
                .eq("status", "needs_human_decision")
                .order("created_at", desc=True)
                .range(start, end)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST list failed: {e.__class__.__name__}: {e}") from e
        data = getattr(resp, "data", None)
        if isinstance(data, list):
            return [dict(r) for r in data if isinstance(r, dict)]
        return []

    if dict_row is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    created_at,
                    updated_at,
                    email_id,
                    email_from,
                    email_to,
                    subject,
                    body,
                    reply,
                    trace,
                    type,
                    status,
                    config
                FROM public.email_quote_records
                WHERE status = 'needs_human_decision'
                ORDER BY created_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                {"limit": int(limit), "offset": int(offset)},
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def update_email_record(
    *,
    record_id: str,
    status: EmailRecordStatus | None = None,
    record_type: EmailRecordType | None = None,
    reply: str | None = None,
    trace: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    record_uuid = _coerce_uuid(record_id)
    if record_uuid is None:
        return None

    if _use_supabase():
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = str(status)
        if record_type is not None:
            updates["type"] = str(record_type)
        if reply is not None:
            updates["reply"] = str(reply)
        if trace is not None:
            updates["trace"] = _jsonable(trace)
        if config is not None:
            updates["config"] = _jsonable(config)

        if not updates:
            return get_email_record(record_id=str(record_uuid))

        client = supabase_client()
        try:
            resp = client.table("email_quote_records").update(updates).eq("id", str(record_uuid)).execute()
        except Exception as e:  # noqa: BLE001
            raise DbNotConfiguredError(f"Supabase REST update failed: {e.__class__.__name__}: {e}") from e

        data = getattr(resp, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return dict(data[0])
        return get_email_record(record_id=str(record_uuid))

    if dict_row is None:
        raise DbNotConfiguredError("psycopg is not installed. Install demo_web_gui requirements to enable DB features.")
    if Jsonb is None:
        raise DbNotConfiguredError("psycopg JSON helpers are unavailable. Install demo_web_gui requirements to enable DB features.")

    sets: list[str] = []
    params: dict[str, Any] = {"id": record_uuid}

    if status is not None:
        sets.append("status = %(status)s")
        params["status"] = str(status)
    if record_type is not None:
        sets.append("type = %(type)s")
        params["type"] = str(record_type)
    if reply is not None:
        sets.append("reply = %(reply)s")
        params["reply"] = str(reply)
    if trace is not None:
        sets.append("trace = %(trace)s")
        params["trace"] = Jsonb(_jsonable(trace))
    if config is not None:
        sets.append("config = %(config)s")
        params["config"] = Jsonb(_jsonable(config))

    if not sets:
        return get_email_record(record_id=str(record_uuid))

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                UPDATE public.email_quote_records
                SET {", ".join(sets)}
                WHERE id = %(id)s
                """,
                params,
            )
            if cur.rowcount == 0:
                return None
            return _fetch_email_record(cur=cur, record_id=record_uuid)
