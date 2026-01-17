-- Postgres schema for storing email quote records (single-table design).
-- Designed for Supabase Postgres.

create extension if not exists pgcrypto;

create table if not exists public.email_quote_records (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    email_id text null,
    email_from text null,
    email_to text null,
    subject text null,
    body text null,

    reply text null,
    trace jsonb null,

    type text not null default 'auto',
    status text not null default 'unprocessed',
    config jsonb null,

    origin_city text null,
    destination_city text null,
    price double precision null,
    currency text null,
    transport_type text null,
    has_route boolean null
);

alter table public.email_quote_records
    add column if not exists origin_city text null;
alter table public.email_quote_records
    add column if not exists destination_city text null;
alter table public.email_quote_records
    add column if not exists price double precision null;
alter table public.email_quote_records
    add column if not exists currency text null;
alter table public.email_quote_records
    add column if not exists transport_type text null;
alter table public.email_quote_records
    add column if not exists has_route boolean null;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'email_quote_records_type_chk') then
        alter table public.email_quote_records
            add constraint email_quote_records_type_chk
            check (type in ('auto', 'human'));
    end if;
end $$;

do $$
begin
    if exists (select 1 from pg_constraint where conname = 'email_quote_records_status_chk') then
        alter table public.email_quote_records
            drop constraint email_quote_records_status_chk;
    end if;

    alter table public.email_quote_records
        add constraint email_quote_records_status_chk
        check (status in ('unprocessed', 'auto_processed', 'needs_human_decision', 'human_confirmed_replied', 'human_rejected'));
end $$;

create index if not exists email_quote_records_created_at_idx on public.email_quote_records (created_at desc);
create index if not exists email_quote_records_status_idx on public.email_quote_records (status);
create index if not exists email_quote_records_type_idx on public.email_quote_records (type);
create index if not exists email_quote_records_email_id_idx on public.email_quote_records (email_id);

create or replace function public.set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

do $$
begin
    if not exists (select 1 from pg_trigger where tgname = 'trg_email_quote_records_set_updated_at') then
        create trigger trg_email_quote_records_set_updated_at
        before update on public.email_quote_records
        for each row
        execute function public.set_updated_at();
    end if;
end $$;
