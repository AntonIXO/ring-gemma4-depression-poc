-- Everything is a Vector 2.0 core schema
create extension if not exists vector;

create table if not exists public.unified_epochs (
  participant_id text not null,
  epoch_start_s bigint not null,
  values double precision[] not null,
  observed_mask boolean[] not null,
  created_at timestamptz not null default now(),
  primary key (participant_id, epoch_start_s)
);

create table if not exists public.day_vectors (
  participant_id text not null,
  day_start_s bigint not null,
  embedding vector(512) not null,
  created_at timestamptz not null default now(),
  primary key (participant_id, day_start_s)
);

create table if not exists public.week_vectors (
  participant_id text not null,
  window_start_s bigint not null,
  embedding vector(512) not null,
  created_at timestamptz not null default now(),
  primary key (participant_id, window_start_s)
);

create table if not exists public.state_embeddings (
  id bigserial primary key,
  participant_id text not null,
  window_start_s bigint not null,
  vector_kind text not null check (vector_kind in ('moment', 'day', 'week', 'fusion')),
  embedding vector(512) not null,
  metadata text not null default '',
  created_at timestamptz not null default now(),
  unique (participant_id, window_start_s, vector_kind)
);

create index if not exists state_embeddings_kind_idx on public.state_embeddings (vector_kind);
create index if not exists state_embeddings_participant_idx on public.state_embeddings (participant_id);
create index if not exists state_embeddings_embedding_idx
  on public.state_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function public.search_state_embeddings(
  query_embedding vector(512),
  match_count integer default 10
)
returns table (
  id text,
  cosine_similarity double precision,
  metadata text
)
language sql
stable
as $$
  select
    se.id::text as id,
    1 - (se.embedding <=> query_embedding) as cosine_similarity,
    se.metadata
  from public.state_embeddings se
  order by se.embedding <=> query_embedding
  limit greatest(match_count, 1)
$$;

