import type { EpochRecord } from "../types.js";
import { getSupabaseAdminClient } from "./client.js";

export async function persistEpochRecords(records: EpochRecord[]): Promise<void> {
  if (records.length === 0) return;
  const supabase = getSupabaseAdminClient();
  const rows = records.map((r) => ({
    participant_id: r.participantId,
    epoch_start_s: r.epochStartS,
    values: r.values,
    observed_mask: r.observedMask
  }));
  const { error } = await supabase.from("unified_epochs").upsert(rows, {
    onConflict: "participant_id,epoch_start_s"
  });
  if (error) throw error;
}

export async function persistStateEmbedding(params: {
  participantId: string;
  windowStartS: number;
  vectorKind: "moment" | "day" | "week" | "fusion";
  embedding: number[];
  metadata: string;
}): Promise<void> {
  const supabase = getSupabaseAdminClient();
  const { error } = await supabase.from("state_embeddings").upsert({
    participant_id: params.participantId,
    window_start_s: params.windowStartS,
    vector_kind: params.vectorKind,
    embedding: params.embedding,
    metadata: params.metadata
  });
  if (error) throw error;
}

export async function semanticSearch(embedding: number[], topK: number): Promise<
  Array<{
    id: string;
    cosine_similarity: number;
    metadata: string;
  }>
> {
  const supabase = getSupabaseAdminClient();
  const { data, error } = await supabase.rpc("search_state_embeddings", {
    query_embedding: embedding,
    match_count: topK
  });
  if (error) throw error;
  return data ?? [];
}

