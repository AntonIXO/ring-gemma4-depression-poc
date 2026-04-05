import type { SearchResult } from "../types.js";
import { cosineSimilarity } from "../utils/stats.js";

interface Entry {
  id: string;
  embedding: number[];
  metadata: string;
}

export class InMemoryVectorDB {
  private readonly entries: Entry[] = [];

  upsert(id: string, embedding: number[], metadata: string): void {
    const idx = this.entries.findIndex((x) => x.id === id);
    if (idx === -1) this.entries.push({ id, embedding: [...embedding], metadata });
    else this.entries[idx] = { id, embedding: [...embedding], metadata };
  }

  semanticStateSearch(query: number[], topK: number): SearchResult[] {
    return this.entries
      .map((e) => ({
        id: e.id,
        cosineSimilarity: cosineSimilarity(query, e.embedding),
        metadata: e.metadata
      }))
      .sort((a, b) => b.cosineSimilarity - a.cosineSimilarity)
      .slice(0, topK);
  }
}

