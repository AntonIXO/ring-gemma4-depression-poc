import { InMemoryVectorDB } from "../vector-db/vector-db.js";

async function main() {
  const db = new InMemoryVectorDB();
  db.upsert("state_01", [0.1, 0.2, 0.3], "high_recovery");
  db.upsert("state_02", [0.2, 0.1, 0.0], "strain_risk");
  const results = db.semanticStateSearch([0.1, 0.2, 0.29], 3);
  for (const r of results) {
    console.log(`${r.id}\t${r.cosineSimilarity.toFixed(6)}\t${r.metadata}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

