import { AdaptiveInheritedMasking } from "../aim/aim.js";
import { DayVectorEncoder, MomentVectorEncoder, WeekVectorEncoder } from "../embeddings/hierarchy.js";
import { PatchTemporalEncoder } from "../fusion/patch-encoder.js";
import { Gemma4Adapter } from "../gemma/gemma4.js";
import type { EpochRecord } from "../types.js";

async function main() {
  const sequence = Array.from({ length: 96 }, (_, t) => [70 + (t % 9) * 0.4, 42 + (t % 5) * 0.8, 10, 0.15, 36.6]);
  const observed = sequence.map(() => [true, true, true, true, true]);
  for (let t = 20; t < 28; t += 1) observed[t][1] = false;

  const aim = new AdaptiveInheritedMasking({ inheritedDecay: 0.92, nmarRunBoost: 1.35 });
  const aimOut = aim.apply(sequence, observed);

  const patchEncoder = new PatchTemporalEncoder({ patchLength: 8, patchStride: 4, hiddenDim: 64 });
  const fused = patchEncoder.encode(aimOut.imputed);

  const epochs: EpochRecord[] = aimOut.imputed.map((values, i) => ({
    participantId: "demo",
    epochStartS: i * 900,
    values,
    observedMask: [true, true, true, true, true]
  }));

  const momentEncoder = new MomentVectorEncoder();
  const moment = momentEncoder.encode(epochs[1], epochs[0]);
  const dayEncoder = new DayVectorEncoder();
  const day = dayEncoder.encode("demo", 0, epochs);
  const weekEncoder = new WeekVectorEncoder();
  const week = weekEncoder.encode("demo", 0, Array.from({ length: 14 }, () => day));

  const gemma = new Gemma4Adapter({
    rank: 16,
    alpha: 32,
    dropout: 0.05,
    targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  });
  const updated = gemma.loraUpdate(new Array(128).fill(0.01), new Array(128).fill(0.0005));
  const tokens = gemma.substitutePadToken([1, 2, 0, 0, 3], 0, 5);
  const cohortOk = gemma.validateCohort({
    expectedSize: 60,
    participantIds: Array.from({ length: 60 }, (_, i) => `ring_${i}`)
  });

  console.log(`fusion_dim=${fused.length}`);
  console.log(`moment_dim=${moment.length}`);
  console.log(`day_dim=${day.values.length}`);
  console.log(`week_dim=${week.values.length}`);
  console.log(`lora_dim=${updated.length}`);
  console.log(`cohort_ok=${cohortOk}`);
  console.log(`pad_tokens=${tokens.join(",")}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

