import { DataFusionPipeline } from "../pipeline/data-fusion.js";
import { UNIFIED_FEATURE_ORDER } from "../config.js";
import { mapSdnnToRmssd, smoothSleepProbabilities } from "../math/harmonization.js";

function sampleRows() {
  return {
    lifeSnaps: [
      { participant_id: "p001", timestamp_s: "1710201600", heart_rate: "71", hrv_sdnn: "48", steps: "12" },
      { participant_id: "p001", timestamp_s: "1710201900", heart_rate: "72", hrv_sdnn: "47", steps: "10" }
    ],
    globem: [
      { participant_id: "p001", timestamp_s: "1710201600", heart_rate: "72", eda: "0.14", skin_temp: "36.5" }
    ],
    tiles: [
      {
        participant_id: "p001",
        timestamp_s: "1710201600",
        hr: "70",
        rmssd: "35",
        sleep_wake: "0.10",
        sleep_light: "0.60",
        sleep_deep: "0.20",
        sleep_rem: "0.10"
      }
    ]
  };
}

async function main() {
  const pipeline = new DataFusionPipeline();
  const rows = sampleRows();
  pipeline.ingestRows(rows.lifeSnaps, {
    name: "LifeSnaps",
    participantColumn: "participant_id",
    timestampColumn: "timestamp_s",
    featureColumnToUnified: { heart_rate: "hr_bpm", hrv_sdnn: "hrv_ms", steps: "steps" }
  });
  pipeline.ingestRows(rows.globem, {
    name: "GLOBEM",
    participantColumn: "participant_id",
    timestampColumn: "timestamp_s",
    featureColumnToUnified: { heart_rate: "hr_bpm", eda: "eda", skin_temp: "temp_c" }
  });
  pipeline.ingestRows(rows.tiles, {
    name: "TILES",
    participantColumn: "participant_id",
    timestampColumn: "timestamp_s",
    featureColumnToUnified: {
      hr: "hr_bpm",
      rmssd: "hrv_ms",
      sleep_wake: "sleep_prob_wake",
      sleep_light: "sleep_prob_light",
      sleep_deep: "sleep_prob_deep",
      sleep_rem: "sleep_prob_rem"
    }
  });

  const epochs = pipeline.buildEpochLake();
  const windows = pipeline.buildLookbackWindows(epochs);
  console.log(`Unified epochs: ${epochs.length} (${UNIFIED_FEATURE_ORDER.length} features)`);
  console.log(`14-day windows: ${windows.length}`);

  const first = epochs.find((x) => x.observedMask[1]);
  if (first) {
    console.log(`HRV SDNN→RMSSD: ${mapSdnnToRmssd(first.values[1]).toFixed(4)}`);
  }

  const sleep = epochs.find((x) => x.values.length >= 9);
  if (sleep) {
    const smoothed = smoothSleepProbabilities(
      [sleep.values[5], sleep.values[6], sleep.values[7], sleep.values[8]],
      [
        [1.05, -0.03, -0.01, -0.01],
        [-0.02, 1.07, -0.03, -0.02],
        [-0.01, -0.04, 1.09, -0.04],
        [-0.02, -0.02, -0.03, 1.07]
      ]
    );
    console.log(`Sleep smoothing: ${smoothed.map((x) => x.toFixed(4)).join(", ")}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

