import {
  EPOCH_SECONDS,
  LOOKBACK_DAYS,
  SECONDS_PER_DAY,
  STEPS_PER_DAY,
  UNIFIED_FEATURE_ORDER
} from "../config.js";
import type { DatasetName, EpochRecord, UnifiedSample } from "../types.js";

export interface DatasetSpec {
  name: DatasetName;
  participantColumn: string;
  timestampColumn: string;
  featureColumnToUnified: Record<string, string>;
}

export type CsvRow = Record<string, string>;

function floorToEpoch(ts: number): number {
  return Math.floor(ts / EPOCH_SECONDS) * EPOCH_SECONDS;
}

export class DataFusionPipeline {
  private readonly samples: UnifiedSample[] = [];

  ingestRows(rows: CsvRow[], spec: DatasetSpec): void {
    for (const row of rows) {
      const participantId = row[spec.participantColumn] ?? "";
      const timestampS = Number.parseInt(row[spec.timestampColumn] ?? "", 10);
      if (!participantId || Number.isNaN(timestampS)) continue;

      const features: Record<string, number> = {};
      for (const [source, target] of Object.entries(spec.featureColumnToUnified)) {
        const v = Number.parseFloat(row[source] ?? "");
        if (!Number.isNaN(v)) features[target] = v;
      }
      if (Object.keys(features).length === 0) continue;

      this.samples.push({
        participantId,
        dataset: spec.name,
        timestampS,
        features
      });
    }
  }

  buildEpochLake(): EpochRecord[] {
    const grouped = new Map<string, Map<number, Map<string, { sum: number; n: number }>>>();
    for (const s of this.samples) {
      if (!grouped.has(s.participantId)) grouped.set(s.participantId, new Map());
      const byParticipant = grouped.get(s.participantId)!;
      const epoch = floorToEpoch(s.timestampS);
      if (!byParticipant.has(epoch)) byParticipant.set(epoch, new Map());
      const byEpoch = byParticipant.get(epoch)!;
      for (const [k, v] of Object.entries(s.features)) {
        const stat = byEpoch.get(k) ?? { sum: 0, n: 0 };
        stat.sum += v;
        stat.n += 1;
        byEpoch.set(k, stat);
      }
    }

    const out: EpochRecord[] = [];
    for (const [participantId, byEpoch] of grouped.entries()) {
      const sortedEpochs = [...byEpoch.keys()].sort((a, b) => a - b);
      if (sortedEpochs.length === 0) continue;
      const firstDay = Math.floor(sortedEpochs[0] / SECONDS_PER_DAY) * SECONDS_PER_DAY;
      const lastDay = Math.floor(sortedEpochs[sortedEpochs.length - 1] / SECONDS_PER_DAY) * SECONDS_PER_DAY;

      for (let dayStart = firstDay; dayStart <= lastDay; dayStart += SECONDS_PER_DAY) {
        for (let step = 0; step < STEPS_PER_DAY; step += 1) {
          const epochStartS = dayStart + step * EPOCH_SECONDS;
          const values = new Array<number>(UNIFIED_FEATURE_ORDER.length).fill(0);
          const observedMask = new Array<boolean>(UNIFIED_FEATURE_ORDER.length).fill(false);
          const stats = byEpoch.get(epochStartS);

          if (stats) {
            UNIFIED_FEATURE_ORDER.forEach((name, i) => {
              const agg = stats.get(name);
              if (agg && agg.n > 0) {
                values[i] = agg.sum / agg.n;
                observedMask[i] = true;
              }
            });
          }

          out.push({ participantId, epochStartS, values, observedMask });
        }
      }
    }

    out.sort((a, b) =>
      a.participantId === b.participantId
        ? a.epochStartS - b.epochStartS
        : a.participantId.localeCompare(b.participantId)
    );
    return out;
  }

  buildLookbackWindows(epochs: EpochRecord[]): EpochRecord[][] {
    const byParticipant = new Map<string, EpochRecord[]>();
    for (const e of epochs) {
      if (!byParticipant.has(e.participantId)) byParticipant.set(e.participantId, []);
      byParticipant.get(e.participantId)!.push(e);
    }

    const windows: EpochRecord[][] = [];
    const lookbackSeconds = LOOKBACK_DAYS * SECONDS_PER_DAY;
    for (const series of byParticipant.values()) {
      series.sort((a, b) => a.epochStartS - b.epochStartS);
      for (let i = 0; i < series.length; i += 1) {
        const start = series[i].epochStartS - lookbackSeconds;
        windows.push(series.filter((x) => x.epochStartS >= start && x.epochStartS <= series[i].epochStartS));
      }
    }
    return windows;
  }
}

