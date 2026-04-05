import type { DayVector, EpochRecord, WeekVector } from "../types.js";
import { mean, stdDev } from "../utils/stats.js";

export class MomentVectorEncoder {
  encode(current: EpochRecord, previous?: EpochRecord): number[] {
    const out: number[] = [];
    for (let i = 0; i < current.values.length; i += 1) {
      const value = current.values[i];
      const delta = previous ? value - (previous.values[i] ?? value) : 0;
      out.push(value, delta, delta / 15);
    }
    return out;
  }
}

export class DayVectorEncoder {
  encode(participantId: string, dayStartS: number, dayEpochs: EpochRecord[]): DayVector {
    if (dayEpochs.length === 0) return { participantId, dayStartS, values: [] };
    const dims = dayEpochs[0].values.length;
    const values: number[] = [];
    for (let d = 0; d < dims; d += 1) {
      const axis = dayEpochs.map((x) => x.values[d]);
      values.push(mean(axis), stdDev(axis), Math.min(...axis), Math.max(...axis));
    }
    return { participantId, dayStartS, values };
  }
}

export class WeekVectorEncoder {
  encode(participantId: string, windowStartS: number, dayVectors: DayVector[]): WeekVector {
    if (dayVectors.length === 0) return { participantId, windowStartS, values: [] };
    const dims = dayVectors[0].values.length;
    const values: number[] = [];
    for (let d = 0; d < dims; d += 1) {
      const axis = dayVectors.map((x) => x.values[d]);
      const trend = axis.length > 1 ? (axis[axis.length - 1] - axis[0]) / (axis.length - 1) : 0;
      values.push(mean(axis), trend);
    }
    return { participantId, windowStartS, values };
  }
}

