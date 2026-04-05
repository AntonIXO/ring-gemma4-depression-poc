export interface AimConfig {
  inheritedDecay: number;
  nmarRunBoost: number;
}

export interface AimOutput {
  imputed: number[][];
  adaptiveWeights: number[][];
}

export class AdaptiveInheritedMasking {
  constructor(private readonly cfg: AimConfig) {
    if (!(cfg.inheritedDecay > 0 && cfg.inheritedDecay <= 1)) {
      throw new Error("inheritedDecay must be in (0,1]");
    }
  }

  apply(signal: number[][], observed: boolean[][]): AimOutput {
    if (signal.length !== observed.length) throw new Error("signal/observed length mismatch");
    if (signal.length === 0) return { imputed: [], adaptiveWeights: [] };

    const tLen = signal.length;
    const dim = signal[0].length;
    const imputed = signal.map((x) => [...x]);
    const adaptiveWeights = Array.from({ length: tLen }, () => new Array<number>(dim).fill(1));
    const lastValue = new Array<number>(dim).fill(0);
    const missingRun = new Array<number>(dim).fill(0);
    const initialized = new Array<boolean>(dim).fill(false);

    for (let t = 0; t < tLen; t += 1) {
      for (let d = 0; d < dim; d += 1) {
        if (observed[t][d]) {
          initialized[d] = true;
          lastValue[d] = signal[t][d];
          missingRun[d] = 0;
          adaptiveWeights[t][d] = 1;
        } else {
          missingRun[d] += 1;
          const decay = this.cfg.inheritedDecay ** missingRun[d];
          const boost = missingRun[d] >= 3 ? this.cfg.nmarRunBoost : 1;
          const w = Math.min(2, boost * decay);
          adaptiveWeights[t][d] = w;
          imputed[t][d] = initialized[d] ? lastValue[d] * w : 0;
        }
      }
    }

    return { imputed, adaptiveWeights };
  }
}

