export interface PatchEncoderConfig {
  patchLength: number;
  patchStride: number;
  hiddenDim: number;
}

export class PatchTemporalEncoder {
  constructor(private readonly cfg: PatchEncoderConfig) {
    if (cfg.patchLength <= 0 || cfg.patchStride <= 0 || cfg.hiddenDim <= 0) {
      throw new Error("Invalid patch encoder config");
    }
  }

  encode(sequence: number[][]): number[] {
    if (sequence.length === 0 || sequence[0].length === 0) return [];
    const featureDim = sequence[0].length;

    const patches: number[][] = [];
    for (
      let start = 0;
      start + this.cfg.patchLength <= sequence.length;
      start += this.cfg.patchStride
    ) {
      const flat: number[] = [];
      for (let t = 0; t < this.cfg.patchLength; t += 1) {
        flat.push(...sequence[start + t].slice(0, featureDim));
      }
      patches.push(flat);
    }
    if (patches.length === 0) return [];

    const projected = patches.map((patch) => {
      const h = new Array<number>(this.cfg.hiddenDim).fill(0);
      for (let i = 0; i < patch.length; i += 1) {
        for (let d = 0; d < this.cfg.hiddenDim; d += 1) {
          h[d] += patch[i] * Math.sin((i + 1) * (d + 3) * 0.001);
        }
      }
      return h;
    });

    const scores = projected.map((v) =>
      v.reduce((acc, x) => acc + x * (1 / this.cfg.hiddenDim), 0)
    );
    const maxScore = Math.max(...scores);
    const exp = scores.map((s) => Math.exp(s - maxScore));
    const denom = exp.reduce((a, b) => a + b, 0);
    const alpha = denom > 1e-12 ? exp.map((x) => x / denom) : exp.map(() => 1 / exp.length);

    const out = new Array<number>(this.cfg.hiddenDim).fill(0);
    for (let i = 0; i < projected.length; i += 1) {
      for (let d = 0; d < this.cfg.hiddenDim; d += 1) out[d] += alpha[i] * projected[i][d];
    }
    return out;
  }
}

