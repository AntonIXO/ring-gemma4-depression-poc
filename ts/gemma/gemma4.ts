export interface LoRAConfig {
  rank: number;
  alpha: number;
  dropout: number;
  targetModules: string[];
}

export interface SmartRingCohort {
  participantIds: string[];
  expectedSize: number;
}

export class Gemma4Adapter {
  constructor(private readonly cfg: LoRAConfig) {
    if (cfg.rank <= 0 || cfg.alpha <= 0 || cfg.dropout < 0 || cfg.dropout >= 1) {
      throw new Error("Invalid LoRA config");
    }
  }

  loraUpdate(baseWeights: number[], gradients: number[]): number[] {
    if (baseWeights.length !== gradients.length) throw new Error("weights/gradients mismatch");
    const scale = this.cfg.alpha / this.cfg.rank;
    return baseWeights.map((w, i) => w - scale * (1 - this.cfg.dropout) * gradients[i]);
  }

  substitutePadToken(tokens: number[], blockedPadToken: number, replacementToken: number): number[] {
    if (blockedPadToken === replacementToken) {
      throw new Error("blocked and replacement token must differ");
    }
    return tokens.map((x) => (x === blockedPadToken ? replacementToken : x));
  }

  validateCohort(cohort: SmartRingCohort): boolean {
    return cohort.participantIds.length === cohort.expectedSize;
  }
}

