export type DatasetName = "LifeSnaps" | "GLOBEM" | "TILES" | "PMData" | "AllOfUs";

export type FeatureMap = Record<string, number>;

export interface UnifiedSample {
  participantId: string;
  dataset: DatasetName;
  timestampS: number;
  features: FeatureMap;
}

export interface EpochRecord {
  participantId: string;
  epochStartS: number;
  values: number[];
  observedMask: boolean[];
}

export interface DayVector {
  participantId: string;
  dayStartS: number;
  values: number[];
}

export interface WeekVector {
  participantId: string;
  windowStartS: number;
  values: number[];
}

export interface SearchResult {
  id: string;
  cosineSimilarity: number;
  metadata: string;
}

