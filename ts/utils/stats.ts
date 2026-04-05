export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((acc, x) => acc + x, 0) / values.length;
}

export function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance =
    values.reduce((acc, x) => {
      const d = x - m;
      return acc + d * d;
    }, 0) /
    (values.length - 1);
  return Math.sqrt(variance);
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * p;
  const low = Math.floor(idx);
  const high = Math.ceil(idx);
  const w = idx - low;
  return sorted[low] * (1 - w) + sorted[high] * w;
}

export function l2Norm(values: number[]): number {
  return Math.sqrt(values.reduce((acc, x) => acc + x * x, 0));
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return -1;
  const na = l2Norm(a);
  const nb = l2Norm(b);
  if (na === 0 || nb === 0) return -1;
  let dot = 0;
  for (let i = 0; i < a.length; i += 1) dot += a[i] * b[i];
  return dot / (na * nb);
}

