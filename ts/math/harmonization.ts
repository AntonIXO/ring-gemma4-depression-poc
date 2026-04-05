import { percentile } from "../utils/stats.js";

export function polynomialMap(x: number, coefficients: number[]): number {
  let acc = 0;
  let xPow = 1;
  for (const c of coefficients) {
    acc += c * xPow;
    xPow *= x;
  }
  return acc;
}

export function mapSdnnToRmssd(sdnnMs: number): number {
  const coeff = [2.731, 0.612, -0.0019, 0.000004];
  return polynomialMap(sdnnMs, coeff);
}

export function mapRmssdToSdnn(rmssdMs: number): number {
  const coeff = [-1.947, 1.381, -0.0047, 0.000008];
  return polynomialMap(rmssdMs, coeff);
}

export function robustZScore14d(current: number, history: number[]): number {
  if (history.length === 0) return 0;
  const med = percentile(history, 0.5);
  const q1 = percentile(history, 0.25);
  const q3 = percentile(history, 0.75);
  const iqr = Math.max(q3 - q1, 1e-6);
  return (current - med) / iqr;
}

export function smoothSleepProbabilities(
  raw: [number, number, number, number],
  inverseConfusion: [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number]
  ]
): [number, number, number, number] {
  const out: number[] = [0, 0, 0, 0];
  for (let i = 0; i < 4; i += 1) {
    let v = 0;
    for (let j = 0; j < 4; j += 1) v += inverseConfusion[i][j] * raw[j];
    out[i] = Math.max(0, v);
  }
  const sum = out.reduce((a, b) => a + b, 0);
  if (sum <= 1e-12) return [0.25, 0.25, 0.25, 0.25];
  return [out[0] / sum, out[1] / sum, out[2] / sum, out[3] / sum];
}

