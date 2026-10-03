/**
 * Average precision for detection from detections already sorted by decreasing confidence, each marked true or false
 * positive, and the number of ground-truth objects.
 */
export type PrCurve = { precision: number[]; recall: number[] }

export function prCurve(isTp: boolean[], groundTruth: number): PrCurve {
  let tp = 0
  let fp = 0
  const precision: number[] = []
  const recall: number[] = []
  for (const hit of isTp) {
    if (hit) tp += 1
    else fp += 1
    precision.push(tp / (tp + fp))
    recall.push(groundTruth ? tp / groundTruth : 0)
  }
  return { precision, recall }
}

/** Interpolated precision at recall r: the highest precision at any recall ≥ r (0 if none). */
export const interpolated = (c: PrCurve, r: number) =>
  Math.max(0, ...c.precision.filter((_, i) => c.recall[i] >= r - 1e-12))

/** All-point AP (PASCAL VOC 2010 onwards): the area under the interpolated, monotone precision envelope. */
export function apAllPoint(c: PrCurve): number {
  let ap = 0
  let prev = 0
  const levels = [...new Set(c.recall)].sort((a, b) => a - b)
  for (const r of levels) {
    ap += (r - prev) * interpolated(c, r)
    prev = r
  }
  return ap
}

/** AP as the mean interpolated precision at n evenly spaced recall levels: 11 (VOC 2007) or 101 (COCO). */
export function apSampled(c: PrCurve, n: number): number {
  let sum = 0
  for (let i = 0; i < n; i++) sum += interpolated(c, i / (n - 1))
  return sum / n
}
