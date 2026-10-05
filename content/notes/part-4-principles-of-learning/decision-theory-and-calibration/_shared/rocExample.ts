import { rocHull, rocPoints, type Point } from './calibration'
import { normal, stream } from 'aifn-compute/foundation/random'

/**
 * A small scored test set shared by the ROC-analysis figures: 25 negatives with scores N(0, 1) and 25 positives with
 * scores N(1.4, 1), drawn once from a fixed seed. Small enough that the empirical ROC curve has visible concavities,
 * so its convex hull differs from it.
 */
export function rocExample(seed = 11) {
  const g = stream(seed)
  const scores: number[] = []
  const labels: boolean[] = []
  for (let i = 0; i < 25; i++) {
    scores.push(normal(g))
    labels.push(false)
  }
  for (let i = 0; i < 25; i++) {
    scores.push(1.4 + normal(g))
    labels.push(true)
  }
  const curve = rocPoints(scores, labels)
  const hull = rocHull(curve)
  return { scores, labels, curve, hull }
}

/** Expected cost per case at ROC point (fpr, tpr) with prevalence π and error costs c_FP, c_FN. */
export const expectedCost = ([fpr, tpr]: Point, pi: number, cfp: number, cfn: number) =>
  pi * cfn * (1 - tpr) + (1 - pi) * cfp * fpr

/** The segment of the line through `at` with slope m that lies inside the unit square. */
export function lineInSquare(at: Point, m: number): { x: number[]; y: number[] } {
  const [x0, y0] = at
  const ys = (x: number) => y0 + m * (x - x0)
  const xs = (y: number) => x0 + (y - y0) / m
  // Candidate crossings with the four sides, kept when inside the square.
  const pts: Point[] = [
    [0, ys(0)],
    [1, ys(1)],
    [xs(0), 0],
    [xs(1), 1],
  ].filter(([x, y]) => x >= -1e-9 && x <= 1 + 1e-9 && y >= -1e-9 && y <= 1 + 1e-9) as Point[]
  if (pts.length < 2) return { x: [], y: [] }
  pts.sort((a, b) => a[0] - b[0])
  const a = pts[0]
  const b = pts[pts.length - 1]
  return { x: [a[0], b[0]], y: [a[1], b[1]] }
}
