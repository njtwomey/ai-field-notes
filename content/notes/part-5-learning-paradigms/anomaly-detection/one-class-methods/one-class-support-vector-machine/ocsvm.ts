import { oneClassSvm, oneClassScore, type OneClassModel } from 'aifn-methods/unsupervised/anomaly'
import { fromData } from 'aifn-compute/foundation/tensor'

export type Point = [number, number]

export const rbf = (gamma: number) => (a: Point, b: Point) =>
  Math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

export type OcSvmFit = {
  /** Dual variables, one per training point, in [0, C] and summing to 1. */
  alpha: number[]
  /** The upper bound C = 1/(νn). */
  C: number
  /** Offset: the decision function is f(x) = Σ αᵢ k(xᵢ, x) − ρ. */
  rho: number
  /** f at each training point. */
  f: number[]
  iterations: number
  _model: OneClassModel
}

/** Train one-class SVM, backed by aifn-methods. */
export function trainOcSvm(x: Point[], nu: number, gamma: number): OcSvmFit {
  const n = x.length
  const flat = Float64Array.from(x.flat())
  const X = fromData(flat, [n, 2])
  const model = oneClassSvm(X, { nu, gamma })
  const C = 1 / (nu * n)
  // Anomaly score is -f(x), so f(x) = -score
  const scores = oneClassScore(model, X)
  const f = Array.from(scores, (s) => -s)
  return {
    alpha: Array.from(model.alpha),
    C,
    rho: model.offset,
    f,
    iterations: 1,
    _model: model,
  }
}

/** Evaluate decision function f(at), backed by aifn-methods. */
export function decision(fit: OcSvmFit, _x: Point[], _gamma: number, at: Point): number {
  const atTensor = fromData(Float64Array.from(at), [1, 2])
  const score = oneClassScore(fit._model, atTensor)[0]
  return -score
}

export type Shape = 'blob' | 'blobs' | 'ring' | 'banana'
type Rng = { uniform: () => number; normal: () => number }

export const N_INLIERS = 110
export const N_OUTLIERS = 6

/** N_INLIERS points of the chosen shape, then N_OUTLIERS points drawn uniformly over [−3.6, 3.6]². */
export function makeData(shape: Shape, g: Rng): Point[] {
  const pts: Point[] = []
  for (let i = 0; i < N_INLIERS; i++) {
    if (shape === 'blob') pts.push([0.8 * g.normal(), 0.8 * g.normal()])
    else if (shape === 'blobs') {
      const c = i % 2 === 0 ? -1.4 : 1.4
      pts.push([c + 0.5 * g.normal(), c + 0.5 * g.normal()])
    } else if (shape === 'ring') {
      const t = 2 * Math.PI * g.uniform()
      const r = 2 + 0.25 * g.normal()
      pts.push([r * Math.cos(t), r * Math.sin(t)])
    } else {
      // A crescent: an arc of the circle of radius 2.2 centred at (0, −1.6), thickest in the middle.
      const t = Math.PI * (0.15 + 0.7 * g.uniform())
      const r = 2.2 + 0.3 * Math.sin(t) * g.normal()
      pts.push([r * Math.cos(t), r * Math.sin(t) - 1.6])
    }
  }
  for (let i = 0; i < N_OUTLIERS; i++) pts.push([-3.6 + 7.2 * g.uniform(), -3.6 + 7.2 * g.uniform()])
  return pts
}

/**
 * The zero level set of a grid by marching squares, as one polyline whose pieces are separated by NaN points, which
 * a line series draws as gaps. z[r][c] is the value at (xs[c], ys[r]).
 */
export function zeroContour(xs: number[], ys: number[], z: number[][]): { x: number[]; y: number[] } {
  const ox: number[] = []
  const oy: number[] = []
  const cross = (x0: number, y0: number, v0: number, x1: number, y1: number, v1: number): Point => {
    const t = v0 / (v0 - v1)
    return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)]
  }
  for (let r = 0; r + 1 < ys.length; r++) {
    for (let c = 0; c + 1 < xs.length; c++) {
      const [xa, xb, ya, yb] = [xs[c], xs[c + 1], ys[r], ys[r + 1]]
      // Corners counter-clockwise from bottom-left, and the edges between consecutive corners.
      const v = [z[r][c], z[r][c + 1], z[r + 1][c + 1], z[r + 1][c]]
      const p: Point[] = [
        [xa, ya],
        [xb, ya],
        [xb, yb],
        [xa, yb],
      ]
      const hits: Point[] = []
      for (let e = 0; e < 4; e++) {
        const a = v[e]
        const b = v[(e + 1) % 4]
        if (a >= 0 !== b >= 0) hits.push(cross(p[e][0], p[e][1], a, p[(e + 1) % 4][0], p[(e + 1) % 4][1], b))
      }
      // Two crossings give one segment; four (a saddle) give two, paired by the order around the cell.
      for (let h = 0; h + 1 < hits.length; h += 2) {
        ox.push(hits[h][0], hits[h + 1][0], NaN)
        oy.push(hits[h][1], hits[h + 1][1], NaN)
      }
    }
  }
  return { x: ox, y: oy }
}
