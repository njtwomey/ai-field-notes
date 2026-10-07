import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dictionaryLearningSteps, type DictionaryUpdate } from 'aifn-compute/signal/sparse'

/** The angles of the three hidden directions, in radians. */
export const HIDDEN = [0.3, 1.25, 2.4]
/** Points per hidden direction. */
const PER = 30

/**
 * The training signals, 2 × 90: point i lies on hidden direction i mod 3, at a signed distance between 0.25 and 1
 * from the origin, plus Gaussian noise of standard deviation `noise` in each coordinate.
 */
export function directionData(noise: number): number[][] {
  const s = stream('dictionary-learning/directions')
  const n = HIDDEN.length * PER
  const size = toFlat(uniform(child(s, 'size'), 0.25, 1, { shape: [n] }))
  const sign = toFlat(uniform(child(s, 'sign'), 0, 1, { shape: [n] }))
  const e = toFlat(normals(child(s, 'noise'), [2, n]))
  const Y: number[][] = [[], []]
  for (let i = 0; i < n; i++) {
    const a = HIDDEN[i % HIDDEN.length]
    const c = size[i] * (sign[i] < 0.5 ? -1 : 1)
    Y[0].push(c * Math.cos(a) + noise * e[i])
    Y[1].push(c * Math.sin(a) + noise * e[n + i])
  }
  return Y
}

/** One round of the fit: the atoms (columns of a 2 × k matrix), the codes and the objective. */
export type Round = { D: number[][]; X: number[][]; objective: number; replaced: number }

/**
 * Dictionary learning with one atom per signal, traced round by round. The start is `k` training signals drawn at
 * random (`signals`) or `k` random Gaussian directions (`directions`), both from the stream named by `start`.
 */
export function learnDirections(
  Y: number[][],
  k: number,
  update: DictionaryUpdate,
  init: 'signals' | 'directions',
  start: number,
): Round[] {
  const s = stream(start)
  const options = {
    atoms: k,
    sparsity: 1,
    update,
    init: init === 'directions' ? normals(child(s, 'init'), [2, k]) : undefined,
  }
  const run = trace(dictionaryLearningSteps(Y, options), undefined, 40, { stream: s })
  return run.steps.map((state) => ({
    D: toArray(state.D) as number[][],
    X: toArray(state.X) as number[][],
    objective: state.objective,
    replaced: state.replaced,
  }))
}

/** The largest angle, in degrees, between a hidden direction and the learnt atom nearest it (atoms are lines). */
export function worstAngle(D: number[][]): number {
  const k = D[0].length
  return Math.max(
    ...HIDDEN.map((a) => {
      let best = 90
      for (let j = 0; j < k; j++) {
        const cos = Math.abs(Math.cos(a) * D[0][j] + Math.sin(a) * D[1][j]) / Math.hypot(D[0][j], D[1][j])
        best = Math.min(best, (Math.acos(Math.min(1, cos)) * 180) / Math.PI)
      }
      return best
    }),
  )
}
