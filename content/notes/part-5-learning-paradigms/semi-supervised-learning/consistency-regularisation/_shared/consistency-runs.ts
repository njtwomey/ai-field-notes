import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { float64Data, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { adamRule } from 'aifn-compute/optim/first-order'
import { blobs, moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { rowArgmax } from 'aifn-methods/learning/semi-supervised'
import {
  consistencyTraining,
  predictProbabilities,
  type ConsistencyMethod,
  type ConsistencyProblem,
  type ConsistencyState,
} from 'aifn-methods/neural/semi-supervised'

/** A few-label problem with its points as plain arrays for plotting. */
export type Problem = ConsistencyProblem & {
  truth: Int32Array
  /** Labelled points, as [x1, x2] rows. */
  labelled: [number, number][]
  /** Unlabelled points, as [x1, x2] rows. */
  unlabelled: [number, number][]
}

const rowsOf = (t: Tensor): [number, number][] => {
  const v = float64Data(t)
  return Array.from({ length: v.length / 2 }, (_, i) => [v[2 * i], v[2 * i + 1]])
}

function asProblem(x: Tensor, y: ArrayLike<number>, perClass: number, classes: number, seed: number): Problem {
  const sp = semiSupervisedSplit(stream(`consistency/split/${seed}`), y, perClass, { x, draw: 'global', strict: true })
  return {
    labelledX: sp.labelledX,
    labelledY: sp.labelledY,
    unlabelledX: sp.unlabelledX,
    classes,
    truth: sp.unlabelledTruth,
    labelled: rowsOf(sp.labelledX),
    unlabelled: rowsOf(sp.unlabelledX),
  }
}

/** Two moons: 200 points with noise 0.1, 4 labelled points per class, the rest unlabelled. */
export function moonsProblem(seed: number): Problem {
  const d = moons(stream(`consistency/moons/${seed}`), { n: 200, noise: 0.1 })
  return asProblem(d.x, toFlat(d.y) as number[], 4, 2, seed)
}

/** Three Gaussian blobs (sd 0.8, centres on a triangle of side 3), 2 labelled points per class. */
export function blobsProblem(sizes: readonly [number, number, number], seed: number): Problem {
  const d = blobs(stream(`consistency/blobs/${seed}`), {
    n: [...sizes],
    centers: [
      [0, 0],
      [3, 0],
      [1.5, 2.6],
    ],
    sd: 0.8,
  })
  return asProblem(d.x, toFlat(d.y) as number[], 2, 3, seed)
}

/** The classifier of every figure in the chapter: two hidden layers of 32 tanh units. */
export function classifier(classes: number): Layer<Params[]> {
  return Mlp([2, 32, 32, classes], { activation: 'tanh', init: xavierUniform() })
}

/** An evaluation grid over a box, with its points as a tensor for one forward pass. */
export type Grid = { xs: number[]; ys: number[]; points: Tensor }

export function grid(x: readonly [number, number], y: readonly [number, number], step: number): Grid {
  const count = (r: readonly [number, number]) => Math.round((r[1] - r[0]) / step) + 1
  const xs = Array.from({ length: count(x) }, (_, j) => x[0] + j * step)
  const ys = Array.from({ length: count(y) }, (_, i) => y[0] + i * step)
  const data = new Float64Array(xs.length * ys.length * 2)
  ys.forEach((b, i) =>
    xs.forEach((a, j) => {
      data[2 * (i * xs.length + j)] = a
      data[2 * (i * xs.length + j) + 1] = b
    }),
  )
  return { xs, ys, points: fromData(data, [xs.length * ys.length, 2]) }
}

/** The box and grid of the moons figures. */
export const MOONS_BOX = { x: [-1.5, 2.5] as const, y: [-1, 1.5] as const }
export const MOONS_GRID = grid(MOONS_BOX.x, MOONS_BOX.y, 0.08)
/** The box and grid of the blobs figures. */
export const BLOBS_BOX = { x: [-2.5, 5.5] as const, y: [-2.5, 5] as const }
export const BLOBS_GRID = grid(BLOBS_BOX.x, BLOBS_BOX.y, 0.15)

/**
 * The field a Raster draws: for two classes, P(class 1) − ½ (a diverging field whose zero contour is the boundary);
 * for more, the most probable class (a categorical field).
 */
export function field(net: Layer<Params[]>, params: Params[], g: Grid, classes: number): number[][] {
  const p = predictProbabilities(net, params, g.points)
  const W = g.xs.length
  if (classes === 2) return g.ys.map((_, i) => g.xs.map((_, j) => p[2 * (i * W + j) + 1] - 0.5))
  const { cls } = rowArgmax(p, classes)
  return g.ys.map((_, i) => g.xs.map((_, j) => cls[i * W + j]))
}

/** Accuracy of the network's arg max on the unlabelled points against their true classes. */
export function accuracy(net: Layer<Params[]>, params: Params[], problem: Problem): number {
  const { cls } = rowArgmax(predictProbabilities(net, params, problem.unlabelledX), problem.classes)
  let right = 0
  for (let i = 0; i < cls.length; i++) if (cls[i] === problem.truth[i]) right++
  return right / cls.length
}

/** Per-class accuracy on the unlabelled points. */
export function classAccuracy(net: Layer<Params[]>, params: Params[], problem: Problem): number[] {
  const { cls } = rowArgmax(predictProbabilities(net, params, problem.unlabelledX), problem.classes)
  return Array.from({ length: problem.classes }, (_, c) => {
    let n = 0
    let right = 0
    for (let i = 0; i < cls.length; i++)
      if (problem.truth[i] === c) {
        n++
        if (cls[i] === c) right++
      }
    return n ? right / n : NaN
  })
}

/** What a figure's training run needs. */
export type RunSetup = {
  problem: Problem
  method: ConsistencyMethod
  steps: number
  rate: number
  weight: number
  rampUp?: number
  weak: number
  strong: number
  batch?: number
  seed: number
  /** Steps between kept checkpoints. */
  every: number
}

/** A kept checkpoint: the step, its training state, and the accuracies of the student (and teacher). */
export type Checkpoint = {
  step: number
  state: ConsistencyState
  accuracy: number
  teacherAccuracy: number
}

/** Trains with `consistencyTraining` and yields a checkpoint every `every` steps, from step 0. */
export function* consistencyRun(s: RunSetup): Generator<Checkpoint> {
  const net = classifier(s.problem.classes)
  const alg = consistencyTraining(s.problem, {
    model: net,
    method: s.method,
    weight: s.weight,
    rampUp: s.rampUp ?? 0,
    steps: s.steps,
    weak: s.weak,
    strong: s.strong,
    batch: s.batch,
    optimizer: adamRule({ stepSize: s.rate }),
  })
  const it = live(alg, { params: net.init(stream(`consistency/init/${s.seed}`)) })
  for (let k = 0; k <= s.steps; k++) {
    const next = it.next()
    if (next.done) return
    const state = next.value.state as ConsistencyState
    if (k % s.every === 0 || k === s.steps || state.diverged) {
      yield {
        step: k,
        state,
        accuracy: accuracy(net, state.params, s.problem),
        teacherAccuracy: state.teacher ? accuracy(net, state.teacher, s.problem) : NaN,
      }
    }
    if (state.diverged) return
  }
}

/** Points split by a per-point class (−1 drops a point), for one Points layer per class. */
export function byClass(points: readonly [number, number][], cls: ArrayLike<number>, c: number) {
  const x: number[] = []
  const y: number[] = []
  points.forEach((p, i) => {
    if (cls[i] === c) {
      x.push(p[0])
      y.push(p[1])
    }
  })
  return { x, y }
}

/** A circle of radius r around a centre, as a closed polyline. */
export function circle(centre: readonly [number, number], r: number, n = 48) {
  const x = Array.from({ length: n + 1 }, (_, k) => centre[0] + r * Math.cos((2 * Math.PI * k) / n))
  const y = Array.from({ length: n + 1 }, (_, k) => centre[1] + r * Math.sin((2 * Math.PI * k) / n))
  return { x, y }
}

export const percent = (v: number | undefined) =>
  v === undefined || !Number.isFinite(v) ? '…' : `${(100 * v).toFixed(1)}%`
