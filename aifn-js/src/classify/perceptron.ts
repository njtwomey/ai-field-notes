/**
 * The perceptron (Rosenblatt, 1958): a linear classifier trained one example at a time, updating only on mistakes.
 * Novikoff's (1962) theorem bounds the number of mistakes on separable data by (R/γ)².
 */

import type { Decides, Estimator, FitOptions, Fitted, Scores, Supervised, Trained } from 'aifn/estimators'
import { permutation, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/trace'
import { classLabels, inputs, matrix, values, vec } from './util'

/** The problem a perceptron run solves: inputs [n, d] and labels ±1. */
export interface PerceptronProblem {
  x: Tensor
  /** Labels −1 or +1, [n]. */
  y: Tensor
  learningRate?: number
  intercept?: boolean
  /** Visit the rows in a fresh random order each epoch (needs a stream at `init`); default false (row order). */
  shuffle?: boolean
}

/** One perceptron state: the weights after visiting one example. */
export interface PerceptronState {
  weights: Tensor
  bias: number
  /** The example visited in this step (−1 at the start), its margin y(w·x + b) before the update, and whether it updated. */
  example: number
  margin: number
  updated: boolean
  /** Mistakes so far, and mistakes in the current epoch. */
  mistakes: number
  epochMistakes: number
  epoch: number
  /** Position within the epoch's visiting order. */
  position: number
  /** The visiting order of the current epoch. */
  order: Tensor
  /** A full epoch passed without a mistake. */
  converged: boolean
  /** The stream for the next epoch's order (when shuffling). */
  stream?: Stream
}

/**
 * The perceptron as a traceable algorithm: each step visits one example and, if y(w·x + b) ≤ 0, sets
 * w ← w + η y x and b ← b + η y. It is done after an epoch without mistakes. `init` takes optional starting weights.
 */
export function perceptronSteps(
  problem: PerceptronProblem,
): Algorithm<{ weights?: Tensor; bias?: number }, PerceptronState> {
  const { n, d, v } = matrix(problem.x, 'perceptronSteps')
  const y = values(problem.y)
  const eta = problem.learningRate ?? 1
  const intercept = problem.intercept ?? true
  const orderFor = (epoch: number, s: Stream | undefined): Tensor =>
    problem.shuffle && s
      ? permutation(s.child('epoch', epoch), n)
      : fromData(
          Int32Array.from({ length: n }, (_, i) => i),
          [n],
        )
  return {
    name: 'perceptron',
    init: ({ weights, bias = 0 }, s) => ({
      weights: weights ?? fromData(new Float64Array(d), [d]),
      bias,
      example: -1,
      margin: NaN,
      updated: false,
      mistakes: 0,
      epochMistakes: 0,
      epoch: 0,
      position: 0,
      order: orderFor(0, s),
      converged: false,
      stream: s,
    }),
    step: (state) => {
      const i = state.order.data[state.position]
      const w = Float64Array.from(state.weights.data as Float64Array)
      let f = state.bias
      for (let j = 0; j < d; j++) f += w[j] * v[i * d + j]
      const margin = y[i] * f
      const updated = margin <= 0
      let bias = state.bias
      if (updated) {
        for (let j = 0; j < d; j++) w[j] += eta * y[i] * v[i * d + j]
        if (intercept) bias += eta * y[i]
      }
      const epochMistakes = state.epochMistakes + (updated ? 1 : 0)
      const last = state.position === n - 1
      return {
        weights: fromData(w, [d]),
        bias,
        example: i,
        margin,
        updated,
        mistakes: state.mistakes + (updated ? 1 : 0),
        epochMistakes: last ? 0 : epochMistakes,
        epoch: last ? state.epoch + 1 : state.epoch,
        position: last ? 0 : state.position + 1,
        order: last ? orderFor(state.epoch + 1, state.stream) : state.order,
        converged: last && epochMistakes === 0,
        stream: state.stream,
      }
    },
    done: (state) => state.converged,
  }
}

/** A fitted binary perceptron (labels 0/1). */
export interface PerceptronModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<PerceptronState> {
  readonly kind: 'perceptron'
  readonly weights: Tensor
  readonly bias: number
  readonly mistakes: number
  readonly epochs: number
  readonly converged: boolean
}

/**
 * The perceptron for labels 0/1 (mapped to ∓1): at most `epochs` passes (default 100), stopping after a pass without
 * mistakes. `score` is w·x + b [m]; `decide` is 1 where it is positive. With `shuffle`, `fit` needs a stream.
 */
export function perceptron(
  params: { epochs?: number; learningRate?: number; intercept?: boolean; shuffle?: boolean } = {},
): Estimator<Supervised<Tensor, Tensor>, PerceptronModel> {
  const { epochs = 100, learningRate = 1, intercept = true, shuffle = false } = params
  return {
    name: 'perceptron',
    params: { epochs, learningRate, intercept, shuffle },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'perceptron')
      const { y: labels, k } = classLabels(y, n, 'perceptron')
      if (k > 2) throw new Error('perceptron: binary labels 0/1 only; use a multiclass reduction')
      const signs = vec(Array.from(labels, (c) => (c === 1 ? 1 : -1)))
      const alg = perceptronSteps({ x, y: signs, learningRate, intercept, shuffle })
      const training: Trace<PerceptronState> = trace(alg, {}, epochs * n, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          mistakes: (s) => s.mistakes,
          ...(options.trace?.record as Record<string, (s: PerceptronState, t: number) => number> | undefined),
        },
      })
      const final = training.steps[training.steps.length - 1]
      const w = final.weights.data as Float64Array
      const score = (q: Tensor) => {
        const { n: m, v } = inputs(q, d, 'perceptron')
        const out = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let s = final.bias
          for (let j = 0; j < d; j++) s += w[j] * v[i * d + j]
          out[i] = s
        }
        return fromData(out, [m])
      }
      return {
        kind: 'perceptron',
        weights: final.weights,
        bias: final.bias,
        mistakes: final.mistakes,
        epochs: final.epoch,
        converged: final.converged,
        training,
        forward: score,
        score,
        decide: (q: Tensor) =>
          fromData(
            Int32Array.from(score(q).data as Float64Array, (s) => (s > 0 ? 1 : 0)),
            [q.shape[0]],
          ),
      }
    },
  }
}
