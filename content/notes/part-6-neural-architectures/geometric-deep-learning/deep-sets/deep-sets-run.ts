/**
 * The training run behind the Deep Sets figure: the same φ–ρ network trained three times, once per pooling (sum, mean,
 * max), on sets of 1 to 10 numbers, then tested on sets of 1 to 30. The layers, the loss, the optimiser, the pooling
 * and the data are all engine calls.
 */
import { stream } from 'aifn-compute/foundation/random'
import { add, mul, reshape, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import { Mlp } from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { padSets, setPool, type PaddedSets, type SetPooling } from 'aifn-compute/nn/sets'
import { numberSets, type SetTarget } from 'aifn-methods/data/synthetic'

export const POOLINGS: readonly SetPooling[] = ['sum', 'mean', 'max']
/** Training sets have 1 to 10 elements; test sets 1 to 30. */
export const TRAIN_MIN = 1
export const TRAIN_MAX = 10
export const TEST_MAX = 30
export const TRAIN_SETS = 64
export const TEST_PER_SIZE = 20
/** Width of φ's hidden layer and of the pooled vector, and of ρ's hidden layer. */
const WIDTH = 16
/** Optimiser steps between checkpoints. */
const EVERY = 50

export type Setup = { target: SetTarget; steps: number; rate: number; seed: number }
export type Checkpoint = {
  step: number
  /** Training loss on standardised targets, per pooling. */
  loss: Record<SetPooling, number>
  /** Root-mean-square test error at each set size 1..TEST_MAX, per pooling. */
  rmse: Record<SetPooling, number[]>
  /** Test predictions at each set size (TEST_PER_SIZE each), per pooling. */
  predictions: Record<SetPooling, number[][]>
}

const PHI = Mlp([1, WIDTH, WIDTH], { outputActivation: 'relu' })
const RHO = Mlp([WIDTH, WIDTH, 1])
type Params = { phi: ReturnType<typeof PHI.init>; rho: ReturnType<typeof RHO.init> }

const TRAIN = numberSets(stream('deep-sets/train'), { count: TRAIN_SETS, minSize: TRAIN_MIN, maxSize: TRAIN_MAX })
const TRAIN_BATCH = padSets(TRAIN.sets.map((s) => s.map((v) => [v])))
/** For each test size n, TEST_PER_SIZE sets of exactly n numbers. */
export const TEST = Array.from({ length: TEST_MAX }, (_, i) =>
  numberSets(stream(`deep-sets/test/${i + 1}`), { count: TEST_PER_SIZE, minSize: i + 1, maxSize: i + 1 }),
)
const TEST_BATCHES = TEST.map((t) => padSets(t.sets.map((s) => s.map((v) => [v]))))

/** ρ(pool(φ(x))) for a padded batch: φ on every element row, the masked pool, then ρ; output `[B]`. */
function predict(p: Params, b: PaddedSets, pooling: SetPooling): Value {
  const [B, K] = b.mask.shape
  const h = reshape(PHI.apply(p.phi, reshape(b.x, [B * K, 1])), [B, K, WIDTH])
  return reshape(RHO.apply(p.rho, setPool(h, b.mask, pooling)), [B])
}

/** Full-batch Adam for each pooling in turn, one step each per round, with a test sweep every few steps. */
export function* deepSetsRun(setup: Setup): Generator<Checkpoint> {
  const y = TRAIN.targets[setup.target]
  const centre = y.reduce((a, v) => a + v, 0) / y.length
  const spread = Math.sqrt(y.reduce((a, v) => a + (v - centre) ** 2, 0) / y.length) || 1
  const yTrain = tensor(y.map((v) => (v - centre) / spread))
  const runs = POOLINGS.map((pooling) => {
    const alg = trainingLoop({
      loss: (p: Params) => meanSquaredErrorLoss(predict(p, TRAIN_BATCH, pooling), yTrain),
      data: { y: yTrain },
      optimizer: adamRule({ stepSize: setup.rate }),
    })
    const s = stream(`deep-sets/init/${setup.seed}`)
    return live(alg, { params: { phi: PHI.init(s), rho: RHO.init(s) } }, { stream: s })
  })
  const state = POOLINGS.map(() => null as null | { params: Params; loss: number })
  const checkpoint = (step: number): Checkpoint => {
    const rec = <T>(f: (k: number, pooling: SetPooling) => T) =>
      Object.fromEntries(POOLINGS.map((q, k) => [q, f(k, q)])) as Record<SetPooling, T>
    const predictions = rec((k, q) =>
      TEST_BATCHES.map((b) => Array.from(toFlat(add(mul(predict(state[k]!.params, b, q), spread), centre) as Tensor))),
    )
    return {
      step,
      loss: rec((k) => state[k]!.loss),
      predictions,
      rmse: rec((_, q) =>
        predictions[q].map((pred, i) => {
          const truth = TEST[i].targets[setup.target]
          return Math.sqrt(pred.reduce((a, v, j) => a + (v - truth[j]) ** 2, 0) / pred.length)
        }),
      ),
    }
  }
  for (let step = 0; step <= setup.steps; step++) {
    runs.forEach((it, k) => {
      const next = it.next()
      if (!next.done && !next.value.state.diverged)
        state[k] = { params: next.value.state.params, loss: next.value.state.loss }
    })
    if (step % EVERY === 0 || step === setup.steps) yield checkpoint(step)
  }
}
