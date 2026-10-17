/**
 * The experiment behind the molecular property figure: synthetic molecules from this sub-chapter's generator, split
 * by scaffold or at random; a count-fingerprint ridge regression (engine `linearRegression`); and a three-layer message
 * passing network with one weight matrix per bond order and a sum readout of per-atom outputs, trained by full-batch
 * Adam (engine `trainingLoop`). Everything numerical is an engine call; this file fixes the recipe.
 */
import { child, normals, permutation, stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  matmul,
  mul,
  tensor,
  toFlat,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import type { Graph } from 'aifn-compute/graph'
import { propagate } from 'aifn-compute/graph/propagation'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import { relu } from 'aifn-compute/nn/functional'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { linearRegression } from 'aifn-methods/learning/linear'
import { batchGraphs, globalPool } from 'aifn-compute/nn/graph'
import {
  ATOM_FEATURES,
  countFingerprints,
  moleculeGraph,
  smallMolecules,
  type Molecule,
} from 'aifn-methods/data/synthetic'

export const COUNT = 180
const HIDDEN = 12
const LAYERS = 3
/** Optimiser steps between checkpoints. */
const EVERY = 10

/** The molecules, drawn once. */
export const MOLECULES: readonly Molecule[] = smallMolecules(stream('molecules/data'), { count: COUNT })

export type SplitKind = 'scaffold' | 'random'
/** Scaffold split: the five-rings are the test set. Random split: the same number of test molecules at random. */
export function splitOf(kind: SplitKind): { train: number[]; test: number[] } {
  const test =
    kind === 'scaffold'
      ? MOLECULES.flatMap((m, i) => (m.scaffold === 'ring5' ? [i] : []))
      : (() => {
          const n = MOLECULES.filter((m) => m.scaffold === 'ring5').length
          const order = toFlat(permutation(stream('molecules/random-split'), COUNT))
          return order.slice(0, n).sort((a, b) => a - b)
        })()
  const inTest = new Set(test)
  return { train: MOLECULES.flatMap((_, i) => (inTest.has(i) ? [] : [i])), test }
}

const rmse = (pred: readonly number[], truth: readonly number[]) =>
  Math.sqrt(pred.reduce((s, p, i) => s + (p - truth[i]) ** 2, 0) / pred.length)

/** Ridge regression on count fingerprints up to a radius; the vocabulary comes from the training molecules. */
export function fingerprintModel(kind: SplitKind, radius: number) {
  const { train, test } = splitOf(kind)
  const trainMols = train.map((i) => MOLECULES[i])
  const testMols = test.map((i) => MOLECULES[i])
  const fit = countFingerprints(trainMols, radius)
  const yTrain = trainMols.map((m) => m.property)
  const model = linearRegression({ l2: 0.1 }).fit(dataset(fit.counts, tensor(yTrain)))
  const predictTrain = toFlat(model.decide(fit.counts))
  const predictTest = toFlat(model.decide(countFingerprints(testMols, radius, trainMols).counts))
  return {
    vocabulary: fit.vocabulary.length,
    predictTest,
    trainRmse: rmse(predictTrain, yTrain),
    testRmse: rmse(
      predictTest,
      testMols.map((m) => m.property),
    ),
  }
}

/** A set of molecules as one batch: two graphs, one per bond order, the atom features and the batch vector. */
function batchOf(mols: readonly Molecule[]) {
  const graphs = mols.map(moleculeGraph)
  const byOrder = (order: number): Graph[] =>
    graphs.map(({ graph }) => ({
      ...graph,
      edges: graph.edges
        .filter((e) => (e as { weight?: number }).weight === order)
        .map((e) => ({ from: e.from, to: e.to })),
    }))
  const single = batchGraphs(byOrder(1))
  const double = batchGraphs(byOrder(2))
  return {
    single: single.graph,
    double: double.graph,
    batch: single.batch,
    graphs: single.graphs,
    x: concat(
      graphs.map((g) => g.x),
      0,
    ) as Tensor,
  }
}
type Batch = ReturnType<typeof batchOf>

type Params = { self: Tensor[]; single: Tensor[]; double: Tensor[]; bias: Tensor[]; out: Tensor; outBias: Tensor }

/**
 * Per-atom outputs: three layers $\hvec' = \mathrm{ReLU}(\hvec\Wmat_0 + \sum_{\text{single}} \hvec_u \Wmat_1 +
 * \sum_{\text{double}} \hvec_u \Wmat_2 + \bvec)$, then a linear map of the concatenated layers to one number per atom.
 */
function atomOutputs(p: Params, b: Batch): Value {
  let h: Value = b.x
  const layers: Value[] = []
  for (let l = 0; l < LAYERS; l++) {
    h = relu(
      add(
        add(
          add(matmul(h, p.self[l]), matmul(propagate(b.single, h), p.single[l])),
          matmul(propagate(b.double, h), p.double[l]),
        ),
        p.bias[l],
      ),
    )
    layers.push(h)
  }
  return add(matmul(concat(layers, 1), p.out), p.outBias)
}

function init(seed: number): Params {
  const s = stream(`molecules/init/${seed}`)
  const w = (name: string, l: number, rows: number) =>
    mul(normals(child(s, name, l), [rows, HIDDEN]), 1 / Math.sqrt(rows)) as Tensor
  const inputs = (l: number) => (l === 0 ? ATOM_FEATURES : HIDDEN)
  const ls = Array.from({ length: LAYERS }, (_, l) => l)
  return {
    self: ls.map((l) => w('self', l, inputs(l))),
    single: ls.map((l) => w('single', l, inputs(l))),
    double: ls.map((l) => w('double', l, inputs(l))),
    bias: ls.map(() => zeros([HIDDEN])),
    out: mul(normals(child(s, 'out'), [LAYERS * HIDDEN, 1]), 0.1 / Math.sqrt(LAYERS * HIDDEN)) as Tensor,
    outBias: zeros([1]),
  }
}

export type Setup = { split: SplitKind; steps: number; rate: number; seed: number }
export type Checkpoint = {
  step: number
  trainRmse: number
  testRmse: number
  /** Predicted properties of the test molecules. */
  predictTest: number[]
  /** Per-atom outputs of every test molecule (their sum is the prediction). */
  atomsTest: number[][]
}

/** Full-batch Adam on the mean squared error of the molecule predictions, with checkpoints. */
export function* gnnRun(setup: Setup): Generator<Checkpoint> {
  const { train, test } = splitOf(setup.split)
  const trainBatch = batchOf(train.map((i) => MOLECULES[i]))
  const testBatch = batchOf(test.map((i) => MOLECULES[i]))
  const yTrain = tensor(train.map((i) => [MOLECULES[i].property]))
  const yTest = test.map((i) => MOLECULES[i].property)
  const predict = (p: Params, b: Batch) => globalPool(atomOutputs(p, b), b.batch, b.graphs, 'sum')
  const alg = trainingLoop({
    loss: (p: Params) => meanSquaredErrorLoss(predict(p, trainBatch), yTrain),
    data: { y: yTrain },
    optimizer: adamRule({ stepSize: setup.rate }),
  })
  for (const { step, state } of live(alg, { params: init(setup.seed) }, { stream: stream('molecules/run') })) {
    const last = step >= setup.steps || state.diverged
    if (step % EVERY === 0 || last) {
      const atoms = toFlat(atomOutputs(state.params, testBatch) as Tensor)
      const predictTest: number[] = new Array(test.length).fill(0)
      const atomsTest: number[][] = test.map(() => [])
      atoms.forEach((a, v) => {
        predictTest[testBatch.batch[v]] += a
        atomsTest[testBatch.batch[v]].push(a)
      })
      yield {
        step,
        trainRmse: Math.sqrt(state.loss),
        testRmse: rmse(predictTest, yTest),
        predictTest,
        atomsTest,
      }
    }
    if (last) return
  }
}
