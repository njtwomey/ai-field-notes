/**
 * Linear node classifiers on three views of a contextual stochastic block model, the experiment shared by the
 * heterophily and benchmark notes. Each view is a fixed propagation of the node features (engine `propagate`), and each
 * classifier is the engine's logistic regression fitted on the labelled nodes, so a whole sweep over homophily runs in
 * a fraction of a second:
 *
 * - `features`: the node's own features $\xvec_v$, a model that ignores the edges (the linear stand-in for an MLP);
 * - `gcn`: the mean over the node and its neighbours, $\frac{1}{d_v + 1}(\xvec_v + \sum_{u \in \Ncal(v)} \xvec_u)$, one
 *   linear graph convolution with self-loops (SGC with one hop, mean normalisation);
 * - `ego-neighbour`: the concatenation $[\xvec_v, \frac{1}{d_v}\sum_{u \in \Ncal(v)} \xvec_u]$, which keeps the node's
 *   own features apart from its neighbours' (H2GCN's first design).
 */
import { child, permutation, stream, type Stream } from 'aifn-compute/foundation/random'
import { concat, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { propagate } from 'aifn-compute/graph/propagation'
import { dataset, takeRows } from 'aifn-compute/learning/estimators'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { normalCdf } from 'aifn-compute/numerics/special'
import { blockModelProbabilities, contextualBlockModel, type BlockModelDataset } from 'aifn-methods/data/synthetic'

export const VIEWS = ['features', 'gcn', 'ego-neighbour'] as const
export type View = (typeof VIEWS)[number]
export const VIEW_NAMES: Record<View, string> = {
  features: 'features only (no edges)',
  gcn: 'mean with self-loop (GCN)',
  'ego-neighbour': 'ego and neighbours apart (H2GCN)',
}

/** The three feature views of a graph's nodes, each a node-by-feature matrix. */
export function views(d: BlockModelDataset): Record<View, Tensor> {
  const mean = propagate(d.graph, d.x, { aggregate: 'mean' }) as Tensor
  return {
    features: d.x,
    gcn: propagate(d.graph, d.x, { aggregate: 'mean', selfLoops: true }) as Tensor,
    'ego-neighbour': concat([d.x, mean], 1) as Tensor,
  }
}

/** `perClass` labelled nodes of each class, drawn at random, and the rest as test nodes. */
export function labelledSplit(s: Stream, y: readonly number[], perClass: number): { train: number[]; test: number[] } {
  const order = toFlat(permutation(s, y.length))
  const seen = new Map<number, number>()
  const train: number[] = []
  const test: number[] = []
  for (const i of order) {
    const k = seen.get(y[i]) ?? 0
    if (k < perClass) {
      train.push(i)
      seen.set(y[i], k + 1)
    } else test.push(i)
  }
  return { train, test }
}

/** Test accuracy of a lightly penalised logistic regression fitted on the training rows of a view. */
export function linearAccuracy(x: Tensor, y: readonly number[], split: { train: number[]; test: number[] }): number {
  const model = logisticRegression({ l2: 1e-3 }).fit(
    dataset(takeRows(x, split.train), tensor(split.train.map((i) => y[i]))),
  )
  const predicted = toFlat(model.decide(takeRows(x, split.test)))
  return predicted.filter((p, j) => p === y[split.test[j]]).length / split.test.length
}

/** The settings of one graph: two blocks, an expected degree, an edge homophily and a feature signal. */
export type BlockSetting = { blockSize: number; degree: number; homophily: number; signal: number }

/** Draw the two-block contextual model of a setting, with two-dimensional features. */
export function drawBlockModel(s: Stream, b: BlockSetting): BlockModelDataset {
  const { pIn, pOut } = blockModelProbabilities({ classes: 2, ...b })
  return contextualBlockModel(s, { sizes: [b.blockSize, b.blockSize], pIn, pOut, features: 2, signal: b.signal })
}

/** Test accuracy of every view on one drawn graph and one labelled split. */
export function viewAccuracies(d: BlockModelDataset, split: { train: number[]; test: number[] }): Record<View, number> {
  const y = toFlat(d.y)
  const v = views(d)
  return Object.fromEntries(VIEWS.map((k) => [k, linearAccuracy(v[k], y, split)])) as Record<View, number>
}

/** The accuracy of each view against homophily, averaged over a few seeds. */
export function homophilySweep(options: {
  homophilies: readonly number[]
  setting: Omit<BlockSetting, 'homophily'>
  perClass: number
  seeds: number
}): Record<View, number[]> {
  const out = Object.fromEntries(VIEWS.map((k) => [k, [] as number[]])) as Record<View, number[]>
  for (const h of options.homophilies) {
    const total = Object.fromEntries(VIEWS.map((k) => [k, 0])) as Record<View, number>
    for (let seed = 0; seed < options.seeds; seed++) {
      const s = stream(`heterophily/${seed}`)
      const d = drawBlockModel(child(s, 'graph', h), { ...options.setting, homophily: h })
      const split = labelledSplit(child(s, 'split'), toFlat(d.y), options.perClass)
      const acc = viewAccuracies(d, split)
      for (const k of VIEWS) total[k] += acc[k] / options.seeds
    }
    for (const k of VIEWS) out[k].push(total[k])
  }
  return out
}

/**
 * The accuracy each view would reach with every node of degree exactly $d$, Gaussian noise and the best linear rule:
 * $\Phi$ of the separation in noise units. Along the class-difference direction the two-dimensional features are a
 * scalar $\mu s_v + \varepsilon_v$ with $s_v = \pm 1$, $\mu$ the signal over $\sqrt 2$ and $\varepsilon_v$ standard
 * normal; the neighbour mean has mean $\mu (2h - 1) s_v$ and variance $1/d$.
 */
export function idealAccuracy(view: View, b: Omit<BlockSetting, 'blockSize'>): number {
  const mu = b.signal / Math.SQRT2
  const r = 2 * b.homophily - 1
  const d = b.degree
  const separation =
    view === 'features'
      ? mu
      : view === 'gcn'
        ? (mu * Math.abs(1 + d * r)) / Math.sqrt(d + 1)
        : mu * Math.sqrt(1 + d * r * r)
  return Number(normalCdf(separation))
}
