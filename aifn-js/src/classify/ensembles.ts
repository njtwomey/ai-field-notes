/**
 * Tree estimators and tree ensembles:
 *
 * - `decisionTree`, `regressionTree`: CART as estimators (growth traced, optional cost-complexity pruning).
 * - `randomForest`: bagged trees with random feature subsets per node (Breiman, 2001, "Random forests").
 * - `adaBoostSteps`, `adaBoost`: multiclass AdaBoost by SAMME (Zhu, Zou, Rosset and Hastie, 2009), which is
 *   Freund and Schapire's (1997) AdaBoost.M1 for two classes.
 * - `gradientBoostingSteps`, `gradientBoosting`: gradient tree boosting (Friedman, 2001, "Greedy function
 *   approximation: a gradient boosting machine") for squared error and the logistic (binomial and multinomial)
 *   deviance, with Newton leaf values as scikit-learn's `GradientBoostingClassifier`.
 */

import type {
  CategoricalPredictive,
  Decides,
  Estimator,
  FitOptions,
  Fitted,
  Predicts,
  Scores,
  Supervised,
  Trained,
} from 'aifn/estimators'
import { stream as makeStream, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/trace'
import {
  applyTree,
  featureImportances,
  growTree,
  nodeLabel,
  predictTree,
  pruneTree,
  treeGrowthSteps,
  type DecisionTree,
  type TreeGrowthState,
  type TreeParams,
} from './tree'
import { classLabels, inputs, matrix, probabilityModel, sigmoid, softmaxRows, targets, values } from './util'

/** Data with optional per-row sample weights. */
export type WeightedData = Supervised<Tensor, Tensor> & { weights?: Tensor }

/** A fitted classification tree. */
export interface DecisionTreeModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, CategoricalPredictive>,
    Trained<TreeGrowthState> {
  readonly kind: 'decision-tree'
  readonly tree: DecisionTree
  /** The tree before pruning (the same as `tree` when `pruneAlpha` is 0). */
  readonly grown: DecisionTree
  readonly classes: number
  readonly featureImportances: Tensor
  probabilities(x: Tensor): Tensor
  /** The leaf each row reaches. */
  apply(x: Tensor): Tensor
}

/**
 * A CART classification tree for labels 0 … K−1 (Gini by default). `forward`, `score` and `predictive` give the
 * leaf's weighted class shares; `decide` the largest. `pruneAlpha` > 0 prunes the grown tree at that complexity.
 * Growth is traced node by node in `training`.
 */
export function decisionTree(
  params: TreeParams & { pruneAlpha?: number } = {},
): Estimator<WeightedData, DecisionTreeModel> {
  const { pruneAlpha = 0, ...treeParams } = params
  return {
    name: 'decision-tree',
    params,
    fit({ x, y, weights }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'decisionTree')
      const { k: K } = classLabels(y, n, 'decisionTree')
      const problem = { x, y, weights, task: 'classification' as const, classes: K, params: treeParams }
      const training: Trace<TreeGrowthState> = trace(treeGrowthSteps(problem), {}, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          nodes: (s) => s.tree.nodes.length,
          ...(options.trace?.record as Record<string, (s: TreeGrowthState, t: number) => number> | undefined),
        },
      })
      const grown: DecisionTree = training.steps[training.steps.length - 1].tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const head = (q: Tensor) => {
        inputs(q, d, 'decisionTree')
        return values(predictTree(tree, q)).slice()
      }
      return {
        kind: 'decision-tree',
        tree,
        grown,
        classes: K,
        featureImportances: featureImportances(tree),
        training,
        apply: (q: Tensor) => applyTree(tree, q),
        ...probabilityModel(head, (h) => h, K),
      }
    },
  }
}

/** A fitted regression tree. */
export interface RegressionTreeModel extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Trained<TreeGrowthState> {
  readonly kind: 'regression-tree'
  readonly tree: DecisionTree
  readonly grown: DecisionTree
  readonly featureImportances: Tensor
  apply(x: Tensor): Tensor
}

/** A CART regression tree (squared error): each leaf predicts the weighted mean of its targets. */
export function regressionTree(
  params: Omit<TreeParams, 'criterion'> & { pruneAlpha?: number } = {},
): Estimator<WeightedData, RegressionTreeModel> {
  const { pruneAlpha = 0, ...treeParams } = params
  return {
    name: 'regression-tree',
    params,
    fit({ x, y, weights }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'regressionTree')
      targets(y, n, 'regressionTree')
      const problem = {
        x,
        y,
        weights,
        task: 'regression' as const,
        params: { ...treeParams, criterion: 'squared' as const },
      }
      const training = trace(treeGrowthSteps(problem), {}, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
      })
      const grown: DecisionTree = training.steps[training.steps.length - 1].tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const predict = (q: Tensor) => {
        inputs(q, d, 'regressionTree')
        return predictTree(tree, q)
      }
      return {
        kind: 'regression-tree',
        tree,
        grown,
        featureImportances: featureImportances(tree),
        training,
        apply: (q: Tensor) => applyTree(tree, q),
        forward: predict,
        decide: predict,
      }
    },
  }
}

// ── Random forests ───────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted random forest classifier. */
export interface RandomForestModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Predicts<Tensor, CategoricalPredictive> {
  readonly kind: 'random-forest'
  readonly trees: DecisionTree[]
  /** The bootstrap count of each training row in each tree's sample [T, n] (all ones without bootstrap). */
  readonly inBag: Tensor
  readonly classes: number
  /** Mean of the trees' impurity importances [d]. */
  readonly featureImportances: Tensor
  /** Out-of-bag class shares [n, K] (NaN rows for rows in every bag) and the out-of-bag accuracy. */
  readonly outOfBag: { probabilities: Tensor; accuracy: number }
  probabilities(x: Tensor): Tensor
  /** Class shares from the first `t` trees only [m, K], e.g. to show the forest growing. */
  probabilitiesUpTo(x: Tensor, t: number): Tensor
}

/**
 * A random forest (Breiman, 2001): `trees` CART trees (default 100), each grown on a bootstrap sample (as integer
 * sample weights) searching a random subset of `maxFeatures` features (default `sqrt`) at every node. The predictive
 * averages the trees' leaf class shares. Tree t draws from `stream.child('tree', t)`, so adding trees keeps the
 * existing ones. If a node's feature subset has no valid split, the node becomes a leaf (scikit-learn draws further
 * features instead).
 */
export function randomForest(
  params: TreeParams & { trees?: number; bootstrap?: boolean } = {},
): Estimator<Supervised<Tensor, Tensor>, RandomForestModel> {
  const { trees: T = 100, bootstrap = true, maxFeatures = 'sqrt', ...rest } = params
  return {
    name: 'random-forest',
    params: { trees: T, bootstrap, maxFeatures, ...rest },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'randomForest')
      const { y: labels, k: K } = classLabels(y, n, 'randomForest')
      const s = options.stream ?? makeStream('random-forest')
      const inBag = new Float64Array(T * n)
      const trees: DecisionTree[] = []
      for (let t = 0; t < T; t++) {
        const ts = s.child('tree', t)
        const counts = inBag.subarray(t * n, (t + 1) * n)
        if (bootstrap) {
          const b = ts.child('bootstrap')
          for (let r = 0; r < n; r++) counts[b.int(n)]++
        } else counts.fill(1)
        trees.push(
          growTree(
            {
              x,
              y,
              weights: fromData(Float64Array.from(counts), [n]),
              task: 'classification',
              classes: K,
              params: { ...rest, maxFeatures },
            },
            ts.child('features'),
          ),
        )
      }
      const average = (q: Tensor, upTo: number) => {
        const { n: m } = inputs(q, d, 'randomForest')
        const out = new Float64Array(m * K)
        for (let t = 0; t < upTo; t++) {
          const p = values(predictTree(trees[t], q))
          for (let j = 0; j < out.length; j++) out[j] += p[j] / upTo
        }
        return out
      }
      // Out-of-bag: average only the trees whose bag missed the row.
      const oob = new Float64Array(n * K)
      const votes = new Float64Array(n)
      for (let t = 0; t < T; t++) {
        const p = values(predictTree(trees[t], x))
        for (let i = 0; i < n; i++) {
          if (inBag[t * n + i] > 0) continue
          votes[i]++
          for (let c = 0; c < K; c++) oob[i * K + c] += p[i * K + c]
        }
      }
      let correct = 0
      let counted = 0
      for (let i = 0; i < n; i++) {
        if (votes[i] === 0) {
          for (let c = 0; c < K; c++) oob[i * K + c] = NaN
          continue
        }
        let best = 0
        for (let c = 0; c < K; c++) {
          oob[i * K + c] /= votes[i]
          if (oob[i * K + c] > oob[i * K + best]) best = c
        }
        counted++
        if (best === labels[i]) correct++
      }
      const importance = new Float64Array(d)
      for (const tree of trees) {
        const imp = values(featureImportances(tree))
        for (let j = 0; j < d; j++) importance[j] += imp[j] / T
      }
      return {
        kind: 'random-forest',
        trees,
        inBag: fromData(inBag, [T, n]),
        classes: K,
        featureImportances: fromData(importance, [d]),
        outOfBag: { probabilities: fromData(oob, [n, K]), accuracy: counted ? correct / counted : NaN },
        probabilitiesUpTo: (q: Tensor, t: number) => fromData(average(q, Math.max(1, Math.min(t, T))), [q.shape[0], K]),
        ...probabilityModel(
          (q) => average(q, T),
          (h) => h,
          K,
        ),
      }
    },
  }
}

// ── AdaBoost (SAMME) ─────────────────────────────────────────────────────────────────────────────────────────────

/** The problem an AdaBoost run solves. */
export interface AdaBoostProblem {
  x: Tensor
  /** Labels 0 … K−1. */
  y: Tensor
  /** The weak learner's tree parameters (default stumps: `{ maxDepth: 1 }`). */
  base?: TreeParams
  learningRate?: number
}

/** One AdaBoost state: the learners so far, their weights and the current sample weights. */
export interface AdaBoostState {
  learners: DecisionTree[]
  /** The weight α_m of each learner. */
  alphas: number[]
  /** The weighted training error of each learner. */
  errors: number[]
  /** Sample weights for the next round [n] (sum 1). */
  sampleWeights: Tensor
  /** The ensemble's votes Σ α_m 1[h_m(x) = k] on the training rows [n, K], and its training error. */
  votes: Tensor
  trainingError: number
  round: number
  /** A learner had zero error (it is kept) or no better than chance (the run stops without it). */
  stopped: 'perfect' | 'chance' | null
}

/**
 * AdaBoost by SAMME as a traceable algorithm: each round fits a weighted tree, weights it by
 * α = η (log((1 − err)/err) + log(K − 1)), multiplies the weights of misclassified rows by e^α and renormalises. It
 * is done when a learner is perfect or no better than chance (err ≥ 1 − 1/K).
 */
export function adaBoostSteps(problem: AdaBoostProblem): Algorithm<object, AdaBoostState> {
  const { n } = matrix(problem.x, 'adaBoostSteps')
  const { y, k: K } = classLabels(problem.y, n, 'adaBoostSteps')
  const eta = problem.learningRate ?? 1
  const base = problem.base ?? { maxDepth: 1 }
  return {
    name: 'adaboost-samme',
    init: () => ({
      learners: [],
      alphas: [],
      errors: [],
      sampleWeights: fromData(new Float64Array(n).fill(1 / n), [n]),
      votes: fromData(new Float64Array(n * K), [n, K]),
      trainingError: NaN,
      round: 0,
      stopped: null,
    }),
    step: (state) => {
      const w = values(state.sampleWeights)
      const tree = growTree({
        x: problem.x,
        y: problem.y,
        weights: state.sampleWeights,
        task: 'classification',
        classes: K,
        params: base,
      })
      const probs = values(predictTree(tree, problem.x))
      const pred = new Int32Array(n)
      let err = 0
      let total = 0
      for (let i = 0; i < n; i++) {
        let best = 0
        for (let c = 1; c < K; c++) if (probs[i * K + c] > probs[i * K + best]) best = c
        pred[i] = best
        total += w[i]
        if (best !== y[i]) err += w[i]
      }
      err /= total
      if (err >= 1 - 1 / K) return { ...state, round: state.round + 1, stopped: 'chance' }
      const perfect = err <= 0
      // A perfect learner gets weight 1 and ends the run, as in scikit-learn.
      const alpha = perfect ? 1 : eta * (Math.log((1 - err) / err) + Math.log(K - 1))
      const next = new Float64Array(n)
      let z = 0
      for (let i = 0; i < n; i++) z += next[i] = w[i] * (!perfect && pred[i] !== y[i] ? Math.exp(alpha) : 1)
      for (let i = 0; i < n; i++) next[i] /= z
      const votes = Float64Array.from(values(state.votes))
      let wrong = 0
      for (let i = 0; i < n; i++) {
        votes[i * K + pred[i]] += alpha
        let best = 0
        for (let c = 1; c < K; c++) if (votes[i * K + c] > votes[i * K + best]) best = c
        if (best !== y[i]) wrong++
      }
      return {
        learners: [...state.learners, tree],
        alphas: [...state.alphas, alpha],
        errors: [...state.errors, err],
        sampleWeights: fromData(next, [n]),
        votes: fromData(votes, [n, K]),
        trainingError: wrong / n,
        round: state.round + 1,
        stopped: perfect ? 'perfect' : null,
      }
    },
    done: (state) => state.stopped !== null,
  }
}

/** A fitted AdaBoost classifier. */
export interface AdaBoostModel
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Trained<AdaBoostState> {
  readonly kind: 'adaboost'
  readonly learners: DecisionTree[]
  readonly alphas: Tensor
  readonly errors: Tensor
  readonly classes: number
  /** Normalised votes Σ α_m 1[h_m(x) = k] / Σ α_m from the first `rounds` learners [m, K]. */
  votesUpTo(x: Tensor, rounds: number): Tensor
}

/**
 * AdaBoost (SAMME) with `rounds` weak learners (default 50 stumps). `forward` and `score` are the normalised votes
 * [m, K]; `decide` is the class with the most weight.
 */
export function adaBoost(
  params: { rounds?: number; learningRate?: number; base?: TreeParams } = {},
): Estimator<Supervised<Tensor, Tensor>, AdaBoostModel> {
  const { rounds = 50, learningRate = 1, base = { maxDepth: 1 } } = params
  return {
    name: 'adaboost',
    params: { rounds, learningRate, base },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'adaBoost')
      const { k: K } = classLabels(y, n, 'adaBoost')
      const training = trace(adaBoostSteps({ x, y, base, learningRate }), {}, rounds, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        stopOnNonFinite: false,
        record: {
          trainingError: (s) => s.trainingError,
          ...(options.trace?.record as Record<string, (s: AdaBoostState, t: number) => number> | undefined),
        },
      })
      const final = training.steps[training.steps.length - 1]
      const votesUpTo = (q: Tensor, upTo: number) => {
        const { n: m } = inputs(q, d, 'adaBoost')
        const out = new Float64Array(m * K)
        const R = Math.min(upTo, final.learners.length)
        let total = 0
        for (let r = 0; r < R; r++) {
          total += final.alphas[r]
          const p = values(predictTree(final.learners[r], q))
          for (let i = 0; i < m; i++) {
            let best = 0
            for (let c = 1; c < K; c++) if (p[i * K + c] > p[i * K + best]) best = c
            out[i * K + best] += final.alphas[r]
          }
        }
        if (total > 0) for (let j = 0; j < out.length; j++) out[j] /= total
        return out
      }
      const model = probabilityModel(
        (q) => votesUpTo(q, Infinity),
        (h) => h,
        K,
      )
      return {
        kind: 'adaboost',
        learners: final.learners,
        alphas: fromData(Float64Array.from(final.alphas), [final.alphas.length]),
        errors: fromData(Float64Array.from(final.errors), [final.errors.length]),
        classes: K,
        training,
        votesUpTo: (q: Tensor, r: number) => fromData(votesUpTo(q, r), [q.shape[0], K]),
        forward: model.forward,
        score: model.score,
        decide: model.decide,
      }
    },
  }
}

// ── Gradient boosting ────────────────────────────────────────────────────────────────────────────────────────────

/** The loss gradient boosting minimises. */
export type BoostingLoss = 'squared' | 'logistic'

/** The problem a gradient boosting run solves. */
export interface GradientBoostingProblem {
  x: Tensor
  /** Real targets (squared) or labels 0 … K−1 (logistic). */
  y: Tensor
  loss: BoostingLoss
  learningRate?: number
  /** Tree parameters for each stage (default `{ maxDepth: 3 }`). */
  tree?: Omit<TreeParams, 'criterion'>
  /** Fraction of rows each stage fits on, drawn without replacement (stochastic gradient boosting; default 1). */
  subsample?: number
}

/** One state of gradient boosting. */
export interface GradientBoostingState {
  /** The initial raw prediction F₀ ([1] for squared or binary logistic, [K] for multinomial). */
  initial: number[]
  /** Trees per stage: one, or K for the multinomial loss. Leaf values are the stage's Newton steps (before η). */
  stages: DecisionTree[][]
  /** The raw prediction F on the training rows: [n] or [n, K]. */
  raw: Tensor
  /** The negative gradients the latest stage fitted (the pseudo-residuals): [n] or [n, K]. */
  residuals: Tensor
  /** The training loss: half the mean squared error, or the mean negative log-likelihood. */
  loss: number
  stage: number
  stream?: Stream
}

/**
 * Gradient tree boosting as a traceable algorithm. Each stage computes the pseudo-residuals rᵢ = −∂L/∂F(xᵢ) (yᵢ − Fᵢ
 * for squared error, yᵢ − pᵢ for the logistic deviance), fits a regression tree to them (squared-error splits), sets
 * each leaf to a Newton step (the mean residual for squared error; Σr / Σp(1 − p) for the binomial loss, times
 * (K − 1)/K per class for the multinomial one), and adds η times the tree to F.
 */
export function gradientBoostingSteps(problem: GradientBoostingProblem): Algorithm<object, GradientBoostingState> {
  const { n } = matrix(problem.x, 'gradientBoostingSteps')
  const eta = problem.learningRate ?? 0.1
  const treeParams = problem.tree ?? { maxDepth: 3 }
  const subsample = problem.subsample ?? 1
  const logistic = problem.loss === 'logistic'
  const lab = logistic ? classLabels(problem.y, n, 'gradientBoostingSteps') : null
  const K = lab ? lab.k : 1
  const C = logistic && K > 2 ? K : 1 // raw columns
  const y = logistic ? Float64Array.from(lab!.y) : targets(problem.y, n, 'gradientBoostingSteps')
  const probs = (F: Float64Array): Float64Array => (C === 1 ? Float64Array.from(F, sigmoid) : softmaxRows(F, n, C))
  const lossOf = (F: Float64Array): number => {
    let s = 0
    if (!logistic) {
      for (let i = 0; i < n; i++) s += 0.5 * (y[i] - F[i]) ** 2
      return s / n
    }
    if (C === 1) {
      for (let i = 0; i < n; i++) s += Math.max(F[i], 0) + Math.log1p(Math.exp(-Math.abs(F[i]))) - y[i] * F[i]
      return s / n
    }
    for (let i = 0; i < n; i++) {
      let m = -Infinity
      for (let c = 0; c < C; c++) m = Math.max(m, F[i * C + c])
      let z = 0
      for (let c = 0; c < C; c++) z += Math.exp(F[i * C + c] - m)
      s += m + Math.log(z) - F[i * C + y[i]]
    }
    return s / n
  }
  const residualsOf = (F: Float64Array): Float64Array => {
    if (!logistic) return Float64Array.from(F, (f, i) => y[i] - f)
    const p = probs(F)
    if (C === 1) return Float64Array.from(p, (q, i) => y[i] - q)
    return Float64Array.from(p, (q, j) => (y[Math.floor(j / C)] === j % C ? 1 : 0) - q)
  }
  return {
    name: 'gradient-boosting',
    init: (_opts, s) => {
      let initial: number[]
      if (!logistic) initial = [y.reduce((a, b) => a + b, 0) / n]
      else if (C === 1) {
        const p = y.reduce((a, b) => a + b, 0) / n
        initial = [Math.log(p / (1 - p))]
      } else {
        const counts = new Float64Array(C)
        for (const c of y) counts[c]++
        const logs = Array.from(counts, (c) => Math.log(c / n))
        const mean = logs.reduce((a, b) => a + b, 0) / C
        initial = logs.map((l) => l - mean)
      }
      const F = new Float64Array(n * C)
      for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) F[i * C + c] = initial[c]
      const shape = C === 1 ? [n] : [n, C]
      return {
        initial,
        stages: [],
        raw: fromData(F, shape),
        residuals: fromData(residualsOf(F), shape),
        loss: lossOf(F),
        stage: 0,
        stream: s,
      }
    },
    step: (state) => {
      const F = Float64Array.from(values(state.raw))
      const r = residualsOf(F)
      const p = logistic ? probs(F) : null
      // Rows this stage fits on.
      let rowWeights: Float64Array | undefined
      if (subsample < 1) {
        const s = (state.stream ?? makeStream('gradient-boosting')).child('stage', state.stage)
        const order = Array.from({ length: n }, (_, i) => i)
        const take = Math.max(1, Math.floor(subsample * n))
        for (let a = 0; a < take; a++) {
          const b = a + s.int(n - a)
          ;[order[a], order[b]] = [order[b], order[a]]
        }
        rowWeights = new Float64Array(n)
        for (let a = 0; a < take; a++) rowWeights[order[a]] = 1
      }
      const trees: DecisionTree[] = []
      for (let c = 0; c < C; c++) {
        const rc = C === 1 ? r : Float64Array.from({ length: n }, (_, i) => r[i * C + c])
        const tree = growTree({
          x: problem.x,
          y: fromData(rc, [n]),
          weights: rowWeights ? fromData(rowWeights, [n]) : undefined,
          task: 'regression',
          params: { ...treeParams, criterion: 'squared' },
        })
        const leaves = values(applyTree(tree, problem.x))
        if (logistic) {
          // Newton leaf values: Σr / Σ p(1 − p) over the leaf's (in-sample) rows, times (K − 1)/K for K classes.
          const num = new Float64Array(tree.nodes.length)
          const den = new Float64Array(tree.nodes.length)
          for (let i = 0; i < n; i++) {
            if (rowWeights && rowWeights[i] === 0) continue
            const q = p![i * C + c]
            num[leaves[i]] += rc[i]
            den[leaves[i]] += q * (1 - q)
          }
          const factor = C === 1 ? 1 : (C - 1) / C
          tree.nodes = tree.nodes.map((node) =>
            node.children.length
              ? node
              : (() => {
                  const value = [Math.abs(den[node.id]) < 1e-150 ? 0 : (factor * num[node.id]) / den[node.id]]
                  return { ...node, value, label: nodeLabel({ ...node, value }, 'regression') }
                })(),
          )
        }
        for (let i = 0; i < n; i++) F[i * C + c] += eta * tree.nodes[leaves[i]].value[0]
        trees.push(tree)
      }
      const shape = C === 1 ? [n] : [n, C]
      return {
        initial: state.initial,
        stages: [...state.stages, trees],
        raw: fromData(F, shape),
        residuals: fromData(r, shape),
        loss: lossOf(F),
        stage: state.stage + 1,
        stream: state.stream,
      }
    },
  }
}

/** A fitted gradient boosting model. */
export interface GradientBoostingModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Trained<GradientBoostingState> {
  readonly kind: 'gradient-boosting'
  readonly loss: BoostingLoss
  readonly learningRate: number
  readonly initial: number[]
  readonly stages: DecisionTree[][]
  /** Classes for the logistic loss (1 for squared error). */
  readonly classes: number
  /** The raw prediction F(x) after the first `stages` stages: [m] or [m, K]. */
  rawUpTo(x: Tensor, stages: number): Tensor
  /** Logistic loss only: class probabilities [m, K]. */
  probabilities?(x: Tensor): Tensor
  predictive?(x: Tensor): CategoricalPredictive
  score?(x: Tensor): Tensor
}

/**
 * Gradient boosting (default 100 stages of depth-3 trees, η = 0.1). Squared error: `forward` and `decide` give F(x).
 * Logistic: `forward`/`score` give F(x) (the log-odds [m] for two classes, [m, K] scores otherwise), `predictive` the
 * class probabilities, `decide` the most probable class.
 */
export function gradientBoosting(
  params: {
    loss?: BoostingLoss
    stages?: number
    learningRate?: number
    tree?: Omit<TreeParams, 'criterion'>
    subsample?: number
  } = {},
): Estimator<Supervised<Tensor, Tensor>, GradientBoostingModel> {
  const { loss = 'squared', stages = 100, learningRate = 0.1, tree = { maxDepth: 3 }, subsample = 1 } = params
  return {
    name: 'gradient-boosting',
    params: { loss, stages, learningRate, tree, subsample },
    fit({ x, y }, options: FitOptions = {}) {
      const { n, d } = matrix(x, 'gradientBoosting')
      const K = loss === 'logistic' ? classLabels(y, n, 'gradientBoosting').k : 1
      const C = K > 2 ? K : 1
      const training = trace(gradientBoostingSteps({ x, y, loss, learningRate, tree, subsample }), {}, stages, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          loss: (s) => s.loss,
          ...(options.trace?.record as Record<string, (s: GradientBoostingState, t: number) => number> | undefined),
        },
      })
      const final = training.steps[training.steps.length - 1]
      const rawUpTo = (q: Tensor, upTo: number): Float64Array => {
        const { n: m } = inputs(q, d, 'gradientBoosting')
        const F = new Float64Array(m * C)
        for (let i = 0; i < m; i++) for (let c = 0; c < C; c++) F[i * C + c] = final.initial[c]
        const S = Math.min(upTo, final.stages.length)
        for (let s = 0; s < S; s++) {
          for (let c = 0; c < C; c++) {
            const v = values(predictTree(final.stages[s][c], q))
            for (let i = 0; i < m; i++) F[i * C + c] += learningRate * v[i]
          }
        }
        return F
      }
      const shape = (m: number) => (C === 1 ? [m] : [m, C])
      const common = {
        kind: 'gradient-boosting' as const,
        loss,
        learningRate,
        initial: final.initial,
        stages: final.stages,
        classes: K,
        training,
        rawUpTo: (q: Tensor, s: number) => fromData(rawUpTo(q, s), shape(q.shape[0])),
      }
      const forward = (q: Tensor) => fromData(rawUpTo(q, Infinity), shape(q.shape[0]))
      if (loss === 'squared') return { ...common, forward, decide: forward }
      const toProbs = (F: Float64Array, m: number): Float64Array => {
        if (C > 1) return softmaxRows(F, m, C)
        const out = new Float64Array(2 * m)
        for (let i = 0; i < m; i++) {
          out[2 * i + 1] = sigmoid(F[i])
          out[2 * i] = 1 - out[2 * i + 1]
        }
        return out
      }
      const model = probabilityModel((q) => rawUpTo(q, Infinity), toProbs, K)
      return {
        ...common,
        forward,
        score: forward,
        decide: model.decide,
        predictive: model.predictive,
        probabilities: model.probabilities,
      }
    },
  }
}
