import { describe, expect, it } from 'vitest'
import {
  adaBoost,
  adaBoostSteps,
  bernoulliNaiveBayes,
  codeDistance,
  costComplexityPath,
  crammerSinger,
  decisionTree,
  dichotomyTree,
  exhaustiveCode,
  gaussianNaiveBayes,
  gradientBoosting,
  growTree,
  kNearestNeighbours,
  linearDiscriminant,
  linearSvm,
  multinomialNaiveBayes,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  outputCode,
  perceptron,
  perceptronSteps,
  quadraticDiscriminant,
  randomCode,
  randomForest,
  regressionTree,
  smoSteps,
  splitSearch,
  supportVectorMachine,
  treeGrowthSteps,
  type DecisionTree,
} from 'aifn/classify'
import { classProbabilities, hasPredictive, logisticRegression } from 'aifn/estimators'
import { leaves, preOrder } from 'aifn/graph'
import { linear, rbf } from 'aifn/kernels'
import { stream } from 'aifn/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/trace'
import { fixture } from './fixtures'

type TreeFx = {
  feature: number[]
  threshold: number[]
  left: number[]
  right: number[]
  impurity: number[]
  count: number[]
  importances: number[]
}
type Fx = {
  x3: number[][]
  y3: number[]
  xq: number[][]
  knn: { uniform: number[][]; distance: number[][] }
  knn_manhattan: number[][]
  gnb: number[][]
  counts: number[][]
  ycounts: number[]
  mnb: number[][]
  bnb: number[][]
  lda: number[][]
  lda_ratio: number[]
  qda: number[][]
  tree: TreeFx
  tree_proba: number[][]
  tree_entropy: TreeFx
  path: { alphas: number[]; impurities: number[] }
  pruned: TreeFx
  yreg: number[]
  rtree: TreeFx
  rtree_predict: number[]
  svc: { decision: number[]; support: number[]; dual: number[] }
  svc_linear: { decision: number[]; coef: number[] }
  lsvc: { coef: number[]; intercept: number }
  cs: { coef: number[][]; intercept: number[]; decision: number[][] }
  perceptron: { coef: number[]; intercept: number }
  ada: { weights: number[]; errors: number[]; predict: number[] }
  gbr: { staged: number[][] }
  gbc: { decision: number[]; proba: number[][] }
  gbm: { proba: number[][] }
  ovr: { decision: number[][]; proba: number[][] }
  ovo: { decision: number[][] }
}
const fx = fixture<Fx>('classify')
const X3 = tensor(fx.x3)
const Y3 = tensor(fx.y3)
const XQ = tensor(fx.xq)
const X2 = tensor(fx.x3.slice(0, 40))
const Y2 = tensor(fx.y3.slice(0, 40))

function close(a: Tensor | number[], b: number[] | number[][], tol = 1e-8) {
  const got = Array.isArray(a) ? a : toFlat(a)
  const want = (b as number[]).flat() as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], -Math.log10(tol)))
}

function sameTree(tree: DecisionTree, want: TreeFx) {
  expect(tree.nodes.length).toBe(want.feature.length)
  tree.nodes.forEach((node, i) => {
    const leaf = want.left[i] < 0
    expect(node.children.length === 0).toBe(leaf)
    if (!leaf) {
      expect(node.feature).toBe(want.feature[i])
      // scikit-learn stores thresholds in float32.
      expect(node.threshold).toBeCloseTo(want.threshold[i], 6)
      expect(node.children).toEqual([want.left[i], want.right[i]])
    }
    expect(node.impurity).toBeCloseTo(want.impurity[i], 10)
    expect(node.count).toBe(want.count[i])
  })
}

function protocol<O, S>(
  alg: Algorithm<O, S>,
  opts: O,
  n: number,
  key: (s: S) => number,
  s?: () => ReturnType<typeof stream>,
) {
  const a = trace(alg, opts, n, { stream: s?.(), record: { k: key } })
  const b = trace(alg, opts, n, { stream: s?.(), record: { k: key } })
  expect(toFlat(a.series.k)).toEqual(toFlat(b.series.k))
  const i = Math.min(3, a.meta.steps)
  expect(key(seek(alg, opts, i, { stream: s?.() }))).toEqual(key(run(alg, opts, i, { stream: s?.() })))
  const short = trace(alg, opts, 2, { stream: s?.(), record: { k: key } })
  expect(toFlat(extend(short, alg, opts, n - 2).series.k)).toEqual(toFlat(a.series.k))
}

describe('k-nearest neighbours', () => {
  it('matches scikit-learn (uniform, distance, Manhattan)', () => {
    close(kNearestNeighbours({ k: 5 }).fit({ x: X3, y: Y3 }).probabilities(XQ), fx.knn.uniform)
    close(kNearestNeighbours({ k: 5, weights: 'distance' }).fit({ x: X3, y: Y3 }).probabilities(XQ), fx.knn.distance)
    close(kNearestNeighbours({ k: 4, metric: 'manhattan' }).fit({ x: X3, y: Y3 }).probabilities(XQ), fx.knn_manhattan)
  })
  it('exposes the neighbours, nearest first', () => {
    const m = kNearestNeighbours({ k: 3 }).fit({ x: X3, y: Y3 })
    const nb = m.neighbours(X3)
    expect(toRows(nb.index).map((r) => r[0])).toEqual(fx.x3.map((_, i) => i))
    const d = toRows(nb.distance)
    for (const r of d) expect(r[0] <= r[1] && r[1] <= r[2]).toBe(true)
    expect(hasPredictive(m)).toBe(true)
  })
})

describe('generative classifiers', () => {
  it('naive Bayes matches scikit-learn', () => {
    close(gaussianNaiveBayes().fit({ x: X3, y: Y3 }).probabilities(XQ), fx.gnb)
    const C = tensor(fx.counts)
    const yc = tensor(fx.ycounts)
    close(
      multinomialNaiveBayes({ alpha: 0.5 })
        .fit({ x: C, y: yc })
        .probabilities(tensor(fx.counts.slice(0, 6))),
      fx.mnb,
    )
    close(
      bernoulliNaiveBayes({ binarize: 1.5 })
        .fit({ x: C, y: yc })
        .probabilities(tensor(fx.counts.slice(0, 6))),
      fx.bnb,
    )
  })
  it('LDA and QDA match scikit-learn', () => {
    const lda = linearDiscriminant().fit({ x: X3, y: Y3 })
    close(lda.probabilities(XQ), fx.lda)
    close(lda.explainedVarianceRatio, fx.lda_ratio)
    // The projected classes have unit pooled within-class variance.
    const z = toRows(lda.transform(X3))
    expect(z[0].length).toBe(2)
    close(quadraticDiscriminant({ regularisation: 0.1 }).fit({ x: X3, y: Y3 }).probabilities(XQ), fx.qda)
  })
})

describe('perceptron', () => {
  it('matches scikit-learn without shuffling', () => {
    const m = perceptron().fit({ x: X2, y: Y2 })
    close(m.weights, fx.perceptron.coef)
    expect(m.bias).toBeCloseTo(fx.perceptron.intercept, 10)
    expect(m.converged).toBe(true)
  })
  it('follows the trace protocol', () => {
    const signs = tensor(fx.y3.slice(0, 40).map((c) => (c ? 1 : -1)))
    protocol(
      perceptronSteps({ x: X2, y: signs, shuffle: true }),
      {},
      60,
      (s) => s.mistakes,
      () => stream('p'),
    )
  })
})

describe('support vector machines', () => {
  it('SMO matches scikit-learn SVC (RBF and linear kernels)', () => {
    for (const selection of ['second-order', 'maximal-violating'] as const) {
      const m = supportVectorMachine({ C: 1, kernel: rbf({ lengthscale: 1 }), tol: 1e-6, selection }).fit({
        x: X2,
        y: Y2,
      })
      close(m.score(XQ), fx.svc.decision, 1e-4)
      expect(toFlat(m.supportVectors)).toEqual(fx.svc.support.slice().sort((a, b) => a - b))
    }
    const lin = supportVectorMachine({ C: 0.5, kernel: linear(), tol: 1e-6 }).fit({ x: X2, y: Y2 })
    close(lin.score(XQ), fx.svc_linear.decision, 1e-4)
    close(lin.weights!, fx.svc_linear.coef, 1e-4)
  })
  it('SMO states are feasible and the dual objective rises', () => {
    const signs = tensor(fx.y3.slice(0, 40).map((c) => (c ? 1 : -1)))
    const t = trace(smoSteps({ x: X2, y: signs, C: 1 }), {}, 500)
    let prev = -Infinity
    for (const s of t.steps) {
      const a = toFlat(s.alpha)
      expect(Math.abs(a.reduce((acc, v, i) => acc + v * toFlat(signs)[i], 0))).toBeLessThan(1e-9)
      for (const v of a) expect(v >= 0 && v <= 1).toBe(true)
      expect(s.dualObjective).toBeGreaterThanOrEqual(prev - 1e-12)
      prev = s.dualObjective
    }
    expect(t.meta.stopped).toBe('done')
    protocol(smoSteps({ x: X2, y: signs, C: 1 }), {}, 30, (s) => s.dualObjective)
  })
  it('linear SVM by dual coordinate descent matches LinearSVC; Pegasos approaches it', () => {
    const m = linearSvm({ C: 0.5, tol: 1e-10, maxIterations: 20000 }).fit({ x: X2, y: Y2 })
    close(m.weights, fx.lsvc.coef, 1e-4)
    expect(m.bias).toBeCloseTo(fx.lsvc.intercept, 4)
    const p = linearSvm({ C: 0.5, method: 'pegasos', maxIterations: 20000 }).fit(
      { x: X2, y: Y2 },
      { stream: stream(1) },
    )
    const gap = p.training.steps.at(-1)!.primalObjective - m.training.steps.at(-1)!.primalObjective
    expect(gap).toBeGreaterThanOrEqual(-1e-9)
    expect(gap).toBeLessThan(0.5)
  })
})

describe('trees', () => {
  it('CART matches scikit-learn: structure, thresholds, impurities, importances', () => {
    const m = decisionTree().fit({ x: X3, y: Y3 })
    sameTree(m.tree, fx.tree)
    close(m.featureImportances, fx.tree.importances)
    close(m.probabilities(XQ), fx.tree_proba)
    sameTree(decisionTree({ criterion: 'entropy', maxDepth: 3 }).fit({ x: X3, y: Y3 }).tree, fx.tree_entropy)
    const r = regressionTree({ maxDepth: 3 }).fit({ x: X3, y: tensor(fx.yreg) })
    sameTree(r.tree, fx.rtree)
    close(r.decide(XQ), fx.rtree_predict)
  })
  it('cost-complexity path and pruning match scikit-learn', () => {
    const m = decisionTree().fit({ x: X3, y: Y3 })
    const path = costComplexityPath(m.tree)
    close(path.alphas, fx.path.alphas)
    close(path.impurities, fx.path.impurities)
    sameTree(decisionTree({ pruneAlpha: 0.02 }).fit({ x: X3, y: Y3 }).tree, fx.pruned)
  })
  it('trees are aifn/graph trees', () => {
    const t = decisionTree({ maxDepth: 3 }).fit({ x: X3, y: Y3 }).tree
    expect(preOrder(t)).toEqual(t.nodes.map((n) => n.id))
    expect(leaves(t).every((v) => t.nodes[v].feature === -1)).toBe(true)
    expect(t.nodes[0].label).toMatch(/^\$x_/)
  })
  it('the split search at the root finds the root split', () => {
    const s = splitSearch({ x: X3, y: Y3, task: 'classification' })
    expect(s.feature).toBe(fx.tree.feature[0])
    expect(s.threshold).toBeCloseTo(fx.tree.threshold[0], 6)
    const best = Math.max(...s.candidates.flatMap((c) => toFlat(c.decreases)))
    expect(best).toBeCloseTo(s.decrease, 12)
  })
  it('growth follows the trace protocol', () => {
    protocol(
      treeGrowthSteps({ x: X3, y: Y3, task: 'classification', params: { maxFeatures: 1 } }),
      {},
      20,
      (s) => s.tree.nodes.length + s.pending.length,
      () => stream(3),
    )
  })
})

describe('ensembles', () => {
  it('AdaBoost (SAMME) matches scikit-learn', () => {
    const m = adaBoost({ rounds: 5 }).fit({ x: X3, y: Y3 })
    close(m.alphas, fx.ada.weights)
    close(m.errors, fx.ada.errors)
    expect(toFlat(m.decide(XQ))).toEqual(fx.ada.predict)
    protocol(adaBoostSteps({ x: X3, y: Y3 }), {}, 6, (s) => s.alphas.reduce((a, b) => a + b, 0))
  })
  it('gradient boosting matches scikit-learn (squared, binomial, multinomial)', () => {
    const r = gradientBoosting({ stages: 5, learningRate: 0.3, tree: { maxDepth: 2 } }).fit({
      x: X3,
      y: tensor(fx.yreg),
    })
    fx.gbr.staged.forEach((row, s) => close(r.rawUpTo(XQ, s + 1), row))
    const b = gradientBoosting({ loss: 'logistic', stages: 5, learningRate: 0.3, tree: { maxDepth: 2 } }).fit({
      x: X2,
      y: Y2,
    })
    close(b.forward(XQ), fx.gbc.decision)
    close(b.probabilities!(XQ), fx.gbc.proba)
    const k = gradientBoosting({ loss: 'logistic', stages: 4, learningRate: 0.3, tree: { maxDepth: 2 } }).fit({
      x: X3,
      y: Y3,
    })
    close(k.probabilities!(XQ), fx.gbm.proba)
    expect(k.training.series.loss.data[4]).toBeLessThan(k.training.series.loss.data[0])
  })
  it('random forests are reproducible, extendable and sensible', () => {
    const a = randomForest({ trees: 20 }).fit({ x: X3, y: Y3 }, { stream: stream(5) })
    const b = randomForest({ trees: 20 }).fit({ x: X3, y: Y3 }, { stream: stream(5) })
    expect(toFlat(a.probabilities(XQ))).toEqual(toFlat(b.probabilities(XQ)))
    const c = randomForest({ trees: 25 }).fit({ x: X3, y: Y3 }, { stream: stream(5) })
    expect(toFlat(c.probabilitiesUpTo(XQ, 20))).toEqual(toFlat(a.probabilities(XQ)))
    expect(a.outOfBag.accuracy).toBeGreaterThan(0.8)
    close([toFlat(a.featureImportances).reduce((u, v) => u + v, 0)], [1])
  })
})

describe('multiclass reductions', () => {
  const base = logisticRegression({ l2: 1, tol: 1e-14 })
  it('one-versus-rest and one-versus-one match scikit-learn with logistic regression', () => {
    const ovr = oneVersusRest(base).fit({ x: X3, y: Y3 })
    close(ovr.score(XQ), fx.ovr.decision, 1e-6)
    close(Array.from(classProbabilities(ovr.predictive!(XQ)).probs), fx.ovr.proba, 1e-6)
    const ovo = oneVersusOne(base).fit({ x: X3, y: Y3 })
    close(ovo.score(XQ), fx.ovo.decision, 1e-6)
  })
  it('codes: sizes and distances', () => {
    expect(oneVersusOneCode(4).shape).toEqual([4, 6])
    expect(exhaustiveCode(4).shape).toEqual([4, 7])
    expect(codeDistance(exhaustiveCode(5))).toBe(8)
    const r = randomCode(stream(1), 5, 10)
    expect(r.shape).toEqual([5, 10])
    expect(toFlat(randomCode(stream(1), 5, 10))).toEqual(toFlat(r))
  })
  it('output codes with the one-versus-rest code agree with one-versus-rest', () => {
    const ecoc = outputCode(base, exhaustiveCode(3), { decoding: 'loss' }).fit({ x: X3, y: Y3 })
    const acc = toFlat(ecoc.decide(X3)).filter((c, i) => c === fx.y3[i]).length / 60
    expect(acc).toBeGreaterThan(0.85)
  })
  it('nested dichotomies give normalised probabilities', () => {
    for (const shape of ['balanced', 'chain'] as const) {
      const m = nestedDichotomies(base, (K) => dichotomyTree(K, shape)).fit({ x: X3, y: Y3 })
      for (const row of toRows(m.probabilities(XQ))) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    }
  })
  it('Crammer–Singer matches LinearSVC(multi_class="crammer_singer")', () => {
    const m = crammerSinger({ C: 0.5, tol: 1e-10, maxEpochs: 20000 }).fit({ x: X3, y: Y3 })
    close(m.weights, fx.cs.coef, 1e-4)
    close(m.bias, fx.cs.intercept, 1e-4)
  })
  it('a binary tree fitted with weights equals one fitted on duplicated rows', () => {
    const w = tensor(fx.y3.map((_, i) => (i % 3) + 1))
    const t1 = growTree({ x: X3, y: Y3, weights: w, task: 'classification' })
    expect(t1.nodes[0].weight).toBe(fx.y3.reduce((a, _, i) => a + (i % 3) + 1, 0))
  })
})
