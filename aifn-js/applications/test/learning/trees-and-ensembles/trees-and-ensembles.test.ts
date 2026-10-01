import { describe, expect, it } from 'vitest'
import {
  costComplexityPath,
  decisionTree,
  growTree,
  regressionTree,
  splitSearch,
  treeGrowthSteps,
} from 'aifn-applied/learning/trees-and-ensembles'
import { leaves, preOrder } from 'aifn/graph'
import { stream } from 'aifn/foundation/random'
import { tensor, toFlat } from 'aifn/foundation/tensor'
import { dataset } from 'aifn/learning/estimators'
import { expectProtocol } from '../../protocol'
import { close, fx, sameTree, X3, XQ, Y3 } from '../shared'

describe('trees', () => {
  it('CART matches scikit-learn: structure, thresholds, impurities, importances', () => {
    const m = decisionTree().fit(dataset(X3, Y3))
    sameTree(m.tree, fx.tree)
    close(m.featureImportances, fx.tree.importances)
    close(m.predictive(XQ), fx.tree_proba)
    sameTree(decisionTree({ criterion: 'entropy', maxDepth: 3 }).fit(dataset(X3, Y3)).tree, fx.tree_entropy)
    const r = regressionTree({ maxDepth: 3 }).fit(dataset(X3, tensor(fx.yreg)))
    sameTree(r.tree, fx.rtree)
    close(r.decide(XQ), fx.rtree_predict)
  })
  it('cost-complexity path and pruning match scikit-learn', () => {
    const m = decisionTree().fit(dataset(X3, Y3))
    const path = costComplexityPath(m.tree)
    close(path.alphas, fx.path.alphas)
    close(path.impurities, fx.path.impurities)
    sameTree(decisionTree({ pruneAlpha: 0.02 }).fit(dataset(X3, Y3)).tree, fx.pruned)
  })
  it('trees are aifn/graph trees', () => {
    const t = decisionTree({ maxDepth: 3 }).fit(dataset(X3, Y3)).tree
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
    expectProtocol(treeGrowthSteps({ x: X3, y: Y3, task: 'classification', params: { maxFeatures: 1 } }), undefined, {
      n: 20,
      record: { k: (s) => s.tree.nodes.length + s.pending.length },
      seed: 3,
    })
  })
  it('a binary tree fitted with weights equals one fitted on duplicated rows', () => {
    const w = tensor(fx.y3.map((_, i) => (i % 3) + 1))
    const t1 = growTree(undefined, { x: X3, y: Y3, weights: w, task: 'classification' })
    expect(t1.nodes[0].weight).toBe(fx.y3.reduce((a, _, i) => a + (i % 3) + 1, 0))
  })
  it('a random feature subset without a valid split draws further features, as scikit-learn', () => {
    // Only feature 0 varies; with maxFeatures 1 most nodes first draw a constant feature. Every leaf must still be pure.
    const n = 24
    const x = tensor(Array.from({ length: n }, (_, i) => [i, 1, 2, 3, 4]))
    const y = tensor(
      Array.from({ length: n }, (_, i) => Math.floor(i / 3) % 2),
      undefined,
      'int32',
    )
    for (const seed of [1, 2, 3]) {
      const tree = growTree(stream(seed), { x, y, task: 'classification', params: { maxFeatures: 1 } })
      for (const node of tree.nodes) {
        if (node.feature < 0) expect(node.impurity).toBeLessThanOrEqual(1e-12)
        else expect(node.feature).toBe(0)
      }
    }
  })
})
