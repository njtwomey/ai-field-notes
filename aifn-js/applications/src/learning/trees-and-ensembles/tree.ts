/**
 * Decision trees by CART (Breiman, Friedman, Olshen and Stone, 1984, "Classification and Regression Trees"): greedy
 * binary splits x_f ≤ t chosen to maximise the decrease in weighted impurity (Gini, entropy or squared error), grown
 * depth first, and minimal cost-complexity pruning (ibid., ch. 3; Hastie, Tibshirani and Friedman, 2009, §9.2.2).
 *
 * The conventions follow scikit-learn's `DecisionTreeClassifier`/`DecisionTreeRegressor`: thresholds halfway between
 * consecutive distinct values, entropy in bits, nodes numbered in the order they are created (preorder, left first),
 * and the same cost-complexity path. A tree is plain data: `{ nodes, root }`, each node `{ id, parent, children, … }`.
 */

import type { Tree, TreeNode } from 'aifn/graph'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import type {
  AnyUnivariate,
  Decides,
  Estimator,
  Expects,
  FitOptions,
  Fitted,
  Predicts,
  Scores,
  Supervised,
  Trained,
} from 'aifn/learning/estimators'
import type { Status } from 'aifn/foundation/contracts'
import { type Stream, child, integers } from 'aifn/foundation/random'
import { run, trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { classLabels, inputs, matrix, probabilityModel, targets, values } from '../util'

/** The impurity a split decreases: Gini or entropy (classification), squared error (regression). */
export type Criterion = 'gini' | 'entropy' | 'squared'

/** What a decision-tree node records, on top of the `aifn/graph` tree structure. */
export interface SplitData {
  depth: number
  /** The split feature (−1 at a leaf). */
  feature: number
  /** The split threshold (NaN at a leaf); the left child (slot 0) takes x[feature] ≤ threshold. */
  threshold: number
  /** Number of training rows reaching the node, and their total sample weight. */
  count: number
  weight: number
  /** The node's impurity (Gini, entropy in bits, or variance). */
  impurity: number
  /** Classification: the weighted class shares [K]. Regression: the weighted mean target, one entry. */
  value: number[]
  /** Weighted impurity decrease of the split, (w_t·i(t) − w_L·i(L) − w_R·i(R)) / W (0 at a leaf). */
  decrease: number
}

/** A node of a decision tree: an `aifn/graph` `TreeNode` (id, parent, children, slot, label) with `SplitData`. */
export type DecisionNode = TreeNode<SplitData>

/**
 * A decision tree: an `aifn/graph` binary `Tree` (arity 2; `children` are `[left, right]` at a split, `[]` at a leaf)
 * whose nodes carry `SplitData` and a label (the split `$x_j \le t$`, or the leaf's prediction).
 */
export interface DecisionTree extends Tree<SplitData> {
  task: 'classification' | 'regression'
  criterion: Criterion
  /** Number of classes (1 for regression) and of features. */
  classes: number
  features: number
}

/** The node label: the split as TeX, or the leaf's class (or mean). */
export function nodeLabel(node: SplitData, task: 'classification' | 'regression'): string {
  const fmt = (v: number) => Number(v.toPrecision(3)).toString()
  if (node.feature >= 0) return `$x_{${node.feature}} \\le ${fmt(node.threshold)}$`
  if (task === 'regression') return `$${fmt(node.value[0])}$`
  let best = 0
  node.value.forEach((v, c) => {
    if (v > node.value[best]) best = c
  })
  return `class ${best}`
}

/** A tree from nodes (parents before children), with its edge list and arity. */
function asTree(
  nodes: DecisionNode[],
  p: { task: 'classification' | 'regression'; criterion: Criterion; K: number; d: number },
): DecisionTree {
  return {
    kind: 'tree',
    nodes,
    root: 0,
    edges: nodes.map((node) => (node.parent === null ? null : {})),
    arity: 2,
    task: p.task,
    criterion: p.criterion,
    classes: p.K,
    features: p.d,
  }
}

/** Hyperparameters of tree growth. */
export interface TreeParams {
  criterion?: Criterion
  /** Maximum depth (root at 0; default unlimited). */
  maxDepth?: number
  /** A node with fewer rows is not split (default 2). */
  minSamplesSplit?: number
  /** Every child keeps at least this many rows (default 1). */
  minSamplesLeaf?: number
  /** A split must decrease weighted impurity (as a fraction of the total weight) by at least this (default 0). */
  minImpurityDecrease?: number
  /**
   * Features searched at each node: all (default), a count, a fraction, or `sqrt`/`log2` of the number of features.
   * Subsets are drawn from the stream given to growth (random forests).
   */
  maxFeatures?: number | 'sqrt' | 'log2'
}

/** The data a tree is grown on: inputs [n, d], targets (labels 0 … K−1, or real values) and optional sample weights. */
export interface TreeProblem {
  x: Tensor
  y: Tensor
  weights?: Tensor
  task: 'classification' | 'regression'
  /** Number of classes (classification; default one more than the largest label). */
  classes?: number
  params?: TreeParams
}

// ── Impurity ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Impurity of a node from its weighted class totals (classification) or weighted sums (regression). */
function impurityOf(criterion: Criterion, stats: Float64Array, w: number): number {
  if (w <= 0) return 0
  if (criterion === 'squared') {
    // stats = [Σw y, Σw y²]
    const m = stats[0] / w
    return Math.max(stats[1] / w - m * m, 0)
  }
  let s = 0
  if (criterion === 'gini') {
    for (const c of stats) s += (c / w) ** 2
    return 1 - s
  }
  for (const c of stats) if (c > 0) s -= (c / w) * Math.log2(c / w)
  return s
}

/** Candidate splits of one feature at one node: every threshold considered and its weighted impurity decrease. */
export interface FeatureSplits {
  feature: number
  thresholds: Tensor
  /** w_t·i(t) − w_L·i(L) − w_R·i(R), divided by the total training weight (so they sum to the importance). */
  decreases: Tensor
}

/** The best split at a node, and every candidate that was considered (the split search exposed). */
export interface SplitSearch {
  /** −1 when no valid split exists. */
  feature: number
  threshold: number
  decrease: number
  /** Rows going left and right, in the node's row order. */
  left: Int32Array
  right: Int32Array
  candidates: FeatureSplits[]
}

interface Prepared {
  n: number
  d: number
  x: Float64Array
  y: Float64Array
  w: Float64Array
  totalWeight: number
  task: 'classification' | 'regression'
  K: number
  criterion: Criterion
  maxDepth: number
  minSamplesSplit: number
  minSamplesLeaf: number
  minImpurityDecrease: number
  maxFeatures: number
}

function prepare(problem: TreeProblem, where: string): Prepared {
  if (problem.x.shape.length !== 2) throw new Error(`${where}: expected x of shape [n, d]`)
  const [n, d] = problem.x.shape
  const x = values(problem.x)
  const y = values(problem.y)
  if (y.length !== n) throw new Error(`${where}: ${n} rows but ${y.length} targets`)
  const w = problem.weights ? values(problem.weights) : new Float64Array(n).fill(1)
  let totalWeight = 0
  for (const u of w) totalWeight += u
  const p = problem.params ?? {}
  const criterion = p.criterion ?? (problem.task === 'classification' ? 'gini' : 'squared')
  if ((criterion === 'squared') !== (problem.task === 'regression')) {
    throw new Error(`${where}: criterion ${criterion} does not fit a ${problem.task} tree`)
  }
  let K = 1
  if (problem.task === 'classification') {
    let top = 0
    for (const v of y) {
      if (!(Number.isInteger(v) && v >= 0)) throw new Error(`${where}: labels must be integers 0 … K−1`)
      top = Math.max(top, v + 1)
    }
    K = Math.max(problem.classes ?? 0, top, 2)
  }
  const mf = p.maxFeatures
  const maxFeatures =
    mf === undefined
      ? d
      : mf === 'sqrt'
        ? Math.max(1, Math.floor(Math.sqrt(d)))
        : mf === 'log2'
          ? Math.max(1, Math.floor(Math.log2(d)))
          : mf < 1
            ? Math.max(1, Math.floor(mf * d))
            : Math.min(d, Math.floor(mf))
  return {
    n,
    d,
    x,
    y,
    w,
    totalWeight,
    task: problem.task,
    K,
    criterion,
    maxDepth: p.maxDepth ?? Infinity,
    minSamplesSplit: p.minSamplesSplit ?? 2,
    minSamplesLeaf: p.minSamplesLeaf ?? 1,
    minImpurityDecrease: p.minImpurityDecrease ?? 0,
    maxFeatures,
  }
}

/** Weighted statistics of a set of rows: class totals [K], or [Σwy, Σwy²]. */
function statsOf(p: Prepared, rows: ArrayLike<number>): { stats: Float64Array; w: number } {
  const stats = new Float64Array(p.task === 'classification' ? p.K : 2)
  let w = 0
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r]
    w += p.w[i]
    if (p.task === 'classification') stats[p.y[i]] += p.w[i]
    else {
      stats[0] += p.w[i] * p.y[i]
      stats[1] += p.w[i] * p.y[i] * p.y[i]
    }
  }
  return { stats, w }
}

/** scikit-learn skips thresholds between values closer than this. */
const FEATURE_THRESHOLD = 1e-7

/** The best split of `rows` over `features`, with every candidate threshold and its decrease. */
function search(p: Prepared, rows: Int32Array, features: readonly number[]): SplitSearch {
  const { stats: total, w: W } = statsOf(p, rows)
  const parent = W * impurityOf(p.criterion, total, W)
  const S = total.length
  let best = { feature: -1, threshold: NaN, decrease: -Infinity, position: -1, order: new Int32Array(0) }
  const candidates: FeatureSplits[] = []
  const left = new Float64Array(S)
  const right = new Float64Array(S)
  for (const f of features) {
    const order = Int32Array.from(rows).sort((a, b) => p.x[a * p.d + f] - p.x[b * p.d + f] || a - b)
    left.fill(0)
    let wl = 0
    const thresholds: number[] = []
    const decreases: number[] = []
    for (let k = 0; k < order.length - 1; k++) {
      const i = order[k]
      wl += p.w[i]
      if (p.task === 'classification') left[p.y[i]] += p.w[i]
      else {
        left[0] += p.w[i] * p.y[i]
        left[1] += p.w[i] * p.y[i] * p.y[i]
      }
      const here = p.x[i * p.d + f]
      const next = p.x[order[k + 1] * p.d + f]
      if (next <= here + FEATURE_THRESHOLD) continue
      const nl = k + 1
      if (nl < p.minSamplesLeaf || order.length - nl < p.minSamplesLeaf) continue
      for (let s = 0; s < S; s++) right[s] = total[s] - left[s]
      const wr = W - wl
      const decrease = parent - wl * impurityOf(p.criterion, left, wl) - wr * impurityOf(p.criterion, right, wr)
      const threshold = here / 2 + next / 2
      thresholds.push(threshold)
      decreases.push(decrease / p.totalWeight)
      if (decrease > best.decrease) best = { feature: f, threshold, decrease, position: nl, order }
    }
    candidates.push({
      feature: f,
      thresholds: fromData(Float64Array.from(thresholds), [thresholds.length]),
      decreases: fromData(Float64Array.from(decreases), [decreases.length]),
    })
  }
  if (best.feature < 0) {
    return { feature: -1, threshold: NaN, decrease: 0, left: new Int32Array(0), right: new Int32Array(0), candidates }
  }
  // Children keep the node's row order.
  const f = best.feature
  const l: number[] = []
  const r: number[] = []
  for (const i of rows) (p.x[i * p.d + f] <= best.threshold ? l : r).push(i)
  return {
    feature: f,
    threshold: best.threshold,
    decrease: best.decrease / p.totalWeight,
    left: Int32Array.from(l),
    right: Int32Array.from(r),
    candidates,
  }
}

/**
 * The split search at one node: every candidate threshold of every feature with its weighted impurity decrease, and
 * the best one. `rows` defaults to all rows. This is exactly the search tree growth performs at a node.
 */
export function splitSearch(problem: TreeProblem, rows?: ArrayLike<number>): SplitSearch {
  const p = prepare(problem, 'splitSearch')
  const idx = rows ? Int32Array.from(rows) : Int32Array.from({ length: p.n }, (_, i) => i)
  return search(
    p,
    idx,
    Array.from({ length: p.d }, (_, j) => j),
  )
}

// ── Growth ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A node waiting to be expanded. */
interface Pending {
  rows: Int32Array
  depth: number
  parent: number
  side: 0 | 1
}

/** One state of tree growth: the nodes so far and the stack of nodes still to expand. */
export interface TreeGrowthState extends Status {
  /** Nodes created. */
  t: number
  /** The nodes created so far, as a (partial) tree: every created child is linked to its parent. */
  tree: DecisionTree
  /** Nodes waiting to be created, top of the stack last: their rows, depth and parent. */
  pending: { rows: Tensor; depth: number; parent: number; side: 0 | 1 }[]
  /** The node created in this step (−1 at the start), the rows that reached it and the search made there (null at a leaf that was not searched). */
  current: number
  rows: Tensor
  search: SplitSearch | null
  done: boolean
}

function nodeValue(p: Prepared, stats: Float64Array, w: number): number[] {
  if (p.task === 'regression') return [w > 0 ? stats[0] / w : 0]
  return Array.from(stats, (c) => (w > 0 ? c / w : 0))
}

/** The features searched at a node: all, or a random subset drawn from the stream of the step that creates it. */
function featuresFor(p: Prepared, s: Stream): number[] {
  const all = Array.from({ length: p.d }, (_, j) => j)
  if (p.maxFeatures >= p.d) return all
  const sub = child(s, 'features')
  for (let a = 0; a < p.maxFeatures; a++) {
    const b = a + integers(sub, p.d - a)
    ;[all[a], all[b]] = [all[b], all[a]]
  }
  return all.slice(0, p.maxFeatures).sort((a, b) => a - b)
}

/**
 * CART growth as a traceable algorithm: each step pops one node from the stack, decides whether it is a leaf (depth,
 * size, purity or no split worth making) and otherwise searches its best split and pushes the right and then the left
 * child. Nodes are therefore numbered in preorder. When `maxFeatures` is below the number of features, the step that
 * creates a node draws its feature subset from the step's stream. No start.
 */
export function treeGrowthSteps(problem: TreeProblem): Algorithm<void, TreeGrowthState> {
  const p = prepare(problem, 'treeGrowthSteps')
  const toPending = (q: Pending) => ({
    rows: fromData(q.rows, [q.rows.length]),
    depth: q.depth,
    parent: q.parent,
    side: q.side,
  })
  return {
    name: 'cart-growth',
    init: () => ({
      t: 0,
      tree: asTree([], p),
      pending: [toPending({ rows: Int32Array.from({ length: p.n }, (_, i) => i), depth: 0, parent: -1, side: 0 })],
      current: -1,
      rows: fromData(new Int32Array(0), [0]),
      search: null,
      done: false,
    }),
    step: (state, ctx) => {
      if (state.pending.length === 0) return { ...state, t: state.t + 1, done: true }
      const pending = state.pending.slice()
      const top = pending.pop()!
      const rows = Int32Array.from(top.rows.data as Int32Array)
      const id = state.tree.nodes.length
      const { stats, w } = statsOf(p, rows)
      const impurity = impurityOf(p.criterion, stats, w)
      let leaf =
        top.depth >= p.maxDepth ||
        rows.length < p.minSamplesSplit ||
        rows.length < 2 * p.minSamplesLeaf ||
        impurity <= Number.EPSILON
      let found: SplitSearch | null = null
      if (!leaf) {
        found = search(p, rows, featuresFor(p, ctx.stream))
        leaf = found.feature < 0 || found.decrease + Number.EPSILON < p.minImpurityDecrease
      }
      const node: DecisionNode = {
        id,
        parent: top.parent < 0 ? null : top.parent,
        children: [],
        ...(top.parent >= 0 && { slot: top.side }),
        label: '',
        depth: top.depth,
        feature: leaf ? -1 : found!.feature,
        threshold: leaf ? NaN : found!.threshold,
        count: rows.length,
        weight: w,
        impurity,
        value: nodeValue(p, stats, w),
        decrease: leaf ? 0 : found!.decrease,
      }
      node.label = nodeLabel(node, p.task)
      const nodes = state.tree.nodes.slice()
      nodes.push(node)
      if (top.parent >= 0) {
        const parent = { ...nodes[top.parent], children: nodes[top.parent].children.slice() }
        parent.children[top.side] = id
        nodes[top.parent] = parent
      }
      if (!leaf) {
        pending.push(toPending({ rows: found!.right, depth: top.depth + 1, parent: id, side: 1 }))
        pending.push(toPending({ rows: found!.left, depth: top.depth + 1, parent: id, side: 0 }))
      }
      return {
        t: state.t + 1,
        tree: asTree(nodes, p),
        pending,
        current: id,
        rows: fromData(rows, [rows.length]),
        search: found,
        done: pending.length === 0,
      }
    },
    done: (state) => state.done,
  }
}

/**
 * Grow a tree to completion (see `treeGrowthSteps`); `s` draws the feature subsets when `maxFeatures` is below the
 * number of features (a run of `treeGrowthSteps` on that root stream).
 */
export function growTree(s: Stream | undefined, problem: TreeProblem): DecisionTree {
  const n = problem.x.shape[0]
  return run(treeGrowthSteps(problem), undefined, 2 * n + 1, { stream: s }).tree
}

// ── Prediction ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The leaf each row of x [m, d] reaches, int32 [m]. */
export function applyTree(tree: DecisionTree, x: Tensor): Tensor {
  if (x.shape.length !== 2 || x.shape[1] !== tree.features) {
    throw new Error(`applyTree: expected x of shape [m, ${tree.features}]`)
  }
  const [m, d] = x.shape
  const v = values(x)
  const out = new Int32Array(m)
  for (let i = 0; i < m; i++) {
    let node = tree.nodes[tree.root]
    while (node.children.length)
      node = tree.nodes[v[i * d + node.feature] <= node.threshold ? node.children[0] : node.children[1]]
    out[i] = node.id
  }
  return fromData(out, [m])
}

/** The ids of the nodes on each row's path from the root to its leaf. */
export function decisionPath(tree: DecisionTree, x: Tensor): number[][] {
  const [m, d] = x.shape
  const v = values(x)
  return Array.from({ length: m }, (_, i) => {
    const path: number[] = []
    let node = tree.nodes[tree.root]
    path.push(node.id)
    while (node.children.length) {
      node = tree.nodes[v[i * d + node.feature] <= node.threshold ? node.children[0] : node.children[1]]
      path.push(node.id)
    }
    return path
  })
}

/** Leaf values for x [m, d]: class shares [m, K] (classification) or predictions [m] (regression). */
export function predictTree(tree: DecisionTree, x: Tensor): Tensor {
  const leaves = applyTree(tree, x).data as Int32Array
  const m = leaves.length
  if (tree.task === 'regression')
    return fromData(
      Float64Array.from(leaves, (id) => tree.nodes[id].value[0]),
      [m],
    )
  const K = tree.classes
  const out = new Float64Array(m * K)
  for (let i = 0; i < m; i++) for (let c = 0; c < K; c++) out[i * K + c] = tree.nodes[leaves[i]].value[c]
  return fromData(out, [m, K])
}

/** Impurity-based feature importances [d]: each feature's total weighted impurity decrease, normalised to sum to 1. */
export function featureImportances(tree: DecisionTree): Tensor {
  const out = new Float64Array(tree.features)
  for (const node of tree.nodes) if (node.children.length) out[node.feature] += node.decrease
  let s = 0
  for (const v of out) s += v
  if (s > 0) for (let j = 0; j < out.length; j++) out[j] /= s
  return fromData(out, [tree.features])
}

/** Number of leaves and depth of a tree. */
export function treeSize(tree: DecisionTree): { leaves: number; depth: number; nodes: number } {
  let leaves = 0
  let depth = 0
  for (const node of tree.nodes) {
    if (!node.children.length) leaves++
    depth = Math.max(depth, node.depth)
  }
  return { leaves, depth, nodes: tree.nodes.length }
}

// ── Cost-complexity pruning ──────────────────────────────────────────────────────────────────────────────────────

/** The weakest link and every internal node's g(t) = (R(t) − R(T_t)) / (|leaves(T_t)| − 1) in a (partly pruned) tree. */
function weakestLinks(tree: DecisionTree, cut: Set<number>) {
  const total = tree.nodes[tree.root].weight
  const R = (id: number) => (tree.nodes[id].weight / total) * tree.nodes[id].impurity
  const branch = new Map<number, { r: number; leaves: number }>()
  const visit = (id: number): { r: number; leaves: number } => {
    const node = tree.nodes[id]
    if (!node.children.length || cut.has(id)) return { r: R(id), leaves: 1 }
    const a = visit(node.children[0])
    const b = visit(node.children[1])
    const out = { r: a.r + b.r, leaves: a.leaves + b.leaves }
    branch.set(id, out)
    return out
  }
  const root = visit(tree.root)
  let best = -1
  let bestAlpha = Infinity
  for (const [id, b] of [...branch.entries()].sort((u, v) => u[0] - v[0])) {
    const g = (R(id) - b.r) / (b.leaves - 1)
    if (g < bestAlpha) {
      bestAlpha = g
      best = id
    }
  }
  return { best, alpha: bestAlpha, impurity: root.r, leaves: root.leaves }
}

/**
 * The minimal cost-complexity pruning path: the effective α at which each successive weakest link is pruned, and the
 * total leaf impurity Σ (w_t / W) i(t) of the pruned tree at each α (starting with α = 0 and the full tree), as
 * scikit-learn's `cost_complexity_pruning_path`.
 */
export function costComplexityPath(tree: DecisionTree): { alphas: Tensor; impurities: Tensor; leaves: Tensor } {
  const cut = new Set<number>()
  let w = weakestLinks(tree, cut)
  const alphas = [0]
  const impurities = [w.impurity]
  const leaves = [w.leaves]
  while (w.best >= 0) {
    cut.add(w.best)
    const alpha = w.alpha
    w = weakestLinks(tree, cut)
    alphas.push(alpha)
    impurities.push(w.impurity)
    leaves.push(w.leaves)
  }
  return {
    alphas: fromData(Float64Array.from(alphas), [alphas.length]),
    impurities: fromData(Float64Array.from(impurities), [impurities.length]),
    leaves: fromData(Int32Array.from(leaves), [leaves.length]),
  }
}

/**
 * The tree pruned at complexity α: weakest links are cut while their g(t) ≤ α. Nodes are renumbered in preorder, so
 * the result is a tree of the same shape as one grown to that size.
 */
export function pruneTree(tree: DecisionTree, alpha: number): DecisionTree {
  const cut = new Set<number>()
  for (;;) {
    const w = weakestLinks(tree, cut)
    if (w.best < 0 || w.alpha > alpha) break
    cut.add(w.best)
  }
  const nodes: DecisionNode[] = []
  const copy = (id: number, parent: number, side: number) => {
    const src = tree.nodes[id]
    const leaf = !src.children.length || cut.has(id)
    const nid = nodes.length
    const { slot: _slot, ...rest } = src
    void _slot
    nodes.push({
      ...rest,
      id: nid,
      parent: parent < 0 ? null : parent,
      children: [],
      ...(parent >= 0 && { slot: side }),
      feature: leaf ? -1 : src.feature,
      threshold: leaf ? NaN : src.threshold,
      decrease: leaf ? 0 : src.decrease,
      value: src.value.slice(),
    })
    nodes[nid].label = nodeLabel(nodes[nid], tree.task)
    if (parent >= 0) nodes[parent].children[side] = nid
    if (!leaf) {
      copy(src.children[0], nid, 0)
      copy(src.children[1], nid, 1)
    }
  }
  copy(tree.root, -1, 0)
  return { ...tree, nodes, root: 0, edges: nodes.map((node) => (node.parent === null ? null : {})) }
}

/** Data with optional per-row sample weights. */
export type WeightedData = Supervised<Tensor, Tensor> & { weights?: Tensor }

/** A fitted classification tree. */
export interface DecisionTreeModel
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, AnyUnivariate>,
    Trained<TreeGrowthState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'decision-tree'
  readonly tree: DecisionTree
  /** The tree before pruning (the same as `tree` when `pruneAlpha` is 0). */
  readonly grown: DecisionTree
  readonly classes: number
  readonly featureImportances: Tensor
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
      const training: Trace<TreeGrowthState> = trace(treeGrowthSteps(problem), undefined, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          nodes: (s) => s.tree.nodes.length,
          ...(options.trace?.record as Record<string, (s: TreeGrowthState, t: number) => number> | undefined),
        },
      })
      const grown: DecisionTree = training.final.tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const head = (q: Tensor) => {
        inputs(q, d, 'decisionTree')
        return values(predictTree(tree, q)).slice()
      }
      return {
        kind: 'model',
        name: 'decision-tree',
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
export interface RegressionTreeModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Expects<Tensor>, Trained<TreeGrowthState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'regression-tree'
  readonly tree: DecisionTree
  readonly grown: DecisionTree
  readonly featureImportances: Tensor
  apply(x: Tensor): Tensor
}

/**
 * A CART regression tree (squared error): each leaf predicts the weighted mean of its targets. `expect(x, f)` is the
 * weighted mean of f over the training targets in the leaf x reaches (the tree's empirical conditional law of y).
 */
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
      const training = trace(treeGrowthSteps(problem), undefined, 2 * n + 1, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
      })
      const grown: DecisionTree = training.final.tree
      const tree = pruneAlpha > 0 ? pruneTree(grown, pruneAlpha) : grown
      const predict = (q: Tensor) => {
        inputs(q, d, 'regressionTree')
        return predictTree(tree, q)
      }
      // The training targets (and weights) of each leaf, for expectations of functions of y.
      const yv = targets(y, n, 'regressionTree')
      const wv = weights ? values(weights) : null
      const members = new Map<number, number[]>()
      values(applyTree(tree, x)).forEach((leaf, i) => {
        const list = members.get(leaf) ?? []
        list.push(i)
        members.set(leaf, list)
      })
      const expect = (q: Tensor, f?: (y: number) => number) => {
        if (!f) return predict(q)
        const leaves = values(applyTree(tree, q))
        return fromData(
          Float64Array.from(leaves, (leaf) => {
            const rows = members.get(leaf) ?? []
            let total = 0
            let sum = 0
            for (const i of rows) {
              const w = wv ? wv[i] : 1
              total += w
              sum += w * f(yv[i])
            }
            return total > 0 ? sum / total : NaN
          }),
          [leaves.length],
        )
      }
      return {
        kind: 'model',
        name: 'regression-tree',
        tree,
        grown,
        featureImportances: featureImportances(tree),
        training,
        apply: (q: Tensor) => applyTree(tree, q),
        forward: predict,
        decide: predict,
        expect,
      }
    },
  }
}
