/**
 * CART in a few dozen lines, for the figures in this category: greedy binary splits on numeric features, chosen by
 * the largest decrease in weighted impurity. Labels are 0/1 for classification (a leaf stores the fraction of ones)
 * or real numbers for regression (a leaf stores the mean).
 */

export type Criterion = 'gini' | 'entropy' | 'squared'

export type TreeNode =
  | { leaf: true; value: number; n: number }
  | { leaf: false; feature: number; threshold: number; left: TreeNode; right: TreeNode; value: number; n: number }

export type TreeOptions = {
  maxDepth: number
  /** Smallest number of training points allowed in a leaf. */
  minLeaf?: number
  criterion: Criterion
}

/**
 * Impurity of a node times its size, from the count n, the sum s of labels and the sum q of squared labels. Scaling by
 * n makes the impurity of a split the plain sum of its children's values.
 */
function weightedImpurity(criterion: Criterion, n: number, s: number, q: number): number {
  if (n === 0) return 0
  if (criterion === 'squared') return q - (s * s) / n
  const p = s / n
  if (criterion === 'gini') return 2 * n * p * (1 - p)
  const h = (t: number) => (t > 0 ? -t * Math.log2(t) : 0)
  return n * (h(p) + h(1 - p))
}

export function fitTree(X: number[][], y: number[], options: TreeOptions): TreeNode {
  const { maxDepth, minLeaf = 1, criterion } = options
  const d = X[0]?.length ?? 0

  const grow = (idx: number[], depth: number): TreeNode => {
    const n = idx.length
    let s = 0
    let q = 0
    for (const i of idx) {
      s += y[i]
      q += y[i] * y[i]
    }
    const value = n ? s / n : 0
    const parent = weightedImpurity(criterion, n, s, q)
    if (depth >= maxDepth || n < 2 * minLeaf || parent <= 1e-12) return { leaf: true, value, n }

    let best = { gain: 1e-12, feature: -1, threshold: 0 }
    for (let f = 0; f < d; f++) {
      const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f])
      let sl = 0
      let ql = 0
      for (let k = 0; k < n - 1; k++) {
        const i = sorted[k]
        sl += y[i]
        ql += y[i] * y[i]
        const nl = k + 1
        const here = X[i][f]
        const next = X[sorted[k + 1]][f]
        if (nl < minLeaf || n - nl < minLeaf || here === next) continue
        const gain =
          parent - weightedImpurity(criterion, nl, sl, ql) - weightedImpurity(criterion, n - nl, s - sl, q - ql)
        if (gain > best.gain) best = { gain, feature: f, threshold: (here + next) / 2 }
      }
    }
    if (best.feature < 0) return { leaf: true, value, n }
    const left = idx.filter((i) => X[i][best.feature] <= best.threshold)
    const right = idx.filter((i) => X[i][best.feature] > best.threshold)
    return {
      leaf: false,
      feature: best.feature,
      threshold: best.threshold,
      left: grow(left, depth + 1),
      right: grow(right, depth + 1),
      value,
      n,
    }
  }

  return grow(
    y.map((_, i) => i),
    0,
  )
}

export function predictTree(node: TreeNode, x: number[]): number {
  let t = node
  while (!t.leaf) t = x[t.feature] <= t.threshold ? t.left : t.right
  return t.value
}

export function countLeaves(node: TreeNode): number {
  return node.leaf ? 1 : countLeaves(node.left) + countLeaves(node.right)
}
