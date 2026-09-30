/**
 * Parameters as pytrees: nested plain objects and arrays whose leaves are tensors (or numbers). Layers return their
 * parameters as trees, the autodiff transforms differentiate with respect to them, and the optimisers update them leaf
 * by leaf. These helpers walk trees in a fixed order (depth first, object keys in insertion order), the order
 * `aifn/autodiff` uses.
 */

import { isTensor, type Tensor } from 'aifn/tensor'

/**
 * A tree of parameters: tensors and numbers in nested objects and arrays. `undefined` marks an absent optional
 * parameter (a layer without bias); it is not a leaf.
 */
export type Params = Tensor | number | undefined | readonly Params[] | { readonly [key: string]: Params }

function isPlainObject(x: unknown): x is Record<string, unknown> {
  if (typeof x !== 'object' || x === null) return false
  const proto = Object.getPrototypeOf(x) as unknown
  return proto === Object.prototype || proto === null
}

/** A leaf with its path, e.g. `layers[0].weight`. */
export type Leaf = { path: string; value: Tensor | number }

/** The leaves of a tree with their paths, depth first. */
export function treeLeaves(tree: unknown, root = ''): Leaf[] {
  const out: Leaf[] = []
  const walk = (node: unknown, path: string) => {
    if (typeof node === 'number' || isTensor(node)) out.push({ path, value: node })
    else if (Array.isArray(node)) node.forEach((c, i) => walk(c, `${path}[${i}]`))
    else if (isPlainObject(node)) for (const [k, c] of Object.entries(node)) walk(c, path ? `${path}.${k}` : k)
  }
  walk(tree, root)
  return out
}

/** A leaf's value: a tensor or a number. */
export type LeafValue = Tensor | number

/**
 * Map the leaves of several trees of the same structure together: `treeZip([a, b], ([x, y], path) => …)` calls f on
 * each pair of corresponding leaves and rebuilds the structure of the first tree. Anything that is not a leaf, array
 * or plain object is carried over from the first tree.
 */
export function treeZip<T>(trees: readonly [T, ...unknown[]], f: (leaves: LeafValue[], path: string) => LeafValue): T {
  const walk = (nodes: unknown[], path: string): unknown => {
    const [node] = nodes
    if (typeof node === 'number' || isTensor(node)) return f(nodes as LeafValue[], path)
    if (Array.isArray(node)) {
      return node.map((_, i) =>
        walk(
          nodes.map((n) => (n as unknown[])[i]),
          `${path}[${i}]`,
        ),
      )
    }
    if (isPlainObject(node)) {
      return Object.fromEntries(
        Object.keys(node).map((k) => [
          k,
          walk(
            nodes.map((n) => (n as Record<string, unknown>)[k]),
            path ? `${path}.${k}` : k,
          ),
        ]),
      )
    }
    return node
  }
  return walk([...trees], '') as T
}

/** Map the leaves of one tree, keeping its structure. */
export function treeMap<T>(tree: T, f: (leaf: LeafValue, path: string) => LeafValue): T {
  return treeZip([tree], ([leaf], path) => f(leaf, path))
}

/** The number of scalar parameters in a tree. */
export function countParams(tree: unknown): number {
  return treeLeaves(tree).reduce(
    (n, { value }) => n + (typeof value === 'number' ? 1 : value.shape.reduce((a, b) => a * b, 1)),
    0,
  )
}
