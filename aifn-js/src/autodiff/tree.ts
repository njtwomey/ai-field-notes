/**
 * Pytrees: nested arrays and plain objects whose leaves are numbers, tensors or traced values. The transforms flatten
 * their arguments into leaves, differentiate with respect to the leaves and rebuild results of the same structure.
 * Anything else inside a tree (strings, booleans, null, functions) is static: carried through, never differentiated.
 */

import { isTensor, isTraced, zeros, type Tensor, type Value } from 'aifn/tensor'

/** A nested structure of numbers, tensors and traced values (arrays and plain objects). */
export type Tree = Value | readonly Tree[] | { readonly [key: string]: Tree }

/** True for a differentiable leaf: a number, a tensor or a traced value. */
export function isLeaf(x: unknown): x is Value {
  return typeof x === 'number' || isTensor(x) || isTraced(x)
}

function isPlainObject(x: unknown): x is Record<string, unknown> {
  if (typeof x !== 'object' || x === null) return false
  const proto = Object.getPrototypeOf(x) as unknown
  return proto === Object.prototype || proto === null
}

/** A flattened tree: its leaves in depth-first order, their paths, and a function that rebuilds the structure. */
export type Flat = {
  leaves: Value[]
  /** A readable path per leaf, e.g. `x`, `w[1]`, `layer.bias`. */
  paths: string[]
  rebuild: (leaves: readonly unknown[]) => unknown
}

/** Flatten `tree` into its leaves (depth first, object keys in insertion order). `root` names the whole tree. */
export function flattenTree(tree: unknown, root = ''): Flat {
  const leaves: Value[] = []
  const paths: string[] = []
  const walk = (node: unknown, path: string): ((values: readonly unknown[]) => unknown) => {
    if (isLeaf(node)) {
      const k = leaves.length
      leaves.push(node)
      paths.push(path)
      return (values) => values[k]
    }
    if (Array.isArray(node)) {
      const parts = node.map((child, i) => walk(child, `${path}[${i}]`))
      return (values) => parts.map((p) => p(values))
    }
    if (isPlainObject(node)) {
      const parts = Object.entries(node).map(
        ([key, child]) => [key, walk(child, path ? `${path}.${key}` : key)] as const,
      )
      return (values) => Object.fromEntries(parts.map(([key, p]) => [key, p(values)]))
    }
    return () => node
  }
  const rebuild = walk(tree, root)
  return { leaves, paths, rebuild }
}

/** Zero of the same kind and shape as a raw leaf value. */
export function zerosLike(x: number | Tensor): number | Tensor {
  return typeof x === 'number' ? 0 : zeros(x.shape)
}
