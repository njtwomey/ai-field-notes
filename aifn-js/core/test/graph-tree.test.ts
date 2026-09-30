/**
 * Smoke tests for the `Tree` type in aifn/graph and the producers that return one: Huffman trees (aifn/info), the
 * branch-and-bound search tree (aifn/programming) and breadth-/depth-first forests. References are hand-checked.
 */

import { describe, expect, it } from 'vitest'
import {
  ancestors,
  binaryTree,
  depth,
  depths,
  foldTree,
  fromEdges,
  height,
  inOrder,
  lca,
  leaves,
  levelOrder,
  mapTree,
  pathToRoot,
  postOrder,
  preOrder,
  rightChild,
  leftChild,
  spanningForestOf,
  spanningTreeOf,
  subtreeSize,
  treeFromChildren,
  treeFromNested,
  treeFromParents,
} from 'aifn/graph'
import { breadthFirstSearch, depthFirstSearch } from 'aifn/graph/traversal'
import { minimumSpanningTree } from 'aifn/graph/spanning-trees'
import { huffmanCode, huffmanSteps, huffmanTree } from 'aifn-applied/information/coding'
import { branchAndBound, branchAndBoundTree, milp } from 'aifn/optim/programming'
import { extend, run, seek, trace } from 'aifn/foundation/trace'

//        0
//      / | \
//     1  2  3
//    / \     \
//   4   5     6
//             |
//             7
const PARENTS = [-1, 0, 0, 0, 1, 1, 3, 6]

describe('graph: Tree constructors', () => {
  it('parents, children and nested agree', () => {
    const a = treeFromParents(PARENTS, { labels: PARENTS.map((_, i) => `n${i}`) })
    const b = treeFromChildren([[1, 2, 3], [4, 5], [], [6], [], [], [7], []])
    expect(a.root).toBe(0)
    expect(a.nodes.map((n) => n.children)).toEqual(b.nodes.map((n) => n.children))
    expect(a.nodes[4].label).toBe('n4')
    expect(a.edges[0]).toBeNull()
    const c = treeFromNested<{ v?: number }>({
      label: 'r',
      children: [
        { label: 'a', v: 1 },
        { label: 'b', edge: { label: 'e' } },
      ],
    })
    expect(preOrder(c).map((i) => c.nodes[i].label)).toEqual(['r', 'a', 'b'])
    expect(c.edges[2]).toEqual({ label: 'e' })
    expect(c.nodes[1].v).toBe(1)
    // A tree survives JSON.
    expect(JSON.parse(JSON.stringify(a))).toEqual(a)
  })

  it('rejects forests, cycles and unknown ids', () => {
    expect(() => treeFromParents([-1, -1])).toThrow(/one root/)
    expect(() => treeFromChildren([[1], [0]])).toThrow()
    expect(() => treeFromChildren([[3]])).toThrow(/unknown child/)
  })

  it('binary trees keep the side of a lone child', () => {
    const t = binaryTree({ label: '2', left: { label: '1' }, right: { label: '4', left: { label: '3' } } })
    expect(t.arity).toBe(2)
    expect(inOrder(t).map((i) => t.nodes[i].label)).toEqual(['1', '2', '3', '4'])
    const lone = binaryTree({ label: 'a', right: { label: 'b' } })
    expect(leftChild(lone, 0)).toBeNull()
    expect(rightChild(lone, 0)).toBe(1)
    expect(inOrder(lone).map((i) => lone.nodes[i].label)).toEqual(['a', 'b'])
  })
})

describe('graph: Tree queries and traversals', () => {
  const t = treeFromParents(PARENTS)
  it('queries', () => {
    expect(depth(t, 7)).toBe(3)
    expect(depths(t)).toEqual([0, 1, 1, 1, 2, 2, 2, 3])
    expect(height(t)).toBe(3)
    expect(height(t, 1)).toBe(1)
    expect(leaves(t)).toEqual([4, 5, 2, 7])
    expect(ancestors(t, 7)).toEqual([6, 3, 0])
    expect(pathToRoot(t, 5)).toEqual([5, 1, 0])
    expect(lca(t, 4, 5)).toBe(1)
    expect(lca(t, 4, 7)).toBe(0)
    expect(lca(t, 6, 7)).toBe(6)
    expect(subtreeSize(t)).toBe(8)
    expect(subtreeSize(t, 3)).toBe(3)
  })
  it('traversals, map and fold', () => {
    expect(preOrder(t)).toEqual([0, 1, 4, 5, 2, 3, 6, 7])
    expect(postOrder(t)).toEqual([4, 5, 1, 2, 7, 6, 3, 0])
    expect(levelOrder(t)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    const m = mapTree(t, (n) => ({ size: subtreeSize(t, n.id) }))
    expect(m.nodes.map((n) => n.size)).toEqual([8, 3, 1, 3, 1, 1, 2, 1])
    expect(m.nodes[3].children).toEqual([6])
    expect(foldTree(t, (n, xs: number[]) => n.id + xs.reduce((a, b) => a + b, 0))).toBe(28)
  })
})

describe('graph: trees from searches', () => {
  const g = fromEdges(6, [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
  ])
  it('breadth- and depth-first forests', () => {
    const bfs = breadthFirstSearch(g, 0)
    expect(bfs.trees).toHaveLength(1)
    const b = bfs.trees[0]
    expect(b.nodes.map((n) => n.vertex)).toEqual([0, 1, 3, 4, 2])
    expect(b.nodes[0].children.map((c) => b.nodes[c].vertex)).toEqual([1, 2])
    expect(b.edges[1]!.edge).toBe(0)
    const dfs = depthFirstSearch(g)
    // Vertex 5 is isolated: a second tree.
    expect(dfs.trees.map((t) => t.nodes.length)).toEqual([5, 1])
    expect(height(dfs.trees[0])).toBe(3)
  })
  it('a spanning tree from an edge set', () => {
    const w = fromEdges(
      4,
      [
        [0, 1, 1],
        [1, 2, 5],
        [0, 2, 2],
        [2, 3, 1],
      ],
      { directed: false, labels: ['a', 'b', 'c', 'd'] },
    )
    const mst = minimumSpanningTree(w)
    const t = spanningTreeOf(w, { edges: mst.edges.data }, 0)
    expect(t.nodes.map((n) => n.label)).toEqual(['a', 'b', 'c', 'd'])
    expect(t.edges.reduce((s, e) => s + (e?.weight ?? 0), 0)).toBe(mst.weight)
    expect(spanningForestOf(w, { edges: mst.edges.data })).toHaveLength(1)
  })
})

describe('info: Huffman tree and steps', () => {
  const p = [0.25, 0.25, 0.2, 0.15, 0.15]
  it('the tree spells the codewords', () => {
    const h = huffmanCode(p)
    const t = h.tree
    expect(t.root).toBe(t.nodes.length - 1)
    expect(t.arity).toBe(2)
    for (let k = 0; k < p.length; k++) {
      const path = pathToRoot(t, k).reverse()
      const bits = path
        .slice(1)
        .map((v) => t.edges[v]!.label)
        .join('')
      expect(bits).toBe(h.codewords[k])
      expect(t.nodes[k].symbol).toBe(k)
    }
    expect(t.nodes[t.root].probability).toBeCloseTo(1, 12)
    expect(huffmanCode([1]).tree.nodes).toHaveLength(1)
  })
  it('steps: K − 1 merges, codewords grow from the end, trace protocol', () => {
    const t = trace(huffmanSteps, p, 10)
    expect(t.steps).toHaveLength(p.length)
    expect(t.steps[1].merged).toEqual([3, 4])
    expect(t.steps[1].codewords).toEqual(['', '', '', '0', '1'])
    const last = t.steps[t.steps.length - 1]
    expect(last.codewords).toEqual(huffmanCode(p).codewords)
    expect(huffmanTree(last).nodes).toHaveLength(2 * p.length - 1)
    expect(seek(huffmanSteps, p, 2)).toEqual(run(huffmanSteps, p, 2))
    expect(extend(trace(huffmanSteps, p, 1), huffmanSteps, p, 3).steps).toEqual(t.steps)
  })
})

describe('programming: branch-and-bound search tree', () => {
  it('is a binary Tree with bounds and pruning reasons', () => {
    const problem = {
      c: [-5, -8],
      A_ub: [
        [1, 1],
        [5, 9],
      ],
      b_ub: [6, 45],
    }
    const r = milp(problem, { strategy: 'depth-first' })
    const t = r.searchTree
    expect(t.nodes).toHaveLength(r.tree.length)
    expect(t.arity).toBe(2)
    for (const n of t.nodes) {
      expect(n.status).toBe(r.tree[n.id].status)
      if (n.parent !== null) expect(n.slot).toBe(r.tree[n.id].branch!.direction === 'down' ? 0 : 1)
    }
    expect(t.nodes.some((n) => n.prunedBy !== null)).toBe(true)
    expect(t.edges[1]!.label).toMatch(/^\$x_\{\d\} \\(le|ge) \d+\$$/)
    // A partial state's tree.
    const s = run(branchAndBound, { problem }, 2)
    expect(branchAndBoundTree(s.nodes).nodes).toHaveLength(s.nodes.length)
  })
})
