/** The Huffman code's tree (a `Tree` of aifn/graph) spells its codewords; its steps build it merge by merge. */
import { describe, expect, it } from 'vitest'
import { pathToRoot } from 'aifn/graph'
import { huffmanCode, huffmanSteps, huffmanTree } from 'aifn-applied/information/coding'
import { extend, run, seek, trace } from 'aifn/foundation/trace'

describe('Huffman tree and steps', () => {
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
    const t = trace(huffmanSteps(p), undefined, 10)
    expect(t.steps).toHaveLength(p.length)
    expect(t.steps[1].merged).toEqual([3, 4])
    expect(t.steps[1].codewords).toEqual(['', '', '', '0', '1'])
    const last = t.steps[t.steps.length - 1]
    expect(last.codewords).toEqual(huffmanCode(p).codewords)
    expect(huffmanTree(last).nodes).toHaveLength(2 * p.length - 1)
    expect(seek(huffmanSteps(p), undefined, 2)).toEqual(run(huffmanSteps(p), undefined, 2))
    expect(extend(trace(huffmanSteps(p), undefined, 1), huffmanSteps(p), 3).steps).toEqual(t.steps)
  })
})
