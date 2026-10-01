import { describe, expect, it } from 'vitest'
import {
  countParams,
  leafCount,
  ravel,
  treeFlatten,
  treeLeaves,
  treeMap,
  treeUnflatten,
  treeZip,
  zerosLike,
} from 'aifn/foundation/pytree'
import { add, mul, tensor, toFlat, type Tensor, type Value } from 'aifn/foundation/tensor'
import { grad } from 'aifn/foundation/autodiff'

const tree = {
  w: tensor([
    [1, 2],
    [3, 4],
  ]),
  b: 0.5,
  layers: [tensor([1, 2, 3]), { gain: 2 }],
  name: 'net',
}

describe('flatten and unflatten', () => {
  it('leaves depth first with readable paths; statics kept in the structure', () => {
    const f = treeFlatten(tree, 'p')
    expect(f.paths).toEqual(['p.w', 'p.b', 'p.layers[0]', 'p.layers[1].gain'])
    expect(f.leaves).toHaveLength(4)
    expect(leafCount(f.treedef)).toBe(4)
    const back = treeUnflatten<typeof tree>(f.treedef, f.leaves)
    expect(back.name).toBe('net')
    expect(back.b).toBe(0.5)
    expect(toFlat(back.layers[0] as Tensor)).toEqual([1, 2, 3])
    // The structure is plain data.
    expect(JSON.parse(JSON.stringify(f.treedef))).toEqual(f.treedef)
  })
  it('a leaf alone is a tree of one leaf', () => {
    expect(treeFlatten(3).leaves).toEqual([3])
    expect(treeUnflatten(treeFlatten(3).treedef, [7])).toBe(7)
  })
})

describe('maps and zips', () => {
  it('treeMap keeps the structure; treeZip combines trees leaf by leaf', () => {
    const doubled = treeMap(tree, (v) => mul(v, 2))
    expect(doubled.b).toBe(1)
    expect(toFlat(doubled.w)).toEqual([2, 4, 6, 8])
    expect(doubled.name).toBe('net')
    const summed = treeZip([tree, doubled], ([x, y]) => add(x, y))
    expect(toFlat(summed.layers[0] as Tensor)).toEqual([3, 6, 9])
    expect(treeLeaves(tree).map((l) => l.path)).toEqual(['w', 'b', 'layers[0]', 'layers[1].gain'])
  })
  it('zerosLike and countParams', () => {
    const z = zerosLike(tree)
    expect(z.b).toBe(0)
    expect(toFlat(z.w)).toEqual([0, 0, 0, 0])
    expect(countParams(tree)).toBe(4 + 1 + 3 + 1)
  })
})

describe('ravel', () => {
  it('one vector and its inverse', () => {
    const r = ravel(tree)
    expect(Array.from(r.vector)).toEqual([1, 2, 3, 4, 0.5, 1, 2, 3, 2])
    const back = r.unravel(Float64Array.from(r.vector, (v) => -v))
    expect(back.b).toBe(-0.5)
    expect(toFlat(back.w)).toEqual([-1, -2, -3, -4])
    expect(back.name).toBe('net')
  })
})

describe('gradients have the structure of their argument', () => {
  it('grad of a pytree argument', () => {
    const gainOf = (p: typeof tree) => (p.layers[1] as { gain: number }).gain
    const loss = (p: typeof tree) => add(mul(p.b, p.b), mul(gainOf(p), gainOf(p))) as Value
    const g = grad(loss)(tree) as unknown as typeof tree
    expect(g.b).toBe(1)
    expect((g.layers[1] as { gain: number }).gain).toBe(4)
    expect(toFlat(g.w as Tensor)).toEqual([0, 0, 0, 0])
    expect(g.name).toBe('net')
  })
})
