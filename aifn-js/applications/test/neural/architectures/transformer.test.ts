import { describe, expect, it } from 'vitest'
import { TransformerBlock } from 'aifn-applied/neural/architectures'
import { gradCheck, valueAndGrad } from 'aifn/foundation/autodiff'
import { stream } from 'aifn/foundation/random'
import { mul, sum, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn/foundation/tensor'
import { fixture } from '../../fixtures'

type Nested = number | Nested[]
type Tree = { [k: string]: Tree | Nested }
type Case = {
  placement: 'pre' | 'post'
  norm: 'layer' | 'rms'
  causal: boolean
  params: Tree
  y: Nested
  gradX: Nested
}
const F = fixture<{ x: Nested; r: Nested; dModel: number; heads: number; hidden: number; blocks: Case[] }>(
  'neural/architectures',
)

/** JSON arrays → tensors, through a parameter tree. */
const toTensors = (t: Tree): unknown =>
  Object.fromEntries(
    Object.entries(t).map(([k, v]) => [k, Array.isArray(v) ? tensor(v as never) : toTensors(v as Tree)]),
  )
const flat = (v: Nested): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v])

describe('TransformerBlock', () => {
  const x = tensor(Array.from({ length: 4 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.sin(t + j))))

  for (const placement of ['pre', 'post'] as const) {
    it(`keeps its shape and has correct gradients (${placement}-norm)`, () => {
      const block = TransformerBlock(8, { heads: 2, causal: true, hidden: 16, placement })
      const params = block.init(stream('block'))
      expect((unwrap(block.apply(params, x)) as Tensor).shape).toEqual([4, 8])
      const report = gradCheck((p: typeof params) => sum(block.apply(p, x)), params, { rtol: 1e-4, atol: 1e-6 })
      expect(report.ok).toBe(true)
    })
  }

  for (const c of F.blocks)
    it(`matches torch (${c.placement}-norm, ${c.norm} norm, ${c.causal ? 'causal' : 'full'}): output and ∂(Σ y⊙r)/∂x`, () => {
      const block = TransformerBlock(F.dModel, {
        heads: F.heads,
        hidden: F.hidden,
        causal: c.causal,
        placement: c.placement,
        norm: c.norm,
      })
      const params = toTensors(c.params) as ReturnType<typeof block.init>
      const X = tensor(F.x as never)
      const R = tensor(F.r as never)
      const y = toFlat(unwrap(block.apply(params, X)) as Tensor)
      flat(c.y).forEach((v, i) => expect(Math.abs(y[i] - v)).toBeLessThan(1e-12))
      const { grad } = valueAndGrad((x: Value) => sum(mul(block.apply(params, x), R)))(X)
      const g = toFlat(grad as Tensor)
      flat(c.gradX).forEach((v, i) => expect(Math.abs(g[i] - v)).toBeLessThan(1e-11))
    })
})
