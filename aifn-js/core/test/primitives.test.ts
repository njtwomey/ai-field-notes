/**
 * The generated primitive suite (design K §10.1): one block per primitive in `registry.list()`, with inputs drawn from
 * the primitive's declared domains (or cases) on a stream keyed by its id, so every failure is reproducible. Per
 * primitive: number, rank-0, contiguous, transposed and sliced inputs agree; broadcasting over [] [3] [2,3] [2,1]; the
 * dtype rule; the vjp against central differences, and second order where declared; inputs unchanged by the call.
 *
 * Checks that fail today are listed in `KNOWN` with a reason and run as expected failures (`it.fails`), so the suite
 * is green, a fix is noticed (the expected failure then fails), and the list is printed at the end.
 */
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { grad } from 'aifn/foundation/autodiff'
import { stream } from 'aifn/foundation/random'
import {
  allclose,
  astype,
  broadcastShapes,
  broadcastTo,
  copy,
  fromData,
  mul,
  registry,
  reshape,
  shapeOfValue,
  slice,
  stack,
  sum,
  toFlat,
  transpose,
  unwrap,
  zeros,
  type Domain,
  type Draw,
  type Primitive,
  type PrimitiveCase,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn/foundation/tensor'
import { NotDifferentiableError } from 'aifn/foundation/errors'

// Load every core module (the leaves of the tree in modules.json, and the old paths while they exist) so that the
// registry holds all of core's primitives.
type Tree = {
  core: { families: { family: string; modules: { module: string; status?: string }[] }[] }
  aliases: { path: string }[]
}
const spec = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, '../../modules.json'), 'utf8')) as Tree
const coreModules = [
  ...spec.core.families.flatMap((f) =>
    f.modules.length
      ? f.modules.filter((m) => m.status !== 'gap').map((m) => `aifn/${f.family}/${m.module}`)
      : [`aifn/${f.family}`],
  ),
  ...spec.aliases.map((a) => a.path).filter((p) => p.startsWith('aifn/')),
]
await Promise.all(coreModules.map((m) => import(m)))

type Check = 'layouts' | 'broadcasting' | 'dtype' | 'vjp' | 'second order' | 'no rule'

/** Checks that fail today, by primitive id: phase 2 work. Each runs as an expected failure. */
const KNOWN: Readonly<Record<string, Partial<Record<Check, string>>>> = {
  'numerics/special/truncatedNormalVDraw': {
    'second order': 'its derivative is written with truncatedNormalWDraw, which has no derivative rule',
  },
}

const shapes: readonly (readonly number[])[] = [[], [3], [2, 3], [2, 1]]
const DEFAULT: Domain = { lo: -2, hi: 2 }

function drawFor(id: string): Draw {
  const s = stream(`primitives/${id}`)
  return (shape, { lo, hi, integer } = DEFAULT) => {
    const n = shape.reduce((a, b) => a * b, 1)
    const data = new Float64Array(n)
    for (let k = 0; k < n; k++)
      data[k] = integer ? Math.floor(lo + (Math.floor(hi) - lo + 1) * s.uniform()) : lo + (hi - lo) * s.uniform()
    return fromData(data, shape)
  }
}

const domainOf = (p: Primitive, i: number): Domain => {
  const d = p.test.domain
  return (Array.isArray(d) ? (d as Domain[])[i] : (d as Domain | undefined)) ?? DEFAULT
}
const isDiff = (p: Primitive, i: number) =>
  Array.isArray(p.differentiable) ? (p.differentiable[i] ?? true) : p.differentiable !== false
const flat = (x: Value): number[] => {
  const r = unwrap(x)
  return typeof r === 'number' ? [r] : toFlat(r)
}
function same(a: Value, b: Value, rtol = 1e-12, atol = 1e-12): void {
  expect(shapeOfValue(a)).toEqual(shapeOfValue(b))
  expect(allclose(unwrap(a), unwrap(b), { rtol, atol, equalNan: true })).toBe(true)
}

/** The same values as `t` in a non-contiguous layout: a transpose of the reversed layout. */
function transposed(t: Raw): Raw {
  if (typeof t === 'number' || t.shape.length < 2) return t
  return transpose(copy(transpose(t)))
}
/** The same values as `t` read with stride 2 along the last axis. */
function sliced(t: Raw): Raw {
  if (typeof t === 'number' || t.shape.length === 0) return t
  const s = t.shape
  const wide = reshape(stack([t, zeros(s)], -1), [...s.slice(0, -1), 2 * s[s.length - 1]])
  return slice(wide, ...s.slice(0, -1).map(() => null), [0, null, 2])
}

/** Apply, checking that the inputs are unchanged by the call. */
function run(p: Primitive, inputs: readonly Raw[], params: unknown): Raw {
  const before = inputs.map((x) => (typeof x === 'number' ? x : Float64Array.from(x.data)))
  const out = unwrap(p.apply(inputs, params))
  inputs.forEach((x, i) => {
    if (typeof x !== 'number') expect(Float64Array.from(x.data)).toEqual(before[i])
  })
  return out
}

/** The vjp of Σ w ⊙ f against central differences in every differentiable input, and second order if declared. */
function checkVjp(p: Primitive, c: PrimitiveCase, draw: Draw, order: 1 | 2): void {
  const params = c.params
  const out = run(p, c.inputs, params)
  const w = typeof out === 'number' ? 0.7 : draw([...out.shape])
  const rtol = p.test.rtol ?? 1e-5
  c.inputs.forEach((x0, i) => {
    if (!isDiff(p, i)) return
    const withArg = (x: Value) =>
      p.apply(
        c.inputs.map((v, j) => (j === i ? x : v)),
        params,
      )
    const loss = (x: Value) => sum(mul(w, withArg(x)))
    const u = typeof x0 === 'number' ? 1.3 : draw([...x0.shape])
    // Order 1 differentiates the loss; order 2 differentiates ⟨u, ∇loss⟩, so its gradient is a Hessian–vector product.
    const f = order === 1 ? loss : (x: Value) => sum(mul(u, grad(loss)(x) as Value))
    const analytic = flat(grad(f)(x0) as Value)
    const base = flat(x0)
    const numeric = base.map((v, k) => {
      const h = 1e-5 * Math.max(1, Math.abs(v))
      const at = (d: number) => {
        const e = base.slice()
        e[k] = v + d
        return flat(f(typeof x0 === 'number' ? e[0] : fromData(Float64Array.from(e), x0.shape)))[0]
      }
      return (at(h) - at(-h)) / (2 * h)
    })
    numeric.forEach((n, k) => expect(Math.abs(analytic[k] - n)).toBeLessThanOrEqual(1e-6 + rtol * Math.abs(n)))
  })
}

/** The checks of one primitive, each a function that throws on failure. */
function checksOf(p: Primitive): [Check, () => void][] {
  const draw = drawFor(p.id)
  const out: [Check, () => void][] = []
  if (p.kind === 'elementwise') {
    const n = p.arity as number
    const args = Array.from({ length: n }, (_, i) => draw([2, 3], domainOf(p, i)))
    const base = { inputs: args }
    out.push([
      'layouts',
      () => {
        const ref = run(p, args, undefined) as Tensor
        same(run(p, args.map(transposed), undefined), ref)
        same(run(p, args.map(sliced), undefined), ref)
        const values = args.map(toFlat)
        for (let k = 0; k < 6; k++) {
          const at = (flat(ref) as number[])[k]
          same(
            run(
              p,
              values.map((v) => v[k]),
              undefined,
            ),
            at,
          )
          same(
            run(
              p,
              values.map((v) => fromData(Float64Array.of(v[k]), [])),
              undefined,
            ),
            fromData(Float64Array.of(at), []),
          )
        }
      },
    ])
    if (n > 1)
      out.push([
        'broadcasting',
        () => {
          for (const a of shapes)
            for (const b of shapes) {
              const ins = args.map((_, i) => draw(i === 0 ? a : i === 1 ? b : [], domainOf(p, i)))
              const target = broadcastShapes(...ins.map((t) => t.shape))
              const expanded = ins.map((t) => copy(broadcastTo(t, target)))
              same(run(p, ins, undefined), run(p, expanded, undefined))
            }
        },
      ])
    out.push([
      'dtype',
      () => {
        expect(
          (
            run(
              p,
              args.map((t) => astype(t, 'float32')),
              undefined,
            ) as Tensor
          ).dtype,
        ).toBe('float32')
        const ints = args.map((_, i) => domainOf(p, i)).map((d) => ({ ...d, lo: Math.ceil(d.lo), integer: true }))
        if (ints.some((d) => d.lo > Math.floor(d.hi))) return
        const out = run(
          p,
          ints.map((d) => astype(draw([2, 3], d), 'int32')),
          undefined,
        ) as Tensor
        expect(out.dtype).toBe(p.dtype === 'same' ? 'int32' : 'float64')
      },
    ])
    if (p.differentiable !== false) out.push(['vjp', () => checkVjp(p, base, draw, 1)])
    if (p.test.secondOrder) out.push(['second order', () => checkVjp(p, base, draw, 2)])
  } else {
    for (const c of p.test.cases?.(draw) ?? []) {
      out.push([
        'layouts',
        () => {
          const ref = run(p, c.inputs, c.params)
          same(run(p, c.inputs.map(transposed), c.params), ref)
          same(run(p, c.inputs.map(sliced), c.params), ref)
        },
      ])
      if (p.differentiable !== false) out.push(['vjp', () => checkVjp(p, c, draw, 1)])
      if (p.test.secondOrder) out.push(['second order', () => checkVjp(p, c, draw, 2)])
    }
  }
  if (p.differentiable === false) {
    const c = p.kind === 'elementwise' ? { inputs: [draw([2, 3], domainOf(p, 0))] } : p.test.cases?.(draw)[0]
    if (c)
      out.push([
        'no rule',
        () =>
          expect(() => grad((x: Value) => sum(p.apply([x, ...c.inputs.slice(1)], c.params)))(c.inputs[0])).toThrow(
            NotDifferentiableError,
          ),
      ])
  }
  return out
}

const untested: string[] = []
const expected: string[] = []

describe('generated primitive suite', () => {
  it('has registered primitives with module-qualified ids', () => {
    const list = registry.list()
    expect(list.length).toBeGreaterThan(50)
    for (const p of list) expect(p.id).toBe(`${p.module}/${p.name}`)
  })

  for (const p of registry.list()) {
    const checks = checksOf(p)
    if (p.kind === 'general' && !p.test.cases) untested.push(`${p.id}: no test cases declared`)
    if (checks.length === 0) continue
    describe(p.id, () => {
      checks.forEach(([check, body], k) => {
        const reason = KNOWN[p.id]?.[check]
        if (reason) expected.push(`${p.id} (${check}): ${reason}`)
        ;(reason ? it.fails : it)(`${check} #${k}`, body)
      })
    })
  }

  afterAll(() => {
    const lines = [...expected.map((l) => `known failure  ${l}`), ...untested.map((l) => `untested       ${l}`)]
    const head = `primitive suite: ${registry.list().length} primitives, ${lines.length} gaps (phase 2)`
    process.stderr.write(`${[head, ...new Set(lines)].join('\n  ')}\n`)
  })
})
