/**
 * A minimal reverse-mode tape for testing primitives' derivative rules: it records every application, and `gradient`
 * sweeps the records backwards calling each vjp with the recorded (traced) inputs, as `aifn/autodiff` will. Vjps that
 * use primitives on traced inputs record further nodes, so a gradient is itself traced and can be differentiated again.
 */

import { expect } from 'vitest'
import {
  add,
  isTraced,
  mul,
  ones,
  size,
  sum,
  tensor,
  toFlat,
  traced,
  unwrap,
  type Raw,
  type Tape,
  type Tensor,
  type Traced,
  type Value,
  type Vjp,
} from 'aifn/foundation/tensor'

type Node = { name: string; inputs: readonly Value[]; output: Raw; vjp: Vjp | null; leaf: boolean }

export class MockTape implements Tape {
  nodes: Node[] = []

  /** A leaf variable to differentiate with respect to. */
  variable<T extends Raw>(value: T): Traced<T> {
    this.nodes.push({ name: 'variable', inputs: [], output: value, vjp: null, leaf: true })
    return traced(this, this.nodes.length - 1, value)
  }

  record(name: string, inputs: readonly Value[], output: Raw, vjp: Vjp | null): Traced {
    this.nodes.push({ name, inputs, output, vjp, leaf: false })
    return traced(this, this.nodes.length - 1, output)
  }

  /** Cotangents of `wrt` for the scalar `out`, possibly traced (for higher derivatives). */
  gradient(out: Traced, wrt: readonly Traced[]): Value[] {
    const cotangents = new Map<number, Value>()
    const seed = out.value
    cotangents.set(out.id, typeof seed === 'number' ? 1 : ones(seed.shape))
    for (let id = out.id; id >= 0; id--) {
      const g = cotangents.get(id)
      const node = this.nodes[id]
      if (g === undefined || node.leaf) continue
      if (node.vjp === null) throw new Error(`no derivative: ${node.name}`)
      const gs = node.vjp(g, node.inputs, traced(this, id, node.output))
      node.inputs.forEach((x, i) => {
        const gi = gs[i]
        if (gi === null || gi === undefined || !isTraced(x)) return
        const xi = x
        const prev = cotangents.get(xi.id)
        cotangents.set(xi.id, prev === undefined ? gi : add(prev, gi))
      })
    }
    return wrt.map((w) => {
      const g = cotangents.get(w.id)
      if (g !== undefined) return g
      const v = w.value
      return typeof v === 'number' ? 0 : mul(0, v)
    })
  }
}

/** Fixed, distinct weights that turn any output into a scalar test function Σ w·f. */
function weights(shape: readonly number[]): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from({ length: n }, (_, k) => 0.3 + 0.7 * Math.sin(1.7 * k + 0.4)),
    shape,
  )
}

function scalarise(y: Value): Value {
  const raw = unwrap(y)
  if (typeof raw === 'number') return mul(y, 0.9)
  return sum(mul(y, weights(raw.shape)))
}

function flat(v: Value): number[] {
  const raw = unwrap(v)
  return typeof raw === 'number' ? [raw] : toFlat(raw)
}

/** Replace element k of a raw value. */
function bump(x: Raw, k: number, h: number): Raw {
  if (typeof x === 'number') return x + h
  const values = toFlat(x)
  values[k] += h
  return tensor(values, x.shape)
}

/**
 * Check the vjps of `f` against central finite differences of Σ w·f(inputs), with respect to every input (or those
 * listed in `wrt`). Returns the analytic gradients.
 */
export function checkGradient(
  f: (...xs: Value[]) => Value,
  inputs: Raw[],
  { wrt, eps = 1e-6, tol = 1e-6 }: { wrt?: number[]; eps?: number; tol?: number } = {},
): number[][] {
  const tape = new MockTape()
  const vars = inputs.map((x) => tape.variable(x))
  const out = scalarise(f(...vars))
  expect(isTraced(out), 'the output should be traced').toBe(true)
  const which = wrt ?? inputs.map((_, i) => i)
  const grads = tape.gradient(
    out as Traced,
    which.map((i) => vars[i]),
  )
  const analytic = grads.map(flat)
  which.forEach((i, j) => {
    const x = inputs[i]
    const n = typeof x === 'number' ? 1 : size(x)
    expect(analytic[j].length, `gradient ${i} size`).toBe(n)
    for (let k = 0; k < n; k++) {
      const at = (h: number) => {
        const xs = inputs.map((v, q) => (q === i ? bump(v, k, h) : v))
        return unwrap(scalarise(f(...xs))) as number
      }
      const numeric = (at(eps) - at(-eps)) / (2 * eps)
      const scale = Math.max(1, Math.abs(numeric))
      expect(
        Math.abs(analytic[j][k] - numeric) / scale,
        `input ${i}, element ${k}: ${analytic[j][k]} vs ${numeric}`,
      ).toBeLessThan(tol)
    }
  })
  return analytic
}
