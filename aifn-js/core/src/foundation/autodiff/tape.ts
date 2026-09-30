/**
 * The tape behind `aifn/foundation/tensor`'s `Tape` hook, and the reverse sweep over it.
 *
 * Reverse mode follows Griewank and Walther (2008), "Evaluating Derivatives", ch. 3–4: the forward pass records every
 * primitive application in evaluation order (a topological order of the computation graph); the reverse sweep visits
 * the records backwards, pulling the output's cotangent (adjoint) back through each primitive's vector–Jacobian
 * product and summing the contributions that reach each value.
 *
 * Higher derivatives use one shared tape per outermost transform, as in PyTorch's `create_graph` and the tape design
 * of JAX's predecessor Autograd (Maclaurin, Duvenaud and Adams, 2015). A nested transform records its inputs as
 * identity nodes of the enclosing inputs, and runs its reverse sweep *on the tape*: the vjp rules are written with
 * primitives, so with traced inputs they record the backward computation as more nodes, which the outer transform can
 * then differentiate. The outermost sweep records nothing (its rules run against a `NullTape`), so first derivatives
 * cost one forward and one backward pass.
 */

import { NotDifferentiableError } from 'aifn/foundation/errors'
import {
  add,
  isTraced,
  traced,
  unwrap,
  withTape,
  type Tape,
  type Tensor,
  type Traced,
  type Value,
  type Vjp,
} from 'aifn/foundation/tensor'

/** One recorded primitive application (or an input leaf, with no inputs and no rule). */
export type TapeNode = {
  /** The primitive's name; `input` for an input of a transform. */
  op: string
  inputs: readonly Value[]
  /** The raw forward value. */
  output: number | Tensor
  /** The vector–Jacobian product rule, or null for a primitive without a derivative (or an input leaf). */
  vjp: Vjp | null
  leaf: boolean
  /** A label for inputs (their path in the argument tree). */
  label?: string
}

/** A tape that records primitive applications in evaluation order. */
export class GraphTape implements Tape {
  readonly nodes: TapeNode[] = []

  record(op: string, inputs: readonly Value[], output: number | Tensor, vjp: Vjp | null): Traced {
    this.nodes.push({ op, inputs, output, vjp, leaf: false })
    return traced(this, this.nodes.length - 1, output)
  }

  /**
   * An input of a transform. A raw value becomes a leaf; a value already traced on this tape (by an enclosing
   * transform) becomes an identity node of it, so that the inner transform differentiates with respect to this node
   * alone while the outer one still sees the dependence.
   */
  input(value: Value, label?: string): Traced {
    if (isTraced(value)) {
      this.nodes.push({ op: 'input', inputs: [value], output: value.value, vjp: (g) => [g], leaf: false, label })
    } else {
      this.nodes.push({ op: 'input', inputs: [], output: value, vjp: null, leaf: true, label })
    }
    return traced(this, this.nodes.length - 1, unwrap(value))
  }
}

/**
 * A tape that keeps nothing. Active during a reverse sweep that need not be differentiated again: vjp rules with
 * traced inputs record on it and get traced results back, which the sweep unwraps.
 */
class NullTape implements Tape {
  record(_op: string, _inputs: readonly Value[], output: number | Tensor): Traced {
    return traced(this, -1, output)
  }
}

const NULL_TAPE = new NullTape()

/**
 * Pull a cotangent `g` of node `id` back through that node's rule alone, without recording: one raw entry per input of
 * the node, null for a constant input or where the rule gives none. With g = 1 on a scalar node this is the node's
 * local partial derivative with respect to each input.
 */
export function pullback(tape: GraphTape, id: number, g: number | Tensor): (number | Tensor | null)[] {
  const node = tape.nodes[id]
  const vjp = node.vjp
  const onTape = (x: Value) => isTraced(x) && x.tape === tape
  if (vjp === null) return node.inputs.map(() => null)
  const inputs = node.inputs.map((x) => (onTape(x) ? x : unwrap(x)))
  const gs = withTape(NULL_TAPE, () => vjp(g, inputs, node.output))
  return node.inputs.map((x, i) => {
    const gi = gs[i]
    return gi === null || gi === undefined || !onTape(x) ? null : unwrap(gi)
  })
}

/** Raised when the reverse sweep meets a primitive without a derivative rule on a path to the output. */
export { NotDifferentiableError }

/** Options of a reverse sweep. */
export type SweepOptions = {
  /**
   * Run the rules on `tape` with the recorded (traced) inputs, so the cotangents are themselves traced and can be
   * differentiated again. Otherwise the sweep records nothing and returns raw cotangents.
   */
  createGraph: boolean
  /** Keep the cotangent of every visited node (for inspecting a graph). */
  keepAll?: boolean
}

/** The result of a reverse sweep. */
export type Sweep = {
  /** The cotangent of each `wrt` node, or null when the output does not depend on it. */
  cotangents: (Value | null)[]
  /** With `keepAll`, the cotangent of every node that received one, by node id. */
  all: Map<number, Value>
}

/**
 * Pull `seed` (the cotangent of `output`, same kind and shape) back to the nodes `wrt` of `tape`. Only nodes recorded
 * after the first of `wrt` and depending on one of them are visited, so values an enclosing transform traced are
 * treated as constants here. A primitive without a derivative rule on such a path is an error.
 */
export function sweep(
  tape: GraphTape,
  output: Value,
  seed: Value,
  wrt: readonly Traced[],
  options: SweepOptions,
): Sweep {
  const all = new Map<number, Value>()
  const none: Sweep = { cotangents: wrt.map(() => null), all }
  if (!isTraced(output) || output.tape !== tape || wrt.length === 0) return none
  const start = Math.min(...wrt.map((w) => w.id))
  const end = output.id
  if (end < start) return none

  // Forward pass over the ids: which nodes depend on a wrt node.
  const depends = new Uint8Array(end - start + 1)
  for (const w of wrt) depends[w.id - start] = 1
  const isWrt = new Uint8Array(end - start + 1)
  for (const w of wrt) isWrt[w.id - start] = 1
  const onTape = (x: Value): x is Traced => isTraced(x) && x.tape === tape && x.id >= start && x.id <= end
  for (let id = start; id <= end; id++) {
    if (depends[id - start]) continue
    for (const x of tape.nodes[id].inputs) {
      if (onTape(x) && depends[x.id - start]) {
        depends[id - start] = 1
        break
      }
    }
  }
  if (!depends[end - start]) return none

  const { createGraph, keepAll = false } = options
  const cotangent = new Map<number, Value>()
  cotangent.set(end, createGraph ? seed : unwrap(seed))
  const body = () => {
    for (let id = end; id >= start; id--) {
      const g = cotangent.get(id)
      if (g === undefined) continue
      if (keepAll) all.set(id, g)
      // A wrt node collects its cotangent and is not pulled back further: the inputs below it are outside this sweep.
      if (isWrt[id - start]) continue
      const node = tape.nodes[id]
      if (node.leaf) continue
      if (node.vjp === null) throw new NotDifferentiableError(node.op)
      // Inputs that do not depend on wrt are constants here: a first-order sweep passes them raw, so rules skip them
      // (e.g. `pow` computes no log a for a constant exponent). Inputs that do depend stay traced, so that a rule
      // without a derivative in that argument can raise its error.
      const inputs = createGraph
        ? node.inputs
        : node.inputs.map((x) => (onTape(x) && depends[x.id - start] ? x : unwrap(x)))
      const out = createGraph ? traced(tape, id, node.output) : node.output
      const gs = node.vjp(g, inputs, out)
      node.inputs.forEach((x, i) => {
        const gi = gs[i]
        if (gi === null || gi === undefined || !onTape(x) || !depends[x.id - start]) return
        const value = createGraph ? gi : unwrap(gi)
        const prev = cotangent.get(x.id)
        cotangent.set(
          x.id,
          prev === undefined ? value : createGraph ? add(prev, value) : (add(unwrap(prev), value) as Value),
        )
      })
    }
  }
  withTape(createGraph ? tape : NULL_TAPE, body)
  return { cotangents: wrt.map((w) => cotangent.get(w.id) ?? null), all }
}
