/**
 * The registry of primitives: every operation defined with `definePrimitive` or `elementwise` (and so every one made
 * by the `defineOp` wrapper) is recorded here under its id,
 * `module/name` (`foundation/tensor/exp`, `numerics/special/erf`, `numerics/linalg/cholesky`). Registering an id twice throws, so "defined once" is
 * checked when modules load. `registry.list()` feeds the generated primitive tests and the lab's reference pages.
 *
 * A name without a module prefix (`'softplus'` rather than `'nn/functional/softplus'`) defines a local primitive that is not
 * registered: per-call maps (`map`, `map2`), derived derivative primitives and one-off primitives in figures.
 */

import type { Tensor } from './core'
import { AifnError } from 'aifn/foundation/errors'
import type { Value } from './tape'

/** An untraced value: a number or a tensor. */
export type Raw = number | Tensor

/** A general primitive applied to its inputs and parameters. */
export type Op<P> = (inputs: readonly Value[], params: P) => Value

/**
 * A general primitive's derivative rule: `vjp(cotangent, inputs, output, params)` returns one cotangent per input
 * (`null` for a zero cotangent). `params` are the primitive's non-differentiable arguments (axes, shapes, options).
 */
export type OpVjp<P> = (cotangent: Value, inputs: readonly Value[], output: Value, params: P) => (Value | null)[]

/**
 * The dtype rule of an elementwise primitive: `same` keeps the promoted input dtype (int32 stays int32, as for
 * negation or addition); `float` gives float64 for int32 inputs (as for exp or division).
 */
export type DTypeRule = 'same' | 'float'

/** An interval of test inputs for one argument: uniform on [lo, hi], or integers in it when `integer`. */
export type Domain = { readonly lo: number; readonly hi: number; readonly integer?: boolean }

/** Draws test tensors from a keyed stream: a tensor of `shape` with entries in `domain` (default [−2, 2]). */
export type Draw = (shape: readonly number[], domain?: Domain) => Tensor

/** One test input of a general primitive: its inputs and parameters. */
export type PrimitiveCase = { readonly inputs: readonly Raw[]; readonly params?: unknown }

/** What the generated tests need to exercise a primitive (design K §10.1). */
export type PrimitiveTest = {
  /** Elementwise primitives: the domain of each argument (one entry applies to all). Default [−2, 2]. */
  readonly domain?: Domain | readonly Domain[]
  /** General primitives: inputs and parameters, drawn with `draw`. Without cases a general primitive is not tested. */
  readonly cases?: (draw: Draw) => readonly PrimitiveCase[]
  /** Relative tolerance of the derivative checks against central differences (default 1e-5). */
  readonly rtol?: number
  /** The derivative rule is itself differentiable, so second derivatives are checked too. */
  readonly secondOrder?: boolean
}

/** Documentation of a primitive, for the lab's reference pages and the catalog. */
export type PrimitiveDoc = {
  /** One sentence. */
  readonly summary?: string
  /** The defining formula, in TeX. */
  readonly formula?: string
  /** The slug of the site note that defines it. */
  readonly note?: string
  /** Keys of `content/references.yaml`. */
  readonly references?: readonly string[]
}

/** A registered primitive. */
export interface Primitive<P = unknown> {
  /** `module/name`, unique. */
  readonly id: string
  /** The aifn module that defines it (the part of the id before the slash). */
  readonly module: string
  /** The name recorded on tapes and used in error messages (the part of the id after the slash). */
  readonly name: string
  readonly kind: 'elementwise' | 'general'
  /** Number of inputs; `variadic` for a list (concat, einsum). */
  readonly arity: number | 'variadic'
  /** Apply to inputs (numbers, tensors or traced values) and parameters. */
  readonly apply: Op<P>
  /** The forward rule on untraced inputs. */
  readonly impl: (inputs: Raw[], params: P) => Raw
  /** Which inputs have a derivative rule (none when there is no rule at all). */
  readonly differentiable: readonly boolean[] | boolean
  /** Elementwise primitives: the result dtype of int32 inputs. */
  readonly dtype?: DTypeRule
  readonly doc: PrimitiveDoc
  readonly test: PrimitiveTest
}

const table = new Map<string, Primitive>()

/** Split an id into module and name; null for a local (unprefixed) name. */
export function parseId(id: string): { module: string; name: string } | null {
  const slash = id.lastIndexOf('/')
  if (slash < 0) return null
  const module = id.slice(0, slash)
  const name = id.slice(slash + 1)
  if (!/^[a-z][a-z0-9-]*(\/[a-z][a-z0-9-]*)*$/.test(module) || name === '') {
    throw new AifnError('definePrimitive', `definePrimitive: invalid primitive id '${id}' (expected module/name)`)
  }
  return { module, name }
}

/** Add a primitive to the registry; an id already registered throws. For `definePrimitive` and `elementwise`. */
export function register<P>(p: Primitive<P>): void {
  if (table.has(p.id)) {
    throw new AifnError(
      'definePrimitive',
      `definePrimitive: primitive '${p.id}' is already registered (each primitive is defined once)`,
    )
  }
  table.set(p.id, p as Primitive)
}

/** The registered primitives. */
export const registry = {
  /** Every registered primitive, in registration order (the order modules loaded). */
  list(): Primitive[] {
    return [...table.values()]
  },
  /** The primitive with this id, or undefined. */
  get(id: string): Primitive | undefined {
    return table.get(id)
  },
  /** True when a primitive with this id is registered. */
  has(id: string): boolean {
    return table.has(id)
  },
}
