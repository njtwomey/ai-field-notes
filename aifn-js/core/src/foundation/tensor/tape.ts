/**
 * The hook between primitives and `aifn/foundation/autodiff`. A traced value wraps a number or tensor with a node id on a tape.
 * When a primitive receives a traced input it computes its forward value on the underlying data and asks the tape to
 * record the application, with the rule that pulls cotangents back through it (its vector–Jacobian product).
 *
 * This file defines only the protocol; the tape itself and the `grad` transforms live in `aifn/foundation/autodiff`. The design is
 * the tape-based reverse mode of Griewank and Walther (2008), "Evaluating Derivatives", ch. 3–4, with primitives whose
 * vjp rules are written in terms of other primitives, as in JAX (Bradbury et al., 2018), so that the backward pass can
 * itself be traced for higher derivatives.
 */

import type { Tape, Tensor, Traced, TracedBrand, Value } from 'aifn/foundation/contracts'

// The traced-value protocol (`Traced`, `Value`, `Vjp`, `Tape`) is defined once, in `aifn/foundation/contracts`.
export type { Tape, Traced, Value, Vjp } from 'aifn/foundation/contracts'

const TRACED: TracedBrand = Symbol.for('aifn.traced') as TracedBrand

/** Wrap a raw value as node `id` of `tape`. For tape implementations. */
export function traced<T extends number | Tensor>(tape: Tape, id: number, value: T): Traced<T> {
  return { [TRACED]: true, value, id, tape }
}

/** True for a traced value. */
export function isTraced(x: unknown): x is Traced {
  return typeof x === 'object' && x !== null && (x as Record<symbol, unknown>)[TRACED] === true
}

/** The underlying number or tensor of a value (the value itself when it is not traced). */
export function unwrap(x: Value): number | Tensor {
  return isTraced(x) ? x.value : x
}

const tapes: Tape[] = []

/** The innermost tape made active by `withTape`, or null. */
export function currentTape(): Tape | null {
  return tapes.length > 0 ? tapes[tapes.length - 1] : null
}

/**
 * Run `fn` with `tape` active. Primitives record on the active tape whenever an input is traced (for any tape), which
 * lets a transform nest tapes; with no tape active they record on the tape of their first traced input.
 */
export function withTape<R>(tape: Tape, fn: () => R): R {
  tapes.push(tape)
  try {
    return fn()
  } finally {
    tapes.pop()
  }
}
