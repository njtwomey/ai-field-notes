/**
 * The hook between primitives and `aifn/autodiff`. A traced value wraps a number or tensor with a node id on a tape.
 * When a primitive receives a traced input it computes its forward value on the underlying data and asks the tape to
 * record the application, with the rule that pulls cotangents back through it (its vector–Jacobian product).
 *
 * This file defines only the protocol; the tape itself and the `grad` transforms live in `aifn/autodiff`. The design is
 * the tape-based reverse mode of Griewank and Walther (2008), "Evaluating Derivatives", ch. 3–4, with primitives whose
 * vjp rules are written in terms of other primitives, as in JAX (Bradbury et al., 2018), so that the backward pass can
 * itself be traced for higher derivatives.
 */

import type { Tensor } from './core'

const TRACED: unique symbol = Symbol.for('aifn.traced')

/** A number or tensor recorded on a tape: `value` is the underlying data and `id` its node on `tape`. */
export interface Traced<T extends number | Tensor = number | Tensor> {
  readonly [TRACED]: true
  readonly value: T
  readonly id: number
  readonly tape: Tape
}

/** Anything a primitive accepts: a number, a tensor, or either of them traced. */
export type Value = number | Tensor | Traced

/**
 * A vector–Jacobian product rule: given the cotangent of the output (same kind and shape as the output), the inputs
 * and the output, return one cotangent per input, each of the same kind and shape as its input. `null` for an input
 * means its cotangent is zero (the input does not affect the output, or does so only piecewise-constantly).
 * Inputs, output and cotangent may themselves be traced, so rules must use primitives, not raw data, for any quantity
 * that depends on them.
 */
export type Vjp = (cotangent: Value, inputs: readonly Value[], output: Value) => (Value | null)[]

/** Records primitive applications. Implemented by `aifn/autodiff`. */
export interface Tape {
  /**
   * Record that primitive `name` mapped `inputs` (as passed, some traced) to `output` (the raw forward value), and
   * return the traced output. `vjp` is `null` for a primitive without a derivative rule: differentiating through it
   * must then be an error, never a silent zero.
   */
  record(name: string, inputs: readonly Value[], output: number | Tensor, vjp: Vjp | null): Traced
}

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
