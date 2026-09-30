/**
 * Variants: a figure offers several functions (or models, or methods), each with its own parameters. The reader picks
 * one and the controls change to its parameters. Adding a variant is one entry in `defineVariants`.
 *
 *   const functions = defineVariants({
 *     quadratic: { label: 'Quadratic', params: { a: slider(-2, 2, 1) }, f: (x: Value, p) => mul(p.a, square(x)) },
 *     softplus: { label: 'Softplus', params: { beta: slider(0.1, 5, 1) }, f: (x: Value, p) => ... },
 *   })
 *   const v = useVariants(functions)   // v.key, v.params (narrowed by v.key), v.f (bound to v.params), v.set, v.state
 *   <VariantControls variants={v} />   // the choice of variant, then its controls
 *
 * Each variant keeps its own values, so switching away and back restores them. Parameters in `shared` belong to every
 * variant and keep one value. The whole state is plain JSON (`v.state`, `v.setState`), e.g. for a URL.
 */
import { useCallback, useMemo, useState } from 'react'
import {
  coerce,
  coerceAll,
  initialValues,
  type AnyValues,
  type ParamDefs,
  type ParamValue,
  type Values,
} from './params'

/** One variant: a label, its parameters, and optionally a function of an input and its values. */
export type VariantSpec<P extends ParamDefs = ParamDefs, S extends ParamDefs = ParamDefs, X = never, R = unknown> = {
  label: string
  description?: string
  params: P
  f?: (x: X, params: Values<P> & Values<S>) => R
}

/** Declared variants and shared parameters, carrying the types of each variant's values. */
export type Variants<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R> = {
  specs: { [K in keyof V]: VariantSpec<V[K], S, X, R> }
  shared: S
}

/**
 * Declare variants. Every entry's `f` (optional; one signature for the family, e.g. `(x: Value, p) => Value`)
 * receives the entry's own values plus the shared ones, typed. Annotate `x`; put `params` before `f`.
 */
export function defineVariants<
  const V extends Record<string, ParamDefs>,
  X = never,
  R = unknown,
  const S extends ParamDefs = Record<never, never>,
>(specs: { [K in keyof V]: VariantSpec<V[K], S, X, R> }, shared?: S): Variants<V, S, X, R> {
  return { specs, shared: (shared ?? {}) as S }
}

/** The plain-JSON state of a variant set: the chosen key, each variant's values, and the shared values. */
export type VariantsJson = { key: string; values: Record<string, Record<string, ParamValue>>; shared: AnyValues }

/** What `useVariants` gives for one chosen variant K: a member of a union discriminated by `key`. */
export type ChosenVariant<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R, K extends keyof V> = {
  key: K
  spec: VariantSpec<V[K], S, X, R>
  /** The chosen variant's values and the shared ones. */
  params: Values<V[K]> & Values<S>
}

export type VariantsControl<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R> = {
  [K in keyof V]: ChosenVariant<V, S, X, R, K>
}[keyof V] & {
  variants: Variants<V, S, X, R>
  /** The chosen variant's `f` with its values bound, `v.f(x)`; undefined if it has none. */
  f: ((x: X) => R) | undefined
  setKey: (key: keyof V & string) => void
  /** Set one parameter of the chosen variant (or a shared one), clamped and snapped to its definition. */
  set: (name: string, value: ParamValue) => void
  /** The chosen variant's values and the shared ones, as a plain record (for controls and `when`). */
  values: AnyValues
  state: VariantsJson
  /** Restore a saved state; anything invalid falls back to its initial value. */
  setState: (state: unknown) => void
}

type Loose = Variants<Record<string, ParamDefs>, ParamDefs, unknown, unknown>

function initialState(variants: Loose, key?: string) {
  const keys = Object.keys(variants.specs)
  return {
    key: key && keys.includes(key) ? key : keys[0],
    values: Object.fromEntries(keys.map((k) => [k, initialValues(variants.specs[k].params)])),
    shared: initialValues(variants.shared) as AnyValues,
  } satisfies VariantsJson
}

/** The state of a variant set: which is chosen, and every variant's values. */
export function useVariants<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R>(
  variants: Variants<V, S, X, R>,
  initial?: { key?: keyof V & string; state?: unknown },
): VariantsControl<V, S, X, R> {
  const loose = variants as unknown as Loose
  const [state, setRaw] = useState<VariantsJson>(() =>
    initial?.state !== undefined ? restore(loose, initial.state) : initialState(loose, initial?.key),
  )
  const setKey = useCallback((key: string) => setRaw((s) => (key in loose.specs ? { ...s, key } : s)), [loose])
  const set = useCallback(
    (name: string, value: ParamValue) =>
      setRaw((s) => {
        const own = loose.specs[s.key].params[name]
        if (own) return { ...s, values: { ...s.values, [s.key]: { ...s.values[s.key], [name]: coerce(own, value) } } }
        const common = loose.shared[name]
        return common ? { ...s, shared: { ...s.shared, [name]: coerce(common, value) } } : s
      }),
    [loose],
  )
  const setState = useCallback((saved: unknown) => setRaw(restore(loose, saved)), [loose])
  return useMemo(() => {
    const spec = loose.specs[state.key]
    const params = { ...state.values[state.key], ...state.shared }
    const run = spec.f as ((x: unknown, p: unknown) => unknown) | undefined
    const f = run && ((x: unknown) => run(x, params))
    const control = { key: state.key, spec, params, f, variants, setKey, set, values: params, state, setState }
    return control as unknown as VariantsControl<V, S, X, R>
  }, [state, loose, variants, setKey, set, setState])
}

function restore(variants: Loose, saved: unknown): VariantsJson {
  const s = (typeof saved === 'object' && saved !== null ? saved : {}) as Partial<VariantsJson>
  const base = initialState(variants, typeof s.key === 'string' ? s.key : undefined)
  return {
    key: base.key,
    values: Object.fromEntries(
      Object.entries(variants.specs).map(([k, spec]) => [k, coerceAll(spec.params, s.values?.[k])]),
    ),
    shared: coerceAll(variants.shared, s.shared),
  }
}

/**
 * A plain conditional control set, without variants: typed values, a clamping setter, and JSON state. Render it with
 * `<ParamControls {...p} />`.
 */
export function useParams<P extends ParamDefs>(defs: P, saved?: unknown) {
  const [values, setValues] = useState<Values<P>>(() =>
    saved !== undefined ? coerceAll(defs, saved) : initialValues(defs),
  )
  const set = useCallback(
    (name: string, value: ParamValue) => {
      const def = defs[name]
      if (def) setValues((v) => ({ ...v, [name]: coerce(def, value) }))
    },
    [defs],
  )
  const setState = useCallback((s: unknown) => setValues(coerceAll(defs, s)), [defs])
  return useMemo(() => ({ defs, values, set, state: values, setState }), [defs, values, set, setState])
}
