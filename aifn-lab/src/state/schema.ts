/**
 * Figure-state schemas (DESIGN.md §4): one declaration of a figure's parameters that yields its control rows, typed
 * values, drag handles, reset and URL state. The data half of every field is an aifn `Space` dimension
 * (`aifn/foundation/space`, design S §2.4), so a schema converts to a `Space` (`toSpace`) and URL values are decoded
 * through `clampReport`. The lab adds presentation only: labels (ReactNode), rows, `onChart`, and a revealing `toggle`
 * against a `setting` switch.
 *
 *   const state = useFigureState({
 *     input: variants({ normal: { mu: slider(-3, 3, 0), sd: slider(0.1, 3, 1) }, gamma: { … } }, { label: '1 · input' }),
 *     reveal: row('3 · reveal', { jacobian: toggle(false, 'Jacobian scaling'), draws: choice([1000, 5000]) }),
 *     x0: slider(-4, 4, 0.5, { onChart: true }),
 *   })
 *
 * The builders are the ones `useParams` and `defineVariants` always took (`slider`, `number`, `choice`), so old and
 * new code describe parameters one way.
 */
import type { ReactNode } from 'react'
import {
  bool,
  oneOf,
  real,
  space,
  subspace,
  variants as spaceVariants,
  type Condition,
  type DimSpec,
  type Space,
} from 'aifn/foundation/space'
import { niceStep } from '@lab/viz/format'
import { snapToStep } from './step'

export type { Condition } from 'aifn/foundation/space'

export type ParamValue = number | string | boolean
/** The values of a parameter set as a plain record, as a `when` closure sees them. */
export type AnyValues = Readonly<Record<string, unknown>>

type Common = {
  /** Shown above the control; defaults to the parameter's name. TeX is not parsed: pass a `Tex` element for maths. */
  label?: ReactNode
  /** One line of help for the control's tooltip and the catalog. */
  doc?: string
  /**
   * Show the control only when this holds. Prefer data (`when('shape', 'shifted')`), which serialises and lets the URL
   * decoder drop inactive keys; a closure over the set's values still works but is presentation only.
   */
  when?: Condition | ((values: AnyValues) => boolean)
  /**
   * The value has a place on a chart (a start point, a threshold) and is moved there by `state.handle(name)`; no
   * control row is drawn for it.
   */
  onChart?: boolean
}

export type SliderDef = Common & {
  kind: 'slider'
  min: number
  max: number
  initial: number
  /** Omitted: a nice 1-2-5 step from the range. */
  step?: number
  /** Step arrows either side of the track (default true). */
  steppable?: boolean
  /** Formats the value in the slider's field. */
  format?: (v: number) => string
}
export type NumberDef = Common & { kind: 'number'; initial: number; min?: number; max?: number; step?: number }

/** One choice: a value, what to show, and words that search should also match. */
export type ChoiceOption<T extends string | number> = {
  value: T
  label?: ReactNode
  keywords?: string
  disabled?: boolean
}
export type ChoiceDef<T extends string | number = string | number> = Common & {
  kind: 'choice'
  options: readonly (T | ChoiceOption<T>)[]
  initial: T
  /** Force a searchable combobox or a plain select; by default decided by the list's length. */
  searchable?: boolean
}
/** A boolean: a `setting` (a switch, e.g. log axes) or a revealing `toggle` (a button that shows an ingredient). */
export type SwitchDef = Common & { kind: 'switch'; initial: boolean; style?: 'switch' | 'reveal' }

/** One case of a `variants` field: a label, its parameters, and optionally a function of an input and its values. */
export type CaseDef<P extends ParamDefs = ParamDefs, S extends ParamDefs = ParamDefs, X = never, R = unknown> = {
  label: string
  description?: string
  params: P
  // A method, so that cases with different parameters are assignable to the general `CaseDef` (bivariance).
  f?(x: X, params: Values<P> & Values<S>): R
}

/**
 * A family choice whose cases have their own parameters (`variants`, `defineVariants`). Each case remembers its
 * values when the reader switches away and back; `shared` parameters belong to every case.
 */
export type VariantsDef<
  V extends Record<string, ParamDefs> = Record<string, ParamDefs>,
  S extends ParamDefs = ParamDefs,
  X = never,
  R = unknown,
> = {
  kind: 'variants'
  specs: { [K in keyof V]: CaseDef<V[K], S, X, R> }
  shared: S
  /** The case chosen at first (default the first). */
  initial?: keyof V & string
  /** The row's label. */
  label?: ReactNode
  /** The label of the case picker (default 'function'). */
  choiceLabel?: ReactNode
  doc?: string
  when?: Common['when']
  onChart?: undefined
}

/** A labelled row of fields; its values nest under its key (`state.reveal.jacobian`). */
export type RowDef<F extends ParamDefs = ParamDefs> = {
  kind: 'row'
  label?: ReactNode
  fields: F
  doc?: string
  when?: Common['when']
  onChart?: undefined
}

export type LeafDef = SliderDef | NumberDef | ChoiceDef | SwitchDef
export type ParamDef = LeafDef | VariantsDef | RowDef
// An interface, so that the recursion through rows and variants is allowed.
export interface ParamDefs {
  readonly [key: string]: ParamDef
}

// ── Values ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** What a variants field gives for one chosen case K: a member of a union discriminated by `key`. */
export type ChosenCase<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R, K extends keyof V> = {
  key: K
  label: string
  spec: CaseDef<V[K], S, X, R>
  /** The chosen case's values and the shared ones. */
  values: Values<V[K]> & Values<S>
  /** The same as `values` (the name `useVariants` used). */
  params: Values<V[K]> & Values<S>
  /** The chosen case's `f` with its values bound, `v.f(x)`; undefined if it has none. */
  f: ((x: X) => R) | undefined
}
export type VariantValue<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R> = {
  [K in keyof V]: ChosenCase<V, S, X, R, K>
}[keyof V]

/** The value type of one field. */
export type ValueOf<D> =
  D extends ChoiceDef<infer T>
    ? T
    : D extends SwitchDef
      ? boolean
      : D extends VariantsDef<infer V, infer S, infer X, infer R>
        ? VariantValue<V, S, X, R>
        : D extends RowDef<infer F>
          ? Values<F>
          : number
/** The typed values of a parameter set. */
export type Values<P> = { -readonly [K in keyof P]: ValueOf<P[K]> }

// ── Builders ─────────────────────────────────────────────────────────────────────────────────────────────────────────

type Extra<D> = Omit<D, 'kind' | 'min' | 'max' | 'initial' | 'options' | 'style'>
type LabelOr<D> = ReactNode | Extra<D>
const options = <D>(x: LabelOr<D> | undefined): Extra<D> =>
  x !== null && typeof x === 'object' && !('$$typeof' in x) && !Array.isArray(x)
    ? (x as Extra<D>)
    : ({ label: x } as unknown as Extra<D>)

/** A slider from `min` to `max` starting at `initial` (steppable by default). */
export function slider(min: number, max: number, initial: number, extra: Extra<SliderDef> = {}): SliderDef {
  return { kind: 'slider', min, max, initial, ...extra }
}

/** A typed number with − and + buttons, optionally bounded. */
export function number(initial: number, extra: Extra<NumberDef> & { min?: number; max?: number } = {}): NumberDef {
  return { kind: 'number', initial, ...extra }
}

/** A categorical choice among strings or numbers; `initial` defaults to the first option. */
export function choice<const T extends string | number>(
  list: readonly (T | ChoiceOption<T>)[],
  initial?: NoInfer<T>,
  extra: Extra<ChoiceDef<T>> = {},
): ChoiceDef<T> {
  const first = list[0]
  const value = initial ?? (typeof first === 'object' ? first.value : first)
  return { kind: 'choice', options: list, initial: value, ...extra } as ChoiceDef<T>
}

/**
 * A revealing toggle: a button that turns on the ingredient being taught (the Jacobian, the prior). Pass a label or
 * options.
 */
export function toggle(initial = false, extra?: LabelOr<SwitchDef>): SwitchDef {
  return { kind: 'switch', style: 'reveal', initial, ...options<SwitchDef>(extra) }
}

/** An on/off setting drawn as a switch (log axes, show the grid). Pass a label or options. */
export function setting(initial = false, extra?: LabelOr<SwitchDef>): SwitchDef {
  return { kind: 'switch', style: 'switch', initial, ...options<SwitchDef>(extra) }
}

/** A labelled row of fields, nested under its key. */
export function row<const F extends ParamDefs>(label: ReactNode, fields: F): RowDef<F> {
  return { kind: 'row', label, fields }
}

/** The condition "field `key` has the value `equals`", for a field's `when`. */
export function when(key: string, equals: string | number | boolean): Condition {
  return { key, equals }
}

type CaseInput = ParamDefs | CaseDef
type ParamsOfCase<C> = C extends { params: infer P extends ParamDefs; label: string } ? P : C

/**
 * A family choice whose cases have their own parameters: each case is a record of fields, or `{ label, params, f? }`.
 * `options.shared` holds parameters every case has.
 */
export function variants<const C extends Record<string, CaseInput>, const S extends ParamDefs = Record<never, never>>(
  cases: C,
  extra: {
    label?: ReactNode
    choiceLabel?: ReactNode
    initial?: keyof C & string
    shared?: S
    doc?: string
    when?: Common['when']
  } = {},
): VariantsDef<{ [K in keyof C]: ParamsOfCase<C[K]> }, S> {
  const specs = Object.fromEntries(
    Object.entries(cases).map(([key, c]) => [key, isCase(c) ? c : { label: key, params: c }]),
  )
  const { shared, ...rest } = extra
  return { kind: 'variants', specs, shared: (shared ?? {}) as S, ...rest } as unknown as VariantsDef<
    { [K in keyof C]: ParamsOfCase<C[K]> },
    S
  >
}

/**
 * A `{ label, params }` case rather than a bare record of fields. A bare record's fields are definitions (objects), so
 * `label` would not be a string; and `params` holds fields unless it is itself one definition (a string `kind`). A
 * case whose parameters include a field named `kind` is still a case.
 */
export const isCase = (c: CaseInput): c is CaseDef => {
  const { label, params } = c as Partial<CaseDef>
  return (
    typeof label === 'string' &&
    typeof params === 'object' &&
    params !== null &&
    typeof (params as { kind?: unknown }).kind !== 'string'
  )
}

/** Fields from an aifn `Space` (a registry's parameters): reals as sliders, choices, booleans as settings. */
export function fromSpace(s: Space): ParamDefs {
  const out: Record<string, ParamDef> = {}
  for (const [key, dim] of Object.entries(s.dims)) {
    const common = { label: dim.label ?? key, doc: dim.doc, when: dim.when }
    switch (dim.type) {
      case 'real':
        out[key] = slider(dim.min, dim.max, dim.default, { ...common, step: dim.step })
        break
      case 'int':
        out[key] = slider(dim.min, dim.max, dim.default, { ...common, step: 1 })
        break
      case 'choice':
        out[key] = choice(dim.options, dim.default, common)
        break
      case 'bool':
        out[key] = setting(dim.default, common)
        break
      case 'space':
        out[key] = { kind: 'row', label: common.label, fields: fromSpace(dim.of), when: dim.when }
        break
      case 'variants':
        out[key] = {
          kind: 'variants',
          specs: Object.fromEntries(
            Object.entries(dim.cases).map(([name, c]) => [name, { label: name, params: fromSpace(c) }]),
          ),
          shared: {},
          initial: dim.default,
          label: common.label,
          when: dim.when,
        }
        break
    }
  }
  return out
}

// ── Leaves: resolution and coercion ──────────────────────────────────────────────────────────────────────────────────

/** A slider's step: its own, or a nice 1-2-5 step from the range (about 100–250 positions). */
export const sliderStep = (d: SliderDef) => d.step ?? niceStep(d.min, d.max)

/** An option's value. */
export const optionValue = <T extends string | number>(o: T | ChoiceOption<T>): T =>
  typeof o === 'object' ? o.value : o

/** `value` made valid for a leaf (clamped, snapped, a known option), or the initial value if it cannot be. */
export function coerce(def: LeafDef, value: unknown): ParamValue {
  switch (def.kind) {
    case 'slider':
      return typeof value === 'number' && Number.isFinite(value)
        ? snapToStep(value, def.min, def.max, sliderStep(def))
        : snapToStep(def.initial, def.min, def.max, sliderStep(def))
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) return def.initial
      const [min, max] = [def.min ?? -Infinity, def.max ?? Infinity]
      return Number.isFinite(min) && def.step
        ? snapToStep(value, min, max, def.step)
        : Math.min(Math.max(value, min), max)
    }
    case 'choice': {
      const values = def.options.map(optionValue)
      if (values.includes(value as string | number)) return value as string | number
      // A URL gives strings: match a numeric option by its text.
      const byText = values.find((v) => String(v) === String(value))
      return byText ?? def.initial
    }
    case 'switch':
      if (typeof value === 'boolean') return value
      if (value === '1' || value === 'true') return true
      if (value === '0' || value === 'false') return false
      return def.initial
  }
}

/** True when a field applies under `values` (no `when`, or it holds). */
export function isActive(def: ParamDef, values: AnyValues): boolean {
  const w = def.when
  if (!w) return true
  return typeof w === 'function' ? w(values) : values[w.key] === w.equals
}

// ── The schema as an aifn Space ──────────────────────────────────────────────────────────────────────────────────────

const BIG = 1e15

function leafDim(def: LeafDef): DimSpec {
  const when = def.when && typeof def.when !== 'function' ? { when: def.when } : {}
  switch (def.kind) {
    case 'slider':
      return real(def.min, def.max, { default: coerce(def, def.initial) as number, ...when })
    case 'number':
      return real(def.min ?? -BIG, def.max ?? BIG, { default: def.initial, ...when })
    case 'choice':
      return oneOf(def.options.map(optionValue), { default: def.initial, ...when })
    case 'switch':
      return bool({ default: def.initial, ...when })
  }
}

/**
 * The data half of a schema as an aifn `Space`: sliders and numbers as reals, choices, switches as booleans, rows as
 * subspaces, variants as variants (each case's parameters with the shared ones). Closures in `when` are dropped (they
 * are presentation only); data conditions carry over.
 */
export function toSpace(defs: ParamDefs): Space {
  const dims: Record<string, DimSpec> = {}
  for (const [key, def] of Object.entries(defs)) {
    if (def.kind === 'row') dims[key] = subspace(toSpace(def.fields)) as unknown as DimSpec
    else if (def.kind === 'variants')
      dims[key] = spaceVariants(
        Object.fromEntries(
          Object.entries(def.specs).map(([name, c]) => [name, toSpace({ ...c.params, ...def.shared })]),
        ),
        { default: initialCase(def) },
      ) as unknown as DimSpec
    else dims[key] = leafDim(def)
  }
  return space(dims)
}

/** The case a variants field opens on. */
export const initialCase = (def: VariantsDef) => {
  const keys = Object.keys(def.specs)
  return def.initial && keys.includes(def.initial) ? def.initial : keys[0]
}
