/**
 * `aifn/foundation/space`: parameter spaces as data (design S §2.4). One schema describes hyperparameter searches,
 * recipe knobs, distribution and kernel parameters and figure controls; a `Space` holds no closures, so it
 * serialises, and its conditions are data too.
 *
 * - Dimensions: `real` (linear or log scale, optional step), `int`, `oneOf` (a choice among listed values), `bool`,
 *   `subspace` (a nested space) and `variants` (a family choice with parameters per case); `space(dims)` builds a
 *   space and `when(key, equals)` makes a dimension conditional on an earlier one.
 * - Values: `defaults(space)`, `clamp(space, values)` (fill, clip, round and snap into the space), `isDimActive`.
 * - Coordinates: `encode(space, values)` maps values into the unit cube [0, 1]^m (log dimensions in log space, integers
 *   and choices as equal bins) and `decode(space, u)` maps any point of the cube back, so `encodedSize` coordinates
 *   drawn uniformly decode to a uniform draw from the space. Inactive dimensions encode their defaults.
 * - Enumeration and draws: `grid(space, { points })` (every combination, conditions respected) and `sample(s, space)`.
 *
 * The encoding follows the unit-cube convention of Bayesian-optimisation libraries (Snoek, Larochelle and Adams, 2012,
 * "Practical Bayesian optimization of machine learning algorithms", NeurIPS; scikit-optimize's `Space.transform`);
 * grid and random search follow Bergstra and Bengio (2012), "Random search for hyper-parameter optimization", JMLR 13.
 */

import { DomainError, ShapeError } from 'aifn/foundation/errors'
import type { Condition, Dim, DimSpec, Size, Space } from 'aifn/foundation/contracts'
import { uniform, type Stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'

// Types defined once, in `aifn/foundation/contracts`.
export type { Condition, Dim, DimSpec, Space } from 'aifn/foundation/contracts'

// ── Values ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The value of a `variants` dimension: the chosen case and that case's parameters. */
export interface VariantValue {
  readonly case: string
  readonly params: SpaceValues
}

/** The value of one dimension. */
export type SpaceValue = number | string | boolean | SpaceValues | VariantValue

/** A point of a space: one value per active dimension, by key. */
export type SpaceValues = { readonly [key: string]: SpaceValue }

/** The value type of one dimension, read from its literal type. */
export type DimValue<D> = D extends { readonly type: 'real' | 'int' }
  ? number
  : D extends { readonly type: 'choice'; readonly options: readonly (infer O)[] }
    ? O
    : D extends { readonly type: 'bool' }
      ? boolean
      : D extends { readonly type: 'space'; readonly of: infer S extends Space }
        ? ValuesOf<S>
        : D extends { readonly type: 'variants'; readonly cases: infer C }
          ? { [K in keyof C & string]: { case: K; params: C[K] extends Space ? ValuesOf<C[K]> : never } }[keyof C &
              string]
          : never

type Dims<S extends Space> = S['dims']
type Conditional<S extends Space> = {
  [K in keyof Dims<S>]: Dims<S>[K] extends { when: Condition } ? K : never
}[keyof Dims<S>]

/** The values of a space, typed from its literal: conditional dimensions are optional. */
export type ValuesOf<S extends Space> = {
  -readonly [K in Exclude<keyof Dims<S>, Conditional<S>>]: DimValue<Dims<S>[K]>
} & { -readonly [K in Conditional<S>]?: DimValue<Dims<S>[K]> }

// ── Builders ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Presentation-neutral documentation any dimension may carry. */
export type DimDoc = { readonly label?: string; readonly doc?: string; readonly when?: Condition }

/** A space from its dimensions (keys in the order given, which is the order of `encode` and `grid`). */
export function space<const D extends Readonly<Record<string, DimSpec>>>(dims: D): { readonly dims: D } {
  return { dims }
}

/**
 * A real dimension on [min, max] with a default (the midpoint of the scale when omitted). `scale: 'log'` needs
 * min > 0 and spaces grids and draws evenly in log x; `step` snaps values to min + k·step.
 */
export function real(
  min: number,
  max: number,
  options: DimDoc & { default?: number; scale?: 'linear' | 'log'; step?: number; unit?: string } = {},
): DimSpec & { readonly type: 'real' } {
  if (!(max >= min)) throw new DomainError('real', `real: needs min ≤ max, got [${min}, ${max}]`)
  if (options.scale === 'log' && !(min > 0)) throw new DomainError('real', 'real: a log scale needs min > 0')
  const mid = options.scale === 'log' ? Math.sqrt(min * max) : (min + max) / 2
  return { ...options, type: 'real', min, max, default: options.default ?? mid }
}

/** An integer dimension on {min, …, max} with a default (min when omitted). */
export function int(
  min: number,
  max: number,
  options: DimDoc & { default?: number } = {},
): DimSpec & { readonly type: 'int' } {
  if (!(Number.isInteger(min) && Number.isInteger(max) && max >= min))
    throw new DomainError('int', `int: needs integers min ≤ max, got [${min}, ${max}]`)
  return { ...options, type: 'int', min, max, default: options.default ?? min }
}

/** A choice among listed values (strings or numbers), with a default (the first when omitted). */
export function oneOf<const O extends readonly (string | number)[]>(
  options: O,
  doc: DimDoc & { default?: O[number] } = {},
): DimSpec & { readonly type: 'choice'; readonly options: O } {
  if (options.length === 0) throw new DomainError('oneOf', 'oneOf: needs at least one option')
  return { ...doc, type: 'choice', options, default: doc.default ?? options[0] }
}

/** A boolean dimension, default false. */
export function bool(options: DimDoc & { default?: boolean } = {}): DimSpec & { readonly type: 'bool' } {
  return { ...options, type: 'bool', default: options.default ?? false }
}

/** A nested space, e.g. a kernel's sub-kernel parameters. */
export function subspace<const S extends Space>(of: S, doc: DimDoc = {}): DimSpec & { readonly type: 'space'; of: S } {
  return { ...doc, type: 'space', of }
}

/** A family choice whose cases have their own parameters, with a default case (the first when omitted). */
export function variants<const C extends Readonly<Record<string, Space>>>(
  cases: C,
  doc: DimDoc & { default?: keyof C & string } = {},
): DimSpec & { readonly type: 'variants'; readonly cases: C } {
  const names = Object.keys(cases)
  if (names.length === 0) throw new DomainError('variants', 'variants: needs at least one case')
  return { ...doc, type: 'variants', cases, default: doc.default ?? names[0] }
}

/** The condition "dimension `key` has the value `equals`", for `DimDoc.when`. `key` must come earlier in the space. */
export function when(key: string, equals: string | number | boolean): Condition {
  return { key, equals }
}

// ── Values ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** True when a dimension applies under `values`: it has no condition, or its condition holds. */
export function isDimActive(dim: DimSpec, values: SpaceValues): boolean {
  return dim.when === undefined || values[dim.when.key] === dim.when.equals
}

function clampDim(dim: Dim, v: unknown): SpaceValue {
  switch (dim.type) {
    case 'real': {
      if (typeof v !== 'number' || Number.isNaN(v)) return dim.default
      let x = Math.min(dim.max, Math.max(dim.min, v))
      if (dim.step !== undefined && dim.step > 0) {
        x = dim.min + Math.round((x - dim.min) / dim.step) * dim.step
        if (x > dim.max) x -= dim.step
      }
      return x
    }
    case 'int':
      return typeof v === 'number' && !Number.isNaN(v)
        ? Math.min(dim.max, Math.max(dim.min, Math.round(v)))
        : dim.default
    case 'choice':
      return dim.options.includes(v as string | number) ? (v as string | number) : dim.default
    case 'bool':
      return typeof v === 'boolean' ? v : dim.default
    case 'space':
      return clamp(dim.of, typeof v === 'object' && v !== null ? (v as SpaceValues) : {})
    case 'variants': {
      const given = typeof v === 'object' && v !== null ? (v as Partial<VariantValue>) : {}
      const name = typeof given.case === 'string' && given.case in dim.cases ? given.case : dim.default
      return { case: name, params: clamp(dim.cases[name], given.params ?? {}) }
    }
  }
}

/**
 * The nearest point of the space to `values`: missing or invalid values take their defaults, numbers are clipped to
 * their range (integers rounded, stepped reals snapped), unknown keys and inactive dimensions are dropped.
 */
export function clamp<S extends Space>(space: S, values: Readonly<Record<string, unknown>>): ValuesOf<S>
export function clamp(space: Space, values: Readonly<Record<string, unknown>>): SpaceValues
export function clamp(space: Space, values: Readonly<Record<string, unknown>>): SpaceValues {
  const out: Record<string, SpaceValue> = {}
  for (const [key, dim] of Object.entries(space.dims)) if (isDimActive(dim, out)) out[key] = clampDim(dim, values[key])
  return out
}

/** Every active dimension at its default. */
export function defaults<S extends Space>(space: S): ValuesOf<S>
export function defaults(space: Space): SpaceValues
export function defaults(space: Space): SpaceValues {
  return clamp(space, {})
}

// ── Unit-cube coordinates ────────────────────────────────────────────────────────────────────────────────────────────

function dimSize(dim: Dim): Size {
  if (dim.type === 'space') return encodedSize(dim.of)
  if (dim.type === 'variants') return 1 + Object.values(dim.cases).reduce((n, c) => n + encodedSize(c), 0)
  return 1
}

/** The number of unit-cube coordinates of a space (a variants dimension holds one for the case plus every case's). */
export function encodedSize(space: Space): Size {
  return Object.values(space.dims).reduce((n, d) => n + dimSize(d), 0)
}

/** Which of `count` equal bins of [0, 1] holds u (u = 1 falls in the last). */
const bin = (u: number, count: number) => Math.min(count - 1, Math.max(0, Math.floor(u * count)))

function encodeDim(dim: Dim, v: SpaceValue, out: number[]): void {
  switch (dim.type) {
    case 'real': {
      const x = v as number
      if (dim.max === dim.min) out.push(0.5)
      else if (dim.scale === 'log') out.push(Math.log(x / dim.min) / Math.log(dim.max / dim.min))
      else out.push((x - dim.min) / (dim.max - dim.min))
      return
    }
    case 'int':
      out.push(((v as number) - dim.min + 0.5) / (dim.max - dim.min + 1))
      return
    case 'choice':
      out.push((dim.options.indexOf(v as string | number) + 0.5) / dim.options.length)
      return
    case 'bool':
      out.push(v ? 0.75 : 0.25)
      return
    case 'space':
      encodeInto(dim.of, v as SpaceValues, out)
      return
    case 'variants': {
      const names = Object.keys(dim.cases)
      const value = v as VariantValue
      out.push((names.indexOf(value.case) + 0.5) / names.length)
      for (const name of names)
        encodeInto(dim.cases[name], name === value.case ? value.params : defaults(dim.cases[name]), out)
    }
  }
}

function encodeInto(space: Space, values: SpaceValues, out: number[]): void {
  const full = clamp(space, values)
  const fallback = defaults(space)
  for (const [key, dim] of Object.entries(space.dims))
    encodeDim(dim, full[key] ?? fallback[key] ?? clampDim(dim, undefined), out)
}

/** The unit-cube coordinates [encodedSize] of a point (clamped first); inactive dimensions encode their defaults. */
export function encode(space: Space, values: SpaceValues): Float64Array {
  const out: number[] = []
  encodeInto(space, values, out)
  return Float64Array.from(out)
}

function decodeDim(dim: Dim, u: ArrayLike<number>, at: { i: number }): SpaceValue {
  switch (dim.type) {
    case 'real': {
      const t = Math.min(1, Math.max(0, u[at.i++]))
      const x = dim.scale === 'log' ? dim.min * (dim.max / dim.min) ** t : dim.min + t * (dim.max - dim.min)
      return clampDim(dim, x)
    }
    case 'int':
      return dim.min + bin(u[at.i++], dim.max - dim.min + 1)
    case 'choice':
      return dim.options[bin(u[at.i++], dim.options.length)]
    case 'bool':
      return u[at.i++] >= 0.5
    case 'space':
      return decodeFrom(dim.of, u, at)
    case 'variants': {
      const names = Object.keys(dim.cases)
      const chosen = names[bin(u[at.i++], names.length)]
      let params: SpaceValues = {}
      for (const name of names) {
        const p = decodeFrom(dim.cases[name], u, at)
        if (name === chosen) params = p
      }
      return { case: chosen, params }
    }
  }
}

function decodeFrom(space: Space, u: ArrayLike<number>, at: { i: number }): SpaceValues {
  const out: Record<string, SpaceValue> = {}
  for (const [key, dim] of Object.entries(space.dims)) {
    const v = decodeDim(dim, u, at)
    if (isDimActive(dim, out)) out[key] = v
  }
  return out
}

/** The point of the space at unit-cube coordinates u [encodedSize] (each clipped to [0, 1]). */
export function decode<S extends Space>(space: S, u: ArrayLike<number>): ValuesOf<S>
export function decode(space: Space, u: ArrayLike<number>): SpaceValues
export function decode(space: Space, u: ArrayLike<number>): SpaceValues {
  const m = encodedSize(space)
  if (u.length !== m) throw new ShapeError('decode', `decode: the space has ${m} coordinates, got ${u.length}`)
  return decodeFrom(space, u, { i: 0 })
}

// ── Enumeration and draws ────────────────────────────────────────────────────────────────────────────────────────────

function dedupe(values: SpaceValue[]): SpaceValue[] {
  const seen = new Set<string>()
  return values.filter((v) => {
    const k = JSON.stringify(v)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

function dimGrid(dim: Dim, points: Size): SpaceValue[] {
  switch (dim.type) {
    case 'real':
    case 'int': {
      if (dim.type === 'int' && dim.max - dim.min + 1 <= points)
        return Array.from({ length: dim.max - dim.min + 1 }, (_, k) => dim.min + k)
      const at = (t: number) => decodeDim(dim, [dim.type === 'int' ? Math.min(t, 1 - 1e-12) : t], { i: 0 })
      return dedupe(points === 1 ? [dim.default] : Array.from({ length: points }, (_, k) => at(k / (points - 1))))
    }
    case 'choice':
      return [...dim.options]
    case 'bool':
      return [false, true]
    case 'space':
      return grid(dim.of, { points })
    case 'variants':
      return Object.entries(dim.cases).flatMap(([name, c]) =>
        grid(c, { points }).map((params): VariantValue => ({ case: name, params })),
      )
  }
}

/**
 * Every combination of the dimensions' grid values, the last dimension varying fastest: `points` evenly spaced values
 * per real (in log space on a log scale; snapped to its step), every integer when there are at most `points` of them
 * (else `points` spread over the range), every option, both booleans, and for `variants` each case's grid in turn. A
 * conditional dimension joins only the combinations where its condition holds.
 */
export function grid<S extends Space>(space: S, options?: { points?: Size }): ValuesOf<S>[]
export function grid(space: Space, options?: { points?: Size }): SpaceValues[]
export function grid(space: Space, { points = 5 }: { points?: Size } = {}): SpaceValues[] {
  if (!(Number.isInteger(points) && points >= 1)) throw new DomainError('grid', 'grid: points must be an integer ≥ 1')
  let out: Record<string, SpaceValue>[] = [{}]
  for (const [key, dim] of Object.entries(space.dims)) {
    const values = dimGrid(dim, points)
    out = out.flatMap((p) => (isDimActive(dim, p) ? values.map((v) => ({ ...p, [key]: v })) : [p]))
  }
  return out
}

/**
 * A uniform draw from the space: `encodedSize` uniforms from the stream, decoded (so log dimensions are log-uniform,
 * integers and choices uniform over their values).
 */
export function sample<S extends Space>(s: Stream, space: S): ValuesOf<S>
export function sample(s: Stream, space: Space): SpaceValues
export function sample(s: Stream, space: Space): SpaceValues {
  return decode(space, toFlat(uniform(s, 0, 1, { shape: [encodedSize(space)] })))
}
