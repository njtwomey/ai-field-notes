/**
 * Declarative parameters: describe a set of controls once (kind, range, initial value, visibility) and get typed values,
 * a setter that clamps and snaps, and plain-JSON state. `ParamControls` renders a set; `defineVariants` builds on it for
 * figures that switch between several functions, each with its own parameters.
 */
import type { ReactNode } from 'react'
import { niceStep, snapToStep } from '@lab/viz'
import { normalise, type Option, type Options } from './options'

export type ParamValue = number | string | boolean
/** The values of a parameter set as a plain record, as `when` sees them. */
export type AnyValues = Readonly<Record<string, ParamValue>>

type Common = {
  /** Shown above the control; defaults to the parameter's name. */
  label?: ReactNode
  /** Show the control (and use its value) only when this holds for the set's current values. */
  when?: (values: AnyValues) => boolean
}
export type SliderDef = Common & {
  kind: 'slider'
  min: number
  max: number
  initial: number
  /** Omitted: a nice 1-2-5 step from the range. */
  step?: number
  steppable?: boolean
}
export type NumberDef = Common & { kind: 'number'; initial: number; min?: number; max?: number; step?: number }
export type ChoiceDef<T extends string = string> = Common & {
  kind: 'choice'
  options: Options<T>
  initial: T
  /** Force a searchable combobox or a plain select; by default decided by the list's length. */
  searchable?: boolean
}
export type SwitchDef = Common & { kind: 'switch'; initial: boolean }

export type ParamDef = SliderDef | NumberDef | ChoiceDef | SwitchDef
export type ParamDefs = Readonly<Record<string, ParamDef>>

/** The value type of one parameter: its choice's union of strings, a boolean for a switch, else a number. */
export type ValueOf<D> = D extends ChoiceDef<infer T> ? T : D extends SwitchDef ? boolean : number
/** The typed values of a parameter set. */
export type Values<P> = { -readonly [K in keyof P]: ValueOf<P[K]> }

type Extra<D> = Omit<D, 'kind' | 'min' | 'max' | 'initial' | 'options'>

/** A slider from `min` to `max` starting at `initial` (steppable by default). */
export function slider(min: number, max: number, initial: number, options: Extra<SliderDef> = {}): SliderDef {
  return { kind: 'slider', min, max, initial, ...options }
}

/** A typed number with − and + buttons, optionally bounded. */
export function number(initial: number, options: Extra<NumberDef> & { min?: number; max?: number } = {}): NumberDef {
  return { kind: 'number', initial, ...options }
}

/** A categorical choice; `initial` defaults to the first option. */
export function choice<const T extends string>(
  options: readonly (T | Option<T>)[],
  initial?: NoInfer<T>,
  extra: Extra<ChoiceDef<T>> = {},
): ChoiceDef<T> {
  return { kind: 'choice', options, initial: initial ?? normalise(options)[0].value, ...extra }
}

/** An on/off switch. */
export function toggle(initial = false, options: Extra<SwitchDef> = {}): SwitchDef {
  return { kind: 'switch', initial, ...options }
}

/** Every parameter's initial value. */
export function initialValues<P extends ParamDefs>(defs: P): Values<P> {
  return Object.fromEntries(Object.entries(defs).map(([k, d]) => [k, d.initial])) as Values<P>
}

/** `value` made valid for `def` (clamped, snapped, a known option), or the initial value if it cannot be. */
export function coerce(def: ParamDef, value: unknown): ParamValue {
  switch (def.kind) {
    case 'slider':
      return typeof value === 'number' && Number.isFinite(value)
        ? snapToStep(value, def.min, def.max, def.step ?? niceStep(def.min, def.max))
        : def.initial
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) return def.initial
      const [min, max] = [def.min ?? -Infinity, def.max ?? Infinity]
      return Number.isFinite(min) && def.step
        ? snapToStep(value, min, max, def.step)
        : Math.min(Math.max(value, min), max)
    }
    case 'choice':
      return normalise(def.options).some((o) => o.value === value) ? (value as string) : def.initial
    case 'switch':
      return typeof value === 'boolean' ? value : def.initial
  }
}

/** Saved values made valid for `defs`: unknown names dropped, missing or invalid ones reset to their initial value. */
export function coerceAll<P extends ParamDefs>(defs: P, saved: unknown): Values<P> {
  const record = typeof saved === 'object' && saved !== null ? (saved as Record<string, unknown>) : {}
  return Object.fromEntries(
    Object.entries(defs).map(([k, d]) => [k, k in record ? coerce(d, record[k]) : d.initial]),
  ) as Values<P>
}

/** The names of the parameters currently shown: those without `when`, or whose `when` holds. */
export function visibleNames(defs: ParamDefs, values: AnyValues): string[] {
  return Object.entries(defs)
    .filter(([, d]) => !d.when || d.when(values))
    .map(([k]) => k)
}
