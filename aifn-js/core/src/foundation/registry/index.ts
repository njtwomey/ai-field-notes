/**
 * `aifn/foundation/registry`: one pattern for every named entry of aifn (design S §3.1): `define(info, value)`
 * attaches metadata to a value and returns it; `entries(kind, ...namespaces)` collects the entries of one kind from a
 * module's namespaces into a frozen table keyed by `info.key`; `isEntry`. Registries are static and per module.
 */

import { AifnError } from 'aifn/foundation/errors'
import type { Entry, EntryKind, Info, Stability } from 'aifn/foundation/contracts'

export type { Entry, EntryKind, Info, Stability }

/** Attach `info` to `value` (the value itself is returned, with a frozen `info` added). */
export function define<T extends object, I extends Info>(info: I, value: T): Entry<T, I> {
  return Object.assign(value, { info: Object.freeze({ ...info }) }) as Entry<T, I>
}

/** True when `x` carries registry metadata of `kind` (any kind when omitted). */
export function isEntry<I extends Info = Info>(x: unknown, kind?: I['kind']): x is Entry<unknown, I> {
  if ((typeof x !== 'function' && typeof x !== 'object') || x === null || !('info' in x)) return false
  const info = (x as { info?: unknown }).info
  if (typeof info !== 'object' || info === null) return false
  const k = (info as { kind?: unknown }).kind
  return typeof k === 'string' && (kind === undefined || k === kind)
}

/**
 * The entries of `kind` among the values of `namespaces` (module namespaces or plain objects), keyed by `info.key` in
 * definition order. A key defined by two different values throws, so each entry is defined once.
 */
export function entries<I extends Info>(
  kind: I['kind'],
  ...namespaces: readonly object[]
): Readonly<Record<string, Entry<unknown, I>>> {
  const out: Record<string, Entry<unknown, I>> = {}
  for (const ns of namespaces)
    for (const value of Object.values(ns)) {
      if (!isEntry<I>(value, kind)) continue
      const known = out[value.info.key]
      if (known !== undefined && known !== value) {
        throw new AifnError(
          'registry',
          `registry: two ${kind} entries are keyed '${value.info.key}' (each entry is defined once)`,
        )
      }
      out[value.info.key] = value
    }
  return Object.freeze(out)
}
