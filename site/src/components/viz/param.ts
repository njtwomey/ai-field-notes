import { useCallback, useMemo, useState } from 'react'

export type ParamSpec = { min: number; max: number; step?: number }

/**
 * One adjustable value with its range. Every input bound to it (slider, chart handle, button) calls `set`, which
 * clamps to the range and snaps to the step, so all inputs obey the same rules.
 */
export type Param = { value: number; set: (v: number) => void } & Required<ParamSpec>

/** Decimal places in the step, used to strip floating-point noise after snapping (0.1 + 0.2 → 0.3). */
const decimals = (step: number) => (String(step).split('.')[1] ?? '').length

export function useParam(initial: number, { min, max, step = 0.01 }: ParamSpec): Param {
  const [value, setValue] = useState(initial)
  const set = useCallback(
    (v: number) => {
      if (!Number.isFinite(v)) return
      const snapped = min + Math.round((Math.min(Math.max(v, min), max) - min) / step) * step
      setValue(Number(Math.min(snapped, max).toFixed(decimals(step))))
    },
    [min, max, step],
  )
  return useMemo(() => ({ value, set, min, max, step }), [value, set, min, max, step])
}
