import { useCallback, useMemo, useState } from 'react'
import { niceStep, snapToStep } from '@lab/viz'

export type ParamSpec = { min: number; max: number; step?: number }

/**
 * One adjustable value with its range and step. Every input bound to it (slider, number field, chart handle, button)
 * calls `set`, which clamps to the range and snaps to the step, so all inputs obey the same rules.
 */
export type Param = { value: number; set: (v: number) => void } & Required<ParamSpec>

/** A clamped, snapped parameter. Without a step it takes a nice 1-2-5 step from the range (about 100–250 positions). */
export function useParam(initial: number, { min, max, step }: ParamSpec): Param {
  const resolved = step ?? niceStep(min, max)
  const [value, setValue] = useState(() => snapToStep(initial, min, max, resolved))
  const set = useCallback(
    (v: number) => {
      if (Number.isFinite(v)) setValue(snapToStep(v, min, max, resolved))
    },
    [min, max, resolved],
  )
  return useMemo(() => ({ value, set, min, max, step: resolved }), [value, set, min, max, resolved])
}
