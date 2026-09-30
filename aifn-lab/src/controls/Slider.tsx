import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useId, useRef, useState, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { Slider as SliderPrimitive } from '@lab/ui/slider'
import { cn } from '@lab/lib/utils'
import { formatNumber, niceStep, snapToStep } from '@lab/viz'
import { ControlLabel } from './ControlLabel'
import type { Param } from './param'
import { useNumberDraft } from './useNumberDraft'

type Common = {
  label: ReactNode
  /** Formats the value in the field when it is not being edited. */
  format?: (v: number) => string
  /** Step arrows either side of the track (default true). Opt out with `steppable={false}`. */
  steppable?: boolean
  /** @deprecated The site's name for `steppable`; accepted so migrated code keeps working. */
  withArrows?: boolean
  /** Delay (ms) before a drag reaches `onChange`; the thumb moves at once and release always commits. Default 0. */
  debounceMs?: number
  disabled?: boolean
  className?: string
}

export type SliderProps = Common &
  (
    | { param: Param; value?: never; onChange?: never; min?: never; max?: never; step?: never }
    | {
        param?: never
        value: number
        onChange: (value: number) => void
        min: number
        max: number
        /** Omitted: a nice 1-2-5 step from the range (about 100–250 positions). */
        step?: number
      }
  )

/**
 * A slider with a typed value. Click anywhere on the track to jump there, drag to move, step with the arrows, or type
 * into the field (Enter or blur commits, clamped to the range and snapped to the step; Escape reverts; ↑ and ↓ step,
 * × 10 with Shift). Bind it to a `useParam` with `param`, or pass value, onChange, min, max and optionally step.
 */
export function Slider(props: SliderProps) {
  const { label, format = formatNumber, debounceMs = 0, disabled, className } = props
  const steppable = props.steppable ?? props.withArrows ?? true
  const value = props.param ? props.param.value : props.value
  const onChange = props.param ? props.param.set : props.onChange
  const min = props.param ? props.param.min : props.min
  const max = props.param ? props.param.max : props.max
  const step = props.param ? props.param.step : (props.step ?? niceStep(props.min, props.max))
  const id = useId()
  const snap = (v: number) => snapToStep(v, min, max, step)
  const set = (v: number) => {
    const next = snap(v)
    if (next !== value) onChange(next)
  }
  const field = useNumberDraft({ value, onCommit: set, step, format })
  // While a debounced drag is in progress the thumb follows `dragging`; onChange fires after the pause or on release.
  const [dragging, setDragging] = useState<number | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const first = (v: number | readonly number[]) => (Array.isArray(v) ? v[0] : (v as number))

  const track = (
    <SliderPrimitive
      id={id}
      // An array, not a number: the shadcn wrapper renders one thumb per entry and, given a plain number, falls back to
      // [min, max], drawing a second, phantom thumb. Base UI then resolves a track press to the phantom thumb's index,
      // which has no value, and ignores the click.
      value={[dragging ?? value]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={(v) => {
        const n = first(v)
        if (debounceMs <= 0) return set(n)
        setDragging(n)
        clearTimeout(timer.current)
        timer.current = setTimeout(() => set(n), debounceMs)
      }}
      onValueCommitted={(v) => {
        clearTimeout(timer.current)
        setDragging(null)
        set(first(v))
      }}
    />
  )
  return (
    <div className={cn('flex w-full max-w-sm min-w-48 flex-col gap-1.5', className)}>
      <div className="flex items-center justify-between gap-2">
        <ControlLabel htmlFor={id}>{label}</ControlLabel>
        <input
          aria-label={typeof label === 'string' ? `${label} value` : 'value'}
          disabled={disabled}
          // At least w-20, wider for a long formatted value (e.g. a Player's "t = 12 · update").
          style={{ width: `max(5rem, ${String(field.value).length + 2}ch)` }}
          className="h-7 rounded-md border border-input bg-transparent px-1.5 text-right font-mono text-xs tabular-nums outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
          {...field}
        />
      </div>
      {steppable ? (
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            disabled={disabled || value <= min}
            onClick={() => set(value - step)}
            aria-label="Step down"
          >
            <ChevronLeft />
          </Button>
          <div className="min-w-0 flex-1">{track}</div>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            disabled={disabled || value >= max}
            onClick={() => set(value + step)}
            aria-label="Step up"
          >
            <ChevronRight />
          </Button>
        </div>
      ) : (
        track
      )}
    </div>
  )
}
