import { Minus, Plus } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { ButtonGroup } from '@lab/ui/button-group'
import { cn } from '@lab/lib/utils'
import { formatField, snapToStep } from '@lab/state/step'
import { ControlLabel } from './ControlLabel'
import type { Param } from './param'
import { useNumberDraft } from './useNumberDraft'

type Common = {
  label: ReactNode
  format?: (v: number) => string
  disabled?: boolean
  className?: string
}

export type NumberFieldProps = Common &
  (
    | { param: Param; value?: never; onChange?: never; min?: never; max?: never; step?: never }
    | {
        param?: never
        value: number
        onChange: (value: number) => void
        min?: number
        max?: number
        /** The step of the − and + buttons and of ↑/↓; values snap to it when a `min` is given. Default 1. */
        step?: number
      }
  )

/**
 * A typed number with − and + buttons, for values with no natural track (a seed, a sample size up to 10⁶, a tolerance).
 * Enter or blur commits (clamped, and snapped to the step); Escape reverts; ↑ and ↓ step, × 10 with Shift.
 */
export function NumberField(props: NumberFieldProps) {
  const { label, format = formatField, disabled, className } = props
  const value = props.param ? props.param.value : props.value
  const onChange = props.param ? props.param.set : props.onChange
  const min = (props.param ? props.param.min : props.min) ?? -Infinity
  const max = (props.param ? props.param.max : props.max) ?? Infinity
  const step = (props.param ? props.param.step : props.step) ?? 1
  const id = useId()
  const set = (v: number) => {
    const next = Number.isFinite(min) ? snapToStep(v, min, max, step) : Math.min(Math.max(v, min), max)
    if (next !== value) onChange(next)
  }
  const field = useNumberDraft({ value, onCommit: set, step, format })
  return (
    <div className={cn('flex w-full max-w-xs min-w-40 flex-col gap-1.5', className)}>
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
      <ButtonGroup className="w-full">
        <Button
          variant="outline"
          size="icon"
          aria-label="Decrease"
          disabled={disabled || value <= min}
          onClick={() => set(value - step)}
        >
          <Minus />
        </Button>
        <input
          id={id}
          disabled={disabled}
          data-slot="input"
          className="h-8 min-w-0 flex-1 border border-input bg-transparent px-2 text-right font-mono text-xs tabular-nums outline-none focus-visible:z-10 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
          {...field}
        />
        <Button
          variant="outline"
          size="icon"
          aria-label="Increase"
          disabled={disabled || value >= max}
          onClick={() => set(value + step)}
        >
          <Plus />
        </Button>
      </ButtonGroup>
    </div>
  )
}
