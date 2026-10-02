import { ChevronDown, Minus, Plus } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { ButtonGroup } from '@lab/ui/button-group'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@lab/ui/dropdown-menu'
import { cn } from '@lab/lib/utils'
import { checkNumber, formatNumberValue, stepNumber, type NumberOptions } from '@lab/state/number'
import { ControlLabel } from './ControlLabel'
import type { Param } from './param'
import { useNumberDraft } from './useNumberDraft'

type Common = NumberOptions & {
  label: ReactNode
  /** How the value shows when not being edited (default: by type and scale, e.g. 1e-3 on a log10 scale). */
  format?: (v: number) => string
  disabled?: boolean
  className?: string
  /** Text shown as an uncommitted draft at first, validated like typing (for demonstrations). */
  initialDraft?: string
}

export type NumberFieldProps = Common &
  (
    | { param: Param; value?: never; onChange?: never }
    | { param?: never; value: number; onChange: (value: number) => void }
  )

/**
 * A typed number with − and + buttons, for values with no natural track: a seed, an episode count, a learning rate.
 * One field for every typed number: `type` float or int, bounds `gt`/`ge`/`lt`/`le` (`min`/`max` alias `ge`/`le`),
 * `scale` linear or log10, and `suggestions` (a menu of common values beside the field). Rules in `@lab/state/number`.
 *
 * Typing is validated, not clamped: a draft that does not parse or breaks the type or a bound turns the field red with
 * a short message ("must be > 0"), Enter is refused, and Escape or blur reverts to the last valid value. The buttons
 * and ↑/↓ step (linear ± step, × 10 with Shift; log10 × or ÷ 10^step, a decade with Shift) and clamp to the bounds,
 * stopping just inside a strict one. With `param`, its min, max and step are the defaults.
 */
export function NumberField(props: NumberFieldProps) {
  const { label, disabled, className, suggestions } = props
  const value = props.param ? props.param.value : props.value
  const onChange = props.param ? props.param.set : props.onChange
  const options: NumberOptions = {
    type: props.type,
    gt: props.gt,
    ge: props.ge,
    lt: props.lt,
    le: props.le,
    min: props.min ?? finite(props.param?.min),
    max: props.max ?? finite(props.param?.max),
    scale: props.scale,
    step: props.step ?? props.param?.step,
  }
  const format = props.format ?? ((v: number) => formatNumberValue(options, v))
  const id = useId()
  const errorId = useId()
  const commit = (v: number) => {
    if (v !== value) onChange(v)
  }
  const step = (dir: 1 | -1, big = false) => stepNumber(options, value, dir, big)
  const { props: field, error } = useNumberDraft({
    value,
    onCommit: commit,
    step: options.step ?? 1,
    format,
    check: (text) => checkNumber(options, text),
    stepBy: (base, dir, big) => stepNumber(options, base, dir, big),
    initialDraft: props.initialDraft,
  })
  const down = step(-1)
  const up = step(1)
  return (
    <div className={cn('flex w-full max-w-xs min-w-40 flex-col gap-1.5', className)}>
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
      <ButtonGroup className="w-full">
        <Button
          variant="outline"
          size="icon"
          aria-label="Decrease"
          disabled={disabled || down === null}
          onClick={(e) => {
            const next = step(-1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <Minus />
        </Button>
        <input
          id={id}
          disabled={disabled}
          data-slot="input"
          aria-describedby={error ? errorId : undefined}
          className={cn(
            'h-8 min-w-0 flex-1 border border-input bg-transparent px-2 text-right font-mono text-xs tabular-nums outline-none focus-visible:z-10 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30',
            error && 'z-10 border-destructive focus-visible:border-destructive focus-visible:ring-destructive/30',
          )}
          {...field}
        />
        <Button
          variant="outline"
          size="icon"
          aria-label="Increase"
          disabled={disabled || up === null}
          onClick={(e) => {
            const next = step(1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <Plus />
        </Button>
        {suggestions?.length ? (
          <DropdownMenu>
            <DropdownMenuTrigger
              disabled={disabled}
              render={<Button variant="outline" size="icon" aria-label="Suggested values" title="Suggested values" />}
            >
              <ChevronDown />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-auto min-w-24">
              {suggestions.map((s) => (
                <DropdownMenuItem
                  key={s}
                  onClick={() => commit(s)}
                  className={cn('justify-end font-mono text-xs tabular-nums', s === value && 'font-semibold')}
                >
                  {format(s)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
      </ButtonGroup>
      {error ? (
        <p id={errorId} role="alert" className="text-xs leading-none text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  )
}

const finite = (x: number | undefined) => (x !== undefined && Number.isFinite(x) ? x : undefined)
