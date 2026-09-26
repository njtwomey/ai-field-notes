import { useEffect, useId, useState, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useDebouncedCallback } from '@/hooks/use-debounced-callback'
import type { Param } from './param'
import { formatNumber } from './theme'

type SliderOptions = {
  label: ReactNode
  format?: (v: number) => string
  /**
   * Milliseconds between the thumb moving and `onChange` firing. The thumb and label update immediately; the figure
   * updates when the drag pauses, at least every 3× this interval, and always on release. 0 disables debouncing.
   */
  debounceMs?: number
}

/** Either a `param` from useParam (preferred: shared with chart handles), or explicit value, onChange and range. */
type ParamSliderProps = SliderOptions &
  (
    | { param: Param; value?: never; onChange?: never; min?: never; max?: never; step?: never }
    | { param?: never; value: number; onChange: (value: number) => void; min: number; max: number; step?: number }
  )

export function ParamSlider(props: ParamSliderProps) {
  const { label, format = formatNumber, debounceMs = 60 } = props
  const value = props.param ? props.param.value : props.value
  const onChange = props.param ? props.param.set : props.onChange
  const min = props.param ? props.param.min : props.min
  const max = props.param ? props.param.max : props.max
  const step = props.param ? props.param.step : (props.step ?? 0.01)
  const id = useId()
  // Value under the thumb while dragging; null when the slider shows the committed `value`.
  const [dragging, setDragging] = useState<number | null>(null)
  const debounced = useDebouncedCallback(onChange, debounceMs)
  const shown = dragging ?? value
  // When another input (a chart handle) changes the value, drop any pending slider update so it cannot overwrite it.
  useEffect(() => {
    if (dragging === null) debounced.cancel()
  }, [value, dragging, debounced])
  const toNumber = (v: number | readonly number[]) => (Array.isArray(v) ? v[0] : (v as number))
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Label htmlFor={id} className="text-xs font-normal text-muted-foreground">
          {label}
        </Label>
        <span className="font-mono text-xs tabular-nums">{format(shown)}</span>
      </div>
      <Slider
        id={id}
        value={shown}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => {
          const n = toNumber(v)
          if (debounceMs <= 0) return onChange(n)
          setDragging(n)
          debounced(n)
        }}
        onValueCommitted={(v) => {
          debounced.cancel()
          setDragging(null)
          if (toNumber(v) !== value) onChange(toNumber(v))
        }}
      />
    </div>
  )
}

type ParamChoiceProps<T extends string> = {
  label: ReactNode
  value: T
  onChange: (value: T) => void
  options: readonly { value: T; label: ReactNode }[]
}

export function ParamChoice<T extends string>({ label, value, onChange, options }: ParamChoiceProps<T>) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <ToggleGroup
        value={[value]}
        onValueChange={(v) => v[0] && onChange(v[0] as T)}
        variant="outline"
        size="sm"
        spacing={0}
      >
        {options.map((o) => (
          <ToggleGroupItem key={o.value} value={o.value} className="px-3 text-xs">
            {o.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  )
}

export function ParamSwitch({
  label,
  checked,
  onChange,
}: {
  label: ReactNode
  checked: boolean
  onChange: (v: boolean) => void
}) {
  const id = useId()
  return (
    <div className="flex items-center gap-2 self-end">
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
      <Label htmlFor={id} className="text-xs font-normal text-muted-foreground">
        {label}
      </Label>
    </div>
  )
}

export function ParamButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <Button variant="outline" size="sm" onClick={onClick} disabled={disabled} className="self-end">
      {children}
    </Button>
  )
}
