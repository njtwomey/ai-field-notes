import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useEffect, useId, useState, type ReactNode } from 'react'
import { Button } from '@render/ui/button'
import { Label } from '@render/ui/label'
import { Slider } from '@render/ui/slider'
import { Switch } from '@render/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@render/ui/toggle-group'
import { useDebouncedCallback } from './use-debounced-callback'
import type { Param } from './param'
import { formatNumber } from '@render/viz/format'

type SliderOptions = {
  label: ReactNode
  format?: (v: number) => string
  debounceMs?: number
  withArrows?: boolean
}

type ParamSliderProps = SliderOptions &
  (
    | { param: Param; value?: never; onChange?: never; min?: never; max?: never; step?: never }
    | { param?: never; value: number; onChange: (value: number) => void; min: number; max: number; step?: number }
  )

export function ParamSlider(props: ParamSliderProps) {
  const { label, format = formatNumber, debounceMs = 60, withArrows = false } = props
  const value = props.param ? props.param.value : props.value
  const onChange = props.param ? props.param.set : props.onChange
  const min = props.param ? props.param.min : props.min
  const max = props.param ? props.param.max : props.max
  const step = props.param ? props.param.step : (props.step ?? 0.01)
  const id = useId()
  const [dragging, setDragging] = useState<number | null>(null)
  const debounced = useDebouncedCallback(onChange, debounceMs)
  const shown = dragging ?? value

  useEffect(() => {
    if (dragging === null) debounced.cancel()
  }, [value, dragging, debounced])

  const toNumber = (v: number | readonly number[]) => (Array.isArray(v) ? v[0] : (v as number))
  const stepFrom = (v: number, direction: 1 | -1) => {
    const places = (String(step).split('.')[1] ?? '').length
    const next = min + Math.round((v + direction * step - min) / step) * step
    return Number(Math.min(max, Math.max(min, next)).toFixed(places))
  }

  const slider = (
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
  )

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Label htmlFor={id} className="text-xs font-normal text-muted-foreground">
          {label}
        </Label>
        <span className="font-mono text-xs tabular-nums">{format(shown)}</span>
      </div>
      {withArrows ? (
        <div className="flex items-center gap-1.5">
          <StepButton direction={-1} disabled={value <= min} onClick={() => onChange(stepFrom(value, -1))} />
          <div className="min-w-0 flex-1">{slider}</div>
          <StepButton direction={1} disabled={value >= max} onClick={() => onChange(stepFrom(value, 1))} />
        </div>
      ) : (
        slider
      )}
    </div>
  )
}

function StepButton({ direction, disabled, onClick }: { direction: 1 | -1; disabled: boolean; onClick: () => void }) {
  const Icon = direction < 0 ? ChevronLeft : ChevronRight
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      disabled={disabled}
      onClick={onClick}
      aria-label={direction < 0 ? 'Previous' : 'Next'}
    >
      <Icon />
    </Button>
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
        spacing={options.length > 4 ? 1 : 0}
        className={options.length > 4 ? 'flex-wrap' : undefined}
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
