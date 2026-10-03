import { useId, type ReactNode } from 'react'
import { Button } from '@render/ui/button'
import { Label } from '@render/ui/label'
import { Switch } from '@render/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@render/ui/toggle-group'
import type { Param } from './param'
import { type NumberOptions } from '@render/state/number'
import { Slider as ModernSlider } from '@render/controls/numeric/Slider'
import {
  NumberField,
  NumberSelector,
  NumericSelector,
  NumericControl,
  ParamNumberField,
} from '@render/controls/numeric'

export {
  NumberField,
  NumberSelector,
  NumericSelector,
  NumericControl,
  ParamNumberField,
}

type SliderOptions = {
  label: ReactNode
  format?: (v: number) => string
  debounceMs?: number
  withArrows?: boolean
  slider?: boolean
  variant?: 'slider' | 'field' | 'auto'
  logTransform?: boolean | 'value-is-log' | 'value-is-real'
  headerValue?: string | ((v: number) => string)
  spacing?: NumberOptions['spacing']
  increment?: NumberOptions['increment']
  points_per_decade?: number
  points?: number
  scale?: NumberOptions['scale']
  type?: NumberOptions['type']
  suggestions?: readonly number[]
  className?: string
  disabled?: boolean
}

type ParamSliderProps = SliderOptions & {
  param?: Param
  value?: number
  onChange?: (value: number) => void
  min?: number
  max?: number
  step?: number
}

const SLIDER_KEYWORDS =
  /\b(time|t\s*\(s\)|frame|playback|scrub|progress|phase|angle|direction|degrees|radians|rotation|orientation|azimuth|threshold|cutoff|probability|fraction|ratio|proportion|prevalence|leakage|split ratio|quantile|percentile|coverage|confidence level|correlation|rho|blend|mix|interpolation)\b/i

function shouldUseSlider(props: ParamSliderProps): boolean {
  if (props.slider === true || props.variant === 'slider') return true
  if (props.slider === false || props.variant === 'field') return false
  if (typeof props.label === 'string') {
    return SLIDER_KEYWORDS.test(props.label)
  }
  return false
}

export function ParamSlider(props: ParamSliderProps) {
  const useSlider = shouldUseSlider(props)

  if (useSlider) {
    if (props.param) {
      return (
        <ModernSlider
          label={props.label}
          param={props.param}
          format={props.format}
          steppable={props.withArrows ?? true}
          disabled={props.disabled}
          className={props.className}
        />
      )
    }
    return (
      <ModernSlider
        label={props.label}
        value={props.value!}
        onChange={props.onChange!}
        min={props.min ?? 0}
        max={props.max ?? 100}
        step={props.step}
        format={props.format}
        steppable={props.withArrows ?? true}
        disabled={props.disabled}
        className={props.className}
      />
    )
  }

  // Otherwise render NumberField (free-form numeric selector)
  let logTransform = props.logTransform
  let points_per_decade = props.points_per_decade
  if (logTransform === undefined && props.format) {
    try {
      const fnStr = props.format.toString()
      if (fnStr.includes('10 **') || fnStr.includes('10**')) {
        logTransform = 'value-is-log'
        if (points_per_decade === undefined) points_per_decade = 2
      }
    } catch {
      // ignore
    }
  }

  let type = props.type
  if (!type && typeof props.label === 'string') {
    if (
      /\b(seed|draws|steps|iterations|epochs|sample|samples|count|points|terms|layers|depth|clusters|components|neighbours|degree|order|horizon|folds|trees|sweeps|chains|paths|trials|rounds|batches|subsequence)\b/i.test(
        props.label,
      )
    ) {
      type = 'int'
    }
  }

  if (props.param) {
    return (
      <NumberField
        label={props.label}
        param={props.param}
        format={props.format}
        type={type}
        spacing={props.spacing}
        increment={props.increment}
        points_per_decade={points_per_decade}
        points={props.points}
        logTransform={logTransform}
        headerValue={props.headerValue}
        suggestions={props.suggestions}
        disabled={props.disabled}
        className={props.className}
      />
    )
  }

  return (
    <NumberField
      label={props.label}
      value={props.value}
      onChange={props.onChange}
      min={props.min}
      max={props.max}
      step={props.step}
      format={props.format}
      type={type}
      spacing={props.spacing}
      increment={props.increment}
      points_per_decade={points_per_decade}
      points={props.points}
      logTransform={logTransform}
      headerValue={props.headerValue}
      suggestions={props.suggestions}
      disabled={props.disabled}
      className={props.className}
    />
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
