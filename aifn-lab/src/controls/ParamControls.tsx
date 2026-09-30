import type { ReactNode } from 'react'
import { Choice } from './Choice'
import { NumberField } from './NumberField'
import { visibleNames, type AnyValues, type ParamDefs, type ParamValue } from './params'
import { Slider } from './Slider'
import { Switch } from './Switch'

export type ParamControlsProps = {
  defs: ParamDefs
  values: AnyValues
  set: (name: string, value: ParamValue) => void
}

/**
 * One control per parameter, in declaration order, hiding those whose `when` fails. Renders a fragment, so the controls
 * join the surrounding `Controls` grid (or a Figure's controls slot). Pass a `useParams` result: `<ParamControls {...p} />`.
 */
export function ParamControls({ defs, values, set }: ParamControlsProps) {
  return <>{visibleNames(defs, values).map((name) => control(name, defs, values, set))}</>
}

function control(name: string, defs: ParamDefs, values: AnyValues, set: ParamControlsProps['set']): ReactNode {
  const def = defs[name]
  const label = def.label ?? name
  const value = values[name]
  const onChange = (v: ParamValue) => set(name, v)
  switch (def.kind) {
    case 'slider':
      return (
        <Slider
          key={name}
          label={label}
          value={value as number}
          onChange={onChange}
          min={def.min}
          max={def.max}
          step={def.step}
          steppable={def.steppable}
        />
      )
    case 'number':
      return (
        <NumberField
          key={name}
          label={label}
          value={value as number}
          onChange={onChange}
          min={def.min}
          max={def.max}
          step={def.step}
        />
      )
    case 'choice':
      return (
        <Choice
          key={name}
          label={label}
          value={value as string}
          onChange={onChange}
          options={def.options}
          searchable={def.searchable}
        />
      )
    case 'switch':
      return (
        <Switch key={name} label={label} checked={value as boolean} onChange={onChange} className="self-end pb-1" />
      )
  }
}
