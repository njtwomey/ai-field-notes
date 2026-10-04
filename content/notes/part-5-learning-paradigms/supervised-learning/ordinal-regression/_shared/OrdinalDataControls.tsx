import { type Dispatch, type SetStateAction } from 'react'
import { Choice, NumberField, Slider } from 'aifn-render'
import { SHAPE_OPTIONS, type DataSpec } from './ordinal'
import { CELLS, type FillMode, type Resolution } from './useOrdinalData'

const RESOLUTION_OPTIONS = (Object.keys(CELLS) as Resolution[]).map((value) => ({ value, label: value }))
const FILL_OPTIONS = [
  { value: 'decision' as const, label: 'decision' },
  { value: 'expectation' as const, label: 'expectation' },
]

/**
 * The shared controls, always in this order: dataset, classes, noise, seed and points per class, then the display
 * controls, map resolution and (for widgets whose fill shows classes) the fill mode. The spec is held by
 * `useOrdinalData`, shared by every widget in the category, so these controls are placed in a Figure's `controls`
 * rather than declared as its fields.
 */
export function OrdinalDataControls({
  spec,
  setSpec,
  resolution,
  setResolution,
  fill,
  setFill,
}: {
  spec: DataSpec
  setSpec: Dispatch<SetStateAction<DataSpec>>
  resolution: Resolution
  setResolution: (r: Resolution) => void
  fill?: FillMode
  setFill?: (f: FillMode) => void
}) {
  // Functional updates, so a debounced slider never writes back a stale copy of the other fields.
  const update = (patch: Partial<DataSpec>) => setSpec((s) => ({ ...s, ...patch }))
  return (
    <>
      <Choice label="dataset" value={spec.shape} onChange={(shape) => update({ shape })} options={SHAPE_OPTIONS} />
      <NumberField
        label="classes K"
        type="int"
        value={spec.k}
        onChange={(k) => update({ k })}
        min={3}
        max={10}
        suggestions={[3, 5, 7, 10]}
      />
      <Slider
        label="noise σ"
        value={spec.noise}
        onChange={(noise) => update({ noise })}
        min={0}
        max={0.8}
        step={0.02}
      />
      <NumberField label="seed" type="int" value={spec.seed} onChange={(seed) => update({ seed })} min={1} max={100} />
      <NumberField
        label="training points per class"
        type="int"
        value={spec.m}
        onChange={(m) => update({ m })}
        min={10}
        max={100}
        step={10}
        suggestions={[10, 30, 50, 100]}
      />
      <Choice label="map resolution" value={resolution} onChange={setResolution} options={RESOLUTION_OPTIONS} />
      {fill && setFill && <Choice label="fill" value={fill} onChange={setFill} options={FILL_OPTIONS} />}
    </>
  )
}
