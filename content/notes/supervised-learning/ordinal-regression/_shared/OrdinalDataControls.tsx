import { type Dispatch, type SetStateAction } from 'react'
import { ParamChoice, ParamSlider } from 'aifn-render'
import { SHAPE_OPTIONS, type DataSpec } from './ordinal'
import { CELLS, type FillMode, type Resolution } from './useOrdinalData'

const RESOLUTION_OPTIONS = (Object.keys(CELLS) as Resolution[]).map((value) => ({ value, label: value }))
const FILL_OPTIONS = [
  { value: 'decision' as const, label: 'decision' },
  { value: 'expectation' as const, label: 'expectation' },
]

/** Refitting follows every committed change, so the sliders wait a little longer than the default before firing. */
const REFIT_DEBOUNCE_MS = 150

/**
 * The shared controls, always in this order: dataset, classes, noise, seed and points per class, then the display
 * controls, map resolution and (for widgets whose fill shows classes) the fill mode.
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
      <ParamChoice label="dataset" value={spec.shape} onChange={(shape) => update({ shape })} options={SHAPE_OPTIONS} />
      <ParamSlider
        label="classes K"
        value={spec.k}
        onChange={(k) => update({ k })}
        min={3}
        max={10}
        step={1}
        withArrows
        debounceMs={REFIT_DEBOUNCE_MS}
      />
      <ParamSlider
        label="noise σ"
        value={spec.noise}
        onChange={(noise) => update({ noise })}
        min={0}
        max={0.8}
        step={0.02}
        debounceMs={REFIT_DEBOUNCE_MS}
      />
      <ParamSlider
        label="seed"
        value={spec.seed}
        onChange={(seed) => update({ seed })}
        min={1}
        max={20}
        step={1}
        withArrows
        debounceMs={REFIT_DEBOUNCE_MS}
      />
      <ParamSlider
        label="training points per class"
        value={spec.m}
        onChange={(m) => update({ m })}
        min={10}
        max={100}
        step={10}
        withArrows
        debounceMs={REFIT_DEBOUNCE_MS}
      />
      <ParamChoice label="map resolution" value={resolution} onChange={setResolution} options={RESOLUTION_OPTIONS} />
      {fill && setFill && <ParamChoice label="fill" value={fill} onChange={setFill} options={FILL_OPTIONS} />}
    </>
  )
}
