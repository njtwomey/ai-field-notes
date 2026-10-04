import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'

const MAX_N = 128
const NS = Array.from({ length: MAX_N }, (_, i) => i + 1)
const DEVICE_GB = 80

/** Bytes per parameter on one device for mixed-precision Adam (2 + 2 + 12) under each ZeRO stage, with N devices. */
const STAGES = [
  { name: 'data parallel (16Ψ)', bytes: () => 16 },
  { name: 'stage 1: optimiser states', bytes: (n: number) => 4 + 12 / n },
  { name: 'stage 2: + gradients', bytes: (n: number) => 2 + 14 / n },
  { name: 'stage 3: + parameters', bytes: (n: number) => 16 / n },
]

export function ZeroMemory() {
  const state = useFigureState({
    billions: float(7.5, { min: 0.5, max: 200, step: 0.5, label: 'parameters Ψ (billions)' }),
    n: int(64, { min: 1, max: MAX_N, label: 'data-parallel devices N' }),
  })

  const stages = useMemo(
    () => STAGES.map((s, slot) => ({ name: s.name, x: NS, y: NS.map((k) => state.billions * s.bytes(k)), slot })),
    [state.billions],
  )

  const at = (i: number) => `${formatNumber(state.billions * STAGES[i].bytes(state.n))} GB`
  const devices = useAxis({ label: 'data-parallel devices N', range: [1, MAX_N] })
  const memory = useAxis({ label: 'GB per device', log: true, hold: 'union' })

  return (
    <Figure
      title="Model-state memory per device under ZeRO"
      purpose="Change the parameter count and the number of devices to compare model-state memory per device under each ZeRO stage."
      state={state}
      caption="Mixed-precision Adam keeps 16 bytes per parameter: 2 for 16-bit weights, 2 for 16-bit gradients and 12 for the 32-bit master weights and two moments. Each ZeRO stage shards one more of these across the N data-parallel devices. Only stage 3 keeps falling as 1/N. Activations are not included. Drag the vertical line to change N."

      readouts={
        <>
          <Readout label="data parallel" value={at(0)} />
          <Readout label="stage 1" value={at(1)} />
          <Readout label="stage 2" value={at(2)} />
          <Readout label="stage 3" value={at(3)} />
        </>
      }
    >
      <Plot x={devices} y={memory} height={320}>
        {stages.map((line) => (
          <Curve key={line.slot} {...line} />
        ))}
        <Annotation y={DEVICE_GB} text={`${DEVICE_GB} GB device`} dashed muted />
        <Handle {...state.handle('n', { label: 'N' })} />
      </Plot>
    </Figure>
  )
}
