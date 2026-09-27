import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

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
  const [billions, setBillions] = useState(7.5)
  const n = useParam(64, { min: 1, max: MAX_N, step: 1 })

  const series = useMemo((): XYSeries[] => {
    const lines: XYSeries[] = STAGES.map((s, slot) => ({
      name: s.name,
      type: 'line',
      x: NS,
      y: NS.map((k) => billions * s.bytes(k)),
      slot,
    }))
    lines.push({
      name: `${DEVICE_GB} GB device`,
      type: 'line',
      x: [1, MAX_N],
      y: [DEVICE_GB, DEVICE_GB],
      muted: true,
      dashed: true,
    })
    return lines
  }, [billions])

  const at = (i: number) => `${formatNumber(billions * STAGES[i].bytes(n.value))} GB`
  const handles: Handle[] = [{ kind: 'x', at: n.value, label: 'N', onDrag: n.set }]

  return (
    <Interactive
      title="Model-state memory per device under ZeRO"
      caption="Mixed-precision Adam keeps 16 bytes per parameter: 2 for 16-bit weights, 2 for 16-bit gradients and 12 for the 32-bit master weights and two moments. Each ZeRO stage shards one more of these across the N data-parallel devices. Only stage 3 keeps falling as 1/N. Activations are not included. Drag the vertical line to change N."
      controls={
        <>
          <ParamSlider
            label="parameters Ψ (billions)"
            value={billions}
            onChange={setBillions}
            min={0.5}
            max={200}
            step={0.5}
          />
          <ParamSlider label="data-parallel devices N" param={n} />
        </>
      }
      readout={
        <>
          <Readout label="data parallel" value={at(0)} />
          <Readout label="stage 1" value={at(1)} />
          <Readout label="stage 2" value={at(2)} />
          <Readout label="stage 3" value={at(3)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="data-parallel devices N"
        yLabel="GB per device"
        xRange={[1, MAX_N]}
        yLog
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
