import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

const relu = (x: number) => Math.max(0, x)
/** The tent map as a width-2 ReLU layer: 2x on [0, ½], 2 − 2x on [½, 1]. */
const tent = (x: number) => 2 * relu(x) - 4 * relu(x - 0.5)

function sawtooth(x: number, depth: number) {
  let y = x
  for (let i = 0; i < depth; i++) y = tent(y)
  return y
}

const X_RANGE: [number, number] = [0, 1]
const Y_RANGE: [number, number] = [-0.1, 1.1]

export function Sawtooth() {
  const [depth, setDepth] = useState(4)
  const [width, setWidth] = useState(8)

  const { series, error } = useMemo(() => {
    // Every breakpoint of the depth-k sawtooth lies on the grid j / 2^k, so this grid draws it exactly.
    const exact = linspace(0, 1, 2 ** depth + 1)
    // A one-hidden-layer ReLU network with m units and evenly spaced knots: the interpolant at m + 1 points.
    const knots = linspace(0, 1, width + 1)
    const fine = linspace(0, 1, 2049)
    const interp = (x: number) => {
      const j = Math.min(Math.floor(x * width), width - 1)
      const t = x * width - j
      return (1 - t) * sawtooth(knots[j], depth) + t * sawtooth(knots[j + 1], depth)
    }
    const s: XYSeries[] = [
      {
        name: `deep: ${depth} layers × 2 units`,
        type: 'line',
        x: exact,
        y: exact.map((x) => sawtooth(x, depth)),
        slot: 0,
      },
      {
        name: `shallow: 1 layer × ${width} units`,
        type: 'line',
        x: knots,
        y: knots.map((x) => sawtooth(x, depth)),
        slot: 1,
      },
    ]
    return { series: s, error: Math.max(...fine.map((x) => Math.abs(sawtooth(x, depth) - interp(x)))) }
  }, [depth, width])

  return (
    <Interactive
      title="A sawtooth that depth builds cheaply and width does not"
      caption="The deep line is the tent map composed with itself k times. It is computed exactly by k hidden layers of 2 ReLU units and has 2ᵏ linear pieces. The shallow line is a one-hidden-layer ReLU network with m units and evenly spaced knots, which has at most m + 1 pieces. Each extra layer doubles the units the shallow network needs; with fewer than 2ᵏ units its largest error is large."
      controls={
        <>
          <ParamSlider label="depth k of the deep network" value={depth} onChange={setDepth} min={1} max={8} step={1} />
          <ParamSlider
            label="width m of the shallow network"
            value={width}
            onChange={setWidth}
            min={1}
            max={256}
            step={1}
          />
        </>
      }
      readout={
        <>
          <Readout label="pieces of the sawtooth" value={2 ** depth} />
          <Readout label="deep units" value={2 * depth} />
          <Readout label="shallow units needed" value={2 ** depth - 1} />
          <Readout label="largest error of the shallow network" value={formatNumber(error)} />
        </>
      }
    >
      <XYChart series={series} xRange={X_RANGE} yRange={Y_RANGE} xLabel="x" yLabel="output" height={320} />
    </Interactive>
  )
}
