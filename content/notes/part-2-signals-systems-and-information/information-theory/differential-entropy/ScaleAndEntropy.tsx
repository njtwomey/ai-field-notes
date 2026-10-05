import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const log2 = (x: number) => Math.log(x) / Math.LN2
const SQRT12 = Math.sqrt(12)
/** Differential entropies in bits of a Gaussian and a uniform with standard deviation s. */
const hGauss = (s: number) => 0.5 * log2(2 * Math.PI * Math.E * s * s)
const hUniform = (s: number) => log2(SQRT12 * s)

const S_GRID = toFlat(linspace(0.05, 3, 150))
const X_GRID = toFlat(linspace(-6, 6, 481))

/**
 * A Gaussian and a uniform with the same standard deviation s. Both differential entropies fall by one bit each time
 * s halves, pass below zero for small s, and the Gaussian always has the larger entropy.
 */
export function ScaleAndEntropy() {
  const state = useFigureState({
    s: float(1, { min: 0.05, max: 3, step: 0.01, label: 'standard deviation s' }),
  })
  const width = SQRT12 * state.s

  const densities = useMemo(() => {
    const g = X_GRID.map((x) => Math.exp((-0.5 * x * x) / (state.s * state.s)) / (state.s * Math.sqrt(2 * Math.PI)))
    const u = X_GRID.map((x) => (Math.abs(x) <= width / 2 ? 1 / width : 0))
    return [
      { name: 'Gaussian', x: X_GRID, y: g, slot: 0 },
      { name: 'uniform', x: X_GRID, y: u, slot: 1 },
    ] as const
  }, [state.s, width])

  const curves = [
    { name: 'Gaussian', x: S_GRID, y: S_GRID.map(hGauss), slot: 0 },
    { name: 'uniform', x: S_GRID, y: S_GRID.map(hUniform), slot: 1 },
    { name: 'zero', x: [0, 3], y: [0, 0], dashed: true, muted: true },
    {
      name: 'current s',
      x: [state.s, state.s],
      y: [hGauss(state.s), hUniform(state.s)],
      emphasis: true,
    },
  ] as const
  // s is the horizontal position on the entropy curves, so dragging along the axis sets it.

  const xAxis = useAxis({ label: 'x', range: [-6, 6] })
  const yAxis = useAxis({ label: 'density', range: [0, 3] })
  const xAxis2 = useAxis({ label: 'standard deviation s', range: [0, 3] })
  const yAxis2 = useAxis({ label: 'differential entropy (bits)', range: [-3, 4] })
  return (
    <Figure
      title="Differential entropy depends on scale"
      state={state}
      caption="A Gaussian and a uniform distribution with the same standard deviation s. Left: their densities. Right: their differential entropies in bits against s. Halving s lowers both by exactly one bit, so for small s they become negative: a narrow density exceeds 1 and its log is positive. At every s the Gaussian has the higher entropy, by 0.25 bits, because it is the maximum-entropy distribution for a given variance. Drag s or use the slider."

      readouts={
        <>
          <Readout label="h(Gaussian)" value={`${formatNumber(hGauss(state.s))} bits`} />
          <Readout label="h(uniform)" value={`${formatNumber(hUniform(state.s))} bits`} />
          <Readout label="uniform width" value={formatNumber(width)} />
          <Readout label="difference" value={`${formatNumber(hGauss(state.s) - hUniform(state.s))} bits`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...densities[0]} />
          <Curve {...densities[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
          <Curve {...curves[2]} />
          <Points {...curves[3]} />
          <Handle {...state.handle('s', { label: 's' })} />
        </Plot>
      </div>
    </Figure>
  )
}
