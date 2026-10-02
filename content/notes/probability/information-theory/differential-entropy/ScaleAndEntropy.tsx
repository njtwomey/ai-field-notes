import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

const log2 = (x: number) => Math.log(x) / Math.LN2
const SQRT12 = Math.sqrt(12)
/** Differential entropies in bits of a Gaussian and a uniform with standard deviation s. */
const hGauss = (s: number) => 0.5 * log2(2 * Math.PI * Math.E * s * s)
const hUniform = (s: number) => log2(SQRT12 * s)

const S_GRID = linspace(0.05, 3, 150)
const X_GRID = linspace(-6, 6, 481)

/**
 * A Gaussian and a uniform with the same standard deviation s. Both differential entropies fall by one bit each time
 * s halves, pass below zero for small s, and the Gaussian always has the larger entropy.
 */
export function ScaleAndEntropy() {
  const s = useParam(1, { min: 0.05, max: 3, step: 0.01 })
  const width = SQRT12 * s.value

  const densities = useMemo((): XYSeries[] => {
    const g = X_GRID.map((x) => Math.exp((-0.5 * x * x) / (s.value * s.value)) / (s.value * Math.sqrt(2 * Math.PI)))
    const u = X_GRID.map((x) => (Math.abs(x) <= width / 2 ? 1 / width : 0))
    return [
      { name: 'Gaussian', type: 'line', x: X_GRID, y: g, slot: 0 },
      { name: 'uniform', type: 'line', x: X_GRID, y: u, slot: 1 },
    ]
  }, [s.value, width])

  const curves: XYSeries[] = [
    { name: 'Gaussian', type: 'line', x: S_GRID, y: S_GRID.map(hGauss), slot: 0 },
    { name: 'uniform', type: 'line', x: S_GRID, y: S_GRID.map(hUniform), slot: 1 },
    { name: 'zero', type: 'line', x: [0, 3], y: [0, 0], dashed: true, muted: true },
    {
      name: 'current s',
      type: 'scatter',
      x: [s.value, s.value],
      y: [hGauss(s.value), hUniform(s.value)],
      emphasis: true,
    },
  ]
  // s is the horizontal position on the entropy curves, so dragging along the axis sets it.
  const handles: Handle[] = [{ kind: 'x', at: s.value, label: 's', onDrag: (x) => s.set(x) }]

  return (
    <Interactive
      title="Differential entropy depends on scale"
      caption="A Gaussian and a uniform distribution with the same standard deviation s. Left: their densities. Right: their differential entropies in bits against s. Halving s lowers both by exactly one bit, so for small s they become negative: a narrow density exceeds 1 and its log is positive. At every s the Gaussian has the higher entropy, by 0.25 bits, because it is the maximum-entropy distribution for a given variance. Drag s or use the slider."
      controls={<ParamSlider label="standard deviation s" param={s} />}
      readout={
        <>
          <Readout label="h(Gaussian)" value={`${formatNumber(hGauss(s.value))} bits`} />
          <Readout label="h(uniform)" value={`${formatNumber(hUniform(s.value))} bits`} />
          <Readout label="uniform width" value={formatNumber(width)} />
          <Readout label="difference" value={`${formatNumber(hGauss(s.value) - hUniform(s.value))} bits`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={densities} xLabel="x" yLabel="density" xRange={[-6, 6]} yRange={[0, 3]} height={300} />
        <XYChart
          series={curves}
          xLabel="standard deviation s"
          yLabel="differential entropy (bits)"
          xRange={[0, 3]}
          yRange={[-3, 4]}
          handles={handles}
          height={300}
        />
      </div>
    </Interactive>
  )
}
