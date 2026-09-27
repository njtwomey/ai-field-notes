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
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'

const Z = linspace(-4, 4, 401)
const SIGMAS = linspace(0.1, 3, 146)
const pdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)

/** CRPS of N(0, σ²) at observation y, in closed form. */
function crpsGaussian(sigma: number, y: number): number {
  const z = y / sigma
  return sigma * (z * (2 * normalCdf(z) - 1) + 2 * pdf(z) - 1 / Math.sqrt(Math.PI))
}
const logScore = (sigma: number, y: number) =>
  0.5 * Math.log(2 * Math.PI * sigma * sigma) + (y * y) / (2 * sigma * sigma)

/**
 * Left: the predictive cdf F and the observation's step function; the CRPS is the area of their squared gap. Right: CRPS
 * and log score against the forecast's spread for the current observation.
 */
export function CrpsExplorer() {
  const sigma = useParam(1, { min: 0.1, max: 3, step: 0.05 })
  const obs = useParam(0.8, { min: -3.5, max: 3.5, step: 0.05 })
  const s = sigma.value
  const y = obs.value

  const left = useMemo((): XYSeries[] => {
    const cdf = Z.map((z) => normalCdf(z / s))
    return [
      {
        name: 'squared gap (area = CRPS)',
        type: 'line',
        x: Z,
        y: Z.map((z, i) => (cdf[i] - (z >= y ? 1 : 0)) ** 2),
        slot: 2,
        area: true,
      },
      { name: 'forecast cdf F(z)', type: 'line', x: Z, y: cdf, slot: 0 },
      { name: 'observation step', type: 'line', x: [-4, y, y, 4], y: [0, 0, 1, 1], slot: 1, dashed: true },
    ]
  }, [s, y])

  const right = useMemo(
    (): XYSeries[] => [
      { name: 'CRPS', type: 'line', x: SIGMAS, y: SIGMAS.map((v) => crpsGaussian(v, y)), slot: 0 },
      { name: 'log score', type: 'line', x: SIGMAS, y: SIGMAS.map((v) => logScore(v, y)), slot: 1 },
    ],
    [y],
  )

  const handles: Handle[] = [
    { kind: 'x', at: y, label: 'observation', onDrag: (x) => obs.set(Math.min(3.5, Math.max(-3.5, x))) },
  ]
  const spreadHandle: Handle[] = [
    { kind: 'x', at: s, label: 'forecast spread', onDrag: (x) => sigma.set(Math.min(3, Math.max(0.1, x))) },
  ]

  return (
    <Interactive
      title="The continuous ranked probability score"
      caption="The forecast is a normal distribution centred at 0. Left: its cdf, and the step function that jumps at the observed value; the CRPS is the area under their squared difference. Drag the observation. Right: both scores as the forecast's spread changes, for the current observation. The log score punishes a forecast too narrow for the observation far more sharply; the CRPS changes gently and stays in the units of the data."
      controls={
        <>
          <ParamSlider label="forecast spread σ" param={sigma} />
          <ParamSlider label="observation y" param={obs} />
        </>
      }
      readout={
        <>
          <Readout label="CRPS" value={formatNumber(crpsGaussian(s, y))} />
          <Readout label="log score" value={formatNumber(logScore(s, y))} />
          <Readout label="absolute error of the mean" value={formatNumber(Math.abs(y))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={left}
          handles={handles}
          xLabel="z"
          yLabel="probability"
          xRange={[-4, 4]}
          yRange={[0, 1.05]}
          height={300}
        />
        <XYChart
          series={right}
          handles={spreadHandle}
          xLabel="forecast spread σ"
          yLabel="score (lower is better)"
          xRange={[0.1, 3]}
          yRange={[0, 6]}
          height={300}
        />
      </div>
    </Interactive>
  )
}
