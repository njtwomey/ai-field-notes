import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { joinPaths } from '../_shared/sde'

const PATHS = 20
const STEPS = 200
const T = 4
const TIMES = linspace(0, T, STEPS + 1)

/**
 * Ornstein–Uhlenbeck paths dX = −θX dt + σ dW, simulated with the exact Gaussian transition, against the exact mean
 * x₀e^{−θt} and the ±2 sd band, which widens to the stationary ±2σ/√(2θ).
 */
export function OrnsteinUhlenbeckPaths() {
  const x0 = useParam(3, { min: -4, max: 4, step: 0.1 })
  const theta = useParam(1, { min: 0.1, max: 4, step: 0.05 })
  const sigma = useParam(1, { min: 0, max: 2, step: 0.05 })

  // Fixed standard normals: changing a parameter moves every path smoothly instead of redrawing the noise.
  const z = useMemo(() => {
    const { normal } = rng(9)
    return Array.from({ length: PATHS }, () => Float64Array.from({ length: STEPS }, () => normal()))
  }, [])

  const series = useMemo<XYSeries[]>(() => {
    const th = theta.value
    const sg = sigma.value
    const h = T / STEPS
    const decay = Math.exp(-th * h)
    const stepSd = Math.sqrt(((sg * sg) / (2 * th)) * (1 - decay * decay))
    const paths = joinPaths(
      z.map((zi) => {
        const y = [x0.value]
        for (let k = 0; k < STEPS; k++) y.push(y[k] * decay + stepSd * zi[k])
        return { x: TIMES, y }
      }),
    )
    const mean = TIMES.map((t) => x0.value * Math.exp(-th * t))
    const sd = TIMES.map((t) => Math.sqrt(((sg * sg) / (2 * th)) * (1 - Math.exp(-2 * th * t))))
    const stat = sg / Math.sqrt(2 * th)
    return [
      { name: 'paths', type: 'line', ...paths, muted: true },
      { name: 'mean x₀e^{−θt}', type: 'line', x: TIMES, y: mean, slot: 0 },
      {
        name: 'mean ± 2 sd',
        type: 'line',
        ...joinPaths([1, -1].map((s) => ({ x: TIMES, y: mean.map((m, k) => m + 2 * s * sd[k]) }))),
        slot: 1,
      },
      {
        name: 'stationary ± 2σ/√(2θ)',
        type: 'line',
        ...joinPaths([1, -1].map((s) => ({ x: [0, T], y: [2 * s * stat, 2 * s * stat] }))),
        dashed: true,
        emphasis: true,
      },
    ]
  }, [z, x0.value, theta.value, sigma.value])

  const th = theta.value
  const sg = sigma.value
  return (
    <Interactive
      title="Ornstein–Uhlenbeck paths and their Gaussian band"
      caption="Twenty paths of dX = −θX dt + σ dW started at x₀. The mean decays to 0 at rate θ and the band of ±2 standard deviations widens from zero to the stationary ±2σ/√(2θ) (dashed). Large θ forgets the start quickly and holds the paths in a narrow band; large σ widens the band. With θ = ½ and σ = 1 the stationary law is N(0, 1): this is the forward process of a variance-preserving diffusion model. Drag the start point at t = 0."
      controls={
        <>
          <ParamSlider label="pull θ" param={theta} />
          <ParamSlider label="noise σ" param={sigma} />
          <ParamSlider label="start x₀" param={x0} />
        </>
      }
      readout={
        <>
          <Readout label="half-life ln 2 / θ" value={formatNumber(Math.log(2) / th)} />
          <Readout label="stationary sd σ/√(2θ)" value={formatNumber(sg / Math.sqrt(2 * th))} />
          <Readout label="mean at t = 1" value={formatNumber(x0.value * Math.exp(-th))} />
          <Readout
            label="sd at t = 1"
            value={formatNumber(Math.sqrt(((sg * sg) / (2 * th)) * (1 - Math.exp(-2 * th))))}
          />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="t"
        yLabel="X_t"
        series={series}
        xRange={[0, T]}
        yRange={[-5, 5]}
        handles={[{ kind: 'point', at: [0, x0.value], label: 'x₀', onDrag: ([, y]) => x0.set(y) }]}
      />
    </Interactive>
  )
}
