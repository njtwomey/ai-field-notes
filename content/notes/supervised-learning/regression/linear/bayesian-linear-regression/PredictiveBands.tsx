import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { effectiveParameters, logEvidence, maximiseEvidence, parameterVariance, posterior } from './blr'

const MAX_POINTS = 25
/** Noise sd of the generated data; the true noise precision is 1/0.2² = 25. */
const NOISE = 0.2
/** Width of each Gaussian basis function. */
const WIDTH = 0.1
const GRID = linspace(0, 1, 121)

/** A constant plus `m` Gaussian bumps with centres spread evenly over [0, 1]. */
function features(x: number, m: number): number[] {
  const centres = m === 1 ? [0.5] : linspace(0, 1, m)
  return [1, ...centres.map((c) => Math.exp(-((x - c) ** 2) / (2 * WIDTH * WIDTH)))]
}

const fmtLog = (v: number) => formatNumber(10 ** v)

/**
 * The posterior predictive of a Gaussian-basis model fitted to noisy samples of sin(2πx): its mean, a band of two
 * predictive standard deviations, and the narrower band the noise alone would give. The button sets α and β to the
 * values that maximise the evidence.
 */
export function PredictiveBands() {
  const n = useParam(8, { min: 1, max: MAX_POINTS, step: 1 })
  const m = useParam(9, { min: 1, max: 12, step: 1 })
  const logAlpha = useParam(0, { min: -3, max: 3, step: 0.05 })
  const logBeta = useParam(1, { min: -1, max: 3, step: 0.05 })
  const [seed, setSeed] = useState(3)
  const alpha = 10 ** logAlpha.value
  const beta = 10 ** logBeta.value

  const data = useMemo(() => {
    const g = rng(seed)
    return Array.from({ length: MAX_POINTS }, () => {
      const x = g.uniform()
      return { x, y: Math.sin(2 * Math.PI * x) + NOISE * g.normal() }
    }).slice(0, n.value)
  }, [seed, n.value])

  const r = useMemo(() => {
    const phi = data.map((d) => features(d.x, m.value))
    const y = data.map((d) => d.y)
    const post = posterior(phi, y, alpha, beta, m.value + 1)
    const mean: number[] = []
    const sd: number[] = []
    for (const x of GRID) {
      const f = features(x, m.value)
      mean.push(f.reduce((s, v, i) => s + v * post.mean[i], 0))
      sd.push(Math.sqrt(1 / beta + parameterVariance(post, f)))
    }
    return {
      phi,
      y,
      mean,
      sd,
      evidence: logEvidence(phi, y, alpha, beta, post),
      gamma: effectiveParameters(phi, y, alpha, beta),
    }
  }, [data, m.value, alpha, beta])

  const maximise = () => {
    const best = maximiseEvidence(r.phi, r.y, alpha, beta)
    logAlpha.set(Math.log10(best.alpha))
    logBeta.set(Math.log10(best.beta))
  }

  const noise = 2 / Math.sqrt(beta)
  const series: XYSeries[] = [
    { name: 'sin 2πx', type: 'line', x: GRID, y: GRID.map((x) => Math.sin(2 * Math.PI * x)), slot: 2, dashed: true },
    { name: 'noise only (±2/√β)', type: 'line', x: GRID, y: r.mean.map((v) => v + noise), muted: true },
    { name: 'noise only (±2/√β)', type: 'line', x: GRID, y: r.mean.map((v) => v - noise), muted: true },
    {
      name: 'predictive ±2 sd',
      type: 'line',
      x: GRID,
      y: r.mean.map((v, i) => v + 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'predictive ±2 sd',
      type: 'line',
      x: GRID,
      y: r.mean.map((v, i) => v - 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'predictive mean', type: 'line', x: GRID, y: r.mean, slot: 0 },
    { name: 'data', type: 'scatter', x: data.map((d) => d.x), y: data.map((d) => d.y), slot: 1 },
  ]

  return (
    <Interactive
      title="The posterior predictive"
      caption="Noisy samples of sin 2πx fitted with a constant plus Gaussian bumps of width 0.1. The dashed blue band is two predictive standard deviations, noise plus parameter uncertainty; the grey band is the noise alone. Away from the data the parameter term dominates and the band widens. Few points and a weak prior (small α) overfit; a strong prior flattens the fit. Maximise the evidence to let the data choose α and β."
      controls={
        <>
          <ParamSlider label="points N" param={n} format={(v) => String(v)} />
          <ParamSlider label="bumps" param={m} format={(v) => String(v)} />
          <ParamSlider label="prior precision α" param={logAlpha} format={fmtLog} />
          <ParamSlider label="noise precision β" param={logBeta} format={fmtLog} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={20} step={1} />
          <ParamButton onClick={maximise}>Maximise evidence</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="ln p(y | α, β)" value={formatNumber(r.evidence)} />
          <Readout label="effective parameters γ" value={formatNumber(r.gamma)} />
          <Readout label="α, β" value={`${formatNumber(alpha)}, ${formatNumber(beta)}`} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="y" xRange={[0, 1]} yRange={[-2, 2]} height={380} />
    </Interactive>
  )
}
