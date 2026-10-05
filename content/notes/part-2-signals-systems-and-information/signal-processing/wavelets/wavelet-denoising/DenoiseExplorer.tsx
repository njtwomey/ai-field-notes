import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { FILTERS, wavedec, waverec, type Family } from '../_shared/wavelets'
import { normal, stream } from 'aifn-compute/foundation/random'

type Rule = 'soft' | 'hard'

const N = 1024
const LEVELS = 6
const T = Array.from({ length: N }, (_, n) => n / N)
// Donoho and Johnstone's HeaviSine: a sinusoid with two jumps.
const TRUTH = T.map((t) => 4 * Math.sin(4 * Math.PI * t) - Math.sign(t - 0.3) - Math.sign(0.72 - t))
const NOISE = (() => {
  const g = stream(11)
  return Array.from({ length: N }, () => normal(g))
})()

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b)
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2
}

/**
 * Wavelet shrinkage: transform, threshold the detail coefficients, invert. Noise spreads evenly over all coefficients,
 * while a piecewise-smooth signal concentrates in a few large ones, so a threshold removes mostly noise.
 */
export function DenoiseExplorer() {
  const state = useFigureState({
    sigma: float(0.5, { min: 0.1, max: 1.5, step: 0.05, label: 'noise σ' }),
    multiplier: slider(0, 2, 1, { step: 0.05, label: 'threshold (× universal)' }),
    rule: choice<Rule>(
      [
        { value: 'soft', label: 'soft' },
        { value: 'hard', label: 'hard' },
      ],
      'soft',
      { label: 'rule' },
    ),
    family: choice<Family>(
      [
        { value: 'haar', label: 'Haar' },
        { value: 'db4', label: 'db4' },
      ],
      'db4',
      { label: 'wavelet' },
    ),
  })

  const r = useMemo(() => {
    const y = TRUTH.map((v, n) => v + state.sigma * NOISE[n])
    const h = FILTERS[state.family]
    const { approx, details } = wavedec(y, h, LEVELS)
    // Noise level from the finest details: median absolute deviation / 0.6745.
    const sigmaHat = median(Array.from(details[0], Math.abs)) / 0.6745
    const lambda = state.multiplier * sigmaHat * Math.sqrt(2 * Math.log(N))
    let kept = 0
    const shrink = (d: Float64Array) =>
      d.map((v) => {
        const a = Math.abs(v)
        if (a <= lambda) return 0
        kept++
        return state.rule === 'hard' ? v : Math.sign(v) * (a - lambda)
      })
    const cleaned = waverec(approx, details.map(shrink), h)
    const mse = (v: ArrayLike<number>) => TRUTH.reduce((s, t, n) => s + (v[n] - t) ** 2, 0) / N
    const total = details.reduce((s, d) => s + d.length, 0)
    return { y, cleaned: Array.from(cleaned), sigmaHat, lambda, mseNoisy: mse(y), mseClean: mse(cleaned), kept, total }
  }, [state.sigma, state.multiplier, state.rule, state.family])

  const series = [
    { name: 'noisy', x: T, y: r.y, muted: true },
    { name: 'denoised', x: T, y: r.cleaned, slot: 0 },
    { name: 'true signal', x: T, y: TRUTH, slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 't', range: [0, 1] })
  const yAxis = useAxis({ label: 'value', range: [-8, 6] })
  return (
    <Figure
      title="Thresholding wavelet coefficients"
      state={state}
      caption="HeaviSine, a sinusoid with two jumps, plus white Gaussian noise; 1,024 samples and a 6-level orthogonal DWT. Detail coefficients smaller than the threshold λ are set to zero (hard) or shrunk toward zero by λ (soft); the approximation is kept. λ is a multiple of the universal threshold σ̂√(2 ln N), with σ̂ estimated from the finest details by the median absolute deviation. At multiplier 0 nothing is removed; at 1, soft thresholding gives a visibly smooth estimate that keeps the jumps sharp."

      readouts={
        <>
          <Readout label="σ̂ (MAD)" value={formatNumber(r.sigmaHat)} />
          <Readout label="λ" value={formatNumber(r.lambda)} />
          <Readout label="details kept" value={`${r.kept} of ${r.total}`} />
          <Readout label="MSE noisy" value={formatNumber(r.mseNoisy)} />
          <Readout label="MSE denoised" value={formatNumber(r.mseClean)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
