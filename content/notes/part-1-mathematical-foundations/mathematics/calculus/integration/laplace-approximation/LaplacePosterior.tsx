import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { erf, logGamma } from '@/lib/math/special'

const GRID = linspace(0.0005, 0.9995, 600)
const erfc = (x: number) => 1 - erf(x)
const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b)

/**
 * A coin's exact beta posterior against its Laplace approximation, a Gaussian at the MAP estimate with variance equal to
 * the inverse curvature of the negative log posterior there.
 */
export function LaplacePosterior() {
  const [heads, setHeads] = useState(7)
  const [tails, setTails] = useState(3)
  const [alpha, setAlpha] = useState(1)
  const [beta, setBeta] = useState(1)

  const r = useMemo(() => {
    const a = alpha + heads
    const b = beta + tails
    const exact = GRID.map((t) => Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log1p(-t) - logBeta(a, b)))
    const mean = a / (a + b)
    const sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
    // The mode is interior only when both exponents are positive; otherwise the maximum sits on the boundary.
    if (a <= 1 || b <= 1) return { exact, mean, sd, laplace: null }
    const mode = (a - 1) / (a + b - 2)
    const curvature = (a - 1) / mode ** 2 + (b - 1) / (1 - mode) ** 2
    const lsd = 1 / Math.sqrt(curvature)
    const gauss = GRID.map((t) => Math.exp(-0.5 * ((t - mode) / lsd) ** 2) / (lsd * Math.sqrt(2 * Math.PI)))
    // Evidence p(D) = B(a, b) / B(α, β), exactly and by Laplace: ℓ(mode) + ½ log(2π σ²) − log B(α, β).
    const logEvidence = logBeta(a, b) - logBeta(alpha, beta)
    const logLaplace =
      (a - 1) * Math.log(mode) +
      (b - 1) * Math.log1p(-mode) +
      0.5 * Math.log(2 * Math.PI * lsd * lsd) -
      logBeta(alpha, beta)
    const massOutside = 0.5 * (erfc(mode / (lsd * Math.SQRT2)) + erfc((1 - mode) / (lsd * Math.SQRT2)))
    return { exact, mean, sd, laplace: { gauss, mode, lsd, ratio: Math.exp(logLaplace - logEvidence), massOutside } }
  }, [heads, tails, alpha, beta])

  const series: XYSeries[] = [
    { name: 'exact posterior', type: 'line', x: GRID, y: r.exact, slot: 0, area: true },
    ...(r.laplace
      ? [{ name: 'Laplace approximation', type: 'line' as const, x: GRID, y: r.laplace.gauss, slot: 1, dashed: true }]
      : []),
  ]

  return (
    <Interactive
      title="Exact posterior and its Laplace approximation"
      caption="A coin's bias θ after the given heads and tails, under a beta prior. The shaded curve is the exact beta posterior; the dashed curve is the Gaussian at its mode with variance from the curvature there. With few flips near 0 or 1 the posterior is skewed and the Gaussian spills outside [0, 1]; as the flips grow the two curves merge and the evidence ratio approaches 1. With no heads or no tails and a flat prior, the mode is on the boundary and there is no Laplace approximation."
      controls={
        <>
          <ParamSlider label="heads" value={heads} onChange={setHeads} min={0} max={200} step={1} />
          <ParamSlider label="tails" value={tails} onChange={setTails} min={0} max={200} step={1} />
          <ParamSlider label="prior α" value={alpha} onChange={setAlpha} min={0.5} max={10} step={0.5} />
          <ParamSlider label="prior β" value={beta} onChange={setBeta} min={0.5} max={10} step={0.5} />
        </>
      }
      readout={
        <>
          <Readout label="exact mean, sd" value={`${formatNumber(r.mean)}, ${formatNumber(r.sd)}`} />
          <Readout
            label="Laplace mean, sd"
            value={r.laplace ? `${formatNumber(r.laplace.mode)}, ${formatNumber(r.laplace.lsd)}` : 'mode on boundary'}
          />
          <Readout label="evidence, Laplace / exact" value={r.laplace ? formatNumber(r.laplace.ratio) : '—'} />
          <Readout
            label="Laplace mass outside [0, 1]"
            value={r.laplace ? `${(100 * r.laplace.massOutside).toFixed(2)}%` : '—'}
          />
        </>
      }
    >
      <XYChart series={series} xLabel="θ" yLabel="density" xRange={[0, 1]} yRange={[0, undefined]} />
    </Interactive>
  )
}
