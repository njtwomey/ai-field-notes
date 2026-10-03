import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { gaussPdf } from '../_shared/sde'

const XS = linspace(-5, 5, 501)
const DX = XS[1] - XS[0]
const PI = [0.4, 0.6]
const TAU = 0.3

type View = 'densities' | 'denoiser'

/**
 * A prior that is a mixture of two narrow Gaussians, one noisy observation y = x + σε, the exact posterior computed on a
 * grid, and Tweedie's formula y + σ² d/dy log p(y) computed from the marginal alone. The two posterior means agree.
 */
export function TweedieDenoiser() {
  const mu1 = useParam(-1.5, { min: -4, max: 4, step: 0.05 })
  const mu2 = useParam(1.5, { min: -4, max: 4, step: 0.05 })
  const y = useParam(0.6, { min: -4.5, max: 4.5, step: 0.05 })
  const sigma = useParam(0.8, { min: 0.3, max: 2, step: 0.05 })
  const [view, setView] = useState<View>('densities')
  const mus = useMemo(() => [mu1.value, mu2.value], [mu1.value, mu2.value])
  const s2 = sigma.value ** 2

  /** Marginal density of y and its score, analytic: p(y) = Σ π_k N(y; μ_k, τ² + σ²). */
  const marginal = useMemo(
    () => (v: number) => {
      let p = 0
      let dp = 0
      for (let k = 0; k < 2; k++) {
        const w = PI[k] * gaussPdf(v, mus[k], TAU ** 2 + s2)
        p += w
        dp += (w * (mus[k] - v)) / (TAU ** 2 + s2)
      }
      return { p, score: dp / p }
    },
    [mus, s2],
  )

  const prior = useMemo(
    () => XS.map((x) => PI[0] * gaussPdf(x, mus[0], TAU ** 2) + PI[1] * gaussPdf(x, mus[1], TAU ** 2)),
    [mus],
  )

  const post = useMemo(() => {
    const un = prior.map((p, i) => p * gaussPdf(y.value, XS[i], s2))
    const z = un.reduce((a, b) => a + b, 0) * DX
    const dens = un.map((u) => u / z)
    const mean = dens.reduce((a, d, i) => a + d * XS[i], 0) * DX
    const sd = Math.sqrt(dens.reduce((a, d, i) => a + d * (XS[i] - mean) ** 2, 0) * DX)
    return { dens, mean, sd }
  }, [prior, y.value, s2])

  const tweedie = y.value + s2 * marginal(y.value).score

  const densitySeries = useMemo<XYSeries[]>(
    () => [
      { name: 'prior p(x)', type: 'line', x: XS, y: prior, muted: true },
      { name: 'noisy marginal p(y)', type: 'line', x: XS, y: XS.map((v) => marginal(v).p), slot: 1 },
      { name: 'posterior p(x | y)', type: 'line', x: XS, y: post.dens, slot: 0, area: true },
    ],
    [prior, marginal, post.dens],
  )

  const denoiserSeries = useMemo<XYSeries[]>(
    () => [
      { name: 'identity: no denoising', type: 'line', x: XS, y: XS, muted: true, dashed: true },
      {
        name: 'E[x | y] = y + σ² d/dy log p(y)',
        type: 'line',
        x: XS,
        y: XS.map((v) => v + s2 * marginal(v).score),
        slot: 0,
      },
      { name: 'observation', type: 'scatter', x: [y.value], y: [tweedie], emphasis: true },
    ],
    [marginal, s2, y.value, tweedie],
  )

  // The arrow sits just above the tallest curve; the axis grows with the posterior as σ shrinks.
  const top = Math.max(1, ...post.dens, ...prior) * 1.08
  return (
    <Interactive
      title="Tweedie's formula: the posterior mean from the noisy density alone"
      caption="A signal x is drawn from a prior with two narrow bumps (40% and 60%) and observed with Gaussian noise, y = x + σε. The posterior p(x | y) is computed on a grid with Bayes' rule. Tweedie's formula gets its mean from the density of y alone: the arrow starts at y and moves by σ² times the slope of log p(y). Drag y or either prior bump. The denoiser view plots E[x | y] against y: it is pulled towards the bumps, and more strongly for larger σ."
      controls={
        <>
          <ParamChoice
            label="view"
            value={view}
            onChange={setView}
            options={[
              { value: 'densities', label: 'densities' },
              { value: 'denoiser', label: 'denoiser' },
            ]}
          />
          <ParamSlider label="noise σ" param={sigma} />
          <ParamSlider label="observation y" param={y} />
        </>
      }
      readout={
        <>
          <Readout label="posterior mean (Bayes)" value={formatNumber(post.mean)} />
          <Readout label="y + σ² d/dy log p(y)" value={formatNumber(tweedie)} />
          <Readout label="posterior sd" value={formatNumber(post.sd)} />
        </>
      }
    >
      {view === 'densities' ? (
        <XYChart
          height={320}
          xLabel="x or y"
          yLabel="density"
          series={densitySeries}
          xRange={[-5, 5]}
          yRange={[0, Math.ceil(top * 5.5) / 5]}
          vectors={[{ from: [y.value, top], to: [tweedie, top] }]}
          handles={[
            { kind: 'x', at: y.value, label: 'y', onDrag: (v) => y.set(v) },
            { kind: 'x', at: mu1.value, label: 'bump 1', onDrag: (v) => mu1.set(v) },
            { kind: 'x', at: mu2.value, label: 'bump 2', onDrag: (v) => mu2.set(v) },
          ]}
        />
      ) : (
        <XYChart
          height={320}
          xLabel="observation y"
          yLabel="posterior mean E[x | y]"
          series={denoiserSeries}
          xRange={[-5, 5]}
          yRange={[-5, 5]}
          handles={[{ kind: 'x', at: y.value, label: 'y', onDrag: (v) => y.set(v) }]}
        />
      )}
    </Interactive>
  )
}
