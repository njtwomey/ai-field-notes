import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { gaussPdf } from '../_shared/sde'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const XS = toFlat(linspace(-5, 5, 501))
const DX = XS[1] - XS[0]
const PI = [0.4, 0.6]
const TAU = 0.3

type View = 'densities' | 'denoiser'

/**
 * A prior that is a mixture of two narrow Gaussians, one noisy observation y = x + σε, the exact posterior computed on a
 * grid, and Tweedie's formula y + σ² d/dy log p(y) computed from the marginal alone. The two posterior means agree.
 */
export function TweedieDenoiser() {
  const state = useFigureState({
    view: choice<View>(
      [
        { value: 'densities', label: 'densities' },
        { value: 'denoiser', label: 'denoiser' },
      ],
      'densities',
      { label: 'view' },
    ),
    sigma: float(0.8, { min: 0.3, max: 2, step: 0.05, label: 'noise σ' }),
    y: float(0.6, { min: -4.5, max: 4.5, step: 0.05, label: 'observation y' }),
    mu1: slider(-4, 4, -1.5, { step: 0.05, onChart: true }),
    mu2: slider(-4, 4, 1.5, { step: 0.05, onChart: true }),
  })
  const mus = useMemo(() => [state.mu1, state.mu2], [state.mu1, state.mu2])
  const s2 = state.sigma ** 2

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
    const un = prior.map((p, i) => p * gaussPdf(state.y, XS[i], s2))
    const z = un.reduce((a, b) => a + b, 0) * DX
    const dens = un.map((u) => u / z)
    const mean = dens.reduce((a, d, i) => a + d * XS[i], 0) * DX
    const sd = Math.sqrt(dens.reduce((a, d, i) => a + d * (XS[i] - mean) ** 2, 0) * DX)
    return { dens, mean, sd }
  }, [prior, state.y, s2])

  const tweedie = state.y + s2 * marginal(state.y).score

  const densitySeries = useMemo(
    () =>
      [
        { name: 'prior p(x)', x: XS, y: prior, muted: true },
        { name: 'noisy marginal p(y)', x: XS, y: XS.map((v) => marginal(v).p), slot: 1 },
        { name: 'posterior p(x | y)', x: XS, y: post.dens, slot: 0 },
      ] as const,
    [prior, marginal, post.dens],
  )

  const denoiserSeries = useMemo(
    () =>
      [
        { name: 'identity: no denoising', x: XS, y: XS, muted: true, dashed: true },
        {
          name: 'E[x | y] = y + σ² d/dy log p(y)',
          x: XS,
          y: XS.map((v) => v + s2 * marginal(v).score),
          slot: 0,
        },
        { name: 'observation', x: [state.y], y: [tweedie], emphasis: true },
      ] as const,
    [marginal, s2, state.y, tweedie],
  )

  // The arrow sits just above the tallest curve; the axis grows with the posterior as σ shrinks.
  const top = Math.max(1, ...post.dens, ...prior) * 1.08
  const xAxis = useAxis({ label: 'x or y', range: [-5, 5] })
  const yAxis = useAxis({ label: 'density', range: [0, Math.ceil(top * 5.5) / 5] })
  const xAxis2 = useAxis({ label: 'observation y', range: [-5, 5] })
  const yAxis2 = useAxis({ label: 'posterior mean E[x | y]', range: [-5, 5] })
  return (
    <Figure
      title="Tweedie's formula: the posterior mean from the noisy density alone"
      state={state}
      caption="A signal x is drawn from a prior with two narrow bumps (40% and 60%) and observed with Gaussian noise, y = x + σε. The posterior p(x | y) is computed on a grid with Bayes' rule. Tweedie's formula gets its mean from the density of y alone: the arrow starts at y and moves by σ² times the slope of log p(y). Drag y or either prior bump. The denoiser view plots E[x | y] against y: it is pulled towards the bumps, and more strongly for larger σ."

      readouts={
        <>
          <Readout label="posterior mean (Bayes)" value={formatNumber(post.mean)} />
          <Readout label="y + σ² d/dy log p(y)" value={formatNumber(tweedie)} />
          <Readout label="posterior sd" value={formatNumber(post.sd)} />
        </>
      }
    >
      {state.view === 'densities' ? (
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...densitySeries[0]} />
          <Curve {...densitySeries[1]} />
          <Area {...densitySeries[2]} />
          <Vectors vectors={[{ from: [state.y, top], to: [tweedie, top] }]} />
          <Handle {...state.handle('y', { label: 'y' })} />
          <Handle {...state.handle('mu1', { label: 'bump 1' })} />
          <Handle {...state.handle('mu2', { label: 'bump 2' })} />
        </Plot>
      ) : (
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...denoiserSeries[0]} />
          <Curve {...denoiserSeries[1]} />
          <Points {...denoiserSeries[2]} />
          <Handle {...state.handle('y', { label: 'y' })} />
        </Plot>
      )}
    </Figure>
  )
}
