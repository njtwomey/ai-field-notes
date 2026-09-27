import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { makeKernel, samples, gram } from '../../_shared/gp'
import { hgpLogMarginal, hgpPosterior, type Replicate } from '../_shared/hgp'

const GRID = linspace(0, 10, 101)
const X_RANGE: [number, number] = [0, 10]
const Y_RANGE: [number | undefined, number | undefined] = [-3, 3]
const N_REP = 4
/** Replicate 4 is observed only on [0, 5], so its right half must be borrowed from the shared function. */
const COVERAGE: [number, number][] = [
  [0, 10],
  [0, 10],
  [0, 10],
  [0, 5],
]
const trueG = (x: number) => Math.sin(1.1 * x) + 0.5 * Math.cos(0.45 * x + 1)

/** Four replicates of one latent curve: the shared g, a smooth Matérn 3/2 deviation per replicate, and noise. */
const DATA: Replicate[] = (() => {
  const r = rng(7)
  const dev = makeKernel('matern32', { ell: 1.5, sf: 0.4 })
  return COVERAGE.map(([a, b]) => {
    const x = Array.from({ length: 12 }, () => a + (b - a) * r.uniform()).sort((p, q) => p - q)
    const z = x.map(() => r.normal())
    const [h] = samples(
      x.map(() => 0),
      gram(dev, x, x),
      [z],
    )
    return { x, y: x.map((xi, i) => trueG(xi) + h[i] + 0.1 * r.normal()) }
  })
})()

const REPLICATE_OPTIONS = Array.from({ length: N_REP }, (_, i) => ({ value: String(i), label: `${i + 1}` }))

type Theta = { sg: number; logEllG: number; sf: number; logEllF: number; sn: number }

const kernels = ({ sg, logEllG, sf, logEllF }: Theta) => ({
  kg: makeKernel('se', { ell: 10 ** logEllG, sf: sg }),
  kf: makeKernel('matern32', { ell: 10 ** logEllF, sf }),
})

const lml = (t: Theta) => {
  const { kg, kf } = kernels(t)
  return hgpLogMarginal(kg, kf, t.sn * t.sn, DATA)
}

/** Compass search on the log marginal likelihood over the five hyperparameters, from the current values. */
function maximise(start: Theta): Theta {
  const keys = ['sg', 'logEllG', 'sf', 'logEllF', 'sn'] as const
  const logScale = { sg: true, logEllG: false, sf: true, logEllF: false, sn: true }
  // A replicate sd of exactly zero would stay at zero under multiplicative steps.
  let best = { ...start, sf: Math.max(start.sf, 0.05) }
  let bestValue = lml(best)
  for (let step = 0.4; step > 0.01; step /= 2) {
    let improved = true
    while (improved) {
      improved = false
      for (const key of keys) {
        for (const dir of [-1, 1]) {
          const v = best[key]
          const next = { ...best, [key]: logScale[key] ? v * Math.exp(dir * step) : v + dir * step }
          const value = lml(next)
          if (value > bestValue + 1e-9) {
            best = next
            bestValue = value
            improved = true
          }
        }
      }
    }
  }
  return best
}

/**
 * Posterior of the shared function g and of one replicate f_j under a two-level hierarchical GP, with the five
 * hyperparameters on sliders and a button that maximises the marginal likelihood.
 */
export function HgpReplicates() {
  const sg = useParam(1, { min: 0.1, max: 2, step: 0.01 })
  const logEllG = useParam(0, { min: -0.5, max: 1, step: 0.01 })
  const sf = useParam(0.4, { min: 0, max: 1.5, step: 0.01 })
  const logEllF = useParam(0.2, { min: -0.7, max: 1, step: 0.01 })
  const sn = useParam(0.1, { min: 0.02, max: 0.8, step: 0.01 })
  const [focus, setFocus] = useState('3')
  const j = Number(focus)
  const theta: Theta = {
    sg: sg.value,
    logEllG: logEllG.value,
    sf: sf.value,
    logEllF: logEllF.value,
    sn: sn.value,
  }

  const post = useMemo(() => {
    const { kg, kf } = kernels({
      sg: sg.value,
      logEllG: logEllG.value,
      sf: sf.value,
      logEllF: logEllF.value,
      sn: sn.value,
    })
    return hgpPosterior(kg, kf, sn.value ** 2, DATA, GRID)
  }, [sg.value, logEllG.value, sf.value, logEllF.value, sn.value])

  const fit = () => {
    const best = maximise(theta)
    sg.set(best.sg)
    logEllG.set(best.logEllG)
    sf.set(best.sf)
    logEllF.set(best.logEllF)
    sn.set(best.sn)
  }

  const gSd = post.g.variance.map(Math.sqrt)
  const fj = post.f[j]
  const fSd = fj.variance.map(Math.sqrt)
  const others = DATA.filter((_, i) => i !== j)
  const series: XYSeries[] = [
    {
      name: 'other replicates',
      type: 'scatter',
      x: others.flatMap((r) => r.x),
      y: others.flatMap((r) => r.y),
      muted: true,
    },
    {
      name: 'g ± 2 sd',
      type: 'line',
      x: GRID,
      y: post.g.mean.map((m, i) => m + 2 * gSd[i]),
      muted: true,
      dashed: true,
    },
    {
      name: 'g − 2 sd',
      type: 'line',
      x: GRID,
      y: post.g.mean.map((m, i) => m - 2 * gSd[i]),
      muted: true,
      dashed: true,
    },
    { name: 'shared g (mean)', type: 'line', x: GRID, y: post.g.mean, emphasis: true },
    {
      name: `f${j + 1} ± 2 sd`,
      type: 'line',
      x: GRID,
      y: fj.mean.map((m, i) => m + 2 * fSd[i]),
      slot: j,
      dashed: true,
    },
    {
      name: `f${j + 1} − 2 sd`,
      type: 'line',
      x: GRID,
      y: fj.mean.map((m, i) => m - 2 * fSd[i]),
      slot: j,
      dashed: true,
    },
    { name: `replicate f${j + 1} (mean)`, type: 'line', x: GRID, y: fj.mean, slot: j },
    { name: `replicate ${j + 1} data`, type: 'scatter', x: DATA[j].x, y: DATA[j].y, slot: j },
  ]

  const s2g = sg.value ** 2
  const s2f = sf.value ** 2
  const s2n = sn.value ** 2
  return (
    <Interactive
      title="Replicates sharing one latent function"
      caption="Four replicates of one curve, twelve noisy points each; replicate 4 is observed only on [0, 5]. The ink line is the posterior mean of the shared function g (squared exponential kernel k_g), with dashed grey lines two posterior standard deviations either side. The coloured line and band are the posterior of the chosen replicate f_j (Matérn 3/2 deviation kernel k_f). With σ_f = 0 every replicate equals g (complete pooling). A large σ_f relative to σ_n lets f_j follow its own points and shrinks it toward g only where it has no data, as on the right half of replicate 4. The button maximises ln p(ŷ) over all five sliders."
      controls={
        <>
          <ParamChoice label="replicate shown" value={focus} onChange={setFocus} options={REPLICATE_OPTIONS} />
          <ParamSlider label="shared sd σ_g" param={sg} />
          <ParamSlider label="shared length-scale ℓ_g" param={logEllG} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="replicate sd σ_f" param={sf} />
          <ParamSlider label="replicate length-scale ℓ_f" param={logEllF} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="noise sd τ^(−1/2)" param={sn} />
          <ParamButton onClick={fit}>Maximise marginal likelihood</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="ln p(ŷ)" value={formatNumber(post.logMarginal)} />
          <Readout label="shared share σ_g²/(σ_g²+σ_f²+τ⁻¹)" value={formatNumber(s2g / (s2g + s2f + s2n))} />
          <Readout label="weight on own residual σ_f²/(σ_f²+τ⁻¹)" value={formatNumber(s2f / (s2f + s2n))} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="y" xRange={X_RANGE} yRange={Y_RANGE} height={380} />
    </Interactive>
  )
}
