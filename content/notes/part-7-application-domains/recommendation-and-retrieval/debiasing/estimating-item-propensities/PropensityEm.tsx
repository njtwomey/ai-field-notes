import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { logGamma } from '@/lib/math/special'

/** Click counts C_ui for 4 users × 5 items. User 3 clicked item 5 six times, far above what the margins predict. */
const C = [
  [4, 2, 1, 1, 0],
  [6, 3, 2, 1, 1],
  [2, 1, 0, 0, 6],
  [8, 4, 2, 1, 1],
]
const U = C.length
const M = C[0].length
const ROW = C.map((r) => r.reduce((a, b) => a + b, 0))
const COL = C[0].map((_, i) => C.reduce((s, r) => s + r[i], 0))
const USERS = [1, 2, 3, 4]
const ITEMS = [1, 2, 3, 4, 5]
const ITERS = 40

type State = { alpha: number[]; beta: number[] }

const rescale = ({ alpha, beta }: State): State => {
  const m = alpha.reduce((a, b) => a + b, 0) / U
  return { alpha: alpha.map((x) => x / m), beta: beta.map((x) => x * m) }
}

/** Model A: the rank-1 Poisson maximum-likelihood fit, α_u = C_u· / Σβ and β_i = C_·i / Σα, from α = β = 1. */
function ipf(): State {
  let beta = new Array(M).fill(1)
  const alpha = ROW.map((r) => r / beta.reduce((a, b) => a + b, 0))
  beta = COL.map((c) => c / alpha.reduce((a, b) => a + b, 0))
  return rescale({ alpha, beta })
}

const posteriorMean = (s: State, a: number) =>
  C.map((row, u) => row.map((c, i) => (a + c) / (a + s.alpha[u] * s.beta[i])))

/** Model B: EM for C_ui | r_ui ~ Poisson(α_u β_i r_ui), r_ui ~ Gamma(a, a). Returns the state after each iteration. */
function emHistory(a: number): State[] {
  const out = [ipf()]
  for (let t = 0; t < ITERS; t++) {
    const s = out[out.length - 1]
    const Er = posteriorMean(s, a)
    const alpha = ROW.map((r, u) => r / s.beta.reduce((acc, b, i) => acc + b * Er[u][i], 0))
    const beta = COL.map((c, i) => c / alpha.reduce((acc, al, u) => acc + al * Er[u][i], 0))
    out.push(rescale({ alpha, beta }))
  }
  return out
}

/** Marginal log-likelihood under the negative binomial obtained by integrating out r_ui. */
function marginalLogLik(s: State, a: number): number {
  let ll = 0
  C.forEach((row, u) =>
    row.forEach((c, i) => {
      const lam = s.alpha[u] * s.beta[i]
      ll +=
        logGamma(a + c) - logGamma(a) - logGamma(c + 1) + a * Math.log(a / (a + lam)) + c * Math.log(lam / (a + lam))
    }),
  )
  return ll
}

const normalise = (beta: number[]) => {
  const s = beta.reduce((a, b) => a + b, 0)
  return beta.map((b) => b / s)
}

/** IPF against Gamma–Poisson EM on a small click matrix: per-user item propensities and posterior relevance. */
export function PropensityEm() {
  const logA = useParam(0, { min: -0.6, max: 2, step: 0.05 })
  const iter = useParam(ITERS, { min: 0, max: ITERS, step: 1 })
  const a = 10 ** logA.value
  const hist = useMemo(() => emHistory(a), [a])
  const base = hist[0]
  const s = hist[iter.value]
  const Er = useMemo(() => posteriorMean(s, a), [s, a])
  const logEr = useMemo(() => Er.map((row) => row.map(Math.log)), [Er])

  const series: XYSeries[] = [
    { name: 'IPF (model A)', type: 'line', x: ITEMS, y: normalise(base.beta), slot: 0 },
    { name: `EM (model B), iteration ${iter.value}`, type: 'line', x: ITEMS, y: normalise(s.beta), slot: 1 },
  ]

  return (
    <Interactive
      title="Exposure baseline and latent relevance"
      caption="Left: the per-user item propensity π̂_ui = β_i / Σ_j β_j from the rank-1 Poisson fit (IPF) and from Gamma–Poisson EM. Right: the logarithm of the posterior mean relevance E[r_ui], so 0 means average relevance. User 3's six clicks on item 5 are attributed to relevance (E[r] > 1), and the other users' few clicks on item 5 to low relevance, so EM raises item 5's exposure baseline. A large shrinkage a forces every r_ui towards 1 and recovers IPF; a small a lets pairwise relevance absorb more. Step the EM iterations with the arrows."
      controls={
        <>
          <ParamSlider
            label="log₁₀ shrinkage a"
            param={logA}
            format={(v) => `${v.toFixed(2)} (a = ${formatNumber(10 ** v)})`}
          />
          <ParamSlider label="EM iteration" param={iter} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="marginal log-likelihood, IPF" value={formatNumber(marginalLogLik(base, a))} />
          <Readout label="at this iteration" value={formatNumber(marginalLogLik(s, a))} />
          <Readout label="E[r] for user 3, item 5" value={formatNumber(Er[2][4])} />
          <Readout
            label="π̂ for item 5, IPF → EM"
            value={`${formatNumber(normalise(base.beta)[4])} → ${formatNumber(normalise(s.beta)[4])}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={series}
          xLabel="item"
          yLabel="per-user propensity π̂"
          xRange={[1, 5]}
          yRange={[0, 0.5]}
          height={280}
        />
        <Heatmap
          x={ITEMS}
          y={USERS}
          z={logEr}
          scale="diverging"
          range={[-1.2, 1.2]}
          xLabel="item"
          yLabel="user"
          valueLabel="ln E[r_ui]"
          height={280}
        />
      </div>
    </Interactive>
  )
}
